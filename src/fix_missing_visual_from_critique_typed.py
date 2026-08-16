#!/usr/bin/env python3
"""
针对 gemini_video_prompt_critique_typed 中需修帧的条目（missing_visual_element、motion_state）：

  1) 在错误时间区间内均匀抽 N 帧，按 frame_001 … frame_NNN 编号保存；
  2) 问 Gemini：与「错误描述」最 align 的是哪一帧；
  3) 将该帧复制为 aligned_frame.jpg；
  4) 基于错误描述 + 任务 prompt，调用 GPT-Image-2 编辑 → edited_frame.png（默认铅笔素描风格，与身份铅笔图一致）。

不修改 gemini_video_prompt_critique*.py / extract_critique_issue_frames.py / 既有 bash。

依赖：
  export YUNWU_API_KEY=...              # Gemini 选帧（云雾）
  export YUNWU_GPT_IMAGE_API_KEY=...    # GPT-Image-2 修帧（云雾，与上分开）
  # 可选 YUNWU_GPT_IMAGE_BASE_URL=https://yunwu.ai/v1
  ffmpeg / ffprobe

示例：
  python3 fix_missing_visual_from_critique_typed.py \\
    --video .../id014.mp4 \\
    --critique-typed-json .../id014_gemini_critique_typed.json \\
    --prompt-file .../batch_in_gpt_pencil/id014/prompt.txt \\
    --out .../id014/missing_visual_fix
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 片尾 seek 留白，避免 t≈duration 时 ffmpeg 7+ 的 mjpeg 编码报错
_SEEK_END_MARGIN_SEC = 0.08

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import extract_critique_issue_frames as ecf  # noqa: E402
import gemini_video_prompt_critique as gvc  # noqa: E402
from critique_issue_types import needs_frame_fix, normalize_issue_type  # noqa: E402
from gpt_image2_edit_frame_api import (  # noqa: E402
    build_frame_reference_edit_prompt,
    build_pencil_style_transfer_prompt,
    edit_frame_image,
    make_openai_client,
    resolve_image_api_credentials,
)
from public_safety import public_path  # noqa: E402


def clamp_seek_time(t_sec: float, duration_sec: float) -> float:
    """限制在 [0, duration - margin]，避免片尾单帧抽图失败。"""
    if duration_sec <= 0:
        return max(0.0, t_sec)
    upper = max(0.0, duration_sec - _SEEK_END_MARGIN_SEC)
    return max(0.0, min(upper, t_sec))


def sample_timestamps_safe(
    t0: float, t1: float, n: int, duration_sec: float
) -> list[float]:
    """在 sample_timestamps 基础上对片尾再 clamp 一次。"""
    ts = ecf.sample_timestamps(t0, t1, n, duration_sec)
    return [clamp_seek_time(t, duration_sec) for t in ts]


def extract_frame_at_safe(
    video_path: Path,
    t_sec: float,
    out_path: Path,
    *,
    duration_sec: float,
    max_width: int = 0,
) -> float:
    """本流程专用抽帧：片尾 clamp + yuvj420p，兼容新版 ffmpeg mjpeg。"""
    t_seek = clamp_seek_time(t_sec, duration_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf_parts: list[str] = []
    if max_width > 0:
        vf_parts.append(f"scale='min({max_width},iw)':-2")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_seek:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuvj420p",
    ]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    cmd.extend(["-q:v", "2", str(out_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg 抽帧失败 @ {t_seek:.3f}s (请求 {t_sec:.3f}s): {err}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"未生成有效帧文件: {out_path}")
    return t_seek


def normalize_typed_issues(data: dict) -> list[tuple[int, dict]]:
    """返回 [(原始 differences 下标, issue_dict), ...] 需修帧的类型。"""
    raw = data.get("differences") or []
    out: list[tuple[int, dict]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            d = {"point": item.strip(), "issue_type": ""}
        elif isinstance(item, dict):
            d = dict(item)
        else:
            d = {"point": str(item).strip(), "issue_type": ""}
        if not (d.get("point") or "").strip():
            continue
        d["issue_type"] = normalize_issue_type(d.get("issue_type"))
        if needs_frame_fix(d["issue_type"]):
            out.append((i, d))
    return out


def frame_filename_1based(index: int) -> str:
    return f"frame_{index:03d}.jpg"


def build_alignment_prompt(*, point: str, n_frames: int) -> str:
    return "\n".join(
        [
            f"以下按时间顺序附带同一问题时间段内的 {n_frames} 张视频帧图，"
            f"已按先后顺序编号为第 1 帧到第 {n_frames} 帧（frame_001 … frame_{n_frames:03d}）。",
            "",
            "【错误描述】（画面相对文本缺失了哪些视觉元素）",
            point.strip(),
            "",
            "【任务】",
            "哪一帧最适合作为「补全缺失视觉元素」的编辑底图？",
            "选择标准：主体清晰可见、构图便于添加缺失的人/物/道具、遮挡较少、",
            "且该时刻最贴近错误描述所涉及的情节。",
            "",
            "只输出一个 JSON 对象：",
            f'- "best_frame_index"：整数，1 到 {n_frames}；',
            '- "rationale"：一句中文说明；',
            '- "confidence"："high" | "medium" | "low"。',
            "不要其它文字。",
        ]
    )


def clamp_frame_index(idx: int, n: int) -> int:
    try:
        v = int(idx)
    except (TypeError, ValueError):
        v = 1
    return max(1, min(n, v))


async def pick_best_frame_index(
    *,
    candidate_paths: list[Path],
    point: str,
    args: argparse.Namespace,
) -> dict:
    import ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi as ace

    n = len(candidate_paths)
    prompt = build_alignment_prompt(point=point, n_frames=n)
    media = [str(p.resolve()) for p in candidate_paths]
    result = await ace.call_qwen_json(
        prompt,
        media_paths=media,
        temperature=float(args.align_temperature),
        max_retries=args.max_retries,
        timeout=float(args.timeout),
        max_tokens=int(args.align_max_tokens),
        yunwu_text_first=True,
    )
    if not result:
        raise RuntimeError("帧对齐：未得到 JSON（检查 YUNWU_API_KEY）")
    idx = clamp_frame_index(result.get("best_frame_index", 1), n)
    return {
        "best_frame_index": idx,
        "best_frame_file": frame_filename_1based(idx),
        "rationale": str(result.get("rationale") or "").strip(),
        "confidence": str(result.get("confidence") or "").strip() or "medium",
        "n_candidates": n,
    }


async def process_one_issue(
    *,
    issue_list_index: int,
    source_diff_index: int,
    issue: dict,
    video_path: Path,
    duration: float,
    caption: str,
    issue_dir: Path,
    args: argparse.Namespace,
    openai_client,
) -> dict:
    slug = ecf.slug_issue_index(issue_list_index)
    point = str(issue.get("point") or "").strip()
    t0, t1, span_source = ecf.resolve_issue_time_span(
        issue,
        duration_sec=duration,
        uniform_frame_count=max(1, int(args.critique_uniform_frames)),
    )
    n_frames = max(2, int(args.candidate_frames))
    timestamps = sample_timestamps_safe(t0, t1, n_frames, duration)

    candidates_dir = issue_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates_meta: list[dict] = []

    for fi, t in enumerate(timestamps, start=1):
        fname = frame_filename_1based(fi)
        fpath = candidates_dir / fname
        t_actual = extract_frame_at_safe(
            video_path,
            t,
            fpath,
            duration_sec=duration,
            max_width=max(0, int(args.max_width)),
        )
        candidates_meta.append(
            {
                "frame_index": fi,
                "filename": fname,
                "time_sec": round(t_actual, 3),
                "path": str(fpath.relative_to(issue_dir)),
            }
        )
        print(f"  [{slug}] 候选 {fname} @ {t_actual:.2f}s", file=sys.stderr)

    candidate_paths = [candidates_dir / frame_filename_1based(i + 1) for i in range(len(timestamps))]

    alignment: dict
    if args.dry_run:
        alignment = {
            "best_frame_index": 1,
            "best_frame_file": frame_filename_1based(1),
            "rationale": "dry-run",
            "confidence": "low",
            "n_candidates": len(candidate_paths),
            "dry_run": True,
        }
    else:
        alignment = await pick_best_frame_index(
            candidate_paths=candidate_paths,
            point=point,
            args=args,
        )
    alignment["resolved_time_span_sec"] = [round(t0, 3), round(t1, 3)]
    alignment["time_span_source"] = span_source
    (issue_dir / "alignment.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    best_idx = int(alignment["best_frame_index"])
    aligned_src = candidates_dir / frame_filename_1based(best_idx)
    aligned_dst = issue_dir / "aligned_frame.jpg"
    shutil.copy2(aligned_src, aligned_dst)

    pencil_style = not getattr(args, "no_pencil_style", False)
    two_pass = bool(getattr(args, "pencil_two_pass", False)) and pencil_style

    issue_type = normalize_issue_type(issue.get("issue_type"))
    if two_pass:
        edit_prompt = build_frame_reference_edit_prompt(
            point=point,
            caption=caption,
            issue_type=issue_type,
            pencil_style=False,
        )
    else:
        edit_prompt = build_frame_reference_edit_prompt(
            point=point,
            caption=caption,
            issue_type=issue_type,
            pencil_style=pencil_style,
        )
    (issue_dir / "edit_prompt_en.txt").write_text(edit_prompt + "\n", encoding="utf-8")

    edited_path = issue_dir / "edited_frame.png"
    photo_path = issue_dir / "edited_frame_photo.png"
    if args.dry_run:
        print(f"  [{slug}] dry-run：跳过 GPT-Image 编辑", file=sys.stderr)
    else:
        if two_pass:
            edit_frame_image(
                openai_client,
                image_path=aligned_dst,
                prompt=edit_prompt,
                output_path=photo_path,
                image_model=args.image_model,
                size=args.size,
                quality=args.quality,
                input_fidelity=args.input_fidelity or None,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
            print(f"  [{slug}] ✅ edited_frame_photo.png（写实修帧）", file=sys.stderr)
            pencil_prompt = build_pencil_style_transfer_prompt()
            (issue_dir / "pencil_transfer_prompt_en.txt").write_text(
                pencil_prompt + "\n", encoding="utf-8"
            )
            edit_frame_image(
                openai_client,
                image_path=photo_path,
                prompt=pencil_prompt,
                output_path=edited_path,
                image_model=args.image_model,
                size=args.size,
                quality=args.quality,
                input_fidelity=args.input_fidelity or None,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
        else:
            edit_frame_image(
                openai_client,
                image_path=aligned_dst,
                prompt=edit_prompt,
                output_path=edited_path,
                image_model=args.image_model,
                size=args.size,
                quality=args.quality,
                input_fidelity=args.input_fidelity or None,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
        style_note = "铅笔" if pencil_style else "写实"
        print(f"  [{slug}] ✅ edited_frame.png（{style_note}）", file=sys.stderr)

    issue_meta = {
        "issue_list_index": issue_list_index,
        "source_difference_index": source_diff_index,
        "issue_type": issue_type,
        "point": point,
        "approx_time_span_sec": issue.get("approx_time_span_sec"),
        "basis_snapshot_range": issue.get("basis_snapshot_range"),
        "candidates": candidates_meta,
        "alignment": alignment,
        "aligned_frame": str(aligned_dst.relative_to(issue_dir)),
        "edited_frame": str(edited_path.relative_to(issue_dir)) if edited_path.is_file() else None,
        "edited_frame_style": "pencil" if pencil_style else "photo",
        "pencil_two_pass": two_pass,
        "edited_frame_photo": (
            str(photo_path.relative_to(issue_dir)) if photo_path.is_file() else None
        ),
    }
    (issue_dir / "issue.json").write_text(
        json.dumps(issue_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return issue_meta


async def _run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    critique_path = Path(args.critique_typed_json).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not video_path.is_file():
        raise SystemExit(f"找不到视频: {video_path}")
    if not critique_path.is_file():
        raise SystemExit(f"找不到 typed critique: {critique_path}")

    data = json.loads(critique_path.read_text(encoding="utf-8"))
    missing_issues = normalize_typed_issues(data)
    if not missing_issues:
        print("⚠️ 无 missing_visual_element / motion_state 条目，未处理", file=sys.stderr)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest.json").write_text(
            json.dumps(
                {
                    "video": public_path(video_path),
                    "critique_typed_json": str(critique_path),
                    "processed_issues": [],
                    "note": "no frame-fix issues (missing_visual_element / motion_state)",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    caption = ""
    if args.prompt_file:
        caption = Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8").strip()

    duration = gvc.probe_video_duration_sec(video_path)
    if duration is None or duration <= 0:
        duration = float(args.fallback_duration_sec)
        print(f"[warn] 使用 fallback 片长 {duration}s", file=sys.stderr)
    else:
        print(f"[info] 片长 ≈ {duration:.2f}s", file=sys.stderr)

    out_root.mkdir(parents=True, exist_ok=True)
    image_client = None
    if not args.dry_run:
        image_client = make_openai_client(
            base_url=args.image_api_base_url,
            api_key=args.image_api_key,
            timeout=float(args.image_api_timeout),
        )

    processed: list[dict] = []
    for list_i, (src_i, issue) in enumerate(missing_issues):
        issue_dir = out_root / ecf.slug_issue_index(list_i)
        done_flag = issue_dir / "edited_frame.png"
        if done_flag.is_file() and not args.force:
            print(f"[skip] {issue_dir.name} 已存在 edited_frame.png", file=sys.stderr)
            continue
        print(f"======== missing_visual {ecf.slug_issue_index(list_i)} (diff #{src_i}) ========", file=sys.stderr)
        meta = await process_one_issue(
            issue_list_index=list_i,
            source_diff_index=src_i,
            issue=issue,
            video_path=video_path,
            duration=duration,
            caption=caption,
            issue_dir=issue_dir,
            args=args,
            openai_client=image_client,
        )
        processed.append(meta)

    image_api_base_url: str | None = None
    if not args.dry_run:
        _, image_api_base_url = resolve_image_api_credentials(
            api_key=args.image_api_key,
            base_url=args.image_api_base_url,
        )
    manifest = {
        "video": public_path(video_path),
        "critique_typed_json": public_path(critique_path),
        "prompt_file": public_path(args.prompt_file) if args.prompt_file else None,
        "duration_sec": round(duration, 3),
        "candidate_frames_per_issue": int(args.candidate_frames),
        "image_api_base_url": image_api_base_url,
        "processed_issues": processed,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n✅ 完成 {len(processed)} 条 frame-fix → {out_root}", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="missing_visual_element / motion_state：抽帧 → 对齐 → GPT-Image-2 修帧。"
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--critique-typed-json", type=Path, required=True)
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="任务原文（如 batch_in_gpt_pencil/.../prompt.txt），写入图像编辑 prompt",
    )
    p.add_argument("--out", type=Path, required=True, help="如 .../id014/missing_visual_fix")
    p.add_argument(
        "--candidate-frames",
        type=int,
        default=int(os.environ.get("MISSING_VISUAL_CANDIDATE_FRAMES", "8")),
        help="每个 issue 在错误区间内均匀抽帧数（默认 8）",
    )
    p.add_argument(
        "--critique-uniform-frames",
        type=int,
        default=int(os.environ.get("CRITIQUE_FRAME_SCREENSHOTS", "24")),
    )
    p.add_argument("--max-width", type=int, default=0)
    p.add_argument("--fallback-duration-sec", type=float, default=14.0)
    p.add_argument("--align-temperature", type=float, default=0.1)
    p.add_argument("--align-max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--image-model", default=os.environ.get("GPT_IMAGE_MODEL", "gpt-image-2"))
    p.add_argument("--size", default=os.environ.get("GPT_IMAGE_SIZE", "auto"))
    p.add_argument("--quality", default=os.environ.get("GPT_IMAGE_QUALITY", "medium"))
    p.add_argument(
        "--input-fidelity",
        default=os.environ.get("GPT_IMAGE_INPUT_FIDELITY", "high"),
        help="gpt-image-2 的 input_fidelity，空字符串表示不传",
    )
    p.add_argument(
        "--image-api-base-url",
        "--yunwu-base-url",
        dest="image_api_base_url",
        default=None,
        help="GPT-Image 云雾基址，默认 YUNWU_GPT_IMAGE_BASE_URL 或 https://yunwu.ai/v1",
    )
    p.add_argument(
        "--image-api-key",
        "--yunwu-gpt-image-api-key",
        dest="image_api_key",
        default=None,
        help="默认环境变量 YUNWU_GPT_IMAGE_API_KEY（不用 YUNWU_API_KEY）",
    )
    p.add_argument(
        "--image-api-timeout",
        "--openai-timeout",
        dest="image_api_timeout",
        type=float,
        default=600.0,
    )
    p.add_argument("--retry-wait", type=int, default=30)
    p.add_argument(
        "--no-pencil-style",
        action="store_true",
        help="修帧输出写实照片风格（默认输出铅笔素描，供 Seedance 与 Image1 一致）",
    )
    p.add_argument(
        "--pencil-two-pass",
        action="store_true",
        help="先写实修内容再转铅笔（两次 GPT-Image，质量更稳但更慢）",
    )
    args = p.parse_args()
    if args.input_fidelity == "":
        args.input_fidelity = None
    if args.candidate_frames < 2:
        raise SystemExit("--candidate-frames 至少为 2")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
