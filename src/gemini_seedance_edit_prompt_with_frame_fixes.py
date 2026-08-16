#!/usr/bin/env python3
"""
在 missing_visual_fix（GPT-Image 修帧）结果基础上，生成 Seedance 2.0 整片编辑 prompt。

**推荐**：前序结果已齐全时，用纯文本拼装（无 API）：
  assemble_seedance_edit_prompt_with_frame_fixes_v2.py

**Gemini 看图写 prompt**（默认仅铅笔+edited，不传 24 抽帧）：
  gemini_seedance_edit_prompt_with_frame_fixes_edited_only.py

本脚本（gemini_seedance_edit_prompt_with_frame_fixes.py）为旧版：默认铅笔+24 抽帧+edited，较重。

与 gemini_seedance_edit_prompt_from_critique.py 独立，不修改原脚本。

除 critique 外，额外输入：
  - missing_visual_fix/manifest.json + 各 issue_*/edited_frame.png（及 aligned_frame.jpg）
  - 每条修复的 **视频编辑时间段** video_edit_span_sec、**对齐时刻** keyframe_time_sec

方舟下游仍仅为：文本 + 图片1（铅笔）+ 视频1；edited 帧仅作为 Gemini 写 prompt 的视觉目标参考，
须在 seedance_edit_prompt_en 里用分时段英文描述「在 [t0,t1]s 让画面趋近附图目标」。

  export YUNWU_API_KEY=...

  python3 gemini_seedance_edit_prompt_with_frame_fixes.py \\
    --video .../id014.mp4 \\
    --critique-json .../id014_gemini_critique_typed.json \\
    --missing-visual-fix-dir .../id014/missing_visual_fix \\
    --prompt-file .../batch_in_gpt_pencil/id014/prompt.txt \\
    --id id014 \\
    --out .../id014_seedance_edit_prompt_with_frame_fixes.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.resolve()
DEFAULT_PENCIL_ROOT = (_REPO_ROOT / "examples" / "data" / "references").resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gemini_video_prompt_critique as gvc  # noqa: E402
from critique_issue_types import FRAME_FIX_ISSUE_TYPES, normalize_issue_type  # noqa: E402
from gemini_seedance_edit_prompt_from_critique import (  # noqa: E402
    DEFAULT_PENCIL_ROOT,
    SEEDANCE20_EDIT_GUIDE,
    normalize_critique_differences,
    resolve_reference_image,
)


def load_frame_fix_plan(missing_visual_root: Path) -> list[dict]:
    """从 missing_visual_fix 目录读取每条已修帧的计划（含时间区间）。"""
    root = missing_visual_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"找不到 manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan: list[dict] = []
    for item in manifest.get("processed_issues") or []:
        idx = int(item.get("issue_list_index", len(plan)))
        issue_dir = root / f"issue_{idx:03d}"
        alignment = item.get("alignment") or {}
        span = alignment.get("resolved_time_span_sec") or item.get("approx_time_span_sec")
        keyframe_time = None
        best_i = alignment.get("best_frame_index")
        for c in item.get("candidates") or []:
            if c.get("frame_index") == best_i:
                keyframe_time = c.get("time_sec")
                break

        edited_path = issue_dir / "edited_frame.png"
        aligned_path = issue_dir / "aligned_frame.jpg"
        edit_prompt_path = issue_dir / "edit_prompt_en.txt"

        entry = {
            "issue_dir": issue_dir.name,
            "issue_list_index": idx,
            "source_difference_index": item.get("source_difference_index"),
            "point": item.get("point"),
            "video_edit_span_sec": span,
            "keyframe_time_sec": keyframe_time,
            "best_frame_file": alignment.get("best_frame_file"),
            "alignment_rationale": alignment.get("rationale"),
            "aligned_frame": str(aligned_path) if aligned_path.is_file() else None,
            "edited_frame": str(edited_path) if edited_path.is_file() else None,
            "frame_edit_prompt_en": (
                edit_prompt_path.read_text(encoding="utf-8").strip()
                if edit_prompt_path.is_file()
                else None
            ),
        }
        if not entry["edited_frame"]:
            print(f"[warn] 跳过 {issue_dir.name}：无 edited_frame.png", file=sys.stderr)
            continue
        plan.append(entry)
    return plan


def collect_other_differences(
    critique_data: dict,
    *,
    fix_source_indices: set[int],
) -> list[dict]:
    """critique 中不由 frame-fix 负责的条目（motion_process / other 等）。"""
    out: list[dict] = []
    for i, d in enumerate(critique_data.get("differences") or []):
        if not isinstance(d, dict):
            continue
        it = normalize_issue_type(d.get("issue_type"))
        if it in FRAME_FIX_ISSUE_TYPES:
            continue
        row = {
            "source_difference_index": i,
            "issue_type": d.get("issue_type"),
            "point": d.get("point"),
            "approx_time_span_sec": d.get("approx_time_span_sec"),
            "basis_snapshot_range": d.get("basis_snapshot_range"),
        }
        if row.get("point"):
            out.append(row)
    return out


def build_media_paths_for_fixes(
    ref_path: Path,
    video_path: Path,
    frame_n: int,
    fix_plan: list[dict],
    *,
    attach_aligned: bool,
) -> tuple[list[str], tempfile.TemporaryDirectory | None, int]:
    """铅笔图 + 视频抽帧 + 各 issue 的 aligned/edited 附图。"""
    tmp = tempfile.TemporaryDirectory(prefix="seedance_prompt_fix_") if frame_n > 0 else None
    tdir = Path(tmp.__enter__()) if tmp else None
    media: list[str] = [str(ref_path)]
    n_video_frames = 0
    if frame_n > 0:
        assert tdir is not None
        frames = gvc.extract_video_frames_jpg(video_path, frame_n, tdir)
        media.extend(str(p) for p in frames)
        n_video_frames = len(frames)
    else:
        media.append(str(video_path))

    for entry in fix_plan:
        if attach_aligned and entry.get("aligned_frame"):
            media.append(entry["aligned_frame"])
        if entry.get("edited_frame"):
            media.append(entry["edited_frame"])
    return media, tmp, n_video_frames


def build_llm_user_text_with_fixes(
    *,
    original_prompt: str,
    summary: str,
    fix_plan: list[dict],
    other_diffs: list[dict],
    video_duration_sec: float | None,
    n_video_frames: int,
    attach_aligned: bool,
) -> str:
    fix_json = json.dumps(fix_plan, ensure_ascii=False, indent=2)
    other_json = json.dumps(other_diffs, ensure_ascii=False, indent=2)
    dur_line = ""
    if video_duration_sec is not None:
        dur_line = f"原视频时长约 {video_duration_sec:.1f} 秒。\n"

    attach_lines = [
        "【多模态附件顺序】",
        "1) 图片1：人物铅笔参考图（身份锚点，对应下游 Ark 的 reference_image）。",
    ]
    if n_video_frames > 0:
        attach_lines.append(
            f"2) 随后 {n_video_frames} 张为视频1 的均匀抽帧（理解全片时序，编号按时间先后）。"
        )
    else:
        attach_lines.append("2) 随后为完整视频1。")
    k = 3
    for entry in fix_plan:
        label = entry["issue_dir"]
        if attach_aligned:
            attach_lines.append(
                f"{k}) {label}_aligned：原片在 keyframe≈{entry.get('keyframe_time_sec')}s 的对齐底帧；"
                f"应对 video_edit_span_sec={entry.get('video_edit_span_sec')} 时段编辑视频1。"
            )
            k += 1
        attach_lines.append(
            f"{k}) {label}_edited：GPT-Image 铅笔修帧（仅构图/姿态参考，非成片画风）；"
            f"在 video_edit_span_sec={entry.get('video_edit_span_sec')} 内让视频1 **逼真实拍**画面实现同等布局/内容。"
        )
        k += 1

    return "\n".join(
        [
            SEEDANCE20_EDIT_GUIDE.strip(),
            "",
            "\n".join(attach_lines),
            "",
            dur_line + "【用户原始文案】",
            original_prompt.strip() or "（无）",
            "",
            "【critique 总结】",
            (summary or "").strip() or "（无）",
            "",
            "【已由 GPT-Image 修帧处理的 missing_visual / motion_state 条目（须写入 seedance prompt 的分时段指令）】",
            "每条含 video_edit_span_sec：应在该秒级区间内编辑视频1；keyframe_time_sec 为对齐参考时刻。",
            "frame_edit_prompt_en 为修图时用的说明，可提炼为对视频1 的增删改描述。",
            fix_json,
            "",
            "【仍需视频模型直接处理的其它出入（motion_process / other，无 edited 附图）】",
            other_json,
            "",
            "【重要：下游 Ark 实际输入】",
            "方舟编辑接口**只会**收到：本条英文 prompt + Image 1（铅笔）+ Video 1（原片）。",
            "**不会**再收到 issue_XXX_edited 附图。因此 seedance_edit_prompt_en 必须把每张 edited 附图里的"
            "目标画面**用文字写全**（人物/物体/构图/屏幕内容等），不能只写抽象改法而不体现修图结果。",
            "",
            "【你的任务】",
            "写一条给 Seedance 2.0 的英文 seedance_edit_prompt_en（最终编辑模型唯一读到的指令正文）：",
            "- **成片必须是 photorealistic live-action（逼真实拍）**；附图铅笔仅作参考，勿要求成片铅笔/线稿风；",
            "- 显式使用 Video 1 / Image 1；",
            "- 对 fix_plan 中**每一条**，必须包含类似结构（英文）：",
            '  During [t0,t1]s, edit Video 1 to match the photoreal layout/content from the pencil reference (issue_XXX_edited): '
            "<具体实拍可见内容，从附图与 frame_edit_prompt_en 提炼，如 clerk in foreground, manifest on phone screen>；",
            "- 每条都要点明 issue 编号（issue_000_edited / issue_001_edited）或等价说法，表明依据的是修图目标；",
            "- 未在给定时间段内提及的内容默认保持 Video 1 不变；",
            "- 合并 other_differences 中的 motion_process 等要求；",
            "- 结尾强调 photorealistic cinematic quality, no pencil sketch / cartoon；",
            "- 一条连贯 prompt，可分号或短句分时段。",
            "",
            "只输出 JSON：seedance_edit_prompt_en, seedance_edit_prompt_zh, editing_rationale,",
            "以及 frame_fix_intervals（数组，每项含 issue_dir, video_edit_span_sec, keyframe_time_sec, "
            "prompt_excerpt_en；prompt_excerpt_en 须体现 edited 附图中的具体视觉目标）。",
            "不要其它文字。",
        ]
    )


async def _run(args: argparse.Namespace) -> int:
    import ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi as ace

    critique_path = Path(args.critique_json).expanduser().resolve()
    data = json.loads(critique_path.read_text(encoding="utf-8"))
    fix_plan = load_frame_fix_plan(Path(args.missing_visual_fix_dir))
    if not fix_plan:
        print("❌ missing_visual_fix 中无有效 edited_frame", file=sys.stderr)
        return 1

    fix_src = {
        int(x["source_difference_index"])
        for x in fix_plan
        if x.get("source_difference_index") is not None
    }
    other_diffs = collect_other_differences(data, fix_source_indices=fix_src)

    ref_path = resolve_reference_image(
        args.pencil_root.expanduser().resolve(),
        args.id.strip() if args.id else None,
        Path(args.reference_image).expanduser().resolve() if args.reference_image else None,
    )
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"找不到视频: {video_path}")

    prompt_body = ""
    if args.prompt_file:
        pf = Path(args.prompt_file).expanduser().resolve()
        if pf.is_file():
            prompt_body = pf.read_text(encoding="utf-8").strip()

    dur = gvc.probe_video_duration_sec(video_path)
    if dur is not None:
        print(f"[info] video duration ≈ {dur:.1f}s", file=sys.stderr)
    print(f"[info] frame fixes: {len(fix_plan)} 条; other diffs: {len(other_diffs)}", file=sys.stderr)

    frame_n = max(0, int(args.frame_screenshots))
    tmp = None
    result: dict | None = None
    try:
        media_paths, tmp, n_vf = build_media_paths_for_fixes(
            ref_path,
            video_path,
            frame_n,
            fix_plan,
            attach_aligned=bool(args.attach_aligned),
        )
        print(f"[info] 多模态附件共 {len(media_paths)} 项", file=sys.stderr)

        user_text = build_llm_user_text_with_fixes(
            original_prompt=prompt_body,
            summary=str(data.get("summary") or ""),
            fix_plan=fix_plan,
            other_diffs=other_diffs,
            video_duration_sec=dur,
            n_video_frames=n_vf,
            attach_aligned=bool(args.attach_aligned),
        )
        result = await ace.call_qwen_json(
            user_text,
            media_paths=media_paths,
            temperature=float(args.temperature),
            max_retries=args.max_retries,
            timeout=float(args.timeout),
            max_tokens=int(args.max_output_tokens),
            yunwu_text_first=True,
        )
    finally:
        if tmp is not None:
            tmp.__exit__(None, None, None)

    if not result:
        print("❌ 未得到 JSON", file=sys.stderr)
        return 1

    result["input_frame_fix_plan"] = fix_plan
    result["input_other_differences"] = other_diffs
    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(out_text)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        print(f"\n✅ 已写入 {out_path}", file=sys.stderr)
        en = result.get("seedance_edit_prompt_en")
        if isinstance(en, str) and en.strip():
            txt_path = out_path.with_name(out_path.stem + "_en.txt")
            if args.prompt_txt_out:
                txt_path = Path(args.prompt_txt_out).expanduser().resolve()
            txt_path.write_text(en.strip() + "\n", encoding="utf-8")
            print(f"✅ 最终编辑模型用一段话: {txt_path}", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="结合 missing_visual_fix 修帧与时间区间，生成 Seedance 编辑 prompt。"
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--critique-json", type=Path, required=True)
    p.add_argument(
        "--missing-visual-fix-dir",
        type=Path,
        required=True,
        help="fix_missing_visual_from_critique_typed.py 的 --out 目录",
    )
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--reference-image", type=Path, default=None)
    p.add_argument("--pencil-root", type=Path, default=DEFAULT_PENCIL_ROOT)
    p.add_argument("--id", type=str, default=None)
    p.add_argument(
        "--frame-screenshots",
        type=int,
        default=int(os.environ.get("EDIT_PROMPT_FRAME_SCREENSHOTS", "24")),
    )
    p.add_argument(
        "--attach-aligned",
        action="store_true",
        help="除 edited_frame 外，也把 aligned_frame.jpg 送给 Gemini",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--prompt-txt-out",
        type=Path,
        default=None,
        help="另存 seedance_edit_prompt_en 纯文本（默认同目录 <out 主名>_en.txt）",
    )
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.environ.get("EDIT_PROMPT_MAX_OUTPUT_TOKENS", "8192")),
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()
    if args.id:
        args.id = args.id.strip()
    if not args.critique_json.is_file():
        raise SystemExit(f"找不到 critique: {args.critique_json}")
    if not Path(args.missing_visual_fix_dir).is_dir():
        raise SystemExit(f"找不到目录: {args.missing_visual_fix_dir}")

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
