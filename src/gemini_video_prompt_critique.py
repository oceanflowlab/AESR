#!/usr/bin/env python3
"""
多模态：给定本地视频 + 文本描述，让模型对照视频，列出「文本 vs 画面」的出入。

走与 Track1 ACE 相同的 Gemini 路由（默认经云雾）：需 YUNWU_API_KEY（或 Google 直连时的 GEMINI_API_KEY）。

默认经云雾时整段 ``video_url`` 常导致模型**未看到真实画面而胡编**；故本脚本**默认均匀抽 24 帧 JPEG** 再请求（与 ``--frame-screenshots 24`` 等价）。若需整段视频：

  - 传 ``--frame-screenshots 0``，或 ``export CRITIQUE_FRAME_SCREENSHOTS=0``；
  - 或 ``export ACE_GEMINI_TRANSPORT=google`` + ``GEMINI_API_KEY`` 走 Google 直连。

抽帧张数默认可用 ``CRITIQUE_FRAME_SCREENSHOTS`` 覆盖（未设则 24）。

  export YUNWU_API_KEY=...
  python3 gemini_video_prompt_critique.py \\
    --video /path/to.mp4 \\
    --prompt-file /path/to/prompt.txt \\
    --out /path/to/out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def probe_video_duration_sec(video_path: Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            return None
        return float(proc.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return None


def extract_video_frames_jpg(video_path: Path, n: int, out_dir: Path) -> list[Path]:
    """均匀时间间隔抽帧为 JPEG（依赖 ffmpeg）。"""
    dur = probe_video_duration_sec(video_path)
    if dur is None or dur <= 0:
        dur = 10.0
    fps = max(0.001, n / dur)
    out_pattern = str(out_dir / "frame_%04d.jpg")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps},scale=768:-2",
            "-frames:v",
            str(n),
            out_pattern,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg 抽帧失败: {err}")
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg 未生成任何抽帧文件")
    return frames


def build_user_prompt(
    description: str,
    has_image: bool,
    duration_sec: float | None,
    frame_screenshots: int,
) -> str:
    prefix: list[str] = []
    if duration_sec is not None:
        prefix.append(f"（视频时长约 {duration_sec:.1f} 秒。）")
    if frame_screenshots > 0:
        prefix.append(
            f"（以下按时间顺序附带同一视频的 {frame_screenshots} 张均匀抽帧图，请据此推断全片与文本的出入。）"
        )
    if has_image:
        prefix.append("（若附带参考图，仅在与描述相关时作辅助，以视频画面为准。）")
    temporal_hint: list[str] = []
    if duration_sec is not None and frame_screenshots > 0:
        T, N = duration_sec, frame_screenshots
        temporal_hint.append(
            f"时间轴提示：全片约 {T:.1f} 秒；{N} 张抽帧按时间均匀覆盖。"
            f"第 i 张（1≤i≤{N}）大致对应片内时间区间约 [(i-1)/{N}×{T:.1f}, i/{N}×{T:.1f}] 秒（秒数近似即可）。"
        )
    elif frame_screenshots > 0 and duration_sec is None:
        temporal_hint.append(
            f"时间轴提示：当前为按时间先后排列的 {frame_screenshots} 张均匀抽帧；"
            "片长未探测到时 approx_time_span_sec 可填 null 或粗略估计，basis_snapshot_range 尽量给出。"
        )
    elif duration_sec is not None and frame_screenshots == 0:
        temporal_hint.append(f"时间轴提示：当前为整段视频输入，全片约 {duration_sec:.1f} 秒。")

    viewer_lr_guide = (
        "【左右方向约定（重要）】\n"
        "文本与画面中的「左/右」「向左/向右」「左转/右转」等，若**未明确**写「视频中人物自身视角」「"
        "角色视角」「以其自身为参照」等，一律按**观众视角**（镜头/观看者所见画面：画面左侧=左，"
        "画面右侧=右）理解与比对。\n"
        "写 point 时描述画面动作方向也请用观众视角，避免混用人物解剖学左右。\n"
        "因此：文本写「头向左转」且画面中头部移向**画面左侧**（观众左侧）→ 视为一致，"
        "**不要**记入 differences；"
    )

    json_schema = (
        "只输出一个 JSON 对象，键：differences、summary（一句中文）。\n"
        "differences 为数组；每一项须为对象，字段如下：\n"
        '- "point"：该条出入的中文说明（原 differences 字符串数组中每一条的含义）。\n'
        '- "approx_time_span_sec"：该条出入所依据或主要对应的**视频时间段**，用长度为 2 的数组 '
        '[t0, t1] 表示（单位秒，相对片头 0 秒；闭区间近似）。若贯穿全片或与时间弱相关，可用 '
        '[0, T]（T 为片长秒数）并在 point 中简要说明；若无法估计可填 null。\n'
    )
    if frame_screenshots > 0:
        json_schema += (
            f'- "basis_snapshot_range"：长度为 2 的正整数数组 [i, j]，表示主要依据的第 i 到第 j 张抽帧 '
            f'（按时间先后编号 1 到 {frame_screenshots}）；与全片抽样相关时可写 [1, {frame_screenshots}]。\n'
        )
    else:
        json_schema += '- "basis_snapshot_range"：整段视频模式下填 null 或省略。\n'

    # json_schema += (
    #     "必须严格依据所附视频或抽帧画面作答，勿臆造与画面不符的人物、场景或衣着。不要其它说明文字。"
    # )

    main_parts: list[str] = [
        viewer_lr_guide,
        "",
        "【文本描述】",
        description.strip(),
        "【文本描述】与输入视频中的内容有哪些difference。",
        "",
    ]
    main_parts.extend(temporal_hint)
    if temporal_hint:
        main_parts.append("")
    main_parts.append(json_schema)
    main = "\n".join(main_parts)
    if prefix:
        return "\n".join(prefix) + "\n\n" + main
    return main


def _default_frame_screenshots_from_env() -> int:
    raw = os.environ.get("CRITIQUE_FRAME_SCREENSHOTS", "24").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 24
    return max(0, n)


async def _run(args: argparse.Namespace) -> int:
    import ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi as ace

    video_path = Path(args.video).expanduser().resolve()
    prompt_body = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    frame_n = max(0, int(args.frame_screenshots))

    dur = probe_video_duration_sec(video_path)
    if dur is not None:
        print(f"[info] video duration ≈ {dur:.1f}s", file=sys.stderr)

    tmp = (
        tempfile.TemporaryDirectory(prefix="critique_frames_")
        if frame_n > 0
        else None
    )
    tdir = Path(tmp.__enter__()) if tmp else None
    result: dict | None = None
    try:
        media_paths: list[str] = []
        if args.image:
            media_paths.append(str(Path(args.image).expanduser().resolve()))
        if frame_n > 0:
            assert tdir is not None
            try:
                frames = extract_video_frames_jpg(video_path, frame_n, tdir)
            except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"❌ 抽帧失败: {e}", file=sys.stderr)
                return 1
            media_paths.extend(str(p) for p in frames)
            print(
                f"[info] 抽帧：共 {len(frames)} 张 JPEG（均匀采样；整段视频请用 --frame-screenshots 0）",
                file=sys.stderr,
            )
        else:
            media_paths.append(str(video_path))

        user_prompt = build_user_prompt(
            prompt_body,
            has_image=bool(args.image),
            duration_sec=dur,
            frame_screenshots=frame_n,
        )
        result = await ace.call_qwen_json(
            user_prompt,
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
        print("❌ 未得到 JSON（检查密钥与网络）", file=sys.stderr)
        return 1

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n✅ 已写入 {args.out}", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="视频 + 文本：列出二者出入（简单 JSON）。")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--image", type=Path, default=None, help="可选参考图（先图后视频 / 先图后抽帧）")
    p.add_argument(
        "--frame-screenshots",
        type=int,
        default=_default_frame_screenshots_from_env(),
        metavar="N",
        help="默认 24（或环境变量 CRITIQUE_FRAME_SCREENSHOTS）；0=整段 mp4 作 video_url；>0=均匀抽 N 帧 JPEG。",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.environ.get("CRITIQUE_MAX_OUTPUT_TOKENS", "8192")),
        metavar="N",
        help="云雾 max_tokens，默认 8192（本任务输出很短，一般够用）",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"找不到视频: {args.video}")
    if not args.prompt_file.is_file():
        raise SystemExit(f"找不到 prompt 文件: {args.prompt_file}")
    if args.image and not args.image.is_file():
        raise SystemExit(f"找不到参考图: {args.image}")
    if args.frame_screenshots < 0:
        raise SystemExit("--frame-screenshots 不能为负数")

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
