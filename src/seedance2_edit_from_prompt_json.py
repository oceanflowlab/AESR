#!/usr/bin/env python3
"""
第三轮：用已写好的 Seedance 2.0 编辑英文 prompt + 原视频 + 人物参考图，调用方舟
``seedance2_local_video_edit.py`` 同一套 Ark content_generation 流程出成片。

编辑指令来源（二选一）：
  - ``--edit-prompt-json``：``gemini_seedance_edit_prompt_from_critique.py`` 输出的 JSON，读取字段 ``seedance_edit_prompt_en``；
  - ``--prompt-file``：纯文本（单行或多行均可），直接作为编辑指令。

参考图解析与第二轮脚本一致：``--reference-image`` 或 ``--pencil-root`` + ``--id``。
默认 ``--pencil-root`` 为 ``IPVG2026-Test-Track1/out_gpt_image2_pencil_same_crop``（与 Seedance 成片同 crop）。

  export ARK_API_KEY=...

重要：火山方舟当前对「视频编辑」任务通常要求 ``reference_video`` 为 **HTTPS 直链**，
本地 ``--video`` 转 data URL 可能返回 ``reference_video must be provided as a web url``。
此时请把 mp4 上传到 TOS/对象存储等，改用 ``--video-url``。

  python3 seedance2_edit_from_prompt_json.py \\
    --video-url https://example.com/your/id010.mp4 \\
    --edit-prompt-json /path/to_seedance_edit_prompt.json \\
    --id id010 \\
    --out /path/to_edited.mp4

若接口仍接受本地视频（少数环境），可尝试 ``--video /path/to_original.mp4``。

可选：``--duration`` 默认 14（与 Track1 R2V 常见秒数一致），可用环境变量 ``SEEDANCE_EDIT_DURATION`` 覆盖。
默认会在编辑 prompt 正文末尾追加 ``--dur <duration>``（与 ``batch_seedance2_r2v`` 一致）；用 ``--no-cli-suffix`` 可关闭。
其余参数透传给 ``seedance2_local_video_edit.py``（见 ``--help``）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_DIR = Path(__file__).resolve().parent

from batch_seedance2_r2v import append_seedance_text_cli_suffix  # noqa: E402
from gemini_seedance_edit_prompt_from_critique import (  # noqa: E402
    DEFAULT_PENCIL_ROOT,
    resolve_reference_image,
)


def load_instruction(*, edit_prompt_json: Path | None, prompt_file: Path | None) -> str:
    if edit_prompt_json is not None:
        data = json.loads(edit_prompt_json.read_text(encoding="utf-8"))
        for key in ("seedance_edit_prompt_en", "seedance_edit_prompt", "edit_prompt_en"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        raise SystemExit(
            f"在 {edit_prompt_json} 中未找到非空字段 "
            "seedance_edit_prompt_en（或 edit_prompt_en）"
        )
    if prompt_file is not None:
        t = prompt_file.read_text(encoding="utf-8").strip()
        if not t:
            raise SystemExit(f"编辑指令为空: {prompt_file}")
        return t
    raise SystemExit("请指定 --edit-prompt-json 或 --prompt-file")


def main() -> None:
    p = argparse.ArgumentParser(
        description="用 JSON/文本中的编辑 prompt + 原视频 + 参考图调用 Seedance 2.0（方舟）编辑。"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--edit-prompt-json",
        type=Path,
        help="含 seedance_edit_prompt_en 的 JSON（第二轮脚本输出）",
    )
    src.add_argument(
        "--prompt-file",
        type=Path,
        help="纯文本编辑指令（英文），等同 seedance2_local_video_edit --prompt-file",
    )

    vsrc = p.add_mutually_exclusive_group(required=True)
    vsrc.add_argument(
        "--video",
        type=Path,
        default=None,
        help="待编辑原视频（本地）。若 API 报 reference_video 须为 web url，请改用 --video-url",
    )
    vsrc.add_argument(
        "--video-url",
        type=str,
        default=None,
        metavar="URL",
        help="待编辑原视频的 HTTPS 直链（方舟编辑接口通常仅接受此形式）",
    )
    p.add_argument("--reference-image", type=Path, default=None)
    p.add_argument(
        "--pencil-root",
        type=Path,
        default=DEFAULT_PENCIL_ROOT,
        help=f"含 id*/ 参考图的根；默认 {DEFAULT_PENCIL_ROOT}",
    )
    p.add_argument("--id", type=str, default=None, help="与 out_gpt_image2_pencil_same_crop 等下子目录名一致，如 id010")

    p.add_argument("--out", type=Path, required=True, help="输出 mp4 路径")

    p.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("SEEDANCE_EDIT_DURATION", "14")),
        help="生成时长（秒），默认 14 或环境变量 SEEDANCE_EDIT_DURATION",
    )
    p.add_argument("--resolution", default=os.environ.get("SEEDANCE_EDIT_RESOLUTION", "480p"))
    p.add_argument("--ratio", default=os.environ.get("SEEDANCE_EDIT_RATIO", "16:9"))
    p.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2-0-260128"))
    p.add_argument("--base-url", default=None, help="默认 ARK_BASE_URL")
    p.add_argument("--api-key", default=None, help="默认 ARK_API_KEY")
    p.add_argument(
        "--no-cli-suffix",
        action="store_true",
        help="不在 prompt 末尾追加 --dur N（默认会追加，N=--duration）",
    )
    p.add_argument("--generate-audio", action="store_true")
    p.add_argument("--watermark", action="store_true")
    p.add_argument("--max-video-bytes", type=int, default=48 * 1024 * 1024)
    p.add_argument("--poll-interval", type=int, default=30)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-wait", type=int, default=30)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.id:
        args.id = args.id.strip()

    instruction = load_instruction(
        edit_prompt_json=args.edit_prompt_json,
        prompt_file=args.prompt_file,
    )
    if not args.no_cli_suffix:
        instruction = append_seedance_text_cli_suffix(
            instruction,
            SimpleNamespace(duration=args.duration, watermark=args.watermark),
        )
        print(f"[info] prompt 末尾已追加 --dur {args.duration}", file=sys.stderr)

    video_path: Path | None = None
    video_url: str | None = None
    if args.video_url:
        video_url = args.video_url.strip()
        if not (video_url.startswith("https://") or video_url.startswith("http://")):
            print(
                "[warn] --video-url 建议使用 https:// 直链；http 或未加密链接可能被拒。",
                file=sys.stderr,
            )
    else:
        assert args.video is not None
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise SystemExit(f"找不到视频: {video_path}")
        print(
            "[warn] 方舟「视频编辑」接口常要求 reference_video 为公网 HTTPS URL；"
            "若创建任务返回 400 web url，请将本地上传后改用 --video-url。",
            file=sys.stderr,
        )

    ref_path = resolve_reference_image(
        args.pencil_root.expanduser().resolve(),
        args.id,
        args.reference_image.expanduser().resolve() if args.reference_image else None,
    )

    edit_py = _SCRIPT_DIR / "seedance2_local_video_edit.py"
    if not edit_py.is_file():
        raise SystemExit(f"找不到 {edit_py}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_seedance_edit_instruction.txt",
        delete=False,
    ) as tf:
        tf.write(instruction)
        tmp_prompt = Path(tf.name)

    try:
        cmd: list[str] = [
            sys.executable,
            str(edit_py),
            "--prompt-file",
            str(tmp_prompt),
            "--out",
            str(args.out.expanduser().resolve()),
            "--model",
            args.model,
            "--ratio",
            args.ratio,
            "--resolution",
            str(args.resolution),
            "--duration",
            str(args.duration),
            "--max-video-bytes",
            str(args.max_video_bytes),
            "--poll-interval",
            str(args.poll_interval),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
            "--retry-wait",
            str(args.retry_wait),
        ]
        if video_url:
            cmd.extend(["--video-url", video_url])
        else:
            cmd.extend(["--video", str(video_path)])
        cmd.extend(["--image", str(ref_path)])
        if args.base_url:
            cmd.extend(["--base-url", args.base_url])
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.generate_audio:
            cmd.append("--generate-audio")
        if args.watermark:
            cmd.append("--watermark")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.metadata:
            cmd.extend(["--metadata", str(args.metadata.expanduser().resolve())])

        print("[info] 调用 seedance2_local_video_edit.py（子进程）…", file=sys.stderr)
        subprocess.run(cmd, check=True)
    finally:
        tmp_prompt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
