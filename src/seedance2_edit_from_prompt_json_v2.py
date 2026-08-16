#!/usr/bin/env python3
"""
第三轮 v2：读取 assemble_seedance_edit_prompt_with_frame_fixes_v2.py 的 JSON，
调用方舟编辑；可选将 edited_frame 作为额外 reference_image 上传（图片2..N）。

不修改 seedance2_edit_from_prompt_json.py / seedance2_local_video_edit.py。

  python3 seedance2_edit_from_prompt_json_v2.py \\
    --video-url https://.../id014.mp4 \\
    --edit-prompt-json .../id014_seedance_edit_prompt_with_frame_fixes_v2.json \\
    --id id014 \\
    --upload-edited-frames \\
    --out .../id014_edited.mp4

若 API 拒绝多图，去掉 --upload-edited-frames，仅依赖 prompt 中对 Image 2..N 的文字描述。

默认在 prompt 末尾追加 ``--dur <duration>``（``--duration`` / ``SEEDANCE_EDIT_DURATION``）；``--no-cli-suffix`` 可关闭。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from public_safety import public_path, redact_urls

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from batch_seedance2_r2v import (  # noqa: E402
    append_seedance_text_cli_suffix,
    call_with_retries,
    download_url,
    find_video_urls,
    object_to_plain,
)
from gemini_seedance_edit_prompt_from_critique import (  # noqa: E402
    DEFAULT_PENCIL_ROOT,
    resolve_reference_image,
)
from gpt_image2_edit_frame_api import resolve_edited_frame_path  # noqa: E402
from seedance2_edit_from_prompt_json import load_instruction  # noqa: E402
from seedance2_local_video_edit import (  # noqa: E402
    create_edit_task,
    file_to_data_url,
    poll_task,
)


def build_content_v2(
    text: str,
    video_url: str,
    identity_image_url: str | None,
    extra_image_urls: list[str] | None = None,
) -> list:
    parts: list = [{"type": "text", "text": text}]
    if identity_image_url:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": identity_image_url},
                "role": "reference_image",
            }
        )
    for url in extra_image_urls or []:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": url},
                "role": "reference_image",
            }
        )
    parts.append(
        {
            "type": "video_url",
            "video_url": {"url": video_url},
            "role": "reference_video",
        }
    )
    return parts


def load_edited_frame_paths(edit_prompt_json: Path) -> list[str]:
    data = json.loads(edit_prompt_json.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    refs = data.get("edited_frame_references") or []
    for row in refs:
        if not isinstance(row, dict):
            continue
        p = row.get("edited_frame")
        resolved = resolve_edited_frame_path(p) if p else None
        if not resolved:
            continue
        idx = int(row.get("logical_image_index") or len(out) + 2)
        out.append((idx, str(resolved)))
    if not out:
        for i, entry in enumerate(data.get("input_frame_fix_plan") or []):
            if not isinstance(entry, dict):
                continue
            p = entry.get("edited_frame")
            resolved = resolve_edited_frame_path(p) if p else None
            if resolved:
                out.append((i + 2, str(resolved)))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def main() -> None:
    p = argparse.ArgumentParser(description="Seedance 编辑 v2（可选上传修帧 reference_image）")
    p.add_argument("--edit-prompt-json", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-url", type=str, default=None)
    p.add_argument("--reference-image", type=Path, default=None)
    p.add_argument("--pencil-root", type=Path, default=DEFAULT_PENCIL_ROOT)
    p.add_argument("--id", type=str, default=None)
    p.add_argument(
        "--upload-edited-frames",
        action="store_true",
        help="将 JSON 中 edited_frame_references 的 PNG 作为额外 reference_image 上传",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration", type=int, default=int(os.environ.get("SEEDANCE_EDIT_DURATION", "14")))
    p.add_argument("--resolution", default=os.environ.get("SEEDANCE_EDIT_RESOLUTION", "480p"))
    p.add_argument("--ratio", default=os.environ.get("SEEDANCE_EDIT_RATIO", "16:9"))
    p.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2-0-260128"))
    p.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--no-cli-suffix",
        action="store_true",
        help="不在 prompt 末尾追加 --dur N（默认会追加）",
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

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("请设置 ARK_API_KEY")

    json_path = args.edit_prompt_json.expanduser().resolve()
    instruction = load_instruction(edit_prompt_json=json_path, prompt_file=None)

    if args.video_url:
        video_url = args.video_url.strip()
    elif args.video:
        vpath = args.video.expanduser().resolve()
        if not vpath.is_file():
            raise SystemExit(f"找不到视频: {vpath}")
        nbytes = vpath.stat().st_size
        if args.max_video_bytes and nbytes > args.max_video_bytes:
            raise SystemExit("本地视频过大，请用 --video-url")
        print(f"[info] 本地视频 → data URL ({nbytes / (1024 * 1024):.2f} MB)")
        video_url = file_to_data_url(vpath)
    else:
        raise SystemExit("请指定 --video-url 或 --video")

    ref_path = resolve_reference_image(
        args.pencil_root.expanduser().resolve(),
        args.id,
        args.reference_image.expanduser().resolve() if args.reference_image else None,
    )
    identity_url = file_to_data_url(ref_path)

    extra_urls: list[str] = []
    if args.upload_edited_frames:
        for fp in load_edited_frame_paths(json_path):
            print(f"[info] 修帧 reference_image: {Path(fp).name}")
            extra_urls.append(file_to_data_url(Path(fp)))

    ns = SimpleNamespace(duration=args.duration, watermark=args.watermark)
    if not args.no_cli_suffix:
        instruction = append_seedance_text_cli_suffix(instruction, ns)
        print(f"[info] prompt 末尾已追加 --dur {args.duration}", file=sys.stderr)

    content = build_content_v2(instruction, video_url, identity_url, extra_urls)

    if args.dry_run:
        print(json.dumps([{"type": x.get("type"), "role": x.get("role")} for x in content], indent=2))
        print(f"[dry-run] images: 1 identity + {len(extra_urls)} edited")
        return

    try:
        from volcenginesdkarkruntime import Ark
    except ModuleNotFoundError as exc:
        raise SystemExit("请先执行 conda activate aesr，并运行 python -m pip install -r requirements.txt") from exc

    client = Ark(base_url=args.base_url, api_key=api_key)
    ark_args = argparse.Namespace(
        model=args.model,
        ratio=args.ratio,
        resolution=args.resolution,
        duration=args.duration,
        watermark=args.watermark,
        generate_audio=args.generate_audio,
    )
    create_result = call_with_retries(
        "create edit task v2",
        lambda: create_edit_task(client, ark_args, content),
        args.max_retries,
        args.retry_wait,
    )
    task_id = create_result.id
    print(f"task_id={task_id}")
    poll_ns = argparse.Namespace(
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )
    final = poll_task(client, task_id, poll_ns)
    plain = object_to_plain(final)
    urls = find_video_urls(plain)
    if not urls:
        raise SystemExit(f"未解析到视频 URL: {plain}")

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    download_url(urls[0], out)
    meta_out = args.metadata or out.with_name(out.stem + "_metadata.json")
    meta_out.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "video_url": redact_urls(urls[0]),
                "upload_edited_frames": args.upload_edited_frames,
                "n_extra_images": len(extra_urls),
                "edit_prompt_json": public_path(json_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
