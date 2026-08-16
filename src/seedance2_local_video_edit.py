#!/usr/bin/env python3
"""
Seedance 2.0：用方舟 Ark「视频编辑 / 参考视频」能力处理本地视频（可选参考图）。

对齐官方文档中的 tasks.create 多模态 content 示例（text + image_url + video_url）：
https://www.volcengine.com/docs/82379/2291680?lang=zh

依赖：
  conda activate aesr
  python -m pip install -r requirements.txt

认证：
  export ARK_API_KEY=...

本地媒体：
  将本地 mp4/mov 等读成 data:{mime};base64,... 填入 video_url / image_url，
  与仓库内 batch_seedance2_r2v.py 对参考图的用法一致。
  若视频很大，请求体可能超限；请缩小文件或使用可公网访问的 HTTPS URL（--video-url）。
  注意：方舟「视频编辑」接口在多数环境下要求 reference_video 为 **HTTPS 直链**，
  使用本地 --video 转 data URL 可能返回 400（reference_video must be provided as a web url）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from public_safety import public_path, redact_urls

# 与同目录 batch_seedance2_r2v 复用轮询、下载、结果解析
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

MIME_VIDEO = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}
MIME_IMAGE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def file_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    suf = path.suffix.lower()
    mime = MIME_VIDEO.get(suf) or MIME_IMAGE.get(suf) or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def poll_task(client, task_id: str, args) -> object:
    started = time.time()
    while True:
        result = call_with_retries(
            f"get task {task_id}",
            lambda: client.content_generation.tasks.get(task_id=task_id),
            args.max_retries,
            args.retry_wait,
        )
        status = getattr(result, "status", None)
        print(f"  status={status}")
        if status == "succeeded":
            return result
        if status == "failed":
            raise RuntimeError(f"Seedance task failed: {getattr(result, 'error', result)}")
        if time.time() - started > args.timeout:
            raise TimeoutError(f"Timed out waiting for task {task_id}")
        time.sleep(args.poll_interval)


def create_edit_task(client, args, content: list) -> object:
    kwargs = {
        "model": args.model,
        "content": content,
        "ratio": args.ratio,
        "resolution": args.resolution,
        "duration": args.duration,
        "watermark": args.watermark,
    }
    if args.generate_audio:
        kwargs["generate_audio"] = True

    while True:
        try:
            return client.content_generation.tasks.create(**kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            match = re.search(r"unexpected keyword argument ['\"](\w+)['\"]", msg)
            if not match:
                raise
            bad = match.group(1)
            if bad not in kwargs:
                raise
            kwargs.pop(bad)
            if bad == "generate_audio" and args.generate_audio:
                print(
                    "[warn] SDK does not support generate_audio= on tasks.create(); "
                    "flag ignored. Upgrade volcengine-python-sdk[ark] if needed."
                )
            continue


def build_content(text: str, video_url: str, image_url: str | None) -> list:
    """与文档示例顺序一致：先 text，再 reference_image（可选），再 reference_video。"""
    parts: list = [{"type": "text", "text": text}]
    if image_url:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Seedance 2.0：本地视频（或 URL）+ 文本指令 + 可选参考图，调用 Ark content_generation 编辑/生成。"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--video",
        type=Path,
        help="本地参考视频路径（将编码为 data URL）",
    )
    src.add_argument(
        "--video-url",
        help="参考视频 HTTPS 地址（大文件推荐，避免请求体过大）",
    )

    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="本地参考图（可选；对应文档里 image_url + role=reference_image）",
    )
    p.add_argument(
        "--image-url",
        default=None,
        help="参考图 HTTPS 地址（与 --image 二选一）",
    )

    text = p.add_mutually_exclusive_group(required=True)
    text.add_argument("--text", help="编辑指令（中文或英文均可）")
    text.add_argument("--prompt-file", type=Path, help="从文件读取编辑指令")

    p.add_argument("--out", type=Path, default=Path("seedance2_edit_out.mp4"), help="输出 mp4 路径")
    p.add_argument("--metadata", type=Path, default=None, help="写入 metadata.json（默认与 mp4 同目录）")

    p.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2-0-260128"))
    p.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
    p.add_argument("--api-key", default=None, help="默认读环境变量 ARK_API_KEY")

    p.add_argument("--ratio", default="16:9")
    p.add_argument("--resolution", default="480p")
    p.add_argument("--duration", type=int, default=5, help="生成视频时长（秒），以接口为准")
    p.add_argument("--generate-audio", action="store_true")
    p.add_argument("--watermark", action="store_true")

    p.add_argument(
        "--append-cli-suffix",
        action="store_true",
        help="在文本末尾追加与 batch_seedance2_r2v 相同的 --dur N（部分场景与控制台 CLI 习惯一致）",
    )
    p.add_argument(
        "--max-video-bytes",
        type=int,
        default=48 * 1024 * 1024,
        help="本地视频超过该字节数则拒绝使用 data URL（0=不限制）。大文件请用 --video-url。",
    )

    p.add_argument("--poll-interval", type=int, default=30)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-wait", type=int, default=30)
    p.add_argument("--dry-run", action="store_true", help="只打印 content 结构大小，不调用 API")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("请设置 ARK_API_KEY 或传入 --api-key")

    if args.prompt_file:
        instruction = args.prompt_file.read_text(encoding="utf-8").strip()
    else:
        instruction = (args.text or "").strip()
    if not instruction:
        raise SystemExit("编辑指令为空")

    if args.video_url:
        video_url = args.video_url.strip()
    else:
        vpath = args.video.expanduser().resolve()
        if not vpath.is_file():
            raise SystemExit(f"找不到视频文件: {vpath}")
        nbytes = vpath.stat().st_size
        if args.max_video_bytes and nbytes > args.max_video_bytes:
            raise SystemExit(
                f"本地视频过大 ({nbytes} bytes > --max-video-bytes {args.max_video_bytes})。\n"
                "请压缩视频、截短后重试，或使用 --video-url 指向可下载的 HTTPS 地址。"
            )
        print(f"[info] 本地视频 {vpath.name} ({nbytes / (1024 * 1024):.2f} MB) → data URL")
        video_url = file_to_data_url(vpath)

    image_url: str | None = None
    if args.image_url:
        image_url = args.image_url.strip()
    elif args.image:
        ip = args.image.expanduser().resolve()
        if not ip.is_file():
            raise SystemExit(f"找不到参考图: {ip}")
        print(f"[info] 本地参考图 {ip.name} ({ip.stat().st_size / 1024:.1f} KB) → data URL")
        image_url = file_to_data_url(ip)

    ns = SimpleNamespace(
        duration=args.duration,
        watermark=args.watermark,
    )
    if args.append_cli_suffix:
        instruction = append_seedance_text_cli_suffix(instruction, ns)

    content = build_content(instruction, video_url, image_url)

    if args.dry_run:
        approx = len(instruction) + len(video_url) + (len(image_url) if image_url else 0) + 400
        print(f"[dry-run] 估算 HTTP JSON body 约 {approx / (1024 * 1024):.2f} MB（未调用 API）")
        preview = [{"type": p.get("type"), "url_len": len(str(p))} for p in content]
        print(f"[dry-run] content 骨架: {json.dumps(preview, ensure_ascii=False)}")
        return

    try:
        from volcenginesdkarkruntime import Ark
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少依赖: 请先执行 conda activate aesr，并运行 python -m pip install -r requirements.txt"
        ) from exc

    client = Ark(base_url=args.base_url, api_key=api_key)
    print("----- create task -----")
    create_result = call_with_retries(
        "create edit task",
        lambda: create_edit_task(client, args, content),
        args.max_retries,
        args.retry_wait,
    )
    task_id = create_result.id
    print(f"task_id={task_id}")

    print("----- polling -----")
    final = poll_task(client, task_id, args)
    plain = object_to_plain(final)
    urls = find_video_urls(plain)
    if not urls:
        meta_path = args.out.with_suffix(".metadata_failed.json")
        meta_path.write_text(json.dumps(redact_urls(plain), ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"任务成功但未解析到视频 URL，详情已写入: {meta_path}")

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"----- download -> {out} -----")
    download_url(urls[0], out)

    meta_out = args.metadata or out.with_name(out.stem + "_metadata.json")
    meta_out.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "model": args.model,
                "output_video": public_path(out),
                "video_urls": [redact_urls(url) for url in urls],
                "final_result": redact_urls(plain),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ 完成: {out}\nmetadata: {meta_out}")


if __name__ == "__main__":
    main()
