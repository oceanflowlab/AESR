#!/usr/bin/env python3
"""
GPT-Image-2 单帧编辑 API 封装（从 batch_gpt_image2_pencil.py 复制，不修改原文件）。

默认经云雾 OpenAI 兼容接口；**图像编辑使用独立密钥**（与 Gemini 用的 YUNWU_API_KEY 分开）：
  export YUNWU_GPT_IMAGE_API_KEY=...          # 推荐（GPT-Image-2 / images.edit）
  export YUNWU_GPT_IMAGE_BASE_URL=https://yunwu.ai/v1   # 可选

  亦兼容 YUNWU_IMAGE_API_KEY、OPENAI_API_KEY（见 resolve_image_api_credentials）。
  不会回退到 YUNWU_API_KEY，避免与 Gemini 争用同一 key。

  python3 gpt_image2_edit_frame_api.py --image frame.jpg --prompt "..." --out out.png
"""

from __future__ import annotations

import argparse
import base64
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# 与 batch_gpt_image2_pencil_copy.ENGLISH_TASK 一致：修帧参考图须为铅笔素描，与 Image 1 身份铅笔对齐。
PENCIL_SKETCH_STYLE_TRANSFER_EN = (
    "Perform only a style transfer: convert the entire image into a pencil-sketch look "
    "(graphite drawing, visible pencil line texture, black-and-white or soft gray tones). "
    "Strictly preserve the same framing, composition, aspect ratio, pose, and all visible "
    "content elements. Do not zoom out, extend the canvas, or add body parts not in the original. "
    "Keep the face and identity unchanged."
)

PENCIL_MISSING_VISUAL_EDIT_INTRO = (
    "Edit this video frame and render the entire result as a pencil-sketch / graphite drawing "
    "(visible pencil line texture, black-and-white or soft gray tones), matching the style of "
    "Seedance reference identity pencil images. Strictly preserve framing, composition, aspect "
    "ratio, pose, and the existing person identity unless a small adjustment is required to "
    "naturally add the missing elements."
)

PENCIL_MOTION_STATE_EDIT_INTRO = (
    "Edit this video keyframe and render the entire result as a pencil-sketch / graphite drawing "
    "(visible pencil line texture, black-and-white or soft gray tones), matching Seedance reference "
    "identity pencil images. Adjust the subject pose, body orientation, limb positions, or the "
    "visible end-state of camera framing so the still matches the target description. Preserve "
    "identity, scene layout, and lighting unless a small change is required for the correct pose."
)

PHOTO_MOTION_STATE_EDIT_INTRO = (
    "Edit this photorealistic video keyframe. Adjust pose, orientation, or visible end-state of "
    "the subject and framing to match the target description. Preserve identity, lighting, and "
    "perspective unless a small adjustment is required."
)

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: openai. Activate the Conda environment and run: python -m pip install -r requirements.txt"
    ) from exc


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if parsed.netloc == "yunwu.ai" and path == "":
        return "https://yunwu.ai/v1"
    return base_url


def call_with_retries(label, func, max_retries, initial_wait):
    import time

    attempt = 0
    wait_seconds = initial_wait
    while True:
        try:
            return func()
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            print(
                f"[retry] {label}: {type(exc).__name__}, wait {wait_seconds}s ({attempt}/{max_retries})",
                flush=True,
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2


def resolve_image_api_credentials(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    """
    GPT-Image 专用凭证（与 Gemini 的 YUNWU_API_KEY 分离，避免两路争用同一 key）。

    密钥优先级：--api-key > YUNWU_GPT_IMAGE_API_KEY > YUNWU_IMAGE_API_KEY > OPENAI_API_KEY
    基址优先级：--base-url > YUNWU_GPT_IMAGE_BASE_URL > YUNWU_IMAGE_BASE_URL >
                OPENAI_BASE_URL > https://yunwu.ai/v1
    """
    key = (
        api_key
        or os.environ.get("YUNWU_GPT_IMAGE_API_KEY")
        or os.environ.get("YUNWU_IMAGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not key:
        raise SystemExit(
            "请设置 YUNWU_GPT_IMAGE_API_KEY（GPT-Image-2 经云雾，独立于 Gemini 的 YUNWU_API_KEY）"
            "；或 YUNWU_IMAGE_API_KEY / OPENAI_API_KEY。"
        )
    raw_base = (
        base_url
        or os.environ.get("YUNWU_GPT_IMAGE_BASE_URL")
        or os.environ.get("YUNWU_IMAGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://yunwu.ai/v1"
    ).strip()
    bu = normalize_base_url(raw_base) or "https://yunwu.ai/v1"
    return key, bu


def build_frame_reference_edit_prompt(
    *,
    point: str,
    caption: str,
    issue_type: str = "missing_visual_element",
    pencil_style: bool = True,
) -> str:
    """GPT-Image 修帧 prompt：missing_visual_element 补元素；motion_state 改终态姿态/构图。"""
    from critique_issue_types import normalize_issue_type

    it = normalize_issue_type(issue_type or "missing_visual_element")
    cap = (caption or "").strip()
    cap_line = (
        f"The overall scene should remain consistent with this story caption: {cap}"
        if cap
        else (
            "Keep the scene consistent with a pencil-sketch video keyframe."
            if pencil_style
            else "Keep the scene consistent with a photorealistic video frame."
        )
    )
    if it == "motion_state":
        intro = PENCIL_MOTION_STATE_EDIT_INTRO if pencil_style else PHOTO_MOTION_STATE_EDIT_INTRO
        task = f"Target pose, end-state, or framing to match: {point.strip()}"
        tail = (
            "Match the target state in the same pencil-sketch style; "
            "do not add subtitles, logos, or watermarks."
            if pencil_style
            else "Match the target state naturally; do not add subtitles, logos, or watermarks."
        )
    else:
        intro = (
            PENCIL_MISSING_VISUAL_EDIT_INTRO
            if pencil_style
            else (
                "Edit this photorealistic video frame. Preserve the existing person identity, "
                "lighting, color grading, and camera perspective unless a small adjustment "
                "is required to naturally add the missing elements."
            )
        )
        task = f"Missing or incorrect visual elements to fix: {point.strip()}"
        tail = (
            "Integrate new elements seamlessly in the same pencil-sketch style; "
            "do not add subtitles, logos, or watermarks."
            if pencil_style
            else "Integrate new elements seamlessly; do not add subtitles, logos, or watermarks."
        )
    return f"{intro}\n{task}\n{cap_line}\n{tail}"


def build_missing_visual_frame_edit_prompt(
    *,
    point: str,
    caption: str,
    pencil_style: bool = True,
) -> str:
    return build_frame_reference_edit_prompt(
        point=point,
        caption=caption,
        issue_type="missing_visual_element",
        pencil_style=pencil_style,
    )


def build_pencil_style_transfer_prompt() -> str:
    return PENCIL_SKETCH_STYLE_TRANSFER_EN


def resolve_edited_frame_path(path: str | Path) -> Path | None:
    """优先 edited_frame.png；若仅有 photoreal 中间图则回退 edited_frame_photo.png。"""
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    parent = p.parent
    for name in ("edited_frame.png", "edited_frame_pencil.png"):
        cand = parent / name
        if cand.is_file():
            return cand
    photo = parent / "edited_frame_photo.png"
    if photo.is_file():
        return photo
    return None


def make_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
    verbose: bool = True,
) -> OpenAI:
    key, bu = resolve_image_api_credentials(api_key=api_key, base_url=base_url)
    if verbose:
        print(
            f"[info] GPT-Image API: base_url={bu} (密钥: YUNWU_GPT_IMAGE_API_KEY / 云雾 OpenAI 兼容)",
            flush=True,
        )
    return OpenAI(timeout=timeout, api_key=key, base_url=bu)


def edit_frame_image(
    client: OpenAI,
    *,
    image_path: Path,
    prompt: str,
    output_path: Path,
    image_model: str = "gpt-image-2",
    size: str = "auto",
    quality: str = "medium",
    input_fidelity: str | None = "high",
    max_retries: int = 6,
    retry_wait: int = 30,
) -> None:
    request: dict = {
        "model": image_model,
        "prompt": prompt.strip(),
        "size": size,
        "quality": quality,
        "output_format": "png",
    }
    if input_fidelity:
        request["input_fidelity"] = input_fidelity

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with image_path.open("rb") as image_file:
        request["image"] = [image_file]
        result = call_with_retries(
            "gpt-image edit",
            lambda: client.images.edit(**request),
            max_retries,
            retry_wait,
        )

    first = result.data[0]
    if getattr(first, "b64_json", None):
        output_path.write_bytes(base64.b64decode(first.b64_json))
        return
    if getattr(first, "url", None):
        with urllib.request.urlopen(first.url) as response:
            output_path.write_bytes(response.read())
        return
    raise RuntimeError("Image API returned neither b64_json nor url.")


def main() -> None:
    p = argparse.ArgumentParser(description="GPT-Image-2 编辑单张帧图。")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-model", default=os.environ.get("GPT_IMAGE_MODEL", "gpt-image-2"))
    p.add_argument("--size", default=os.environ.get("GPT_IMAGE_SIZE", "auto"))
    p.add_argument("--quality", default=os.environ.get("GPT_IMAGE_QUALITY", "medium"))
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--retry-wait", type=int, default=30)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt.strip()
    else:
        raise SystemExit("请提供 --prompt 或 --prompt-file")

    if not args.image.is_file():
        raise SystemExit(f"找不到图片: {args.image}")

    client = make_openai_client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    edit_frame_image(
        client,
        image_path=args.image.expanduser().resolve(),
        prompt=prompt,
        output_path=args.out.expanduser().resolve(),
        image_model=args.image_model,
        size=args.size,
        quality=args.quality,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
    )
    print(f"✅ {args.out}")


if __name__ == "__main__":
    main()
