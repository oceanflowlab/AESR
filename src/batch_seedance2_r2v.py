import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from public_safety import public_path, redact_urls


DEFAULT_GITHUB_RAW_TEMPLATE = (
    "https://raw.githubusercontent.com/OWNER/REPOSITORY/main/"
    "test_50_mixed_gpt_image2_pencil/{id}/pencil_full_body.png"
)

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def file_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    mime = MIME_BY_SUFFIX.get(image_path.suffix.lower(), "application/octet-stream")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _pil_encode_for_api(image_path: Path, max_long_edge, jpeg_quality):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install Pillow for --local-image-max-long-edge / --local-image-jpeg-quality: "
            "`python -m pip install Pillow`."
        ) from exc

    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in getattr(img, "info", {})):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max_long_edge is not None and max_long_edge > 0:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )

    buffer = io.BytesIO()
    if jpeg_quality is not None:
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("--local-image-jpeg-quality must be between 1 and 95")
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    img.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue(), "image/png"


def build_local_data_url(image_path: Path, args) -> str:
    """Base64 data URL; optionally downscale / re-encode to keep request size smaller."""
    path = Path(image_path)
    max_edge = getattr(args, "local_image_max_long_edge", None)
    jpeg_q = getattr(args, "local_image_jpeg_quality", None)
    if max_edge is None and jpeg_q is None:
        return file_to_data_url(path)

    raw, mime = _pil_encode_for_api(path, max_edge, jpeg_q)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def natural_sample_id_sort(ids):
    def key_fn(x):
        if isinstance(x, str) and x.startswith("id") and x[2:].isdigit():
            return (0, int(x[2:]))
        return (1, str(x))

    return sorted(ids, key=key_fn)


def caption_from_json_entry(entry):
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        if not entry:
            raise ValueError("empty caption object")

        def inner_key_order(k):
            s = str(k)
            try:
                return (0, int(s))
            except ValueError:
                return (1, s)

        ordered = sorted(entry.keys(), key=inner_key_order)
        return str(entry[ordered[0]]).strip()
    raise ValueError(f"unsupported caption entry type: {type(entry)}")


def object_to_plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [object_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [object_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): object_to_plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return object_to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return object_to_plain(value.dict())
    if hasattr(value, "__dict__"):
        return {
            key: object_to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def find_video_urls(value):
    urls = []
    if isinstance(value, str):
        if value.startswith(("http://", "https://")) and any(
            token in value.lower() for token in [".mp4", ".mov", "video"]
        ):
            urls.append(value)
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(find_video_urls(item))
        return urls
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                if "video" in key_lower or any(ext in item.lower() for ext in [".mp4", ".mov"]):
                    urls.append(item)
            urls.extend(find_video_urls(item))
    return urls


def download_url(url, output_path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response:
        tmp_path.write_bytes(response.read())
    tmp_path.replace(output_path)


def call_with_retries(label, func, max_retries, retry_wait):
    attempt = 0
    wait_seconds = retry_wait
    while True:
        try:
            return func()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            print(
                f"[retry] {label}: {type(exc).__name__}: {exc} "
                f"(wait {wait_seconds}s, {attempt}/{max_retries})"
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2


def build_video_prompt(caption, prompt_mode):
    if prompt_mode == "raw":
        return caption
    if prompt_mode == "strong_reference":
        return (
            "Based on Image 1, generate a realistic video. "
            "Use the reference image as the character appearance reference, preserving the person's identity and facial features as much as possible. "
            "Transform the pencil sketch reference into a natural, lifelike real-world video style; do not keep the pencil sketch style in the final video. "
            "The character's action, clothing, role identity, scene, and atmosphere should follow this caption exactly: "
            f"{caption}"
        )
    if prompt_mode == "caption_first":
        return (
            "Use image 1 as the character appearance reference, preserving the person's facial features."
            "Do not copy Image 1 as the first frame, last frame, pose, framing, background, or pencil-sketch style."
            "Generate a realistic, continuous video following this caption. "
            f"{caption}"
        )
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def append_seedance_text_cli_suffix(prompt: str, args) -> str:
    """Ark / Seedance 官方习惯：在文本末尾追加 CLI 风格参数（与 tasks.create 的 duration/watermark 一致）。"""
    parts = []
    if getattr(args, "watermark", False):
        parts.append("--wm true")
    parts.append(f"--dur {int(args.duration)}")
    return prompt.rstrip() + " " + " ".join(parts)


def parse_skip_ids(skip_ids_str):
    if not skip_ids_str or not str(skip_ids_str).strip():
        return set()
    return {p.strip() for p in str(skip_ids_str).split(",") if p.strip()}


def id_num_in_range(item_id, id_num_min, id_num_max):
    """Keep folder keys like id001..id200 when --id-num-min / --id-num-max are set."""
    if id_num_min is None and id_num_max is None:
        return True
    m = re.fullmatch(r"id(\d+)", str(item_id))
    if not m:
        return False
    n = int(m.group(1))
    if id_num_min is not None and n < id_num_min:
        return False
    if id_num_max is not None and n > id_num_max:
        return False
    return True


def iter_items(
    input_dir,
    captions_json=None,
    prompt_filename="prompt.txt",
    reference_image_dir=None,
    reference_image_filename="pencil_full_body.png",
    skip_ids=None,
    id_num_min=None,
    id_num_max=None,
):
    input_dir = Path(input_dir)
    image_root = Path(reference_image_dir) if reference_image_dir else input_dir
    skip_ids = skip_ids or set()

    if captions_json:
        data = json.loads(Path(captions_json).read_text(encoding="utf-8"))
        for item_id in natural_sample_id_sort(data.keys()):
            if not id_num_in_range(item_id, id_num_min, id_num_max):
                continue
            if item_id in skip_ids:
                print(f"[skip] {item_id}: excluded by --skip-ids")
                continue
            image_path = image_root / item_id / reference_image_filename
            if not image_path.is_file():
                print(f"[warn] skip {item_id}: missing {image_path}")
                continue
            caption = caption_from_json_entry(data[item_id])
            yield item_id, image_path, caption
        return

    for item_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        item_id = item_dir.name
        if not id_num_in_range(item_id, id_num_min, id_num_max):
            continue
        if item_id in skip_ids:
            print(f"[skip] {item_id}: excluded by --skip-ids")
            continue
        prompt_path = item_dir / prompt_filename
        image_path = image_root / item_id / reference_image_filename
        if not prompt_path.is_file():
            print(f"[warn] skip {item_id}: missing {prompt_filename} in {item_dir}")
            continue
        if not image_path.is_file():
            print(f"[warn] skip {item_id}: missing reference image {image_path}")
            continue
        caption = prompt_path.read_text(encoding="utf-8").strip()
        yield item_id, image_path, caption


def create_task(client, args, image_url, prompt):
    """Call Ark tasks.create; drop kwargs the installed SDK does not support (version drift)."""
    content = [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url,
            },
            "role": "reference_image",
        },
    ]
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
                    "audio generation flag ignored. Upgrade volcengine-python-sdk[ark] if you need it."
                )
            continue


def poll_task(client, task_id, args):
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


def process_one(client, args, item_id, image_path, caption, output_root):
    output_dir = output_root / item_id
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / f"{item_id}.mp4"
    metadata_path = output_dir / "metadata.json"
    prompt_out_path = output_dir / "video_prompt.txt"

    # 将本次作为 caption 读入的正文落盘，便于输出目录自包含（与送 API 的 video_prompt.txt 区分）
    mirror_name = (
        "video_prompt_ace_enhanced.txt"
        if getattr(args, "prompt_filename", "") == "video_prompt_ace_enhanced.txt"
        else "caption_input_snapshot.txt"
    )
    (output_dir / mirror_name).write_text(caption.strip() + "\n", encoding="utf-8")

    if video_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"[skip] {item_id}: output exists")
        return

    video_prompt = append_seedance_text_cli_suffix(
        build_video_prompt(caption, args.prompt_mode), args
    )
    if args.image_source == "local":
        image_url = build_local_data_url(Path(image_path), args)
    else:
        image_url = args.github_raw_template.format(id=item_id)

    metadata = {
        "id": item_id,
        "model": args.model,
        "image_source": args.image_source,
        "reference_image_path": public_path(image_path),
        "image_url": image_url
        if args.image_source != "local"
        else "data:image/png;base64,<omitted>",
        "ratio": args.ratio,
        "resolution": args.resolution,
        "duration": args.duration,
        "generate_audio": args.generate_audio,
        "watermark": args.watermark,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.captions_json:
        metadata["captions_json"] = public_path(Path(args.captions_json))
    if args.reference_image_dir:
        metadata["reference_image_dir"] = public_path(Path(args.reference_image_dir))
    metadata["reference_image_filename"] = args.reference_image_filename
    if args.local_image_max_long_edge is not None:
        metadata["local_image_max_long_edge"] = args.local_image_max_long_edge
    if args.local_image_jpeg_quality is not None:
        metadata["local_image_jpeg_quality"] = args.local_image_jpeg_quality
    prompt_out_path.write_text(video_prompt, encoding="utf-8")

    if args.dry_run:
        metadata["dry_run"] = True
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        hint = (
            f"local {image_path}"
            if args.image_source == "local"
            else args.github_raw_template.format(id=item_id)
        )
        print(f"[dry-run] {item_id}: {hint}")
        return

    create_result = call_with_retries(
        f"create task {item_id}",
        lambda: create_task(client, args, image_url, video_prompt),
        args.max_retries,
        args.retry_wait,
    )
    task_id = create_result.id
    metadata["task_id"] = task_id
    metadata["create_result"] = redact_urls(object_to_plain(create_result))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  task_id={task_id}")

    final_result = poll_task(client, task_id, args)
    final_plain = object_to_plain(final_result)
    video_urls = find_video_urls(final_plain)
    metadata["final_result"] = redact_urls(final_plain)
    metadata["video_urls"] = [redact_urls(url) for url in video_urls]

    if not video_urls:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("Task succeeded but no video URL was found in the result.")

    download_url(video_urls[0], video_path)
    metadata["output_video"] = public_path(video_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {item_id}: {video_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate realistic videos with Seedance 2 from pencil reference images and prompts."
    )
    parser.add_argument("--input-dir", default="runs/pencil_references")
    parser.add_argument("--output-dir", default="runs/seedance_videos")
    parser.add_argument(
        "--captions-json",
        default=None,
        help="Use face captions from this JSON (id -> caption or id -> {prompt_index: caption}). "
        "When set, per-folder prompt files are not used (--prompt-filename is ignored).",
    )
    parser.add_argument(
        "--prompt-filename",
        default="prompt.txt",
        help="When not using --captions-json, read this file under each id folder as the caption / prompt body.",
    )
    parser.add_argument(
        "--reference-image-dir",
        default=None,
        help="Directory containing <id>/<reference-image-filename>. "
        "Default: same as --input-dir (reference image next to the prompt file).",
    )
    parser.add_argument(
        "--reference-image-filename",
        default="pencil_full_body.png",
        help="Reference image basename under each id folder (see --reference-image-dir).",
    )
    parser.add_argument(
        "--image-source",
        choices=["github", "local"],
        default="github",
        help="github: reference image URL from --github-raw-template; "
        "local: read the reference image file as a base64 data URL.",
    )
    parser.add_argument(
        "--local-image-max-long-edge",
        type=int,
        default=None,
        metavar="PX",
        help="When --image-source local: shrink image so max(width,height) <= PX before base64 "
        "(needs Pillow). Reduces JSON body size and avoids oversized requests.",
    )
    parser.add_argument(
        "--local-image-jpeg-quality",
        type=int,
        default=None,
        metavar="1-95",
        help="When --image-source local: re-encode as JPEG at this quality before base64 "
        "(needs Pillow). Usually much smaller than PNG; omit to keep PNG after any resize.",
    )
    parser.add_argument("--github-raw-template", default=DEFAULT_GITHUB_RAW_TEMPLATE)
    parser.add_argument("--base-url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Ark API key. Defaults to ARK_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2-0-260128"),
        help="Ark 模型 id；未传参时默认读环境变量 SEEDANCE_MODEL，再退回 doubao-seedance-2-0-260128。",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["caption_first", "strong_reference", "raw"],
        default="caption_first",
        help="caption_first / strong_reference wrap the text with the built-in reference-image template; "
        "raw sends the file or JSON caption to the model unchanged.",
    )
    parser.add_argument("--ratio", default="16:9")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument(
        "--duration",
        type=int,
        default=4,
        help="Seconds for Seedance; also appended to prompt text as '--dur N' (Ark CLI-style suffix).",
    )
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--id-num-min",
        type=int,
        default=None,
        metavar="N",
        help="Only process folders whose name matches idNNN with numeric part >= N (e.g. 1 for id001).",
    )
    parser.add_argument(
        "--id-num-max",
        type=int,
        default=None,
        metavar="N",
        help="Only process idNNN with numeric part <= N (e.g. 20 for id020).",
    )
    parser.add_argument(
        "--skip-ids",
        default="",
        metavar="ID,...",
        help="Comma-separated sample folder names to exclude (e.g. id006,id113).",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Maximum number of Seedance tasks running at once.")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_client(args, api_key):
    if args.dry_run:
        return None
    try:
        from volcenginesdkarkruntime import Ark
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: volcenginesdkarkruntime. Install it with "
                    "Activate the Conda environment and run `python -m pip install -r requirements.txt`."
        ) from exc
    return Ark(base_url=args.base_url, api_key=api_key)


def run_item(args, api_key, output_root, item_index, total_items, item):
    item_id, image_path, caption = item
    print(f"[{item_index}/{total_items}] {item_id}")
    try:
        client = make_client(args, api_key)
        process_one(client, args, item_id, image_path, caption, output_root)
        return item_id, None
    except Exception as exc:
        error_path = output_root / item_id / "error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        print(f"[error] {item_id}: {exc}")
        return item_id, exc


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Missing Ark API key. Set `export ARK_API_KEY='your_key'` or pass --api-key.")

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    ref_img_root = Path(args.reference_image_dir) if args.reference_image_dir else None
    if args.reference_image_dir and not ref_img_root.is_dir():
        raise FileNotFoundError(f"Reference image directory not found: {ref_img_root}")

    skip_ids = parse_skip_ids(args.skip_ids)
    if skip_ids:
        print(f"Excluding sample id(s): {', '.join(sorted(skip_ids, key=str))}")

    items = list(
        iter_items(
            input_dir,
            args.captions_json,
            prompt_filename=args.prompt_filename,
            reference_image_dir=args.reference_image_dir,
            reference_image_filename=args.reference_image_filename,
            skip_ids=skip_ids,
            id_num_min=args.id_num_min,
            id_num_max=args.id_num_max,
        )
    )
    if args.limit is not None:
        items = items[: args.limit]

    worker_count = min(args.concurrency, len(items)) if items else 0
    print(
        f"Processing {len(items)} folders from {input_dir} -> {output_root} "
        f"with concurrency={worker_count}"
    )
    if not items:
        print("No items to process.")
        return

    failures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_item, args, api_key, output_root, index, len(items), item)
            for index, item in enumerate(items, start=1)
        ]
        for future in as_completed(futures):
            item_id, exc = future.result()
            if exc is not None:
                failures.append(item_id)

    if failures:
        print(f"Finished with {len(failures)} failed item(s): {', '.join(sorted(failures))}")
    else:
        print("Finished successfully.")


if __name__ == "__main__":
    main()
