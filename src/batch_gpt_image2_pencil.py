import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from public_safety import public_path

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: openai. Activate the Conda environment and run `python -m pip install -r requirements.txt`."
    ) from exc


CHINESE_TASK = (
    "图片是人物的ID图，文本是视频的caption描述，保持ID图整个图像结构不变但是风格变为铅笔画风格"
    "即整个脸部不变化，生成符合文本的合理的人物的全身图，注意参考文本里的服饰和角色身份，"
    "生成的人体身材和比例要自然、正常、符合人体解剖结构"
)

ENGLISH_TASK = (
    "The image is a person's ID reference image, and the text is the caption description of a video. "
    "Keep the entire structure and identity of the ID image unchanged, but convert the style into a pencil sketch. "
    "Do not change the face at all. Generate a reasonable full-body image of the person that matches the text. "
    "Pay close attention to the clothing and role identity described in the text. "
    "Ensure the generated full-body figure has natural, normal, anatomically plausible body shape and proportions."
)


def image_to_data_url(image_path):
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_json(text):
    text = text.strip()
    if text.startswith("<!DOCTYPE html>") or text.startswith("<html"):
        raise ValueError(
            "The API returned an HTML page instead of JSON. "
            "Your base URL is probably the website root; use an OpenAI-compatible API URL such as https://yunwu.ai/v1."
        )
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON in model output: {text[:300]}")
    return json.loads(text[start : end + 1])


def get_response_text(response):
    if isinstance(response, str):
        return response
    if hasattr(response, "output_text"):
        return response.output_text
    if isinstance(response, dict):
        if "output_text" in response:
            return response["output_text"]
        if "choices" in response and response["choices"]:
            message = response["choices"][0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise TypeError(f"Could not read text from response type: {type(response).__name__}")


def call_with_retries(label, func, max_retries, initial_wait):
    attempt = 0
    wait_seconds = initial_wait
    while True:
        try:
            return func()
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            print(f"[retry] {label}: {type(exc).__name__}, wait {wait_seconds}s ({attempt}/{max_retries})")
            time.sleep(wait_seconds)
            wait_seconds *= 2


def build_prompt_payload(client, model, image_path, original_prompt, max_retries, retry_wait):
    data_url = image_to_data_url(image_path)
    instruction = f"""
You prepare prompts for an image generation/editing model.

Task:
1. Translate this Chinese image instruction into natural English, preserving its meaning:
{CHINESE_TASK}

2. Preserve the original video caption content. Only fix English gendered pronouns if they clearly conflict with the visible person in the ID image.
   - If the visible person appears feminine and the caption uses he/him/his, change those pronouns to she/her/her.
   - If the visible person appears masculine and the caption uses she/her/hers, change those pronouns to he/him/his.
   - If the visual presentation is unclear, do not change pronouns.
   - Do not summarize, rewrite, shorten, or otherwise change the caption.

3. Return JSON only, with these fields:
   original_prompt: the exact original caption
   corrected_prompt: the caption after pronoun correction, or the original caption if no correction is needed
   apparent_presentation: "feminine", "masculine", or "unclear"
   pronoun_changed: true or false
   generation_prompt_en: the final English prompt to send to the image model

The final generation_prompt_en must include the translated image instruction and the corrected caption verbatim.

Original caption:
{original_prompt}
""".strip()

    response = call_with_retries(
        "prepare prompt",
        lambda: client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "prompt_payload",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "original_prompt": {"type": "string"},
                            "corrected_prompt": {"type": "string"},
                            "apparent_presentation": {
                                "type": "string",
                                "enum": ["feminine", "masculine", "unclear"],
                            },
                            "pronoun_changed": {"type": "boolean"},
                            "generation_prompt_en": {"type": "string"},
                        },
                        "required": [
                            "original_prompt",
                            "corrected_prompt",
                            "apparent_presentation",
                            "pronoun_changed",
                            "generation_prompt_en",
                        ],
                    },
                }
            },
        ),
        max_retries,
        retry_wait,
    )
    return extract_json(get_response_text(response))


def edit_image(client, args, image_path, generation_prompt, output_path):
    request = {
        "model": args.image_model,
        "prompt": generation_prompt,
        "size": args.size,
        "quality": args.quality,
        "output_format": "png",
    }
    if args.input_fidelity:
        request["input_fidelity"] = args.input_fidelity

    with image_path.open("rb") as image_file:
        request["image"] = [image_file]
        result = call_with_retries(
            "generate image",
            lambda: client.images.edit(**request),
            args.max_retries,
            args.retry_wait,
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


def find_items(input_dir):
    for item_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        image_path = item_dir / "image.png"
        prompt_files = sorted(item_dir.glob("prompt*.txt"))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image.png in {item_dir}")
        if len(prompt_files) != 1:
            raise ValueError(f"Expected exactly one prompt*.txt in {item_dir}, found {len(prompt_files)}")
        yield item_dir.name, image_path, prompt_files[0]


def process_one(client, args, item_id, image_path, prompt_path, output_root):
    item_output_dir = output_root / item_id
    item_output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = item_output_dir / "pencil_full_body.png"
    metadata_path = item_output_dir / "metadata.json"

    if generated_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"[skip] {item_id}: output exists")
        return

    original_prompt = prompt_path.read_text(encoding="utf-8").strip()
    payload = build_prompt_payload(
        client,
        args.text_model,
        image_path,
        original_prompt,
        args.max_retries,
        args.retry_wait,
    )

    original_out = item_output_dir / "original_prompt.txt"
    corrected_out = item_output_dir / "prompt.txt"
    generation_prompt_out = item_output_dir / "generation_prompt_en.txt"

    original_out.write_text(payload["original_prompt"], encoding="utf-8")
    corrected_out.write_text(payload["corrected_prompt"], encoding="utf-8")
    generation_prompt_out.write_text(payload["generation_prompt_en"], encoding="utf-8")

    metadata = {
        "id": item_id,
        "source_image": public_path(image_path),
        "source_prompt_file": public_path(prompt_path),
        "image_model": args.image_model,
        "text_model": args.text_model,
        "size": args.size,
        "quality": args.quality,
        "input_fidelity": args.input_fidelity,
        "chinese_task": CHINESE_TASK,
        "english_task_reference": ENGLISH_TASK,
        "apparent_presentation": payload.get("apparent_presentation"),
        "pronoun_changed": payload.get("pronoun_changed"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.dry_run:
        metadata["dry_run"] = True
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[dry-run] {item_id}: prompt prepared")
        return

    edit_image(client, args, image_path, payload["generation_prompt_en"], generated_path)
    metadata["output_image"] = public_path(generated_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {item_id}: {generated_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate pencil sketch full-body images from test_50_mixed ID images and prompts."
    )
    parser.add_argument("--input-dir", default="examples/data/input", help="Folder containing id subfolders.")
    parser.add_argument(
        "--output-dir",
        default="runs/pencil_references",
        help="Folder for generated images and prompts.",
    )
    parser.add_argument("--image-model", default="gpt-image-2", help="OpenAI image edit model.")
    parser.add_argument(
        "--text-model",
        default="gpt-5.5",
        help="Vision-capable model for prompt prep via responses.create (default: gpt-5.5).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL") or "https://yunwu.ai/v1",
        help="OpenAI-compatible API base URL. Default: https://yunwu.ai/v1. Override with this flag or OPENAI_BASE_URL.",
    )
    parser.add_argument("--size", default="1024x1536", help="Generated image size, e.g. 1024x1536 or auto.")
    parser.add_argument("--quality", default="medium", help="Image quality: high, medium, low, or auto.")
    parser.add_argument(
        "--input-fidelity",
        default="",
        help="Optional input fidelity for models that support it. Leave empty for gpt-image-2.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of folders to process.")
    parser.add_argument("--max-retries", type=int, default=6, help="Retries for 429/timeouts/connection errors.")
    parser.add_argument("--retry-wait", type=int, default=30, help="Initial wait seconds before retrying.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts and metadata without generating images.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds per API call (vision + JSON prompt prep is slow). Default: 600.",
    )
    return parser.parse_args()


def normalize_base_url(base_url):
    if not base_url:
        return None
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if parsed.netloc == "yunwu.ai" and path == "":
        return "https://yunwu.ai/v1"
    return base_url


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.input_fidelity == "":
        args.input_fidelity = None

    output_root.mkdir(parents=True, exist_ok=True)
    client_kwargs = {"timeout": args.client_timeout}
    args.base_url = normalize_base_url(args.base_url)
    if args.base_url:
        print(f"Using API base URL: {args.base_url}")
        client_kwargs["base_url"] = args.base_url
    print(f"Using HTTP client timeout: {args.client_timeout}s")
    client = OpenAI(**client_kwargs)

    items = list(find_items(input_dir))
    if args.limit is not None:
        items = items[: args.limit]

    print(f"Processing {len(items)} folders from {input_dir} -> {output_root}")
    for index, (item_id, image_path, prompt_path) in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item_id}")
        try:
            process_one(client, args, item_id, image_path, prompt_path, output_root)
        except Exception as exc:
            error_path = output_root / item_id / "error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            print(f"[error] {item_id}: {exc}")


if __name__ == "__main__":
    main()
