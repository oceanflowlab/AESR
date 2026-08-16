#!/usr/bin/env python3
"""Compute the GME text-alignment score for videos and matching prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


def find_prompt(video: Path, prompts_dir: Path) -> Path | None:
    candidates = [prompts_dir / f"{video.stem}.txt"]
    if "_" in video.stem:
        candidates.append(prompts_dir / f"{video.stem.split('_', 1)[0]}.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def iter_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_SUFFIXES else []
    return sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def load_runtime(device_name: str, model_source: str, local_files_only: bool) -> tuple[Any, Any, Any, Any]:
    try:
        import av
        import numpy as np
        import torch
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing GME dependencies. Run: pip install -r evaluation/requirements.txt"
        ) from exc

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device_name not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported device: {device_name}")

    dtype = torch.float16 if device_name == "cuda" else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if device_name == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(model_source, **model_kwargs)
    if device_name == "cpu":
        model.to("cpu")
    model.eval()
    return av, np, torch, model


def extract_frames(av: Any, np: Any, video: Path, max_frames: int) -> list[Any]:
    container = av.open(str(video))
    try:
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames and total_frames > max_frames:
            indices = set(np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist())
            frames = [
                frame.to_image()
                for index, frame in enumerate(container.decode(video=0))
                if index in indices
            ]
        else:
            frames = [frame.to_image() for frame in container.decode(video=0)]
    finally:
        container.close()

    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[index] for index in indices]


def score_video(
    model: Any,
    av: Any,
    np: Any,
    torch: Any,
    video: Path,
    prompt: str,
    max_frames: int,
    batch_size: int,
) -> tuple[float, list[float]]:
    frames = extract_frames(av, np, video, max_frames)
    if not frames:
        raise ValueError(f"No video frames could be decoded: {video}")

    instruction = "Find frames that match the given text description."
    similarities: list[float] = []
    with torch.inference_mode():
        text_embedding = model.get_text_embeddings(texts=[prompt], instruction=instruction)
        for start in range(0, len(frames), batch_size):
            image_embeddings = model.get_image_embeddings(
                images=frames[start : start + batch_size], is_query=False
            )
            similarities.extend((text_embedding @ image_embeddings.T).tolist()[0])
    return float(np.mean(similarities)), [float(value) for value in similarities]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, help="Video file or directory")
    parser.add_argument("prompts", type=Path, help="Directory containing matching .txt prompts")
    parser.add_argument("results", type=Path, help="Directory for per-video result folders")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--model-path", default=None, help="Local model path overriding --model")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_frames < 1 or args.batch_size < 1:
        parser.error("--max-frames and --batch-size must be positive")
    videos = iter_videos(args.videos.resolve())
    if not videos:
        parser.error(f"No supported video files found in {args.videos}")
    prompts_dir = args.prompts.resolve()
    if not prompts_dir.is_dir():
        parser.error(f"Prompt directory does not exist: {prompts_dir}")

    try:
        av, np, torch, model = load_runtime(
            args.device, args.model_path or args.model, args.local_files_only
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    for video in videos:
        prompt_path = find_prompt(video, prompts_dir)
        if prompt_path is None:
            print(f"Skipping {video.name}: no matching prompt")
            continue
        output = args.results.resolve() / video.stem / "GME.json"
        if output.exists() and not args.overwrite:
            print(f"Skipping {video.name}: {output} already exists")
            continue
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            print(f"Skipping {video.name}: prompt is empty")
            continue
        try:
            score, frame_scores = score_video(
                model, av, np, torch, video, prompt, args.max_frames, args.batch_size
            )
        except Exception as exc:
            print(f"Failed to score {video.name}: {exc}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": video.name,
            "prompt": prompt,
            "GME-Score": score,
            "GME_frame_similarities": frame_scores,
            "model": args.model_path or args.model,
            "max_frames": args.max_frames,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{video.name}: GME-Score={score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
