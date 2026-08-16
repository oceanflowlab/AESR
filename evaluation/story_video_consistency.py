#!/usr/bin/env python3
"""Compute the MSVBench-style Story-Video Consistency metric used by AESR."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
CAPTION_PROMPT = """Give a detailed video caption of the input video. Describe characters, environments, dynamics, camera movement, and fine-grained element interactions.
You should output 150-250 words."""


def load_api_keys() -> list[str]:
    keys: list[str] = []
    raw = os.environ.get("GEMINI_API_KEYS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.split(",")
        if isinstance(parsed, list):
            keys.extend(str(value).strip() for value in parsed if str(value).strip())
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        keys.append(key)
    return list(dict.fromkeys(keys))


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


def load_components(msvbench_root: Path, kalm_model_path: Path, device: str) -> tuple[Any, Any, Any]:
    tools_dir = msvbench_root / "Tools"
    if not (tools_dir / "gemini_api.py").is_file():
        raise RuntimeError(
            "MSVBench Tools/gemini_api.py was not found. Set --msvbench-root to a complete "
            "MSVBench checkout. See evaluation/README.md."
        )
    if not kalm_model_path.exists():
        raise RuntimeError(f"KaLM embedding model was not found: {kalm_model_path}")
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing story-metric dependencies. Run: pip install -r evaluation/requirements.txt"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    sys.path.insert(0, str(tools_dir))
    try:
        gemini_module = importlib.import_module("gemini_api")
    except ImportError as exc:
        raise RuntimeError(f"Could not import {tools_dir / 'gemini_api.py'}: {exc}") from exc
    gemini_api = getattr(gemini_module, "GeminiAPI", None)
    if gemini_api is None:
        raise RuntimeError(f"GeminiAPI was not defined by {tools_dir / 'gemini_api.py'}")

    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = SentenceTransformer(
        str(kalm_model_path),
        trust_remote_code=True,
        device=device,
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = 512
    return np, gemini_api, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, help="Video file or directory")
    parser.add_argument("prompts", type=Path, help="Directory containing matching .txt prompts")
    parser.add_argument("results", type=Path, help="Directory for per-video result folders")
    parser.add_argument(
        "--msvbench-root",
        type=Path,
        default=os.environ.get("EVAL_MSVBENCH_ROOT"),
        help="Complete MSVBench checkout containing Tools/gemini_api.py",
    )
    parser.add_argument(
        "--kalm-model-path",
        type=Path,
        default=os.environ.get("EVAL_KALM_MODEL_PATH"),
        help="Local KaLM embedding model directory",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--proxy", default=None, help="Optional HTTP(S) proxy for the upstream Gemini helper")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.msvbench_root is None:
        parser.error("Set --msvbench-root or EVAL_MSVBENCH_ROOT")
    msvbench_root = args.msvbench_root.resolve()
    kalm_model_path = (
        args.kalm_model_path.resolve()
        if args.kalm_model_path is not None
        else msvbench_root
        / "Metrics"
        / "StoryVideoAlignment"
        / "tools"
        / "KaLM-embedding-multilingual-mini-instruct-v2"
    )
    videos = iter_videos(args.videos.resolve())
    if not videos:
        parser.error(f"No supported video files found in {args.videos}")
    prompts_dir = args.prompts.resolve()
    if not prompts_dir.is_dir():
        parser.error(f"Prompt directory does not exist: {prompts_dir}")
    keys = load_api_keys()
    if not keys:
        parser.error("Set GEMINI_API_KEY or GEMINI_API_KEYS in your local environment")

    try:
        np, gemini_api, kalm = load_components(msvbench_root, kalm_model_path, args.device)
        gemini = gemini_api(api_keys=keys, proxy=args.proxy)
    except RuntimeError as exc:
        parser.error(str(exc))

    for video in videos:
        prompt_path = find_prompt(video, prompts_dir)
        if prompt_path is None:
            print(f"Skipping {video.name}: no matching prompt")
            continue
        output = args.results.resolve() / video.stem / "story_video_consistency.json"
        if output.exists() and not args.overwrite:
            print(f"Skipping {video.name}: {output} already exists")
            continue
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            print(f"Skipping {video.name}: prompt is empty")
            continue
        started = time.monotonic()
        try:
            caption = gemini.generate_from_videos([str(video)], CAPTION_PROMPT)
            prompt_embedding = kalm.encode([prompt], normalize_embeddings=True, show_progress_bar=False)[0]
            caption_embedding = kalm.encode([caption], normalize_embeddings=True, show_progress_bar=False)[0]
            similarity = float(np.dot(prompt_embedding, caption_embedding))
        except Exception as exc:
            print(f"Failed to score {video.name}: {exc}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": video.name,
            "prompt": prompt,
            "prompt_file": prompt_path.name,
            "gemini_caption": caption,
            "story_video_consistency": similarity,
            "duration_seconds": time.monotonic() - started,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{video.name}: story_video_consistency={similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
