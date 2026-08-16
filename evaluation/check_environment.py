#!/usr/bin/env python3
"""Check local prerequisites for the complete six-metric AESR evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


PACKAGES = {
    "PyTorch": "torch",
    "PyAV": "av",
    "NumPy": "numpy",
    "Transformers": "transformers",
    "Sentence Transformers": "sentence_transformers",
    "InsightFace": "insightface",
    "ONNX": "onnx",
    "ONNX Runtime": "onnxruntime",
}


def env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not require a CUDA-enabled PyTorch build",
    )
    args = parser.parse_args()

    problems: list[str] = []
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is required")

    for display_name, import_name in PACKAGES.items():
        available = importlib.util.find_spec(import_name) is not None
        print(f"{display_name}: {'OK' if available else 'NOT FOUND'}")
        if not available:
            problems.append(f"missing Python package: {display_name}")

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        has_cuda = bool(torch.cuda.is_available())
        print(f"CUDA: {'available' if has_cuda else 'not available'}")
        if not has_cuda and not args.allow_cpu:
            problems.append("CUDA is unavailable; use --allow-cpu only for slow debugging runs")

    roots = {
        "EVAL_CONSISID_ROOT": "cal_face_sim.py",
        "EVAL_VBENCH_ROOT": "evaluate.py",
        "EVAL_MSVBENCH_ROOT": "Tools/gemini_api.py",
    }
    for variable, relative_path in roots.items():
        root = env_path(variable)
        found = root is not None and (root / relative_path).is_file()
        print(f"{variable}: {str(root) if root else 'NOT SET'}")
        if not found:
            problems.append(f"set {variable} to a checkout containing {relative_path}")

    kalm_path = env_path("EVAL_KALM_MODEL_PATH")
    if kalm_path is None:
        msvbench_root = env_path("EVAL_MSVBENCH_ROOT")
        if msvbench_root is not None:
            kalm_path = (
                msvbench_root
                / "Metrics"
                / "StoryVideoAlignment"
                / "tools"
                / "KaLM-embedding-multilingual-mini-instruct-v2"
            )
    print(f"KaLM model: {str(kalm_path) if kalm_path else 'NOT SET'}")
    if kalm_path is None or not kalm_path.exists():
        problems.append("set EVAL_KALM_MODEL_PATH to the local KaLM embedding model")

    has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip()) or bool(
        os.environ.get("GEMINI_API_KEYS", "").strip()
    )
    print(f"Gemini credential: {'set' if has_key else 'NOT SET'}")
    if not has_key:
        problems.append("set GEMINI_API_KEY or GEMINI_API_KEYS locally for story-video captioning")

    if problems:
        print("\nEvaluation environment check failed:")
        for problem in problems:
            print(f"- {problem}")
        print("\nSee evaluation/README.md for Conda setup and external evaluator paths.")
        return 1

    print("\nEvaluation environment check passed. Credentials were not printed or validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
