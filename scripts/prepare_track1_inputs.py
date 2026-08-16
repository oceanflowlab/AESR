#!/usr/bin/env python3
"""Prepare per-sample Stage I inputs from the official Track 1 archive."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Tuple


DEFAULT_DATASET = Path("data/IPVG2026-Test-Track1")
SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def find_eval_json(dataset: Path) -> Path:
    candidates = [dataset / "eval.json", dataset / "IPVG2026-Test-Track1" / "eval.json"]
    candidates.extend(path for path in dataset.rglob("eval.json") if path.is_file())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find eval.json below {dataset}")


def find_image(dataset: Path, eval_path: Path, image_value: str) -> Path:
    image_name = Path(image_value).name
    candidates = [
        dataset / image_value,
        eval_path.parent / image_value,
        dataset / "images" / image_name,
        eval_path.parent / "images" / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Reference image not found for {image_value}")


def prepare_inputs(dataset: Path, output: Path, overwrite: bool = False) -> Tuple[int, int]:
    dataset = dataset.resolve()
    output = output.resolve()
    eval_path = find_eval_json(dataset)
    records = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list in {eval_path}")

    prepared = 0
    skipped = 0
    sample_ids = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {eval_path} is not an object")
        image_value = str(record.get("img", "")).strip()
        prompt = str(record.get("prompt", "")).strip()
        sample_id = Path(image_value).stem
        if not image_value or not prompt:
            raise ValueError(f"Record {index} is missing img or prompt")
        if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise ValueError(f"Unsafe sample ID in record {index}: {sample_id}")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample ID in {eval_path}: {sample_id}")
        sample_ids.add(sample_id)

        source_image = find_image(dataset, eval_path, image_value)
        suffix = source_image.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"Unsupported reference image format: {source_image}")

        sample_dir = output / sample_id
        prompt_path = sample_dir / "prompt.txt"
        reference_path = sample_dir / f"reference{suffix}"
        existing_references = list(sample_dir.glob("reference.*")) if sample_dir.is_dir() else []
        if prompt_path.is_file() and reference_path.is_file() and not overwrite:
            skipped += 1
            continue
        if not overwrite and (prompt_path.exists() or existing_references):
            raise FileExistsError(f"Incomplete or conflicting output for {sample_id}; use --overwrite")

        sample_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for existing in existing_references:
                existing.unlink()
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        shutil.copy2(source_image, reference_path)
        prepared += 1

    return prepared, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Extracted Track 1 directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Prepared input directory (default: <dataset>/inputs)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.dataset / "inputs"
    prepared, skipped = prepare_inputs(args.dataset, output, args.overwrite)
    print(f"Prepared samples: {prepared}; existing samples kept: {skipped}")
    print(f"Stage I input directory: {output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
