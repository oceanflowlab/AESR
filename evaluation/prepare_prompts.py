#!/usr/bin/env python3
"""Write evaluator-friendly prompt text files from an IPVG Track 1 eval.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_eval_json(dataset: Path) -> Path:
    candidates = [dataset / "eval.json", dataset / "IPVG2026-Test-Track1" / "eval.json"]
    candidates.extend(path for path in dataset.rglob("eval.json") if path.is_file())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find eval.json below {dataset}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted Track 1 test-set directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for <sample-id>.txt prompts")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        eval_path = find_eval_json(args.dataset.resolve())
        records = json.loads(eval_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not isinstance(records, list) or not records:
        parser.error(f"Expected a non-empty JSON list in {eval_path}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            parser.error(f"Record {index} in {eval_path} is not an object")
        image_name = Path(str(record.get("img", ""))).name
        prompt = str(record.get("prompt", "")).strip()
        if not image_name or not prompt:
            parser.error(f"Record {index} is missing img or prompt")
        target = output / f"{Path(image_name).stem}.txt"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        target.write_text(prompt + "\n", encoding="utf-8")
        written += 1

    print(f"Prompts written: {written}; existing prompts kept: {skipped}")
    print(f"Prompt directory: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
