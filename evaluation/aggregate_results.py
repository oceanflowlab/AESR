#!/usr/bin/env python3
"""Collect per-video AESR evaluation files into one portable JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vbench_value(value: Any) -> float | None:
    if isinstance(value, list) and value:
        return _as_number(value[0])
    return _as_number(value)


def collect_video_metrics(video_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metric_files = {
        "GME.json": ("GME-Score",),
        "CLIP.json": ("CLIP-Score",),
        "face_similarity.json": ("cur_score", "arc_score", "fid_score"),
        "story_video_consistency.json": ("story_video_consistency",),
        "state_shift_persistence.json": ("state_shift_persistence",),
    }

    for filename, keys in metric_files.items():
        path = video_dir / filename
        if not path.is_file():
            continue
        payload = _load_json(path)
        for key in keys:
            value = _as_number(payload.get(key))
            if value is not None:
                metrics[key] = value

    for path in sorted(video_dir.glob("*_Vbench_eval_results.json")):
        payload = _load_json(path)
        for key, value in payload.items():
            number = _vbench_value(value)
            if number is not None:
                metrics[key] = number

    return metrics


def aggregate(results_dir: Path) -> dict[str, Any]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    results: list[dict[str, Any]] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for video_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metrics = collect_video_metrics(video_dir)
        if not metrics:
            continue
        results.append({"video_name": video_dir.name, **metrics})
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    means = {key: sums[key] / counts[key] for key in sorted(sums)}
    return {"video_count": len(results), "indicator_means": means, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = aggregate(args.results_dir.resolve())
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Collected {report['video_count']} videos into {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
