#!/usr/bin/env python3
"""Select the highest-scoring candidate for each IPVG Track 1 sample."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from aggregate_results import collect_video_metrics


REQUIRED_METRICS = (
    "GME-Score",
    "story_video_consistency",
    "cur_score",
    "arc_score",
    "motion_smoothness",
    "imaging_quality",
)


def extract_id_number(video_stem: str) -> int | None:
    if not video_stem.startswith("id"):
        return None
    number = video_stem[2:].split("_", 1)[0]
    return int(number) if number.isdigit() and int(number) > 0 else None


def safe_suffix(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return sanitized.strip("._-") or "method"


def load_report_metrics(results_dir: Path) -> dict[str, dict[str, float]]:
    report_path = results_dir / "final_results.json"
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {report_path}: {exc}") from exc
    report: dict[str, dict[str, float]] = {}
    for item in payload.get("results", []):
        if not isinstance(item, dict) or not isinstance(item.get("video_name"), str):
            continue
        values: dict[str, float] = {}
        for key in REQUIRED_METRICS:
            try:
                value = float(item[key])
            except (KeyError, TypeError, ValueError):
                continue
            values[key] = value
        report[item["video_name"]] = values
    return report


def candidate_video_path(videos_dir: Path, video_stem: str) -> Path | None:
    direct = videos_dir / f"{video_stem}.mp4"
    if direct.is_file():
        return direct
    sample_id = extract_id_number(video_stem)
    if sample_id is None:
        return None
    matches = sorted(videos_dir.glob(f"id{sample_id:03d}_*.mp4"))
    return matches[0] if matches else None


def score(metrics: dict[str, float]) -> float:
    return (
        0.15 * (metrics["GME-Score"] + metrics["story_video_consistency"])
        + 0.20 * (metrics["cur_score"] + metrics["arc_score"])
        + 0.15 * (metrics["motion_smoothness"] + metrics["imaging_quality"])
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-root",
        type=Path,
        required=True,
        help="Directory containing method video folders and matching Results_<folder> folders",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for selected id<N>.mp4 files")
    parser.add_argument("--id-start", type=int, default=1)
    parser.add_argument("--id-end", type=int, default=200)
    parser.add_argument("--copy-videos", action="store_true", help="Copy selected videos into --output")
    args = parser.parse_args()

    root = args.candidates_root.resolve()
    if not root.is_dir():
        parser.error(f"Candidate root does not exist: {root}")
    if args.id_start < 1 or args.id_end < args.id_start:
        parser.error("Use valid --id-start and --id-end values")

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for results_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("Results_")):
        method_dir = root / results_dir.name.removeprefix("Results_")
        if not method_dir.is_dir():
            skipped.append({"result_dir": str(results_dir), "video_name": "", "reason": "matching video folder not found"})
            continue
        report_metrics = load_report_metrics(results_dir)
        for video_result_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            sample_id = extract_id_number(video_result_dir.name)
            if sample_id is None or not args.id_start <= sample_id <= args.id_end:
                continue
            video = candidate_video_path(method_dir, video_result_dir.name)
            if video is None:
                skipped.append({"result_dir": str(results_dir), "video_name": video_result_dir.name, "reason": "source video not found"})
                continue
            metrics = collect_video_metrics(video_result_dir)
            metrics.update(report_metrics.get(video_result_dir.name, {}))
            missing = [key for key in REQUIRED_METRICS if key not in metrics]
            if missing:
                skipped.append({"result_dir": str(results_dir), "video_name": video_result_dir.name, "reason": "missing metrics: " + ", ".join(missing)})
                continue
            candidates.append(
                {
                    "id": sample_id,
                    "video_name": video_result_dir.name,
                    "method_name": results_dir.name,
                    "source_video": str(video),
                    "score": score(metrics),
                    **metrics,
                }
            )

    best: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        existing = best.get(candidate["id"])
        if existing is None or candidate["score"] > existing["score"]:
            best[candidate["id"]] = candidate
    selected = [best[sample_id] for sample_id in sorted(best)]

    args.output.mkdir(parents=True, exist_ok=True)
    if args.copy_videos:
        for candidate in selected:
            shutil.copy2(candidate["source_video"], args.output / f"id{candidate['id']}.mp4")
    for sample_id in range(args.id_start, args.id_end + 1):
        if sample_id not in best:
            skipped.append({"result_dir": "", "video_name": f"id{sample_id}", "reason": "no completely scored candidate"})

    fields = ["id", "video_name", "method_name", "score", *REQUIRED_METRICS, "source_video"]
    write_csv(args.output / "selection_manifest.csv", selected, fields)
    (args.output / "selection_manifest.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "selection_skipped.csv", skipped, ["result_dir", "video_name", "reason"])
    print(f"Complete candidates: {len(candidates)}")
    print(f"Selected samples: {len(selected)}")
    print(f"Selection manifest: {args.output / 'selection_manifest.json'}")
    if args.copy_videos:
        print(f"Copied submission videos: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
