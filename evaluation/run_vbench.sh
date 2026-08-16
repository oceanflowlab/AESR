#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <videos-dir> <results-dir>" >&2
  exit 2
fi

VIDEOS_DIR="$1"
RESULTS_DIR="$2"
VBENCH_ROOT="${EVAL_VBENCH_ROOT:-}"

if [[ -z "$VBENCH_ROOT" ]]; then
  echo "Set EVAL_VBENCH_ROOT to a VBench checkout containing evaluate.py." >&2
  exit 2
fi
if [[ ! -f "$VBENCH_ROOT/evaluate.py" ]]; then
  echo "VBench evaluator not found: $VBENCH_ROOT/evaluate.py" >&2
  exit 2
fi
if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "Video directory not found: $VIDEOS_DIR" >&2
  exit 2
fi

shopt -s nullglob
videos=("$VIDEOS_DIR"/*.mp4)
if [[ ${#videos[@]} -eq 0 ]]; then
  echo "No .mp4 files found in $VIDEOS_DIR" >&2
  exit 2
fi

for video in "${videos[@]}"; do
  name="$(basename "$video" .mp4)"
  output_dir="$RESULTS_DIR/$name"
  result_json="$output_dir/${name}_Vbench_eval_results.json"
  if [[ -f "$result_json" ]]; then
    echo "Skipping $name: $result_json already exists"
    continue
  fi
  mkdir -p "$output_dir"
  python "$VBENCH_ROOT/evaluate.py" \
    --dimension motion_smoothness imaging_quality \
    --videos_path "$video" \
    --mode custom_input \
    --output_path "$output_dir"
done
