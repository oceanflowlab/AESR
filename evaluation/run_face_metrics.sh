#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <videos-dir> <reference-images-dir> <results-dir>" >&2
  exit 2
fi

VIDEOS_DIR="$1"
REFERENCES_DIR="$2"
RESULTS_DIR="$3"
CONSISID_ROOT="${EVAL_CONSISID_ROOT:-}"

if [[ -z "$CONSISID_ROOT" ]]; then
  echo "Set EVAL_CONSISID_ROOT to a ConsisID checkout containing cal_face_sim.py." >&2
  exit 2
fi
if [[ ! -f "$CONSISID_ROOT/cal_face_sim.py" ]]; then
  echo "Face evaluator not found: $CONSISID_ROOT/cal_face_sim.py" >&2
  exit 2
fi
if [[ ! -d "$VIDEOS_DIR" || ! -d "$REFERENCES_DIR" ]]; then
  echo "Video or reference-image directory does not exist." >&2
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
  sample_id="${name%%_*}"
  reference=""
  for extension in png jpg jpeg webp; do
    if [[ -f "$REFERENCES_DIR/$sample_id.$extension" ]]; then
      reference="$REFERENCES_DIR/$sample_id.$extension"
      break
    fi
  done
  if [[ -z "$reference" ]]; then
    echo "Skipping $name: reference image for $sample_id was not found" >&2
    continue
  fi
  output_dir="$RESULTS_DIR/$name"
  output_json="$output_dir/face_similarity.json"
  if [[ -f "$output_json" ]]; then
    echo "Skipping $name: $output_json already exists"
    continue
  fi
  mkdir -p "$output_dir"
  python "$CONSISID_ROOT/cal_face_sim.py" "$video" "$reference" "$output_json"
done
