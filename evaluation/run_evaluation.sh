#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <videos-dir> <prompts-dir> <reference-images-dir>" >&2
  exit 2
fi

EVALUATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOS_DIR="$1"
PROMPTS_DIR="$2"
REFERENCES_DIR="$3"
RESULTS_DIR="${EVAL_RESULTS_DIR:-$(dirname "$VIDEOS_DIR")/Results_$(basename "$VIDEOS_DIR")}" 
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$VIDEOS_DIR" || ! -d "$PROMPTS_DIR" || ! -d "$REFERENCES_DIR" ]]; then
  echo "Video, prompt, and reference-image directories must exist." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"
"$PYTHON_BIN" "$EVALUATION_ROOT/check_environment.py"

GME_ARGS=()
if [[ -n "${EVAL_GME_MODEL_PATH:-}" ]]; then
  GME_ARGS+=(--model-path "$EVAL_GME_MODEL_PATH")
elif [[ -n "${EVAL_GME_MODEL:-}" ]]; then
  GME_ARGS+=(--model "$EVAL_GME_MODEL")
fi
if [[ "${EVAL_GME_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  GME_ARGS+=(--local-files-only)
fi

"$PYTHON_BIN" "$EVALUATION_ROOT/gme_score.py" "$VIDEOS_DIR" "$PROMPTS_DIR" "$RESULTS_DIR" "${GME_ARGS[@]}"
"$PYTHON_BIN" "$EVALUATION_ROOT/story_video_consistency.py" "$VIDEOS_DIR" "$PROMPTS_DIR" "$RESULTS_DIR"
bash "$EVALUATION_ROOT/run_face_metrics.sh" "$VIDEOS_DIR" "$REFERENCES_DIR" "$RESULTS_DIR"
bash "$EVALUATION_ROOT/run_vbench.sh" "$VIDEOS_DIR" "$RESULTS_DIR"
"$PYTHON_BIN" "$EVALUATION_ROOT/aggregate_results.py" \
  --results-dir "$RESULTS_DIR" \
  --output-json "$RESULTS_DIR/final_results.json"

echo "Evaluation complete: $RESULTS_DIR/final_results.json"
