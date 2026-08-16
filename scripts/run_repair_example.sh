#!/usr/bin/env bash
set -euo pipefail

# Template for one sample. Replace these paths with authorized local inputs.
ID="${ID:-id001}"
VIDEO="${VIDEO:-examples/data/draft_videos/${ID}.mp4}"
PROMPT="${PROMPT:-examples/data/input/${ID}/prompt.txt}"
REFERENCE="${REFERENCE:-examples/data/references/${ID}/pencil.png}"
OUT="${OUT:-runs/${ID}}"

mkdir -p "$OUT"

python src/gemini_video_prompt_critique.py \
  --video "$VIDEO" \
  --prompt-file "$PROMPT" \
  --out "$OUT/${ID}_critique.json"

python src/gemini_video_prompt_critique_typed.py \
  --critique-json "$OUT/${ID}_critique.json" \
  --video "$VIDEO" \
  --prompt-file "$PROMPT" \
  --out "$OUT/${ID}_critique_typed.json"

python src/fix_missing_visual_from_critique_typed.py \
  --video "$VIDEO" \
  --critique-typed-json "$OUT/${ID}_critique_typed.json" \
  --prompt-file "$PROMPT" \
  --out "$OUT/missing_visual_fix"

python src/assemble_seedance_edit_prompt_with_frame_fixes_v2.py \
  --critique-json "$OUT/${ID}_critique_typed.json" \
  --missing-visual-fix-dir "$OUT/missing_visual_fix" \
  --prompt-file "$PROMPT" \
  --out "$OUT/${ID}_edit_prompt.json"

echo "Prompt assembly complete: $OUT/${ID}_edit_prompt.json"
echo "To call Ark, pass the generated JSON, $VIDEO, and $REFERENCE to seedance2_edit_from_prompt_json_v2.py."
