# AESR Evaluation

This directory packages the automatic evaluation workflow used for the AESR
experiments. It is derived from the provided `ID_Eval` package, but removes
machine-specific paths, does not set a download mirror, and never stores API
keys in files or command-line arguments.

The paper reports six automatic metrics:

| Category | Metric | Runner |
| --- | --- | --- |
| Text alignment | `GME-Score` | `gme_score.py` |
| Text alignment | `story_video_consistency` | `story_video_consistency.py` |
| Identity consistency | `cur_score`, `arc_score` | `run_face_metrics.sh` + ConsisID |
| Video quality | `motion_smoothness`, `imaging_quality` | `run_vbench.sh` + VBench |

The candidate score is

```text
0.15 * (GME-Score + story_video_consistency)
+ 0.20 * (cur_score + arc_score)
+ 0.15 * (motion_smoothness + imaging_quality)
```

`state_shift_persistence`, `CLIP-Score`, and `fid_score` are collected when
present but are not part of this formula. The final challenge ranking also
included human evaluation; this repository cannot reproduce that component.

## What is included

- GME scoring with configurable local or Hugging Face model paths.
- MSVBench-style story-video consistency scoring, including Gemini captioning.
- Robust result aggregation and candidate selection.
- Thin launchers for the original ConsisID and VBench implementations.

## Required external components

The supplied `ID_Eval.zip` did **not** include these components or their model
weights, so they are deliberately not copied into this repository:

1. A compatible ConsisID checkout providing `cal_face_sim.py`.
2. A compatible VBench checkout providing `evaluate.py`.
3. A complete MSVBench checkout providing `Tools/gemini_api.py` and the local
   `KaLM-embedding-multilingual-mini-instruct-v2` model directory.
4. The GME model `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct` (or an equivalent local
   checkout) and a CUDA-capable PyTorch installation for practical execution.

Install the upstream projects and their licensed model weights in local paths
outside this repository. Set only local environment variables:

```bash
export EVAL_CONSISID_ROOT=/path/to/ConsisID
export EVAL_VBENCH_ROOT=/path/to/VBench
export EVAL_MSVBENCH_ROOT=/path/to/MSVBench
export EVAL_KALM_MODEL_PATH=/path/to/KaLM-embedding-multilingual-mini-instruct-v2
export GEMINI_API_KEY=...  # keep in a local .env or shell, never commit it
```

The `GEMINI_API_KEY` is used only by the upstream `GeminiAPI` helper during
story-video captioning. It is read from the environment and is not written into
JSON results. If your helper supports a local proxy, pass `--proxy` directly to
`story_video_consistency.py`; do not encode credentials into a proxy URL that
will be logged.

## Installation

Create the base Conda environment from the repository root, then install the
evaluation extras and a PyTorch build appropriate for the machine:

```bash
conda env create -f environment.yml
conda activate aesr
pip install -r evaluation/requirements.txt
# Install PyTorch following the selector at pytorch.org for your CUDA version.
```

Before running any expensive metric, verify all requirements without exposing
the credential value:

```bash
python evaluation/check_environment.py
```

For a CPU-only functional check, use `python evaluation/check_environment.py
--allow-cpu`; GME inference is substantially slower without CUDA.

## Input layout

For generated candidates named `id001_prompt0.mp4`, use a matching prompt
named either `id001_prompt0.txt` or `id001.txt`. Reference images are matched
by the prefix before the first underscore, such as `id001.png`.

```text
local-data/
├── videos/
│   ├── id001_prompt0.mp4
│   └── id002_prompt0.mp4
├── prompts/
│   ├── id001.txt
│   └── id002.txt
└── reference_images/
    ├── id001.png
    └── id002.png
```

For the official Track 1 archive, create evaluator prompt files once:

```bash
python evaluation/prepare_prompts.py data/IPVG2026-Test-Track1 \
  --output data/IPVG2026-Test-Track1/eval_prompts
```

Then use `data/IPVG2026-Test-Track1/images` as the reference-image directory.
After preparation, the relevant paths are:

```text
data/IPVG2026-Test-Track1/
├── eval.json
├── images/
│   ├── id001.webp
│   └── ...
└── eval_prompts/
    ├── id001.txt
    └── ...
```

Generated candidates may be stored separately, but their sample-ID prefix must
match the prompt and reference image:

```text
local-data/videos/
├── id001_prompt0.mp4
└── id002_prompt0.mp4
```

For this layout, run:

```bash
bash evaluation/run_evaluation.sh \
  local-data/videos \
  data/IPVG2026-Test-Track1/eval_prompts \
  data/IPVG2026-Test-Track1/images
```

## Run metrics

The all-in-one runner resumes whenever its per-video output JSON already
exists. It produces `Results_<videos-dir>/final_results.json` by default:

```bash
bash evaluation/run_evaluation.sh \
  local-data/videos \
  local-data/prompts \
  local-data/reference_images
```

Set `EVAL_GME_MODEL_PATH` to an already downloaded GME model directory, or set
`EVAL_GME_LOCAL_FILES_ONLY=1` to prohibit model downloads. Otherwise the GME
runner uses the public Hugging Face model ID declared in `gme_score.py`.

For debugging, run each component individually:

```bash
python evaluation/gme_score.py local-data/videos local-data/prompts local-data/Results_videos
python evaluation/story_video_consistency.py \
  local-data/videos local-data/prompts local-data/Results_videos \
  --msvbench-root "$EVAL_MSVBENCH_ROOT"
bash evaluation/run_face_metrics.sh local-data/videos local-data/reference_images local-data/Results_videos
bash evaluation/run_vbench.sh local-data/videos local-data/Results_videos
python evaluation/aggregate_results.py \
  --results-dir local-data/Results_videos \
  --output-json local-data/Results_videos/final_results.json
```

## Select candidates

Place each method's generated videos beside its matching results directory:

```text
candidate-runs/
├── baseline/
├── Results_baseline/
├── aesr/
└── Results_aesr/
```

Select the highest complete automatic score for each ID and optionally copy the
selected videos to the `id1.mp4`, `id2.mp4`, ... submission layout:

```bash
python evaluation/score_candidates.py \
  --candidates-root candidate-runs \
  --output candidate-runs/Submission \
  --copy-videos
```

The script writes the metric manifest and a separate skipped-record report. It
does not fill missing metrics with arbitrary values.
