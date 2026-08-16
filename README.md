# Identity-Preserving Text-to-Video Generation via Agentic Enhancement and Semantic Repair

<p align="center">
  <strong>AESR · Agentic Enhancement and Semantic Repair</strong><br />
  A training-free, API-oriented pipeline for improving identity-preserving text-to-video generation.
</p>

<p align="center">
  Jiayi Gao<sup>*</sup> · Changcheng Hua<sup>*</sup> · Jiaqi Tang · Yuxin Peng · Yang Liu<sup>†</sup>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#dataset">Dataset</a> ·
  <a href="#playbooks">Playbooks</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#citation">Citation</a>
</p>

## Introduction

Identity-preserving text-to-video generation is sensitive to both the quality of the input prompt and the visual state produced by the first draft. AESR improves these two controllable signals without training a new video model:

1. **Agentic Enhancement (global level)** uses a playbook-backed loop to generate, analyze, reflect, and consolidate prompt improvements.
2. **Semantic Repair (sample level)** critiques a draft video with a typed vision-language evaluator, edits selected keyframes, and feeds the repaired visual evidence into a video-editing model.

The public release is organized from the accompanying ACM Multimedia paper. It contains the reusable orchestration code, editable playbooks, provider adapters, tests, and a small runnable example. Private datasets, generated media, provider response dumps, signed URLs, and local credentials are not included.

## Installation

The code was prepared for **Python 3.10 or newer**. Stage II also requires the system executables `ffmpeg` and `ffprobe` on `PATH`.

### Conda environment setup

```bash
# Install Anaconda or Miniconda first if `conda` is not available.
# The environment file installs Python, FFmpeg/FFprobe, and Python packages.
conda env create -f environment.yml
conda activate aesr

conda env update -f environment.yml --prune  # use this when updating an existing env
python --version
ffmpeg -version
ffprobe -version
```

If you prefer to create the environment without the YAML file, use `conda create -n aesr python=3.10 ffmpeg pip` followed by `pip install -r requirements.txt` after activating `aesr`.

Copy `configs/.env.example` to `.env` and export the variables in your shell. Do not commit `.env`.

## Dataset

Download the official IPVG 2026 Track 1 test set from the
[challenge release](https://github.com/HiDream-ai/ipvg-challenge-2026.github.io/releases/download/testset/IPVG2026-Test-Track1.zip):

```bash
conda activate aesr
python scripts/download_testset.py
```

The script verifies the official archive checksum and extracts it to
`data/IPVG2026-Test-Track1/`. The downloaded data remains ignored by Git. A
text-only smoke-test prompt is also provided at
`examples/data/input/id001/prompt.txt`.

## Playbooks

- `playbooks/playbook_final.json`: final competition playbook after combining
  HOI priors, official-document guidance, and Track 1 warmup updates.
- `playbooks/initial_playbook.json`: merged HOI and official-document playbook
  before Track 1 warmup.
- `playbooks/empty_playbook.json`: empty structure for learning from scratch.

## Inference

### Prompt enhancement only

This command exercises the Stage I orchestration and writes an enhanced prompt and JSON record:

```bash
python src/ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi.py \
  --mode enhance_prompt_only \
  --enhance-input-root examples/data/input \
  --playbook-file playbooks/playbook_final.json \
  --enhance-output-txt enhanced_prompt.txt \
  --enhance-output-json enhanced_prompt.json \
  --limit 1
```

### Complete repair workflow

`scripts/run_repair_example.sh` is a template for one sample. Set `VIDEO`, `PROMPT`, `REFERENCE`, and `OUT` to authorized local paths before running it:

```bash
VIDEO=/path/to/draft.mp4 \
PROMPT=/path/to/prompt.txt \
REFERENCE=/path/to/reference.png \
OUT=runs/id001 \
bash scripts/run_repair_example.sh
```

The script runs critique, typed issue normalization, missing-visual repair, and edit-prompt assembly. The final Ark video-editing call is intentionally left explicit; see `src/seedance2_edit_from_prompt_json_v2.py` and `src/seedance2_local_video_edit.py` for the provider-backed invocation.

### Tests and validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q src evaluation
```

For the complete benchmark setup and metric commands, see
[`evaluation/README.md`](evaluation/README.md).

## Acknowledgement

We thank the authors and maintainers of the commercial model APIs and open-source tools used by this release. Please follow the license and usage terms of every upstream dependency and provider.

## License and release notes

The authors still need to choose and add the software license, final paper/project URLs, and any approved benchmark links before publishing a definitive GitHub release. Until a license is added, treat this repository as source-available for inspection only. See [`docs/release_checklist.md`](docs/release_checklist.md).

## Citation

```bibtex
@inproceedings{gao2026aesr,
  title={Identity-Preserving Text-to-Video Generation via Agentic Enhancement and Semantic Repair},
  author={Gao, Jiayi and Hua, Changcheng and Tang, Jiaqi and Peng, Yuxin and Liu, Yang},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```
