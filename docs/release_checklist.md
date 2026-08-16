# Public release checklist

- [ ] Choose and add a software license.
- [ ] Add the final paper, project, and benchmark links.
- [ ] Confirm all authors and affiliations are correct.
- [ ] Confirm commercial API and model terms allow the intended use.
- [ ] Confirm every example image, video, prompt, and dataset may be redistributed.
- [ ] Run a secret scan over the complete Git history before pushing.
- [ ] Run `python -m compileall -q src` inside the Conda environment.
- [ ] Run `python -m compileall -q src evaluation` inside the Conda environment.
- [ ] Run `python -m unittest discover -s tests -v` inside the Conda environment.
- [ ] Run all dry-run commands without credentials.
- [ ] Keep `.env`, generated media, provider responses, signed URLs, model weights, and private evaluation data out of Git.
- [ ] Confirm the evaluation README names all non-redistributed upstream evaluators and human-evaluation limits.
