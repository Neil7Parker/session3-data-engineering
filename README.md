# Session 3 — Data Engineering Coursework

Author: LI HaoRan

This repository contains three independent course deliverables.

## Verified outputs

- [Research report: successful LaTeX build and separate English / Chinese PDF artifacts](https://github.com/Neil7Parker/session3-data-engineering/actions/runs/45923077410)
- [IMDB model: successful automated training, reload and publication](https://github.com/Neil7Parker/session3-data-engineering/actions/runs/45924877833)
- [Hugging Face model files and measured results](https://huggingface.co/Li7HaoRan/imdb-sentiment-mlp/tree/main)
- [Pickle examples: successful execution and saved outputs](https://github.com/Neil7Parker/session3-data-engineering/actions/runs/45922527446)

The verified IMDB run used the instructor's 500-review CSV, removed one duplicate, and split the remaining reviews into 299 training, 100 validation and 100 test examples. The validation-selected checkpoint achieved **84.0% test accuracy** and **0.8394 test macro-F1**. These are held-out results for this small course dataset, not a general benchmark claim. The Hugging Face model card and `config.json` record the exact source commit of each subsequent automatic update.

| Deliverable | Source | Verification and output |
| --- | --- | --- |
| Research report on continual VLA fine-tuning | `research-report/main.tex`, `research-report/main-zh.tex`, `research-report/references.bib` | The report workflow compiles separate English and Chinese PDFs as Actions artifacts. |
| IMDB neural sentiment classifier | Root Python files and `data/` | The training workflow trains on the instructor CSV, evaluates a held-out test set, and uploads the model bundle to Hugging Face. See [IMDB instructions](IMDB-README.md). |
| Pickle and joblib examples | `pickle-practice/` | Run the examples and load saved objects in a separate process. The workflow also regenerates their artifacts. |

## Report

The review compares ten papers first posted after January 2026. It distinguishes old robot skills, in-domain retention and shifted-domain robustness, and discusses evidence limitations. Numerical findings are attributed to the original studies; no robot-training experiment is claimed.

The two language versions are separate documents. Compile with:

```bash
cd research-report
tectonic --untrusted main.tex
tectonic --untrusted main-zh.tex
```

For the required CI-generated submission, open **Actions → Build Independent English and Chinese Reports**, wait for both jobs to succeed, and download the corresponding PDF artifacts. Rename the downloaded PDF locally according to the instructor's `StudentID-Name.pdf` convention.

## IMDB setup

1. The verified instructor-provided `imdb_top_500.csv` is included in `data/` (500 reviews, 250 per class before deduplication).
2. Create the intended Hugging Face model repository.
3. Store a narrowly scoped model-write token as the GitHub Actions secret `HF_TOKEN`.
4. The default target is `Li7HaoRan/imdb-sentiment-mlp`. Set the optional Actions variable `HF_REPO_ID` only to override it.
5. Push the data/source to `main` or run **Train and Upload to Hugging Face** manually.

The data source and checksum are recorded in `data/README.md`. Model updates are performed by GitHub Actions; the measured results above come from the linked completed run.

## Submission

- Research report: the relevant CI-generated PDF, renamed locally as requested by the instructor.
- Hugging Face practice: name, this GitHub URL, and the Hugging Face model URL after a successful automatic upload.
- Pickle practice: this repository URL, with code and saved files under `pickle-practice/`.

Credentials, private local paths, paper PDFs and internal reading caches are excluded from this repository.
