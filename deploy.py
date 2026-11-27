"""Publish the complete model bundle from GitHub Actions."""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Deployment is reserved for GitHub Actions. Push the source code instead.")
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not token or not repo_id or repo_id.count("/") != 1:
        raise SystemExit("Configure the HF_TOKEN Actions secret and HF_REPO_ID Actions variable (username/repository).")
    folder = Path("model")
    for filename in ["model.pt", "vectorizer.pkl", "config.json", "metrics.json"]:
        if not (folder / filename).is_file():
            raise SystemExit(f"Training artifact missing: {filename}")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    source = "https://github.com/" + os.environ["GITHUB_REPOSITORY"]
    card = f"""---
language: en
pipeline_tag: text-classification
tags:
- pytorch
- tf-idf
- sentiment-analysis
- github-actions
---

# IMDB TF-IDF Neural Network

A two-layer feedforward neural network trained automatically by GitHub Actions.

## Source and reproducibility

- Code: {source}
- Source commit: `{config['source_commit']}`
- Dataset file: `{config['dataset']['filename']}`
- Dataset SHA-256: `{config['dataset']['sha256']}`
- Unique reviews: {config['dataset']['unique_reviews']}
- Train / validation / test sizes: {config['split']['train']} / {config['split']['validation']} / {config['split']['test']}
- Seed: {config['seed']}
- Validation-selected epoch: {config['best_epoch']}

TF-IDF is fitted on training reviews only. Duplicate normalized reviews are removed before stratified splitting.
The validation macro-F1 selects the checkpoint; the test set is evaluated once after checkpoint selection.

## Held-out results

- Test accuracy: {metrics['test_accuracy']:.4f}
- Test macro-F1: {metrics['test_macro_f1']:.4f}

Full confusion matrix, class metrics and training history are in `metrics.json`.
These results concern this held-out subset; they do not establish general movie-review or cross-domain performance.

## Usage

Install the source repository's dependencies. Download these four files into its `model/` directory:
`model.pt`, `vectorizer.pkl`, `config.json`, `metrics.json`.
Run `python predict.py "This movie was wonderful"` in the source repository.

This is a custom PyTorch model; it does not provide a Transformers pipeline or hosted inference endpoint.
Load only this trusted workflow's pickle file. Use the library versions recorded in `config.json`.
Softmax scores are not calibrated confidence estimates. Short texts, sarcasm and unseen vocabulary can fail.
"""
    (folder / "README.md").write_text(card, encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
                      allow_patterns=["model.pt", "vectorizer.pkl", "config.json", "metrics.json", "README.md"],
                      commit_message=f"Train from {config['source_commit'][:12]}")
    print(f"Published https://huggingface.co/{repo_id}/tree/main")


if __name__ == "__main__":
    main()
