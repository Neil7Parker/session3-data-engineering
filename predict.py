"""Predict with a trusted artifact directory downloaded from this project's CI run."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from data_io import clean_text
from network import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--model-dir", default="model")
    args = parser.parse_args()
    if not clean_text(args.text):
        parser.error("Review must contain text.")
    folder = Path(args.model_dir)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    # Pickle can execute code: use only artifacts from the trusted project workflow.
    with (folder / "vectorizer.pkl").open("rb") as handle:
        vectorizer = pickle.load(handle)
    model = build_model(config["input_dim"], config["hidden_dim"], config["dropout"])
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    features = vectorizer.transform([clean_text(args.text)])
    if features.nnz == 0:
        parser.error("No review tokens occur in the trained vocabulary; prediction would be unsupported.")
    with torch.no_grad():
        scores = model(torch.from_numpy(features.toarray().astype(np.float32))).softmax(1)[0].tolist()
    print(json.dumps({"sentiment": config["labels"][int(np.argmax(scores))],
                      "probabilities": dict(zip(config["labels"], scores))}, indent=2))


if __name__ == "__main__":
    main()
