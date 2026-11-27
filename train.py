"""Train only from the instructor-provided IMDB dataset; deployment runs in CI."""
import argparse
import copy
import importlib.metadata
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from data_io import load_reviews
from network import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--output", default="model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.data:
        data = Path(args.data)
    else:
        candidates = [p for p in (Path("data/imdb_balanced_10k.csv"), Path("data/imdb_top_500.csv")) if p.is_file()]
        if len(candidates) != 1:
            parser.error("Place exactly one instructor CSV in data/ or specify --data.")
        data = candidates[0]
    try:
        texts, labels, provenance = load_reviews(data, args.text_column, args.label_column)
    except ValueError as error:
        parser.error(str(error))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    indices = np.arange(len(texts))
    train_ids, other_ids = train_test_split(indices, test_size=0.4, random_state=args.seed, stratify=labels)
    val_ids, test_ids = train_test_split(other_ids, test_size=0.5, random_state=args.seed,
                                         stratify=np.asarray(labels)[other_ids])
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, lowercase=True, strip_accents="unicode")
    train_x = vectorizer.fit_transform([texts[i] for i in train_ids])
    val_x = vectorizer.transform([texts[i] for i in val_ids])
    test_x = vectorizer.transform([texts[i] for i in test_ids])
    all_y = np.asarray(labels, dtype=np.int64)
    model = build_model(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    def predict(x):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(x[i:i+64].toarray().astype(np.float32))).argmax(1).numpy()
                                   for i in range(0, x.shape[0], 64)])

    best_score, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_x.shape[0])
        total_loss = 0.0
        for start in range(0, len(order), 64):
            batch = order[start:start+64]
            x = torch.from_numpy(train_x[batch].toarray().astype(np.float32))
            y = torch.from_numpy(all_y[train_ids][batch])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        score = float(f1_score(all_y[val_ids], predict(val_x), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ids), "validation_macro_f1": score})
        print(json.dumps(history[-1]), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    predictions = predict(test_x)
    versions = {name: importlib.metadata.version(name) for name in ["torch", "scikit-learn", "numpy", "huggingface-hub"]}
    config = {"architecture": "TF-IDF + Linear-ReLU-Dropout-Linear", "input_dim": train_x.shape[1],
              "hidden_dim": 64, "dropout": 0.2, "labels": ["negative", "positive"], "seed": args.seed,
              "best_epoch": best_epoch, "dataset": provenance, "versions": versions,
              "source_commit": os.environ.get("GITHUB_SHA", "local"), "split": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)}}
    metrics = {"test_accuracy": float(accuracy_score(all_y[test_ids], predictions)),
               "test_macro_f1": float(f1_score(all_y[test_ids], predictions, average="macro", zero_division=0)),
               "validation_macro_f1": best_score, "test_confusion_matrix": confusion_matrix(all_y[test_ids], predictions, labels=[0, 1]).tolist(),
               "test_classification_report": classification_report(all_y[test_ids], predictions, labels=[0, 1],
                    target_names=config["labels"], output_dict=True, zero_division=0), "history": history}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    with (output / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["test_accuracy", "test_macro_f1"]}))


if __name__ == "__main__":
    main()
