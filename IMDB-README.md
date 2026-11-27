# IMDB Sentiment Classification with GitHub Actions and Hugging Face

This coursework project implements Option A: TF-IDF review features followed by a feedforward PyTorch neural network. A push to `main` or `master` trains, evaluates and publishes the model bundle to Hugging Face automatically.

## Project structure

```text
data/                            Instructor IMDB CSV
model/                           CI-generated artifacts
data_io.py                       CSV validation and text cleaning
network.py                       Neural network architecture
train.py                         Training and held-out evaluation
predict.py                       Reload and predict
deploy.py                        CI-only model publication
requirements.txt                 Python dependencies
.github/workflows/train-and-upload.yml
```

## One-time setup

1. The instructor's verified `imdb_top_500.csv` is already in `data/`. Keep the original CSV unchanged; its provenance and checksum are in `data/README.md`.
2. Create a GitHub repository and push this source tree. Enable GitHub Actions if prompted.
3. Create a Hugging Face account, verify its email and create a model repository once, for example `USERNAME/imdb-sentiment-nn`.
4. Create a Hugging Face access token with write access to that model repository. Add it directly to GitHub under **Settings → Secrets and variables → Actions → Secrets**, named `HF_TOKEN`. Never commit or paste the token into code, logs or chat.
5. The configured default target is `Li7HaoRan/imdb-sentiment-mlp`. Under **Actions → Variables**, `HF_REPO_ID` may optionally override that target. Other optional variables are `DATA_FILE` (CSV path), `TEXT_COLUMN` and `LABEL_COLUMN` (exact header names). Defaults detect the included CSV and its schema.
6. Push to `main` / `master`, or select **Actions → Train and Upload to Hugging Face → Run workflow** after configuring secrets. All model updates are performed by the workflow.

The workflow fails clearly when configuration or data is missing. Model files are not trained locally for deployment and are never uploaded manually.

## Learning method

Review text is HTML-unescaped, stripped of markup and whitespace-normalized. Identical normalized reviews are deduplicated before splitting; conflicting labels are rejected. The stratified split allocates approximately 60% training, 20% validation and 20% test data using seed 42.

The TF-IDF vocabulary (maximum 10,000 unigram/bigram features) and inverse-document frequencies are fitted using training reviews only. The network maps TF-IDF features through a 64-unit hidden layer, ReLU, dropout 0.2 and two output logits. It learns with cross-entropy and AdamW. Training uses up to 30 epochs, selecting the checkpoint with the best validation macro-F1 and stopping after five epochs without improvement. The final test set is evaluated only after checkpoint selection.

`metrics.json` records held-out accuracy, macro-F1, per-class metrics, a confusion matrix in negative/positive order and training history. `config.json` records dataset SHA-256, split sizes, package versions, seed, selected epoch and source commit. No accuracy is claimed before an actual CI run. TF-IDF handles word occurrence rather than full context; sarcasm and domain shift remain limitations. A small held-out sample can have substantial statistical uncertainty.

## Model files on Hugging Face

- `model.pt`: network state dictionary
- `vectorizer.pkl`: fitted TF-IDF transform
- `config.json`: architecture, labels and training provenance
- `metrics.json`: measured evaluation results
- `README.md`: model card generated from the actual run

This is a custom PyTorch model, so no hosted inference widget is required. To demonstrate inference, download its four artifact files from the Hugging Face repository into this source tree's `model/` folder, install the recorded dependency versions and run:

```bash
python predict.py "The story was engaging and the acting was excellent."
```

Use only trusted project pickles. Reviews containing no known vocabulary are rejected instead of presenting an unsupported prediction. Softmax probabilities are uncalibrated model scores.

## Delivery evidence

Confirm a green completed GitHub Actions run, then open the Hugging Face **Files and versions** tab and confirm the four required files and generated model card. Check the model card's source commit matches the workflow run. Submit your name, GitHub repository URL and Hugging Face model repository URL to the class group.

## References

- [Course practice](https://gitee.com/lundechen/data_engineering_master_2026_fall/blob/master/session-3/practice-huggingface-with-github-ci-cd.md)
- [Hugging Face Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [GitHub setup-python action](https://github.com/actions/setup-python)
- [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
