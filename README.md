# LexForge AI — The Legal Language Lab

LexForge is a Python 3.11 terminal game for law students. Complete contracts in your own words, inspect model-generated suggestions, and compare GPT-2’s clause category with the exercise target. Play drafting, clause guessing, and clause revision challenges across five contract topics.

**For educational use only. LexForge does not provide legal advice.** A model’s confidence measures its preference among its supported categories. It does not measure legal correctness, enforceability, or drafting quality.

## Run the lab

From the extracted project directory, create and activate a Python 3.11 environment. See the [installation guide](docs/install.md) for Ubuntu and Windows PowerShell.

```text
python -m pip install ".[ml,test]"
python scripts/fetch_base.py
lexforge --player Ada
python -m pytest -q
```

The distribution includes a CUAD-trained bigram/trigram model and compact, fine-tuned GPT-2 weights. The fetch step caches the pinned pretrained GPT-2 base, about 525 MiB. Later sessions use the cache offline. For a lightweight install, run `python -m pip install .` and `lexforge`. Drafting then awards the writing points it can assess and leaves the GPT-2 check ungraded.

## Assignment milestones

| Milestone | Implementation and evidence |
| --- | --- |
| 1 · Prepare the legal corpus | `scripts/prepare_data.py` fetches pinned CUAD contracts and LexGLUE LEDGAR provisions. It cleans and segments text, writes split JSONL with SHA-256 fingerprints, and removes duplicate or conflicting examples. N-gram vocabulary is fit on training sentences only. [Dataset guide](docs/datasets.md) |
| 2 · Predict the next word | `NGram` implements word bigrams and trigrams, add-one unigram probabilities, recursive smoothing, OOV handling, and exact top-k word ranking. The trained model and held-out perplexity are in `src/lexforge/assets/`. The same model serves the predictor and drafting modes. [Evaluation](docs/evaluation.md) |
| 3 · Fine-tune GPT-2 | `scripts/train_gpt2.py` fine-tunes the final transformer block, final normalization, and sequence classification head on five real legal labels. The checkpoint records validation accuracy, macro F1, precision, recall, and the confusion matrix. `scripts/evaluate_gpt2.py` reloads saved weights for independent evaluation. [Training and Colab](docs/training.md) |
| 4 · Play in the terminal | `lexforge` opens the Rich and Typer menu. Draft freely at beginner, intermediate, and advanced levels. Replay sets, analyze a complete sentence, guess categories, revise examples, and browse saved progress. [User guide](docs/user-guide.md) |
| 5 · Classify and score drafts | The game sends the combined prompt and continuation to GPT-2, displays the predicted category and softmax confidence, compares it with the target, and itemizes the writing and category points. SQLite saves attempts and XP. [User guide](docs/user-guide.md) |

The lab has 15 drafting challenges, five guessing examples, five revision exercises, a free-drafting analyzer, a model insights view, per-profile settings, six achievements, and a local leaderboard.

## Rebuild or evaluate the models

```text
python -m pip install -e ".[train,test]"
python scripts/prepare_data.py
python scripts/train_ngram.py --tune
python scripts/train_gpt2.py
python scripts/evaluate_gpt2.py --output models/gpt2/test-metrics.json
lexforge --ngram models/ngram.json.gz --gpt-checkpoint models/gpt2
python -m pytest -q
```

The full-data GPT-2 command is intended for a GPU. The training guide also documents the smaller delivered run, Colab instructions, and checkpoint transfer. Recorded metrics and dataset manifests are shipped with the models.

- [Installation](docs/install.md)
- [Dataset preparation](docs/datasets.md)
- [Model training and checkpoint import](docs/training.md)
- [Evaluation methodology](docs/evaluation.md)
- [User guide](docs/user-guide.md)
- [Architecture and future full-stack roadmap](docs/architecture.md)
- [Viva preparation](docs/viva.md)
