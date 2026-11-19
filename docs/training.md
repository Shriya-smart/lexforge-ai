# Model training and checkpoint import

From the project root, install `.[train]` and run `python scripts/prepare_data.py`. Use a new output directory for each GPT-2 run.

## Bigram and trigram

```text
python scripts/train_ngram.py --tune
python scripts/train_ngram.py --evaluate-only
```

The first command trains both orders from the same tokenized training corpus. It chooses the shared smoothing strength by validation trigram perplexity and saves `models/ngram.json.gz` and `models/ngram-metrics.json`. The second command reloads the counts and measures validation and test perplexity. Supply `--metrics ANOTHER_FILE` to preserve the original report during re-evaluation. To use the new model, launch `lexforge --ngram models/ngram.json.gz`.

## GPT-2 sequence classification

```text
python scripts/fetch_base.py
python scripts/train_gpt2.py --epochs 3 --device auto --output models/gpt2
python scripts/evaluate_gpt2.py --checkpoint models/gpt2 --output models/gpt2/test-metrics.json
```

The trainer starts from the pinned MIT-licensed [GPT-2 base](https://huggingface.co/openai-community/gpt2), with a new five-way sequence classification head. It optimizes cross-entropy with AdamW. By default it fine-tunes the last transformer block, final layer normalization, and classification head, leaving the earlier weights frozen. `--blocks` selects how many final transformer blocks to train. `--train-per-label` and `--validation-per-label` cap their respective splits using seeded sampling. The default full split preserves source label frequencies. Reduce `--batch-size` to fit constrained hardware. `--bf16` enables bfloat16 autocast on supported hardware.

Training selects the epoch with the best validation macro F1 after reloading the compact float16 checkpoint. The holdout test is evaluated with the saved best weights. The two import files are `checkpoint.json` and `classifier.safetensors`. The first records labels, base revision, training configuration, and validation metrics. The second stores the actual fine-tuned parameters. Both files belong together. The delivered checkpoint's exact reproduction command and results appear in [Evaluation](evaluation.md).

## Google Colab

1. Upload the source ZIP to a Colab notebook. Select a GPU runtime.
2. Run the following in a notebook cell, replacing the ZIP filename if needed. The scripts use `--device auto` to select CUDA when PyTorch detects it.

```python
!unzip -q /content/lexforge-ai.zip -d /content
%cd /content/lexforge-ai
%pip install -e ".[train]"
!python scripts/prepare_data.py
!python scripts/fetch_base.py
!python scripts/train_gpt2.py --epochs 3 --batch-size 16 --device auto --output models/colab-gpt2
!python scripts/evaluate_gpt2.py --checkpoint models/colab-gpt2 --output models/colab-gpt2/test-metrics.json
!zip -q -j /content/colab-gpt2.zip models/colab-gpt2/*

from google.colab import files
files.download("/content/colab-gpt2.zip")
```

3. Extract the result locally into `models/colab-gpt2/`. Install `.[ml]`, run `python scripts/fetch_base.py` on that machine, then run `lexforge --gpt-checkpoint models/colab-gpt2`. To use a separate local base directory, add `--gpt-base PATH`.

The CLI loads weights lazily on the first classification request. Missing files or optional dependencies produce an actionable message. You can still play all rule-based modes and use the N-gram model.
