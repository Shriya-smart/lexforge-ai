# Evaluation

## Word prediction

The shipped model uses 83,730 CUAD training sentences and a vocabulary of 15,824 entries, including unknown and sentence-end tokens. Both orders share the same fitted vocabulary and counts over the lower orders. A unigram adds one to every outcome. For a higher-order context, the numerator is the observed count plus a smoothing strength times the lower-order probability. The denominator is the context count plus the smoothing strength. Unseen contexts use the lower-order probability.

The preparation pipeline assigns contracts to splits before fitting. `--tune` chose strength 128 from the predefined grid by validation trigram perplexity. For each evaluation token, the scorer takes the negative log of the probability of the actual next word in its observed context. Perplexity is the exponential of that mean. End markers are evaluated and unknown words map to the fitted unknown token. Suggestions omit special tokens and show individual word probabilities without rescaling the displayed list.

| CUAD split | Evaluated tokens | Unknown words | Bigram perplexity | Trigram perplexity |
| --- | ---: | ---: | ---: | ---: |
| Validation | 257,176 | 8,588 | 146.75 | 104.81 |
| Test | 444,727 | 13,187 | 126.68 | 87.29 |

See `src/lexforge/assets/ngram-metrics.json` for exact values and file hashes.

## Clause classification

The trained classifier uses five classes and the last non-padding position for GPT-2 sequence classification. Classification takes the highest softmax score. Accuracy counts correct predictions, macro averaging gives each category equal weight, and the confusion matrix uses actual labels for rows and predicted labels for columns. The epoch is selected on validation macro F1 before the test is evaluated. Tests using synthetic data confirm wiring and do not contribute to benchmark metrics.

The delivered CPU run uses 240 randomly sampled training examples and 60 validation examples per class, seed 42, and up to 96 GPT-2 subword tokens per input. The final block, final normalization, and classifier head are updated over three epochs. The benchmark test uses all 1,103 held-out examples, with the source label proportions.

```text
python scripts/prepare_data.py
python scripts/train_gpt2.py --epochs 3 --train-per-label 240 --validation-per-label 60 --max-length 96 --batch-size 16 --learning-rate 0.00015 --bf16 --device cpu
python scripts/evaluate_gpt2.py --checkpoint models/gpt2 --output models/gpt2/test-metrics.json
```

The delivered run used CPU hardware with native bfloat16 support. Omit `--bf16` for wider hardware compatibility. Results can vary with the hardware and numerical precision. The exact artifact and validation report are in `src/lexforge/assets/`.

| Split | Examples | Accuracy | Macro F1 | Macro precision | Macro recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 300 | 93.67% | 93.71% | 93.89% | 93.67% |
| Test | 1,103 | 93.47% | 90.35% | 88.47% | 93.20% |

The exact confusion matrices and scores by category are in `src/lexforge/assets/checkpoint.json` and `src/lexforge/assets/test-metrics.json`.

## Interpretation

The annotated datasets contain contract provisions, often spanning several sentences. Drafting entries are typically shorter and may use different phrasing. GPT-2 classifies the first 96 subwords for the shipped checkpoint and always returns one of the five supported categories. In the source mapping, every liability example comes from CUAD’s liability-cap annotations, so its results speak more directly to caps than to all ways contracts limit liability. Similar contract templates may persist after fingerprint deduplication. Robustness to new drafting styles, legal quality, and confidence calibration each require separate evaluation.
