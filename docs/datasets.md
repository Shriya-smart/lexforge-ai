# Datasets and preparation

## Sources and attribution

- **Contract Understanding Atticus Dataset (CUAD) v1** by Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball with The Atticus Project. The release contains 510 publicly filed commercial contracts and lawyer-supervised annotations. This project uses the full contract contexts and selected positive answer spans. The dataset is published under CC BY 4.0. [Dataset](https://huggingface.co/datasets/theatticusproject/cuad) · [Paper](https://arxiv.org/abs/2103.06268) · [License](https://creativecommons.org/licenses/by/4.0/)
- **LEDGAR** by Don Tuggener, Pius von Däniken, Thomas Peetz, and Mark Cieliebak, as released in **LexGLUE** by Ilias Chalkidis and colleagues. Its English provisions come from contracts publicly filed with the US Securities and Exchange Commission. The LexGLUE dataset card lists CC BY 4.0. This project uses only its LEDGAR configuration. The original release labels derive from provision headings. [LEDGAR paper](https://aclanthology.org/2020.lrec-1.155/) · [LexGLUE dataset](https://huggingface.co/datasets/coastalcph/lex_glue) · [License](https://creativecommons.org/licenses/by/4.0/)

The source download revisions are pinned in `src/lexforge/data.py` and recorded in the preparation manifest. The project distributes model parameters, aggregate results, and original game examples. The scripts obtain the datasets from their sources.

## Prepare the inputs

```text
python -m pip install -e ".[train]"
python scripts/prepare_data.py --raw data/raw --output data/prepared --seed 42
```

The raw directory caches the original JSON and three Parquet files. The output contains `corpus/` and `clauses/`, each with `train.jsonl`, `validation.jsonl`, and `test.jsonl`. `manifest.json` lists split counts, label counts, provenance URLs, and SHA-256 hashes. For the corpus alone, use `python scripts/prepare_data.py --corpus-only` with the lightweight install.

CUAD preparation normalizes Unicode and quotes, collapses whitespace, segments at sentence punctuation while protecting common abbreviations, and keeps segments with 5–160 word tokens. Tokenization lowercases text, preserves internal apostrophes and hyphens, and recognizes formatted numbers. At training time, words seen fewer than three times map to the unknown token by default. Classifier spans retain casing and punctuation and use GPT-2 byte-level subword tokenization.

| LexForge category | Source label or labels |
| --- | --- |
| Confidentiality | LEDGAR Confidentiality |
| Termination | LEDGAR Terminations, CUAD Termination For Convenience |
| Governing Law | LEDGAR Governing Laws, CUAD Governing Law |
| Indemnification | LEDGAR Indemnifications and Indemnity |
| Limitation of Liability | CUAD Cap On Liability |

CUAD documents are assigned to approximate 80/10/10 partitions by hashing their titles with the seed. The same assignment applies to full text and labeled spans. LEDGAR keeps the published release splits. Both pipelines discard texts whose normalized fingerprints are present in more than one split, discard conflicting labels, and retain one copy of same-split duplicates. The fingerprint ignores case and punctuation outside tokens and replaces numeric values with a placeholder. Classifier deduplication operates across both sources. The released LEDGAR table lacks document IDs, so overlap checks cover fingerprint matches rather than complete contract-level overlap or paraphrases across sources.
