"""Train or evaluate the LexForge bigram and trigram language models."""

import argparse
import hashlib
import json
from pathlib import Path

from lexforge.data import load_rows
from lexforge.ngram import NGram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/corpus"))
    parser.add_argument("--model", type=Path, default=Path("models/ngram.json.gz"))
    parser.add_argument("--metrics", type=Path, default=Path("models/ngram-metrics.json"))
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--tune", action="store_true",
                        help="Select shared smoothing strength by validation trigram perplexity.")
    args = parser.parse_args()
    if args.evaluate_only and args.tune:
        parser.error("--tune requires fitting a new model.")
    validation = [row["text"] for row in load_rows(args.data, "validation")]
    candidates = []
    if args.evaluate_only:
        model = NGram.load(args.model)
    else:
        model = NGram.fit((row["text"] for row in load_rows(args.data, "train")),
                         args.min_count, args.strength)
        if args.tune:
            for strength in (2, 4, 8, 16, 32, 64, 128):
                model.strength = strength
                candidates.append({"strength": strength,
                                   "trigram_perplexity": model.evaluate(validation, 3)["perplexity"]})
            model.strength = min(candidates, key=lambda row: row["trigram_perplexity"])["strength"]
        model.save(args.model)
    report = {"vocabulary": len(model.vocabulary), "strength": model.strength,
              "fit_min_count": None if args.evaluate_only else args.min_count,
              "validation_strength_search": candidates,
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "prepared_sha256": {split: hashlib.sha256((args.data / f"{split}.jsonl").read_bytes()).hexdigest()
                                  for split in ("train", "validation", "test")}}
    for split in ("validation", "test"):
        texts = validation if split == "validation" else [row["text"] for row in load_rows(args.data, split)]
        report[split] = {str(order): model.evaluate(texts, order) for order in (2, 3)}
    metrics = json.dumps(report, indent=2) + "\n"
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(metrics, encoding="utf-8")
    print(metrics, end="")


if __name__ == "__main__":
    main()
