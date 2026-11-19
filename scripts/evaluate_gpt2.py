"""Evaluate a saved GPT-2 checkpoint on a named holdout."""

import argparse
import hashlib
import json
from pathlib import Path

from lexforge.classifier import GPT2Classifier
from lexforge.data import load_rows
from lexforge.evaluation import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/clauses"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gpt2"))
    parser.add_argument("--base")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(GPT2Classifier(args.checkpoint, args.base),
                       load_rows(args.data, args.split), args.batch_size)
    report = {
        "split": args.split,
        "dataset_sha256": hashlib.sha256((args.data / f"{args.split}.jsonl").read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((args.checkpoint / "classifier.safetensors").read_bytes()).hexdigest(),
        **metrics,
    }
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
