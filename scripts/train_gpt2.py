"""Fine-tune GPT-2 on the prepared legal classification splits."""

import argparse
import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from lexforge.classifier import GPT2Classifier, load_base, trainable_parameters
from lexforge.config import BASE_MODEL, BASE_REVISION, LABELS, MAX_LENGTH
from lexforge.data import balanced_sample, load_rows
from lexforge.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/clauses"))
    parser.add_argument("--output", type=Path, default=Path("models/gpt2"))
    parser.add_argument("--base", default=BASE_MODEL, help="Pinned GPT-2 or local copy thereof.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--train-per-label", type=int, default=0, help="Zero uses all rows.")
    parser.add_argument("--validation-per-label", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=1, help="Number of GPT-2 blocks to fine-tune.")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on supported hardware.")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.threads) < 1 or
            min(args.train_per_label, args.validation_per_label) < 0 or
            not 8 <= args.max_length <= 1024 or not 0 <= args.blocks <= 12 or
            args.learning_rate <= 0):
        parser.error("Invalid training arguments.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    training = balanced_sample(load_rows(args.data, "train"), args.train_per_label, args.seed)
    validation = balanced_sample(load_rows(args.data, "validation"),
                                 args.validation_per_label, args.seed)
    for rows in (training, validation):
        if set(row["label"] for row in rows) != set(LABELS):
            parser.error("Both training and validation must contain all five classes.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory to preserve existing checkpoints.")
    model, tokenizer = load_base(args.base, BASE_REVISION, local_only=False)
    names = trainable_parameters(model, args.blocks)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate)
    print(f"Device {device} | train {len(training)} | validation {len(validation)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": 1, "labels": list(LABELS), "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "max_length": args.max_length,
        "trainable_blocks": args.blocks,
        "training": {
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "device": device, "threads": args.threads,
            "bf16_autocast": args.bf16,
            "train_per_label": args.train_per_label, "validation_per_label": args.validation_per_label,
            "train_examples": len(training), "validation_examples": len(validation),
            "data_sha256": {split: hashlib.sha256((args.data / f"{split}.jsonl").read_bytes()).hexdigest()
                            for split in ("train", "validation")},
            "torch": str(torch.__version__),
        },
    }
    with TemporaryDirectory(dir=args.output.parent) as directory:
        stage = Path(directory)
        selected = stage / "selected"
        selected.mkdir()
        best = float("-inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            # Keep the frozen backbone deterministic during training.
            model.transformer.drop.eval()
            for layer in model.transformer.h[:model.config.n_layer - args.blocks]:
                layer.eval()
            random.shuffle(training)
            loss_sum = 0.0
            for start in range(0, len(training), args.batch_size):
                batch = training[start:start + args.batch_size]
                encoded = tokenizer([row["text"] for row in batch], truncation=True,
                                    max_length=args.max_length, padding=True, return_tensors="pt")
                targets = torch.tensor([LABELS.index(row["label"]) for row in batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                precision = torch.autocast(device, dtype=torch.bfloat16) if args.bf16 else nullcontext()
                with precision:
                    loss = model(**encoded.to(device), labels=targets).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(batch)
                if start // args.batch_size % 10 == 0:
                    print(f"Epoch {epoch} · fitted {start + len(batch)}/{len(training)}", flush=True)
            # Select using the exact serialized precision served by the CLI.
            candidate = stage / "candidate"
            candidate.mkdir(exist_ok=True)
            state = model.state_dict()
            weights = {name: state[name].detach().cpu().half().contiguous()
                       for name in names}
            save_file(weights, candidate / "classifier.safetensors")
            (candidate / "checkpoint.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            classifier = GPT2Classifier(candidate, base=args.base, device=device)
            report = evaluate(classifier, validation, args.batch_size)
            del classifier
            print(f"Epoch {epoch} · loss {loss_sum / len(training):.4f} · "
                  f"validation macro F1 {report['macro_f1']:.4f}", flush=True)
            if report["macro_f1"] > best:
                best = report["macro_f1"]
                os.replace(candidate / "classifier.safetensors", selected / "classifier.safetensors")
                metadata["selected_epoch"] = epoch
                metadata["validation"] = report
                (selected / "checkpoint.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
        if args.output.exists():
            args.output.rmdir()
        selected.rename(args.output)
    print(f"Saved fine-tuned checkpoint in {args.output}")


if __name__ == "__main__":
    main()
