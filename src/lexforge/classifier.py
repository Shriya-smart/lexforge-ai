"""Lazy, local GPT-2 inference from a fine-tuned LexForge checkpoint."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import BASE_MODEL, BASE_REVISION, LABELS


class ClassifierUnavailable(RuntimeError):
    """A usable local checkpoint or its runtime is missing."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class ClauseClassifier(Protocol):
    def predict(self, text: str, top_k: int = 1) -> list[Prediction]: ...


def load_base(reference: str, revision: str, *, local_only: bool):
    from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast, logging

    options = {"local_files_only": local_only}
    if not Path(reference).is_dir():
        options["revision"] = revision
    tokenizer = GPT2TokenizerFast.from_pretrained(reference, **options)
    tokenizer.pad_token = tokenizer.eos_token
    loggers = (
        logging.get_logger("transformers.modeling_utils"),
        logging.get_logger("transformers.integrations.tensor_parallel"),
    )
    previous_levels = [logger.level for logger in loggers]
    if local_only:
        # Saved weights supply the head. Inference uses one device.
        for logger in loggers:
            logger.setLevel(logging.ERROR)
    try:
        model = GPT2ForSequenceClassification.from_pretrained(
            reference,
            use_safetensors=True,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={label: index for index, label in enumerate(LABELS)},
            pad_token_id=tokenizer.pad_token_id,
            **options,
        )
    finally:
        for logger, level in zip(loggers, previous_levels):
            logger.setLevel(level)
    model.config.use_cache = False
    return model, tokenizer


def trainable_parameters(model, blocks: int) -> list[str]:
    prefixes = ["score.", "transformer.ln_f."]
    prefixes.extend(f"transformer.h.{i}." for i in range(model.config.n_layer - blocks,
                                                       model.config.n_layer))
    names = []
    for name, parameter in model.named_parameters():
        enabled = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    return names


class GPT2Classifier:
    def __init__(self, checkpoint: Path, base: str | None = None, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.base = base or BASE_MODEL
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        metadata_file = self.checkpoint / "checkpoint.json"
        weights_file = self.checkpoint / "classifier.safetensors"
        if not metadata_file.is_file() or not weights_file.is_file():
            raise ClassifierUnavailable(f"GPT-2 checkpoint unavailable at {self.checkpoint}. "
                                        "See docs/training.md to train or import one.")
        try:
            import torch
            from safetensors import SafetensorError
            from safetensors.torch import load_file

            if self.device == "cpu":
                torch.set_num_threads(min(torch.get_num_threads(), 4))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            blocks = metadata["trainable_blocks"]
            if (metadata["format"] != 1 or tuple(metadata["labels"]) != LABELS
                    or metadata["base_revision"] != BASE_REVISION
                    or metadata["base_model"] != BASE_MODEL
                    or not isinstance(blocks, int) or not 0 <= blocks <= 12
                    or not 8 <= metadata["max_length"] <= 1024):
                raise ValueError("incompatible checkpoint metadata")
            try:
                model, tokenizer = load_base(self.base, BASE_REVISION, local_only=True)
                weights = load_file(weights_file)
            except SafetensorError as error:
                raise ValueError("unreadable model weights") from error
            if set(weights) != set(trainable_parameters(model, blocks)):
                raise ValueError("incomplete fine-tuned weights")
            model.load_state_dict(weights, strict=False)
            model.to(self.device).eval()
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            raise ClassifierUnavailable(
                "GPT-2 could not load. Install lexforge-ai[ml] and cache the pinned GPT-2 "
                "base with scripts/fetch_base.py, or pass --gpt-base PATH. "
                f"Details: {error}"
            ) from error
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self._max_length = metadata["max_length"]

    def probabilities(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Provide at least one nonblank sentence.")
        self._load()
        encoded = self._tokenizer(
            texts, truncation=True, max_length=self._max_length, padding=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
        return logits.softmax(dim=-1).cpu().tolist()

    def predict(self, text: str, top_k: int = 1) -> list[Prediction]:
        if not 1 <= top_k <= len(LABELS):
            raise ValueError(f"top_k must be between 1 and {len(LABELS)}.")
        scores = self.probabilities([text])[0]
        ranked = sorted(zip(LABELS, scores), key=lambda pair: -pair[1])
        return [Prediction(label, score) for label, score in ranked[:top_k]]
