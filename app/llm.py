"""Construct a local GPT-2 clause classifier for the game."""

from pathlib import Path

from lexforge.classifier import GPT2Classifier
from lexforge.config import ASSETS


def load_classifier(checkpoint: Path = ASSETS, base: str | None = None) -> GPT2Classifier:
    """Create a classifier that loads the fine-tuned weights on first use."""
    return GPT2Classifier(checkpoint, base)
