"""Implement the five assignment entry points using LexForge's services."""

from pathlib import Path

from lexforge.classifier import ClauseClassifier
from lexforge.config import ASSETS
from lexforge.content import Challenge
from lexforge.data import load_cuad, prepare_corpus, write_splits
from lexforge.game import DraftResult, complete_draft
from lexforge.ngram import NGram

from .cli import main
from .llm import load_classifier


def milestone_1(source: Path, output: Path, seed: int = 42) -> dict:
    """Prepare grouped CUAD sentence splits and write the JSONL files."""
    return write_splits(prepare_corpus(load_cuad(source), seed), output)


def milestone_2(prefix: str, order: int = 3, top_k: int = 5,
                model_path: Path = ASSETS / "ngram.json.gz") -> list[tuple[str, float]]:
    """Predict the next word from the saved CUAD model."""
    return NGram.load(model_path).suggest(prefix, top_k=top_k, order=order)


def milestone_3(text: str, checkpoint: Path = ASSETS, base: str | None = None,
                top_k: int = 1):
    """Classify text with the fine-tuned GPT-2 model."""
    return load_classifier(checkpoint, base).predict(text, top_k=top_k)


def milestone_4(argv: list[str] | None = None) -> None:
    """Open the interactive drafting game."""
    main(argv)


def milestone_5(challenge: Challenge, completion: str,
                classifier: ClauseClassifier | None = None) -> DraftResult:
    """Classify a full draft and return the rubric and feedback."""
    return complete_draft(challenge, completion,
                          classifier if classifier is not None else load_classifier())
