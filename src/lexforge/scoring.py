"""Transparent writing and category checks for educational feedback."""

import re
from dataclasses import dataclass

from .classifier import Prediction
from .content import Challenge
from .data import clean, tokenize


@dataclass(frozen=True)
class Criterion:
    description: str
    available: int
    earned: int | None


@dataclass(frozen=True)
class Score:
    criteria: tuple[Criterion, ...]

    @property
    def points(self) -> int:
        return sum(item.earned or 0 for item in self.criteria)


def writing_score(challenge: Challenge, text: str, *, draft: bool = False,
                  prediction: Prediction | None = None) -> Score:
    text = clean(text)
    detail_value = 20 if draft else 35
    criteria = [
        Criterion("Uses at least four words", 20, 20 if len(tokenize(text)) >= 4 else 0),
        Criterion("Ends with sentence punctuation", 10,
                  10 if re.search(r'[.!?]["\x27)]*$', text) else 0),
    ]
    for check in challenge.checks:
        matched = any(re.search(pattern, text, flags=re.I) for pattern in check.patterns)
        criteria.append(Criterion(check.description, detail_value, detail_value if matched else 0))
    if draft:
        match = None if prediction is None else 30 if prediction.label == challenge.label else 0
        criteria.append(Criterion("GPT-2 category matches the target", 30, match))
    return Score(tuple(criteria))


def guess_score(expected: str, chosen: str) -> int:
    return 100 if expected == chosen else 0


def level(xp: int) -> int:
    return xp // 250 + 1
