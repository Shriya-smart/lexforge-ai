"""Game orchestration independent of Rich and Typer."""

from dataclasses import dataclass

from .classifier import ClassifierUnavailable, ClauseClassifier, Prediction
from .content import DIFFICULTIES, Challenge
from .data import clean
from .scoring import Score, writing_score


@dataclass(frozen=True)
class DraftResult:
    sentence: str
    prediction: Prediction | None
    score: Score
    model_error: str | None


def complete_draft(challenge: Challenge, completion: str,
                   classifier: ClauseClassifier) -> DraftResult:
    if not completion.strip():
        raise ValueError("Enter some text to complete the sentence.")
    sentence = clean(f"{challenge.text} {completion}")
    prediction = None
    error = None
    try:
        prediction = classifier.predict(sentence, top_k=1)[0]
    except ClassifierUnavailable as unavailable:
        error = str(unavailable)
    score = writing_score(challenge, completion, draft=True, prediction=prediction)
    return DraftResult(sentence, prediction, score, error)


def unlocked_difficulties(completed_drafts: int) -> tuple[str, ...]:
    return DIFFICULTIES[:1 + (completed_drafts >= 3) + (completed_drafts >= 6)]


def achievements(attempts: list[dict]) -> list[str]:
    distinct = {(row["mode"], row["challenge"]) for row in attempts}
    badges = []
    if any(mode == "draft" for mode, _ in distinct):
        badges.append("First Draft")
    if len(distinct) >= 5:
        badges.append("Clause Collector")
    if any(row["mode"] == "guess" and row["score"] == 100 for row in attempts):
        badges.append("Sharp Issue Spotter")
    if any(row["mode"] == "revision" and row["score"] == 100 for row in attempts):
        badges.append("Precision Editor")
    if any(row["mode"] == "draft" and row["matched"] == 1 for row in attempts):
        badges.append("Category Match")
    if {mode for mode, _ in distinct} == {"draft", "guess", "revision"}:
        badges.append("Lab Generalist")
    return badges
