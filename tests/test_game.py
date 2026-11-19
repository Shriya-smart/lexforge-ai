from lexforge.classifier import ClassifierUnavailable, Prediction
from lexforge.content import DRAFTS, GUESSES, REVISIONS
from lexforge.game import achievements, complete_draft, unlocked_difficulties
from lexforge.scoring import guess_score, level, writing_score


class RecordingClassifier:
    """Synthetic interface double that records input; no model inference."""

    def predict(self, text, top_k=1):
        self.input = (text, top_k)
        return [Prediction("Termination", 0.8)]


def test_entire_draft_reaches_classifier_and_scores():
    prediction = RecordingClassifier()
    completion = "by giving 30 days' written notice."
    result = complete_draft(DRAFTS[1], completion, prediction)
    assert prediction.input == (f"{DRAFTS[1].text} {completion}", 1)
    assert result.sentence == prediction.input[0]
    assert result.prediction.label == "Termination"
    assert result.score.points == 100


def test_category_match_ignores_confidence_and_unavailable_stays_ungraded():
    completion = "by giving 30 days' written notice."
    mismatched = writing_score(DRAFTS[1], completion, draft=True,
                               prediction=Prediction("Confidentiality", 0.99))
    matched = writing_score(DRAFTS[1], completion, draft=True,
                            prediction=Prediction("Termination", 0.21))
    assert mismatched.points == 70
    assert matched.points == 100

    class Missing:
        def predict(self, text, top_k=1):
            raise ClassifierUnavailable("checkpoint not found")

    result = complete_draft(DRAFTS[1], completion, Missing())
    assert result.score.points == 70
    assert result.score.criteria[-1].earned is None
    assert result.model_error == "checkpoint not found"


def test_revision_and_guess_rubric():
    revision = "Either party may cancel on 30 days' written notice."
    assert writing_score(REVISIONS[1], revision).points == 100
    assert writing_score(REVISIONS[1], "Cancel soon").points == 0
    expected = GUESSES[0][1]
    assert guess_score(expected, expected) == 100
    assert guess_score(expected, "Termination") == 0


def test_level_and_achievement_thresholds():
    assert level(249) == 1
    assert level(250) == 2
    assert unlocked_difficulties(2) == ("Beginner",)
    assert unlocked_difficulties(3) == ("Beginner", "Intermediate")
    assert unlocked_difficulties(6) == ("Beginner", "Intermediate", "Advanced")
    attempts = [
        {"mode": "draft", "challenge": "d01", "score": 100, "matched": 1},
        {"mode": "guess", "challenge": "g01", "score": 100, "matched": 1},
        {"mode": "revision", "challenge": "r01", "score": 100, "matched": None},
    ]
    assert achievements(attempts) == [
        "First Draft", "Sharp Issue Spotter", "Precision Editor", "Category Match", "Lab Generalist",
    ]
