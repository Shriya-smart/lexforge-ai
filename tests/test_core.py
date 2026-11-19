"""Check the template entry points against LexForge's underlying services."""

import json

import pytest

from app import core
from lexforge.classifier import ClassifierUnavailable, Prediction
from lexforge.config import LABELS
from lexforge.content import DRAFTS


def test_prepares_a_corpus(tmp_path):
    source = tmp_path / "cuad.json"
    source.write_text(json.dumps({"data": [{"title": "Synthetic contract", "paragraphs": [
        {"context": "Either party may terminate this contract."},
    ]}]}), encoding="utf-8")
    result = core.milestone_1(source, tmp_path / "prepared")
    assert sum(split["rows"] for split in result.values()) == 1


def test_predicts_with_the_bundled_ngram():
    suggestions = core.milestone_2("Either party", top_k=3)
    assert len(suggestions) == 3
    assert all(word and 0 < probability <= 1 for word, probability in suggestions)


def test_missing_checkpoint_reports_unavailability(tmp_path):
    with pytest.raises(ClassifierUnavailable, match="checkpoint unavailable"):
        core.milestone_3("This is a synthetic test clause.", checkpoint=tmp_path)


def test_launches_the_template_entry_point(capsys):
    core.milestone_4(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_classifies_the_combined_sentence():
    challenge = DRAFTS[1]
    completion = "by giving 30 days' written notice."

    class SyntheticClassifier:
        def predict(self, text, top_k=1):
            assert text == f"{challenge.text} {completion}"
            return [Prediction(LABELS[1], 0.9)]

    result = core.milestone_5(challenge, completion, SyntheticClassifier())
    assert result.score.points == 100
