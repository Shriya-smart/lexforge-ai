"""Interface tests with synthetic probabilities, plus opt-in real inference."""

import os
from pathlib import Path

import pytest

from lexforge.classifier import ClassifierUnavailable, GPT2Classifier, Prediction
from lexforge.config import ASSETS, LABELS


def test_missing_checkpoint_is_graceful(tmp_path):
    with pytest.raises(ClassifierUnavailable, match="checkpoint unavailable"):
        GPT2Classifier(tmp_path).predict("A synthetic legal example.")


def test_interface_orders_labels_and_checks_limit(tmp_path, monkeypatch):
    classifier = GPT2Classifier(tmp_path)
    monkeypatch.setattr(classifier, "probabilities", lambda texts: [[0.1, 0.05, 0.4, 0.15, 0.3]])
    assert classifier.predict("Synthetic.", 3) == [
        Prediction(LABELS[2], 0.4), Prediction(LABELS[4], 0.3), Prediction(LABELS[3], 0.15)
    ]
    with pytest.raises(ValueError):
        classifier.predict("Synthetic.", 6)


@pytest.mark.skipif(not os.getenv("LEXFORGE_TEST_GPT_BASE"),
                    reason="Opt in with a local GPT-2 base using LEXFORGE_TEST_GPT_BASE.")
def test_real_checkpoint_forward_pass():
    checkpoint = Path(os.environ.get("LEXFORGE_TEST_GPT_CHECKPOINT", str(ASSETS)))
    classifier = GPT2Classifier(checkpoint, os.environ["LEXFORGE_TEST_GPT_BASE"])
    governing = "This agreement shall be governed by the laws of the State of Delaware."
    results = classifier.predict(governing, 5)
    assert results[0].label == "Governing Law"
    assert {prediction.label for prediction in results} == set(LABELS)
    assert sum(prediction.confidence for prediction in results) == pytest.approx(1.0)
    assert results == sorted(results, key=lambda item: -item.confidence)
    batched = classifier.probabilities([governing, governing + " This clause survives."])
    assert batched[0][2] == pytest.approx(results[0].confidence, abs=1e-5)
