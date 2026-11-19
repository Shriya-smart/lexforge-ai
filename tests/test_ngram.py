import math

import pytest

from lexforge.ngram import END, START, UNKNOWN, NGram


@pytest.mark.parametrize("order", [2, 3])
def test_distribution_is_normalized(synthetic_model, order):
    for context in ([], ["the", "party"], ["out", "of-vocabulary"]):
        probabilities = [synthetic_model.probability(w, context, order)
                         for w in synthetic_model.vocabulary]
        assert sum(probabilities) == pytest.approx(1.0)
        assert min(probabilities) > 0


def test_trigram_ranking_uses_context(synthetic_model):
    assert synthetic_model.predict("the party") == "may"
    assert synthetic_model.predict("party may") == "terminate"
    model = NGram.fit(["a party may act", "the party shall act"], min_count=1)
    assert model.predict("a party", 3) == "may"
    assert model.predict("the party", 3) == "shall"


def test_rank_matches_exhaustive_distribution(synthetic_model):
    for order in (2, 3):
        exhaustive = sorted(
            ((word, synthetic_model.probability(word, ["the", "party"], order))
             for word in synthetic_model.vocabulary - {END, UNKNOWN}),
            key=lambda pair: (-pair[1], pair[0]),
        )
        assert synthetic_model.suggest("The party", 3, order) == exhaustive[:3]


def test_round_trip_and_held_out_perplexity(synthetic_model, tmp_path):
    path = tmp_path / "synthetic.json.gz"
    synthetic_model.save(path)
    loaded = NGram.load(path)
    assert loaded.suggest("the party") == synthetic_model.suggest("the party")
    assert START not in loaded.vocabulary
    report = loaded.evaluate(["party not-in-training"])
    assert report["unknown_tokens"] == 1
    assert report["tokens_including_eos"] == 3
    expected = (loaded.probability("party", []) *
                loaded.probability(UNKNOWN, ["party"]) *
                loaded.probability(END, ["party", UNKNOWN])) ** (-1 / 3)
    assert report["perplexity"] == pytest.approx(expected)
    assert math.isfinite(report["perplexity"])


def test_invalid_inputs(synthetic_model):
    with pytest.raises(ValueError):
        synthetic_model.suggest("test", 0)
    with pytest.raises(ValueError):
        synthetic_model.suggest("test", order=4)
    with pytest.raises(ValueError):
        synthetic_model.evaluate([])
