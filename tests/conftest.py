"""Synthetic fixtures for deterministic tests. No results are model benchmarks."""

import pytest

from lexforge.ngram import NGram


@pytest.fixture
def synthetic_model():
    return NGram.fit(
        ["the party may terminate", "the party may terminate", "the party shall notify",
         "the client may inspect", "vendor must notify"],
        min_count=2,
    )
