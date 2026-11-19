"""Word bigrams and trigrams with hierarchical smoothing."""

import gzip
import io
import json
import math
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from .data import tokenize

START = "<s>"
END = "</s>"
UNKNOWN = "<unk>"


class NGram:
    def __init__(self, unigram: dict, bigram: dict, trigram: dict, strength: float):
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("Smoothing strength must be finite and positive.")
        self.unigram, self.bigram, self.trigram = unigram, bigram, trigram
        self.strength = strength
        self.vocabulary = frozenset(unigram)
        self.total = sum(unigram.values())
        self.bigram_totals = {key: sum(counts.values()) for key, counts in bigram.items()}
        self.trigram_totals = {key: sum(counts.values()) for key, counts in trigram.items()}
        self.common = sorted(self.vocabulary - {UNKNOWN, END},
                             key=lambda word: (-unigram[word], word))

    @classmethod
    def fit(cls, texts: Iterable[str], min_count: int = 3, strength: float = 2.0) -> "NGram":
        if min_count < 1:
            raise ValueError("Minimum count must be positive.")
        tokenized = [words for text in texts if (words := tokenize(text))]
        if not tokenized:
            raise ValueError("Training requires tokenized sentences.")
        totals = Counter(chain.from_iterable(tokenized))
        vocabulary = {word for word, count in totals.items() if count >= min_count}
        unigram = Counter({UNKNOWN: 0})
        bigram, trigram = defaultdict(Counter), defaultdict(Counter)
        for words in tokenized:
            left = previous = START
            for word in chain((word if word in vocabulary else UNKNOWN for word in words), (END,)):
                unigram[word] += 1
                bigram[previous][word] += 1
                trigram[f"{left} {previous}"][word] += 1
                left, previous = previous, word
        return cls(unigram, dict(bigram), dict(trigram), strength)

    def _map(self, word: str) -> str:
        return word if word in self.vocabulary else UNKNOWN

    def _context(self, context: list[str], order: int) -> tuple[str, str | None]:
        if order not in (2, 3):
            raise ValueError("Choose bigram (2) or trigram (3).")
        left, right = ([START, START] + [self._map(word) for word in context[-2:]])[-2:]
        return right, f"{left} {right}" if order == 3 else None

    def _probability(self, word: str, bigram: str, trigram: str | None) -> float:
        probability = (self.unigram.get(word, 0) + 1) / (self.total + len(self.vocabulary))
        bigram_counts = self.bigram.get(bigram, {})
        probability = (bigram_counts.get(word, 0) + self.strength * probability) / (
            self.bigram_totals.get(bigram, 0) + self.strength
        )
        if trigram is None:
            return probability
        trigram_counts = self.trigram.get(trigram, {})
        return (trigram_counts.get(word, 0) + self.strength * probability) / (
            self.trigram_totals.get(trigram, 0) + self.strength
        )

    def probability(self, word: str, context: list[str], order: int = 3) -> float:
        return self._probability(self._map(word), *self._context(context, order))

    def suggest(self, prefix: str, top_k: int = 5, order: int = 3) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        bigram, trigram = self._context(tokenize(prefix), order)
        candidates = set(self.common[:top_k])
        candidates.update(self.bigram.get(bigram, {}))
        if trigram is not None:
            candidates.update(self.trigram.get(trigram, {}))
        ranked = [(word, self._probability(word, bigram, trigram))
                  for word in candidates - {END, UNKNOWN}]
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:top_k]

    def predict(self, prefix: str, order: int = 3) -> str | None:
        suggestions = self.suggest(prefix, top_k=1, order=order)
        return suggestions[0][0] if suggestions else None

    def evaluate(self, texts: Iterable[str], order: int = 3) -> dict:
        loss = 0.0
        count = unknown = 0
        for text in texts:
            words = tokenize(text)
            if not words:
                continue
            unknown += sum(word not in self.vocabulary for word in words)
            context: list[str] = []
            for word in chain(words, (END,)):
                loss -= math.log(self.probability(word, context, order))
                context = [*context[-1:], word]
                count += 1
        if not count:
            raise ValueError("No held-out tokens.")
        return {"perplexity": math.exp(loss / count), "tokens_including_eos": count,
                "unknown_tokens": unknown}

    def save(self, path: Path) -> None:
        data = {"version": 1, "strength": self.strength, "unigram": self.unigram,
                "bigram": self.bigram, "trigram": self.trigram}
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            with staged.open("wb") as raw, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, separators=(",", ":"))
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "NGram":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            data = json.load(file)
        if data["version"] != 1:
            raise ValueError("Unsupported N-gram model format.")
        return cls(data["unigram"], data["bigram"], data["trigram"], data["strength"])
