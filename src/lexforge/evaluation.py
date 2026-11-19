"""Evaluate held-out GPT-2 predictions without updating model parameters."""

from .config import LABELS


def classification_report(expected: list[int], predicted: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    indices = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=indices, zero_division=0
    )
    return {
        "examples": len(expected),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "labels": list(LABELS),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=indices).tolist(),
        "per_category": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
    }


def evaluate(classifier, rows: list[dict], batch_size: int = 16) -> dict:
    if not rows or batch_size < 1:
        raise ValueError("Evaluation requires examples and a positive batch size.")
    predicted = []
    for start in range(0, len(rows), batch_size):
        texts = [row["text"] for row in rows[start:start + batch_size]]
        for scores in classifier.probabilities(texts):
            predicted.append(max(range(len(scores)), key=scores.__getitem__))
    return classification_report([LABELS.index(row["label"]) for row in rows], predicted)
