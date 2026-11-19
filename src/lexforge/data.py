"""Load pinned source datasets and prepare grouped or release-defined splits."""

import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from .config import LABELS

CUAD_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad/resolve/"
    "a3c393f5d103fd0c516374e4fdff676c8176dcb1/CUAD_v1/CUAD_v1.json"
)
LEDGAR_URL = (
    "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/"
    "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/ledgar/"
)
SPLITS = ("train", "validation", "test")
LEDGAR_LABELS = {
    "Confidentiality": LABELS[0],
    "Terminations": LABELS[1],
    "Governing Laws": LABELS[2],
    "Indemnifications": LABELS[3],
    "Indemnity": LABELS[3],
}
CUAD_LABELS = {
    "Termination For Convenience": LABELS[1],
    "Governing Law": LABELS[2],
    "Cap On Liability": LABELS[4],
}
TOKEN = re.compile(r"[a-z]+(?:['-][a-z]+)*|[0-9]+(?:[.,][0-9]+)*", re.I)
ABBREVIATION = re.compile(
    r"\b(?:U\.S|U\.K|e\.g|i\.e|Inc|Corp|Ltd|Co|No|Sec|Art|Mr|Ms|Dr)\.",
    re.I,
)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("‘’“”", "\x27\x27\"\""))
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(clean(text).lower())


def sentences(text: str) -> list[str]:
    protected = ABBREVIATION.sub(lambda m: m[0].replace(".", "\ue000"), clean(text))
    return [
        sentence.replace("\ue000", ".").strip()
        for sentence in re.split(r"(?<=[.!?;])\s+", protected)
        if sentence.strip()
    ]


def fingerprint(text: str) -> str:
    normalized = re.sub(r"\d+", "0", " ".join(tokenize(text)))
    return hashlib.sha256(normalized.encode()).hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=90) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_cuad(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_split(title: str, seed: int = 42) -> str:
    digest = hashlib.sha256(f"{seed}:{clean(title).casefold()}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def isolate(rows: list[dict]) -> dict[str, list[dict]]:
    """Remove cross-split duplicates and conflicting labels before evaluation."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[fingerprint(row["text"])].append(row)
    result = {split: [] for split in SPLITS}
    for copies in grouped.values():
        if len({row["split"] for row in copies}) == 1 and len(
            {row.get("label") for row in copies}
        ) == 1:
            row = copies[0]
            result[row["split"]].append(row)
    return result


def prepare_corpus(cuad: dict, seed: int) -> dict[str, list[dict]]:
    rows = []
    for contract in cuad["data"]:
        title = contract["title"]
        for paragraph in contract["paragraphs"]:
            for text in sentences(paragraph["context"]):
                if 5 <= len(tokenize(text)) <= 160:
                    rows.append({"text": text, "split": contract_split(title, seed),
                                 "document": title, "source": "CUAD"})
    return isolate(rows)


def prepare_clauses(cuad: dict, raw: Path, seed: int) -> dict[str, list[dict]]:
    import pandas as pd
    import pyarrow.parquet as parquet

    rows = []
    for split in SPLITS:
        path = raw / f"ledgar-{split}.parquet"
        metadata = parquet.ParquetFile(path).schema_arrow.metadata
        names = json.loads(metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        frame = pd.read_parquet(path, columns=["text", "label"])
        for text, index in frame.itertuples(index=False, name=None):
            if label := LEDGAR_LABELS.get(names[index]):
                rows.append({"text": clean(text), "label": label, "split": split,
                             "source": "LEDGAR"})
    for contract in cuad["data"]:
        title = contract["title"]
        for paragraph in contract["paragraphs"]:
            for question in paragraph["qas"]:
                label = CUAD_LABELS.get(question["id"].rsplit("__", 1)[-1])
                if label:
                    for answer in question["answers"]:
                        rows.append({"text": clean(answer["text"]), "label": label,
                                     "split": contract_split(title, seed),
                                     "source": "CUAD", "document": title})
    return isolate([row for row in rows if len(tokenize(row["text"])) >= 4])


def write_splits(splits: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "labels": dict(Counter(row["label"] for row in rows if "label" in row))}
    return summary


def load_rows(directory: Path, split: str) -> list[dict]:
    with (directory / f"{split}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def balanced_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit == 0:
        return rows.copy()
    randomizer = random.Random(seed)
    result = []
    for label in LABELS:
        examples = [row for row in rows if row["label"] == label]
        result.extend(randomizer.sample(examples, min(limit, len(examples))))
    randomizer.shuffle(result)
    return result
