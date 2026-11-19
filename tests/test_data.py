"""Synthetic preprocessing inputs exercise leakage controls."""

import pytest

from lexforge.data import contract_split, fingerprint, isolate, prepare_clauses, sentences, tokenize


def test_normalization_and_sentence_boundaries():
    assert sentences("Acme Inc. acts.\nSend 1.5 units; Then stop.") == [
        "Acme Inc. acts.", "Send 1.5 units;", "Then stop."
    ]
    assert tokenize("Party’s NON-PUBLIC 1,200.00!") == ["party's", "non-public", "1,200.00"]


def test_document_split_and_duplicate_quarantine():
    assert contract_split("Contract A") == contract_split(" contract a ")
    assert fingerprint("Wait 30 days.") == fingerprint("WAIT 45 DAYS!")
    rows = [
        {"text": "Wait 30 days.", "split": "train", "label": "a"},
        {"text": "WAIT 45 DAYS!", "split": "test", "label": "a"},
        {"text": "Separate example.", "split": "validation", "label": "a"},
        {"text": "Separate example.", "split": "validation", "label": "a"},
        {"text": "Conflict", "split": "train", "label": "a"},
        {"text": "Conflict", "split": "train", "label": "b"},
    ]
    result = isolate(rows)
    assert result == {"train": [], "validation": [rows[2]], "test": []}


def test_cuad_label_mapping_matches_release_spelling(tmp_path):
    import json

    pytest.importorskip("pandas")
    pyarrow = pytest.importorskip("pyarrow")
    from pyarrow import parquet

    table = pyarrow.table({"text": [""], "label": [0]})
    metadata = {"info": {"features": {"label": {"names": ["Unmapped label"]}}}}
    table = table.replace_schema_metadata({b"huggingface": json.dumps(metadata).encode()})
    for split in ("train", "validation", "test"):
        parquet.write_table(table, tmp_path / f"ledgar-{split}.parquet")
    cuad = {"data": [{"title": "Synthetic test contract", "paragraphs": [{"qas": [
        {"id": "example__Termination For Convenience",
         "answers": [{"text": "Either party may end the agreement."}]},
    ]}]}]}
    groups = prepare_clauses(cuad, tmp_path, 42)
    assert groups[contract_split("Synthetic test contract")][0]["label"] == "Termination"
