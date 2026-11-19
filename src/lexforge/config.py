"""Shared model labels and filesystem conventions."""

import os
from pathlib import Path

LABELS = (
    "Confidentiality",
    "Termination",
    "Governing Law",
    "Indemnification",
    "Limitation of Liability",
)
BASE_MODEL = "openai-community/gpt2"
BASE_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MAX_LENGTH = 160
ASSETS = Path(__file__).resolve().parent / "assets"


def state_path() -> Path:
    root = Path(os.environ.get("LEXFORGE_HOME", Path.home() / ".lexforge"))
    return root / "progress.sqlite3"
