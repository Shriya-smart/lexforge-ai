"""Check the module command and persistence exposed by the template."""

import subprocess
import sys

from app.storage import ProgressStore


def test_module_runs_hello():
    result = subprocess.run([sys.executable, "-m", "app.cli", "hello"],
                            check=True, capture_output=True, text=True)
    assert "Legal Sentence Game is ready" in result.stdout


def test_progress_store_round_trip(tmp_path):
    path = tmp_path / "progress.sqlite3"
    ProgressStore(path).ensure_profile("Student")
    assert ProgressStore(path).leaderboard()[0]["name"] == "Student"
