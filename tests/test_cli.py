from typer.testing import CliRunner

from lexforge.classifier import Prediction
from lexforge.cli import app
from lexforge.storage import ProgressStore


def test_every_menu_at_narrow_width(tmp_path, synthetic_model, monkeypatch):
    import lexforge.cli as cli

    calls = []

    class SyntheticClassifier:
        """Stores inputs for CLI wiring assertions without making real predictions."""

        def __init__(self, checkpoint, base):
            self.checkpoint = checkpoint

        def predict(self, text, top_k=1):
            calls.append((text, top_k))
            choices = [Prediction("Confidentiality", 0.8), Prediction("Termination", 0.15),
                       Prediction("Governing Law", 0.05)]
            return choices[:top_k]

    monkeypatch.setattr(cli, "GPT2Classifier", SyntheticClassifier)
    model = tmp_path / "synthetic.json.gz"
    synthetic_model.save(model)
    db = tmp_path / "test.sqlite3"
    script = "\n".join([
        "q", "1", "1", "secret from third parties at all times.", "n",
        "2", "the party", "3", "This is a synthetic example of sensitive information.",
        "4", "1", "n", "5", "The recipient will disclose confidential information only to authorized personnel.", "n",
        "6", "7", "8", "9", "1", "2", "the party", "0", "",
    ])
    result = CliRunner().invoke(app, ["--db", str(db), "--ngram", str(model),
                                     "--gpt-checkpoint", str(tmp_path)], input=script,
                                env={"COLUMNS": "42", "NO_COLOR": "1"})
    assert result.exit_code == 0, result.output
    for display in ("LEXFORGE AI", "Trigram next words", "Bigram next words", "Top three clause",
                    "Category performance", "Recent activity", "Local leaderboard",
                    "Two models, two questions", "preference saved", "Session closed"):
        assert display in result.output
    assert calls == [
        ("The receiving party shall keep confidential information secret from third parties at all times.", 1),
        ("This is a synthetic example of sensitive information.", 3),
    ]
    reopened = ProgressStore(db)
    assert reopened.order("Guest") == 2
    assert {item["mode"] for item in reopened.attempts("Guest")} == {"draft", "guess", "revision"}


def test_absent_models_and_eof_return_cleanly(tmp_path):
    result = CliRunner().invoke(app, ["--db", str(tmp_path / "test.sqlite3"),
                                     "--ngram", str(tmp_path / "missing.gz"),
                                     "--gpt-checkpoint", str(tmp_path)],
                                input="1\n1\nsecret from third parties.\nn\n3\nLegal example.\n")
    assert result.exit_code == 0
    assert "N-gram model unavailable" in result.output
    assert "GPT-2 checkpoint unavailable" in result.output
    assert "GPT-2 check unavailable" in result.output
    assert "Completed attempts are saved" in result.output


def test_difficulty_unlocks_follow_distinct_drafts(tmp_path):
    db = tmp_path / "test.sqlite3"
    store = ProgressStore(db)
    store.ensure_profile("Guest")
    options = ["--db", str(db), "--ngram", str(tmp_path / "missing.gz")]
    locked = CliRunner().invoke(app, options, input="1\n2\n0\n")
    assert "unlock this difficulty" in locked.output
    for i in range(3):
        store.record("Guest", "draft", f"d0{i + 1}", "Confidentiality", 50)
    unlocked = CliRunner().invoke(app, options, input="1\n2\n:back\n0\n")
    assert unlocked.exit_code == 0
    assert "Intermediate · 1/5" in unlocked.output
    assert "Continue the sentence" in unlocked.output


def test_keyboard_interrupt_closes_cleanly(tmp_path, monkeypatch):
    from lexforge.cli import Prompt

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Prompt, "ask", interrupt)
    result = CliRunner().invoke(app, ["--db", str(tmp_path / "test.sqlite3")])
    assert result.exit_code == 0
    assert "Completed attempts are saved" in result.output
