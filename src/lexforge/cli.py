"""Rich terminal interface for LexForge AI."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .classifier import ClassifierUnavailable, GPT2Classifier
from .config import ASSETS, LABELS, state_path
from .content import DIFFICULTIES, DRAFTS, GUESSES, REVISIONS
from .game import achievements, complete_draft, unlocked_difficulties
from .ngram import NGram
from .scoring import Score, guess_score, level, writing_score
from .storage import ProgressStore

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
theme = Theme({"accent": "bold medium_purple1", "guide": "cyan",
               "success": "bright_green", "warning": "yellow"})
MENU = (
    ("1", "Start Legal Drafting Challenge"),
    ("2", "AI Next-Word Predictor"),
    ("3", "Analyze My Legal Sentence"),
    ("4", "Guess the Clause"),
    ("5", "Correct the Clause"),
    ("6", "My Progress"),
    ("7", "Leaderboard"),
    ("8", "Model Insights"),
    ("9", "Settings and Help"),
    ("0", "Exit"),
)


@dataclass
class Session:
    console: Console
    store: ProgressStore
    player: str
    ngram_path: Path
    classifier: GPT2Classifier
    ngram: NGram | None = None

    def panel(self, title: str, content) -> None:
        self.console.print(Panel(content, title=title, title_align="left",
                                 border_style="medium_purple1", padding=(0, 1)))

    def ask(self, question: str, choices=None, default=None) -> str:
        return Prompt.ask(question, choices=choices, default=default,
                          show_choices=False, console=self.console).strip()

    def ask_text(self, question: str) -> str | None:
        while True:
            text = self.ask(f"{question} [dim](:back)[/]")
            if text.casefold() == ":back":
                return None
            if text and len(text) <= 8000:
                return text
            self.console.print("Enter 1–8,000 characters, or type :back.", style="warning")

    def welcome(self) -> None:
        header = Text("LEXFORGE AI", style="accent")
        subtitle = Text("THE LEGAL LANGUAGE LAB", style="guide")
        signed_in = Text(f"Player  {self.player}", style="dim")
        note = Text("For educational use only. LexForge does not provide legal advice.")
        self.panel("Welcome", Group(header, subtitle, signed_in, note))

    def run(self) -> None:
        actions = (self.draft, self.predict, self.analyze, self.guess, self.revise,
                   self.dashboard, self.leaderboard, self.insights, self.settings)
        self.welcome()
        while True:
            menu = Table.grid(padding=(0, 2))
            menu.add_column(style="guide", justify="right")
            menu.add_column()
            for key, label in MENU:
                menu.add_row(key, label)
            self.panel("Main menu", menu)
            choice = self.ask("Select an option", choices=[key for key, _ in MENU])
            if choice == "0":
                self.console.print("Session closed. Your progress is saved.", style="guide")
                return
            try:
                actions[int(choice) - 1]()
            except sqlite3.Error as error:
                self.console.print(Text(f"Could not access saved progress: {error}", style="warning"))

    def suggestions(self, prefix: str) -> None:
        if self.ngram is None:
            try:
                self.ngram = NGram.load(self.ngram_path)
            except (OSError, ValueError, KeyError, TypeError, EOFError):
                self.console.print(Text(f"N-gram model unavailable at {self.ngram_path}. "
                                        "See docs/training.md.", style="warning"))
                return
        order = self.store.order(self.player)
        table = Table(title=f"{'Bigram' if order == 2 else 'Trigram'} next words",
                      title_justify="left", header_style="guide", expand=False)
        table.add_column("Next word", style="accent")
        table.add_column("Probability", justify="right")
        for word, probability in self.ngram.suggest(prefix, top_k=5, order=order):
            table.add_row(Text(word), f"{probability:.2%}")
        self.console.print(table)

    def pick(self, title: str, options: list[str]) -> int | None:
        menu = Table.grid(padding=(0, 2))
        menu.add_column(style="guide", justify="right")
        menu.add_column()
        for number, option in enumerate(options, 1):
            menu.add_row(str(number), Text(option))
        menu.add_row("0", "Back")
        self.panel(title, menu)
        choice = int(self.ask("Select an option", choices=[str(i) for i in range(len(options) + 1)]))
        return choice - 1 if choice else None

    def draft(self) -> None:
        attempts = self.store.attempts(self.player)
        played = {row["challenge"] for row in attempts if row["mode"] == "draft"}
        unlocked = unlocked_difficulties(len(played))
        options = [name if name in unlocked else f"{name} · unlocks after {i * 3} distinct drafts"
                   for i, name in enumerate(DIFFICULTIES)]
        difficulty = self.pick("Drafting difficulty", options)
        if difficulty is None:
            return
        if DIFFICULTIES[difficulty] not in unlocked:
            self.console.print("Complete more distinct drafts to unlock this difficulty.", style="warning")
            return
        challenges = [challenge for challenge in DRAFTS if challenge.difficulty == DIFFICULTIES[difficulty]]
        challenges.sort(key=lambda challenge: challenge.id in played)
        for index, challenge in enumerate(challenges, 1):
            details = "\n".join(f"• {check.description}" for check in challenge.checks)
            self.panel(f"{challenge.difficulty} · {index}/{len(challenges)} · {challenge.label}",
                       Group(Text(challenge.text + " …", style="accent"),
                             Text(details, style="guide")))
            self.suggestions(challenge.text)
            completion = self.ask_text("Continue the sentence")
            if completion is None:
                return
            with self.console.status("Analyzing the complete sentence…", spinner_style="guide"):
                result = complete_draft(challenge, completion, self.classifier)
            self.panel("Your draft", Text(result.sentence))
            if result.prediction:
                predicted = result.prediction
                matches = predicted.label == challenge.label
                self.console.print(
                    Text(f"GPT-2 predicts {predicted.label} · {predicted.confidence:.2%} confidence")
                )
                self.console.print(Text(f"Target: {challenge.label} · "
                                        f"{'category match' if matches else 'category differs'}",
                                        style="success" if matches else "warning"))
                self.console.print("Confidence measures model preference. Legal correctness requires review.",
                                   style="dim")
            else:
                matches = None
                self.console.print(Text(result.model_error or "", style="warning"))
                self.console.print(Text(f"Target: {challenge.label} · GPT-2 check unavailable"))
            self.show_score(result.score)
            xp = self.store.record(self.player, "draft", challenge.id, challenge.label,
                                   result.score.points, matches)
            self.after_score(result.score.points, xp)
            if index < len(challenges) and not self.continue_playing("Next drafting challenge?"):
                return
        self.console.print("Set complete. Return to drafting to replay or choose another difficulty.",
                           style="guide")

    def predict(self) -> None:
        prefix = self.ask_text("Enter a clause prefix")
        if prefix is not None:
            self.suggestions(prefix)

    def analyze(self) -> None:
        sentence = self.ask_text("Enter a complete legal sentence")
        if sentence is None:
            return
        try:
            with self.console.status("Classifying your sentence…", spinner_style="guide"):
                predictions = self.classifier.predict(sentence, top_k=3)
        except ClassifierUnavailable as error:
            self.console.print(Text(str(error), style="warning"))
            return
        table = Table(title="GPT-2 · Top three clause categories",
                      title_justify="left", header_style="guide")
        table.add_column("Clause type")
        table.add_column("Confidence", justify="right")
        for prediction in predictions:
            table.add_row(prediction.label, f"{prediction.confidence:.2%}")
        self.console.print(table)
        self.console.print("Confidence measures model preference. Review legal meaning independently.",
                           style="dim")

    def guess(self) -> None:
        played = {row["challenge"] for row in self.store.attempts(self.player)
                  if row["mode"] == "guess" and row["score"] == 100}
        examples = sorted(GUESSES, key=lambda example: example[0] in played)
        for index, (identifier, expected, sentence) in enumerate(examples, 1):
            self.panel(f"Guess the clause · {index}/{len(examples)}", Text(sentence))
            chosen = self.pick("Which category fits?", list(LABELS))
            if chosen is None:
                return
            score = guess_score(expected, LABELS[chosen])
            xp = self.store.record(self.player, "guess", identifier, expected, score, bool(score))
            feedback = "Correct" if score else f"You chose {LABELS[chosen]}"
            self.console.print(Text(f"{feedback} · Expected category: {expected}",
                                    style="success" if score else "warning"))
            self.after_score(score, xp)
            if index < len(examples) and not self.continue_playing("Next clause?"):
                return
        self.console.print("Set complete. Select Guess the Clause again to replay.", style="guide")

    def revise(self) -> None:
        played = {row["challenge"] for row in self.store.attempts(self.player)
                  if row["mode"] == "revision" and row["score"] == 100}
        challenges = sorted(REVISIONS, key=lambda challenge: challenge.id in played)
        for index, challenge in enumerate(challenges, 1):
            self.panel(f"Correct the clause · {index}/{len(challenges)} · {challenge.label}",
                       Group(Text(challenge.text, style="accent"),
                             Text("\n".join(f"• {check.description}" for check in challenge.checks),
                                  style="guide")))
            text = self.ask_text("Rewrite the entire clause")
            if text is None:
                return
            score = writing_score(challenge, text)
            xp = self.store.record(self.player, "revision", challenge.id, challenge.label, score.points)
            self.show_score(score)
            self.after_score(score.points, xp)
            if index < len(challenges) and not self.continue_playing("Next revision?"):
                return
        self.console.print("Set complete. Select Correct the Clause again to replay.", style="guide")

    def continue_playing(self, label: str) -> bool:
        return Confirm.ask(label, default=True, console=self.console)

    def show_score(self, score: Score) -> None:
        table = Table(header_style="guide")
        table.add_column("Rubric")
        table.add_column("Points", justify="right")
        for criterion in score.criteria:
            points = "Unavailable" if criterion.earned is None else f"{criterion.earned}/{criterion.available}"
            table.add_row(criterion.description, points)
        self.console.print(table)
        self.console.print("Writing checks look for listed details and selected expressions.", style="dim")

    def after_score(self, points: int, xp: int) -> None:
        attempts = self.store.attempts(self.player)
        total = sum(row["xp"] for row in attempts)
        self.console.print(Text(f"Score {points}/100  ·  +{xp} XP  ·  Level {level(total)}",
                                style="success"))
        if xp == 0:
            self.console.print("XP is awarded for improving your best score on a challenge.", style="dim")

    def dashboard(self) -> None:
        attempts = self.store.attempts(self.player)
        total = sum(row["xp"] for row in attempts)
        distinct = len({(row["mode"], row["challenge"]) for row in attempts})
        drafts = len({row["challenge"] for row in attempts if row["mode"] == "draft"})
        average = mean(row["score"] for row in attempts) if attempts else 0
        progress = Progress(TextColumn("Next level"), BarColumn(bar_width=None),
                            TextColumn("{task.completed:.0f}/{task.total:.0f} XP"),
                            expand=True, auto_refresh=False, console=self.console)
        progress.add_task("level", completed=total % 250, total=250)
        badges = ", ".join(achievements(attempts)) or "None yet"
        self.panel("My progress", Group(
            Text(f"{self.player} · Level {level(total)} · {total} XP", style="accent"),
            Text(f"{distinct} challenges completed · {len(attempts)} attempts · Average {average:.1f}/100"),
            Text(f"Drafting unlocked: {', '.join(unlocked_difficulties(drafts))}"),
            progress,
            Text(f"Achievements: {badges}"),
        ))
        categories = Table(title="Category performance", title_justify="left", header_style="guide")
        for name in ("Category", "Attempts", "Average"):
            categories.add_column(name, justify="left" if name == "Category" else "right")
        for label in LABELS:
            scores = [row["score"] for row in attempts if row["category"] == label]
            categories.add_row(label, str(len(scores)), f"{mean(scores):.1f}/100" if scores else "—")
        self.console.print(categories)
        if attempts:
            recent = Table(title="Recent activity", title_justify="left", header_style="guide")
            for column in ("Mode / category", "Score", "XP"):
                recent.add_column(column)
            for row in attempts[:5]:
                recent.add_row(f"{row['mode'].title()} / {row['category']}",
                               str(row["score"]), f"+{row['xp']}")
            self.console.print(recent)

    def leaderboard(self) -> None:
        table = Table(title="Local leaderboard", title_justify="left", header_style="guide")
        for column in ("Rank", "Player", "XP", "Level"):
            table.add_column(column)
        for rank, row in enumerate(self.store.leaderboard(), 1):
            table.add_row(str(rank), Text(row["name"]), str(row["xp"]), str(level(row["xp"])))
        self.console.print(table)

    def insights(self) -> None:
        self.panel("Two models, two questions", Group(
            Text("N-gram · What word comes next? It estimates a probability from nearby word counts."),
            Text("GPT-2 · What type of clause is this? It assigns scores across five categories."),
            Text("The predictor shows individual word probabilities. GPT-2 confidence is its softmax "
                 "score for a category. It does not measure legal validity.", style="dim"),
        ))
        ngram = read_metrics(self.ngram_path.with_name("ngram-metrics.json"))
        if matches_digest(self.ngram_path, ngram.get("model_sha256")):
            table = Table(title="N-gram · Held-out perplexity (lower is better)", header_style="guide")
            for column in ("Split", "Bigram", "Trigram"):
                table.add_column(column)
            for split in ("validation", "test"):
                values = ngram.get(split, {})
                if all(key in values for key in ("2", "3")):
                    table.add_row(split.title(), f"{values['2']['perplexity']:.2f}",
                                  f"{values['3']['perplexity']:.2f}")
            self.console.print(table)
        checkpoint = self.classifier.checkpoint
        weights = checkpoint / "classifier.safetensors"
        report = read_metrics(checkpoint / "checkpoint.json") if weights.is_file() else {}
        validation = report.get("validation")
        metrics = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
        keys = {"examples", *metrics}
        if not isinstance(validation, dict) or not keys <= validation.keys():
            self.console.print("GPT-2 evaluation metrics are unavailable for this checkpoint.",
                               style="warning")
            return

        test = read_metrics(checkpoint / "test-metrics.json")
        show_test = (test.get("split") == "test" and keys <= test.keys()
                     and matches_digest(weights, test.get("checkpoint_sha256")))
        table = Table(title="GPT-2 · Held-out classification", title_justify="left", header_style="guide")
        table.add_column("Metric")
        table.add_column("Validation", justify="right")
        if show_test:
            table.add_column("Test", justify="right")
        reports = [validation, test] if show_test else [validation]
        table.add_row("Examples", *(str(result["examples"]) for result in reports))
        for name in metrics:
            table.add_row(name.replace("_", " ").title(),
                          *(f"{result[name]:.1%}" for result in reports))
        self.console.print(table)
        self.console.print("The model reports include per-category results and confusion matrices.",
                           style="dim")

    def settings(self) -> None:
        order = self.store.order(self.player)
        self.panel("Settings and help", Group(
            Text(f"Current predictor: {'bigram' if order == 2 else 'trigram'}"),
            Text(f"Save file: {self.store.path}"),
            Text("Type :back at text prompts or use 0 in submenus. Ctrl+C closes the lab."),
            Text("Earn XP by improving a personal best. Every 250 XP raises your level."),
            Text("Drafting awards 70 writing points and 30 category-match points. "
                 "Missing GPT-2 leaves the category check ungraded."),
        ))
        selection = self.pick("Settings", ["Use bigram", "Use trigram", "Change player profile"])
        if selection is None:
            return
        if selection < 2:
            self.store.set_order(self.player, selection + 2)
            self.console.print("Predictor preference saved.", style="success")
            return
        name = self.ask_text("Profile name")
        if name is not None:
            try:
                self.player = self.store.ensure_profile(name)
                self.console.print(Text(f"Playing as {self.player}", style="success"))
            except ValueError as error:
                self.console.print(Text(str(error), style="warning"))


def read_metrics(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def matches_digest(path: Path, expected: str | None) -> bool:
    if not expected:
        return False
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected
    except OSError:
        return False


@app.command()
def play(
    player: str = typer.Option("Guest", help="Name of the local player profile."),
    db: Path | None = typer.Option(None, help="Path to a SQLite progress file."),
    ngram: Path = typer.Option(ASSETS / "ngram.json.gz", help="Saved N-gram model."),
    gpt_checkpoint: Path = typer.Option(ASSETS, help="Fine-tuned GPT-2 checkpoint directory."),
    gpt_base: str | None = typer.Option(None, envvar="LEXFORGE_GPT_BASE",
                                        help="Path to a local copy of the pinned GPT-2 base."),
) -> None:
    """Start LexForge AI, the legal language lab."""
    console = Console(theme=theme)
    try:
        store = ProgressStore(db or state_path())
        session = Session(console, store, store.ensure_profile(player),
                          ngram, GPT2Classifier(gpt_checkpoint, gpt_base))
        session.run()
    except (EOFError, KeyboardInterrupt):
        console.print("\nSession closed. Completed attempts are saved.", style="guide")
    except (sqlite3.Error, OSError, ValueError) as error:
        console.print(Text(f"Could not start LexForge: {error}", style="warning"))
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
