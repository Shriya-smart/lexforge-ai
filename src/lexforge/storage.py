"""Transactional SQLite progress for multiple local player profiles."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    name TEXT PRIMARY KEY,
    ngram_order INTEGER NOT NULL DEFAULT 3 CHECK (ngram_order IN (2, 3))
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY,
    player TEXT NOT NULL REFERENCES profiles(name),
    mode TEXT NOT NULL CHECK (mode IN ('draft', 'guess', 'revision')),
    challenge TEXT NOT NULL,
    category TEXT NOT NULL,
    score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    xp INTEGER NOT NULL CHECK (xp BETWEEN 0 AND 100),
    matched INTEGER CHECK (matched IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS attempts_player_challenge
ON attempts(player, mode, challenge, score);
"""


class ProgressStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def ensure_profile(self, name: str) -> str:
        name = name.strip()
        if not 1 <= len(name) <= 28 or not all(character.isprintable() for character in name):
            raise ValueError("Choose a printable player name of 1–28 characters.")
        with self.connection() as connection:
            connection.execute("INSERT OR IGNORE INTO profiles(name) VALUES (?)", (name,))
        return name

    def order(self, player: str) -> int:
        with self.connection() as connection:
            row = connection.execute("SELECT ngram_order FROM profiles WHERE name = ?", (player,)).fetchone()
        return row["ngram_order"]

    def set_order(self, player: str, order: int) -> None:
        with self.connection() as connection:
            connection.execute("UPDATE profiles SET ngram_order = ? WHERE name = ?", (order, player))

    def record(self, player: str, mode: str, challenge: str, category: str,
               score: int, matched: bool | None = None) -> int:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            best = connection.execute(
                """SELECT COALESCE(MAX(score), 0) FROM attempts
                   WHERE player=? AND mode=? AND challenge=?""", (player, mode, challenge)
            ).fetchone()[0]
            xp = max(0, score - best)
            connection.execute(
                """INSERT INTO attempts(player, mode, challenge, category, score, xp, matched)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""", (player, mode, challenge, category, score, xp, matched)
            )
        return xp

    def attempts(self, player: str) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM attempts WHERE player=? ORDER BY id DESC", (player,))
            return [dict(row) for row in rows]

    def leaderboard(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT profiles.name, COALESCE(SUM(attempts.xp), 0) AS xp
                   FROM profiles LEFT JOIN attempts ON profiles.name = attempts.player
                   GROUP BY profiles.name ORDER BY xp DESC, profiles.name COLLATE NOCASE, profiles.name
                   LIMIT 20"""
            )
            return [dict(row) for row in rows]
