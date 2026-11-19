from concurrent.futures import ThreadPoolExecutor

from lexforge.storage import ProgressStore


def test_progress_and_settings_persist(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = ProgressStore(path)
    assert store.ensure_profile(" Ada ") == "Ada"
    store.ensure_profile("Grace")
    store.set_order("Ada", 2)
    assert store.record("Ada", "draft", "d01", "Confidentiality", 70) == 70
    assert store.record("Ada", "draft", "d01", "Confidentiality", 20, False) == 0
    assert store.record("Ada", "draft", "d01", "Confidentiality", 100, True) == 30
    assert store.record("Grace", "guess", "g01", "Confidentiality", 0, False) == 0

    reopened = ProgressStore(path)
    assert reopened.order("Ada") == 2
    assert reopened.order("Grace") == 3
    assert [row["matched"] for row in reopened.attempts("Ada")] == [1, 0, None]
    assert reopened.leaderboard() == [{"name": "Ada", "xp": 100}, {"name": "Grace", "xp": 0}]


def test_concurrent_replay_cannot_duplicate_xp(tmp_path):
    store = ProgressStore(tmp_path / "lab.sqlite3")
    store.ensure_profile("Ada")

    def replay(score):
        return store.record("Ada", "guess", "g01", "Confidentiality", score)

    with ThreadPoolExecutor(max_workers=4) as pool:
        awarded = list(pool.map(replay, [50, 100, 100, 60]))
    assert sum(awarded) == 100
    assert len(store.attempts("Ada")) == 4
