# Architecture and full-stack roadmap

```text
Rich / Typer CLI
  ├─ game.py + content.py + scoring.py
  │    └─ ClauseClassifier interface → GPT2Classifier → local checkpoint + base
  ├─ ngram.py → compressed word counts
  └─ storage.py → SQLite profiles and attempts

data.py → reproducible JSONL → scripts/
                               ├─ train_ngram.py → model + perplexity
                               └─ train_gpt2.py → checkpoint + validation report
                                                   └─ evaluate_gpt2.py → independent test report
```

The CLI owns input, presentation, and navigation. Draft orchestration, scoring, and models accept regular Python data without importing UI code. `ClauseClassifier` describes the prediction interface so a future service can reuse the orchestrator or run deterministic interface tests. Heavy machine learning imports and checkpoint loading occur on first inference. N-gram training and inference use the standard library.

SQLite writes each attempt and its XP in a single immediate transaction. The previous best score is read inside the transaction so simultaneous replays cannot grant duplicate XP. Each profile retains its own predictor setting and attempts. Dashboard summaries derive from stored attempts, so gains and achievements can be recalculated after reopening the game.

## Future full-stack roadmap

1. Wrap the game services in FastAPI routes for predictions, drafting, the five-category analyzer, and profile summaries. Load immutable models during service startup and bound batch sizes and input lengths at the API.
2. Add authenticated learner accounts and tenant-scoped storage. Introduce proper request idempotency keys and database migrations before moving concurrent hosted progress to PostgreSQL.
3. Build a React client around the same draft flow, rubric, difficulty rules, and progression. Add keyboard-first interaction, accessible confidence displays, and model status for partial availability.
4. Version model artifacts and publish evaluation reports with each release. Track held-out class support, category errors, and latency before changing the default model.
