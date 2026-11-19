# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/kj4ug9nh9e/badge.svg)](https://ask.thili.ai/b/kj4ug9nh9e)

This project creates a command-line interface game designed for law students to practice drafting legal sentences. Players will complete partial legal sentences using n-gram prediction, receiving instant feedback on the generated clause types from a fine-tuned GPT-2 model. It's a fun way to hone legal drafting skills and understand common clause structures.

## What you are building

**Goal.** Develop a CLI game where players complete legal sentences using n-gram predictions, and then have their completed sentences classified by a fine-tuned GPT-2 model.

**Done when.** You will have built a command-line interface game where players complete legal sentences using your n-gram model's predictions. The game will then classify the player's completed sentences using a fine-tuned GPT-2 model and provide instant feedback on the generated clause types.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a legal text corpus for n-gram model training.**  
      Hint: Use text processing techniques to prepare your legal text corpus for n-gram training.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Implement the core logic for predicting the next word based on the n-gram model you built.
- [ ] **3. Load the fine-tuned GPT-2 model for legal clause classification.**  
      Hint: Load the pre-trained and fine-tuned GPT-2 model for legal clause classification, as shown in the drill.
- [ ] **4. Create a CLI game interface presenting partial legal sentences for completion.**  
      Hint: Design a simple command-line interface to present partial sentences and capture user input.
- [ ] **5. After user completion, classify the full sentence and provide feedback.**  
      Hint: Integrate the GPT-2 model to classify the user's completed sentence and display the predicted clause type as feedback.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement a scoring system based on how closely the player's completed sentence matches the expected clause type or a 'correct' completion.
- Allow players to input their own full legal sentences for classification by the GPT-2 model, outside of the completion game.
- Add more game modes, such as identifying the clause type of a given sentence or correcting errors in a provided legal sentence.
