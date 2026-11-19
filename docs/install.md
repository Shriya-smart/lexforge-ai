# Installation

Use Python 3.11 in a virtual environment. The CLI runs in a UTF-8 terminal. Linux and Windows use the same CLI and training commands after activation.

## Ubuntu

```bash
cd lexforge-ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[ml,test]"
python scripts/fetch_base.py
lexforge --player Ada
```

## Windows PowerShell

```powershell
Set-Location lexforge-ai
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ".[ml,test]"
.\.venv\Scripts\python.exe scripts/fetch_base.py
.\.venv\Scripts\lexforge.exe --player Ada
```

Using the executable paths directly avoids shell activation policy changes. Use Windows Terminal with a monospace font that supports box-drawing characters.

## Installation options

- `python -m pip install .` installs the CLI and the bundled N-gram model without the optional PyTorch runtime.
- `python -m pip install ".[ml]"` enables GPT-2 inference. Cache the base once with `python scripts/fetch_base.py`.
- `python -m pip install -e ".[train,test]"` enables preprocessing, training, evaluation, and tests. For a smaller Linux CPU installation, install PyTorch from its CPU wheel index first with `python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu`.
- The fine-tuned weights and reports are bundled. The pretrained GPT-2 base is separate and uses about 525 MiB. An existing copy of the pinned base works with `lexforge --gpt-base PATH`. That directory needs `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`.
- If the Hugging Face transfer service is unavailable, set `HF_HUB_DISABLE_XET=1` in the shell before running the fetch script. Keep the same Hugging Face cache location for fetching and for launching the CLI.
- Run `lexforge --help` to see all paths and options. Without activation on Windows, substitute `.\.venv\Scripts\python.exe` for `python` and `.\.venv\Scripts\lexforge.exe` for `lexforge`.

For a local test run, install the `test` extra and call `python -m pytest -q`. For the optional real-checkpoint test, set `LEXFORGE_TEST_GPT_BASE` to the snapshot path printed by the fetch script, or to `openai-community/gpt2` to use the local Hugging Face cache. The remaining tests use explicitly synthetic fixtures or exercise the missing-checkpoint path.
