"""Download CUAD and LexGLUE LEDGAR data and write reproducible JSONL splits."""

import argparse
import json
from pathlib import Path

from lexforge.data import (CUAD_URL, LEDGAR_URL, SPLITS, download, load_cuad,
                           prepare_clauses, prepare_corpus, write_splits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-only", action="store_true")
    args = parser.parse_args()
    sources = {"cuad.json": CUAD_URL}
    if not args.corpus_only:
        sources.update({f"ledgar-{s}.parquet": f"{LEDGAR_URL}{s}-00000-of-00001.parquet"
                        for s in SPLITS})
    for basename, url in sources.items():
        path = args.raw / basename
        if not path.is_file():
            print(f"Downloading {basename}", flush=True)
            download(url, path)
    cuad = load_cuad(args.raw / "cuad.json")
    result = {"seed": args.seed, "sources": sources, "source_revision_policy": "pinned URLs",
              "corpus": write_splits(prepare_corpus(cuad, args.seed), args.output / "corpus")}
    if not args.corpus_only:
        result["clauses"] = write_splits(prepare_clauses(cuad, args.raw, args.seed),
                                        args.output / "clauses")
    report = json.dumps(result, indent=2)
    (args.output / "manifest.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
