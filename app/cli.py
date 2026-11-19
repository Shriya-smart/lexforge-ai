"""Run the LexForge menu through the template's command-line entry point."""

import sys

from lexforge.cli import app


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args == ["hello"]:
        print("Legal Sentence Game is ready. Run python -m app.cli to play LexForge.")
        return
    app(args=args, prog_name="python -m app.cli")


if __name__ == "__main__":
    main()
