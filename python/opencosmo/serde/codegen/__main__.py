"""Regenerate message modules, or check that they are up to date."""

from __future__ import annotations

import argparse
import sys

from .config import SPECS
from .emit import render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m opencosmo.serde.codegen")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit with an error instead of writing when a module is out of date",
    )
    args = parser.parse_args(argv)
    stale = []
    for spec in SPECS:
        source = render(spec)
        current = spec.output.read_text() if spec.output.exists() else None
        if current == source:
            continue
        stale.append(spec.output)
        if not args.check:
            spec.output.write_text(source)
            print(f"Wrote {spec.output}")
    if args.check and stale:
        for path in stale:
            print(f"Out of date: {path}", file=sys.stderr)
        print("Run `python -m opencosmo.serde.codegen` to regenerate.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
