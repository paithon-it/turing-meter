"""Command line: turing-meter validate <dir>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from turing_meter.tasks import validate_tree


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turing-meter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate every task under a directory")
    v.add_argument("dir", type=Path)
    args = ap.parse_args(argv)

    results = validate_tree(args.dir)
    if not results:
        print(f"no tasks found under {args.dir}", file=sys.stderr)
        return 1
    bad = 0
    for name, errs in results:
        if errs:
            bad += 1
            print(f"FAIL {name}: " + " | ".join(errs))
        else:
            print(f"ok   {name}")
    return 1 if bad else 0
