"""Command line: turing-meter validate <dir>; turing-meter pair <sheet.csv> --options M."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from turing_meter.tasks import validate_tree


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turing-meter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate every task under a directory")
    v.add_argument("dir", type=Path)
    p = sub.add_parser("pair", help="measure one person and one machine on a sheet of the person's own cases")
    p.add_argument("sheet", type=Path, help="CSV with one row per case (docs/PAIR-PROTOCOL.md, section 8)")
    p.add_argument("--options", type=int, required=True, help="number of options per case (2 for yes/no)")
    p.add_argument("--resamples", type=int, default=2000, help="resamplings of the cases for the intervals")
    p.add_argument("--seed", type=int, default=534)
    p.add_argument("--delegate-seconds", type=float, default=None,
                   help="seconds per case of someone who takes the machine's answer without checking")
    p.add_argument("--json", type=Path, default=None, help="also write the measurement as JSON")
    args = ap.parse_args(argv)

    if args.cmd == "pair":
        return _pair(args)

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


def _pair(args) -> int:
    from turing_meter.pair import SheetError, analyze, read_sheet, report

    try:
        ref = analyze(read_sheet(args.sheet), args.options, args.resamples, args.seed, args.delegate_seconds)
    except (SheetError, OSError) as e:
        print(f"turing-meter pair: {e}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(ref, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(report(ref))
    return 0
