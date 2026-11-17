"""Claude Haiku 5.5 told a random letter: the random anchor, through Claude Code in headless mode.

Usage, from the experiment folder:

  python -m angle.anchor items                                           # first, the items with their random letter
  python -m closed.anchor_claude MODEL NAME EFFORT [--limit N] [--workers N]

Everything as the told condition of closed/devil_claude.py (same isolation, same prompt, same one-field schema),
on data/items/chatbench-anchor.jsonl, where the hypothesis is the random letter. Output:
devil-results/chatbench-anchor__NAME__told.jsonl. Resumes where it stopped; stops on the plan's limit.
"""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from angle.anchor import ITEMS
from closed.devil_claude import MISSING, RESULTS, ask
from closed.machines_claude import append, report


def run(model: str, name: str, effort: str, limit: int | None, workers: int) -> None:
    lock, stop = threading.Lock(), threading.Event()
    file = RESULTS / f"chatbench-anchor__{name}__told.jsonl"
    done = {json.loads(l)["id"] for l in open(file, encoding="utf-8")} if file.exists() else set()
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    random.Random(534).shuffle(items)
    rest = [c for c in items if c["id"] not in done][:limit]

    def one(c: dict) -> None:
        if stop.is_set():
            return
        line, missing = ask(c, "told", model, effort)
        if missing and "limit" in (missing.get("head", "") + missing["why"]).lower():
            stop.set()
            print(f"STOPPED on a limit: {missing.get('head', missing['why'])[:160]}", flush=True)
            return
        if line:
            line["person_answer"] = c["person_answer"]
        else:
            missing["task"] = "chatbench-anchor"
        append(file if line else MISSING, line or missing, lock)

    t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, rest))
    report(file, f"chatbench-anchor {name}")
    print(f"{len(rest)} asked in {time.time() - t:.0f} s", flush=True)
    if stop.is_set():
        sys.exit(4)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    model, name, effort = sys.argv[1:4]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    run(model, name, effort, limit, workers)


if __name__ == "__main__":
    main()
