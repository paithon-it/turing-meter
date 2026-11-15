"""A closed model in the two new conditions of the devil's advocate, through Claude Code in headless mode.

Usage, from the experiment folder:

  python -m closed.devil_claude MODEL NAME MODE EFFORT [--limit N] [--workers N]     # MODE: told or devil

Same isolation as closed/machines_claude.py (no tools, no MCP servers, no inherited settings, an empty folder, a
fixed system prompt), one item of data/items/chatbench-devil.jsonl per call. The prompt is angle.devil.prompt.
`told` constrains the letter with a one-field schema, as the other Claude runs; `devil` with a two-field schema,
the arguments first and then the letter, so that the arguments are written before the verdict. The arguments are
kept in `head`. The condition alone is the run already in closed-results/.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from angle.devil_prompt import LETTERS, prompt
from closed.machines_claude import append, command, environment, read_letter, report

RESULTS = Path("devil-results")
ITEMS = Path("data/items/chatbench-devil.jsonl")
MISSING = RESULTS / "_missing.jsonl"


def schema(k: int, mode: str) -> dict:
    letter = {"type": "string", "enum": list(LETTERS[:k])}
    if mode == "told":
        return {"type": "object", "properties": {"answer": letter}, "required": ["answer"], "additionalProperties": False}
    return {"type": "object", "properties": {"arguments": {"type": "string"}, "answer": letter},
            "required": ["arguments", "answer"], "additionalProperties": False}


def launch(text: str, model: str, effort: str, sch: dict, timeout: int = 300) -> str:
    c = command(text, model, effort) + ["--json-schema", json.dumps(sch)]
    with tempfile.TemporaryDirectory(prefix="turing-closed-") as cwd:
        r = subprocess.run(c, cwd=cwd, env=environment(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=timeout)
    if r.returncode != 0 and not r.stdout.strip():
        raise RuntimeError(f"claude -p exited with {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def ask(c: dict, mode: str, model: str, effort: str) -> tuple[dict | None, dict | None]:
    """The line to write, or the reason why there is none."""
    k = len(c["options"])
    base = {"id": c["id"], "task": "chatbench-devil", "mode": mode, "model": model, "effort": effort}
    try:
        v = json.loads(launch(prompt(c, mode), model, effort, schema(k, mode)))
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return None, {**base, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    if v.get("is_error"):
        return None, {**base, "why": "error", "head": str(v.get("result"))[:200]}
    u = v.get("usage", {})
    spent = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
             "output_tokens": u.get("output_tokens", 0), "answered_by": sorted(v.get("modelUsage", {}))}
    out = v.get("structured_output") or {}
    answer, arguments = out.get("answer"), str(out.get("arguments", ""))
    if answer in LETTERS[:k] and len(answer) == 1:
        choice = LETTERS.index(answer)
    else:                                   # the schema was not used: the letter is read from the text
        choice = read_letter(v.get("result", ""), k)
        arguments = arguments or v.get("result", "")
    if choice is None:
        return None, {**base, "why": "unreadable", "head": v.get("result", "")[:120], **spent}
    prob = [0.0] * k
    prob[choice] = 1.0
    return {"id": c["id"], "prob": prob, "correct_option": c["correct_option"], "hypothesis": c["hypothesis"],
            "length": len(arguments), "head": arguments[:1000], "model": model, "effort": effort, **spent}, None


def run(model: str, name: str, mode: str, effort: str, limit: int | None, workers: int) -> None:
    lock, stop = threading.Lock(), threading.Event()
    file = RESULTS / f"chatbench-devil__{name}__{mode}.jsonl"
    done = {json.loads(l)["id"] for l in open(file, encoding="utf-8")} if file.exists() else set()
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    random.Random(534).shuffle(items)
    rest = [c for c in items if c["id"] not in done][:limit]

    def one(c: dict) -> None:
        if stop.is_set():
            return
        line, missing = ask(c, mode, model, effort)
        if missing and "limit" in (missing.get("head", "") + missing["why"]).lower():
            stop.set()                              # the plan's limit: the items left are not written as failures
            print(f"STOPPED on a limit: {missing.get('head', missing['why'])[:160]}", flush=True)
            return
        append(file if line else MISSING, line or missing, lock)

    t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, rest))
    report(file, f"chatbench-devil {name} {mode}")
    print(f"{len(rest)} asked in {time.time() - t:.0f} s", flush=True)
    if stop.is_set():
        sys.exit(4)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    model, name, mode, effort = sys.argv[1:5]
    if mode not in ("told", "devil"):
        sys.exit(f"unknown mode: {mode}")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    run(model, name, mode, effort, limit, workers)


if __name__ == "__main__":
    main()
