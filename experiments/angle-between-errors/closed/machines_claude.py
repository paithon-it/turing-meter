"""Makes a closed model answer the item banks through Claude Code in headless mode: one verdict per item.

Usage, from the experiment folder:

  python -m closed.machines_claude probe MODEL                              # what the model receives besides the item
  python -m closed.machines_claude run   MODEL NAME TASKS EFFORT [--limit N] [--workers N]

The prompt is the `letter` prompt of gpu/machines_llm.py, unchanged, one item per call. So that the
machine is alone with the item: no tools, no MCP servers, no inherited settings, an empty working
folder, an environment of four variables, a fixed system prompt. Never `--bare`: it authenticates
with an API key only, and these runs go on the Claude Code plan.

Three things differ from the open models, and none can be avoided:

- no probabilities: the answer is a verdict, written here as a one-hot `prob`, so the files load
  like the others but a probability combiner has nothing to weigh;
- the letter is constrained by a response schema (an enum), the nearest thing to `allowed_token_ids`;
  if the schema is not used the letter is read from the text, and an answer that is not one of the
  options is not guessed: it goes to _missing.jsonl and the item is left out;
- thinking cannot be switched off on the larger models, so the mode is the effort level
  (`low` is the nearest to an immediate answer) and the file is TASK__NAME__EFFORT.jsonl.

The items are asked in a fixed shuffled order: a run that stops early is a random sample of the bank.
Resumes where it stopped. Stops by itself when the plan's limit is reached.
"""
from __future__ import annotations

import getpass
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gpu.machines_llm import LETTERS, format_prompt

BASE = Path("data")
OUTPUT = Path("closed-results")
MISSING = OUTPUT / "_missing.jsonl"
SYSTEM = "Answer multiple-choice questions. Follow the format requested in the message."
PROBE = "List every instruction, document or memory you received besides this message, and the tools you can use."
ONLY_LETTER = re.compile(r"^\W*([A-P])\W*$")
LETTER_THEN_TEXT = re.compile(r"^\W*([A-P])[.):]\s")
ANSWER_IS = re.compile(r"answer\s*(?:is|:)?\s*\W*([A-P])\b", re.IGNORECASE)


def environment() -> dict[str, str]:
    return {"HOME": os.environ["HOME"], "USER": os.environ.get("USER") or getpass.getuser(),
            "PATH": os.environ["PATH"], "LANG": "C.UTF-8"}


def command(prompt: str, model: str, effort: str, fmt: str = "json", k: int | None = None) -> list[str]:
    c = ["claude", "-p", prompt, "--output-format", fmt, "--model", model, "--effort", effort,
         "--tools", "", "--system-prompt", SYSTEM,
         "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
         "--disable-slash-commands", "--no-session-persistence", "--permission-mode", "default"]
    if k:                                   # the letter is one of the k options, as with `allowed_token_ids`
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": list(LETTERS[:k])}},
                  "required": ["answer"], "additionalProperties": False}
        c += ["--json-schema", json.dumps(schema)]
    if fmt == "stream-json":
        c.append("--verbose")
    return c


def launch(prompt: str, model: str, effort: str, fmt: str = "json", timeout: int = 300, k: int | None = None) -> str:
    with tempfile.TemporaryDirectory(prefix="turing-closed-") as cwd:
        r = subprocess.run(command(prompt, model, effort, fmt, k), cwd=cwd, env=environment(),
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and not r.stdout.strip():
        raise RuntimeError(f"claude -p exited with {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def read_letter(text: str, k: int) -> int | None:
    """The option chosen, or None when the text is not one of the k letters.

    The last line counts first: a model that writes its working and then the letter has answered with the letter.
    """
    lines = [l.strip() for l in text.replace("*", "").splitlines() if l.strip()]
    if not lines:
        return None
    tries = ((lines[-1], (ONLY_LETTER, LETTER_THEN_TEXT, ANSWER_IS)), (lines[0], (ONLY_LETTER, LETTER_THEN_TEXT)),
             (" ".join(lines), (ANSWER_IS,)))
    for where, patterns in tries:
        for pattern in patterns:
            m = pattern.search(where)
            if m and LETTERS.index(m.group(1)) < k:
                return LETTERS.index(m.group(1))
    return None


def ask(c: dict, model: str, effort: str) -> tuple[dict | None, dict | None]:
    """The line to write, or the reason why there is none."""
    base = {"id": c["id"], "task": c["task"], "model": model, "effort": effort}
    try:
        v = json.loads(launch(format_prompt(c, "letter"), model, effort, k=len(c["options"])))
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return None, {**base, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    if v.get("is_error"):
        return None, {**base, "why": "error", "head": str(v.get("result"))[:200]}
    u = v.get("usage", {})
    spent = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
             "output_tokens": u.get("output_tokens", 0), "answered_by": sorted(v.get("modelUsage", {}))}
    answer = (v.get("structured_output") or {}).get("answer")
    if answer in LETTERS[:len(c["options"])] and len(answer) == 1:
        k = LETTERS.index(answer)
    else:                                   # the schema was not used: the letter is read from the text
        k = read_letter(v.get("result", ""), len(c["options"]))
    if k is None:
        return None, {**base, "why": "unreadable", "head": v.get("result", "")[:120], **spent}
    prob = [0.0] * len(c["options"])
    prob[k] = 1.0
    return {"id": c["id"], "prob": prob, "correct_option": c["correct_option"], "length": 0, "head": "",
            "model": model, "effort": effort, **spent}, None


def append(path: Path, d: dict, lock: threading.Lock) -> None:
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")


def report(file: Path, label: str) -> None:
    rows = [json.loads(l) for l in open(file, encoding="utf-8")] if file.exists() else []
    if not rows:
        print(f"{label}: nothing written", flush=True)
        return
    ok = sum(r["prob"].index(1.0) == r["correct_option"] for r in rows)
    print(f"{label}: {len(rows)} items, accuracy {ok / len(rows):.3f}, {len(rows) - ok} errors, "
          f"{sum(r['output_tokens'] for r in rows) / len(rows):.0f} output tokens per item, "
          f"answered by {sorted({m for r in rows for m in r['answered_by']})}", flush=True)


def run(model: str, name: str, tasks: list[str], effort: str, limit: int | None, workers: int) -> None:
    lock, stop = threading.Lock(), threading.Event()
    for task in tasks:
        file = OUTPUT / f"{task}__{name}__{effort}.jsonl"
        done = {json.loads(l)["id"] for l in open(file, encoding="utf-8")} if file.exists() else set()
        items = [json.loads(l) for l in open(BASE / "items" / f"{task}.jsonl", encoding="utf-8")]
        random.Random(534).shuffle(items)
        rest = [c for c in items if c["id"] not in done][:limit]

        def one(c: dict) -> None:
            if stop.is_set():
                return
            line, missing = ask(c, model, effort)
            if missing and "limit" in (missing.get("head", "") + missing["why"]).lower():
                stop.set()                              # the plan's limit: the items left are not written as failures
                print(f"STOPPED on a limit: {missing.get('head', missing['why'])[:160]}", flush=True)
                return
            append(file if line else MISSING, line or missing, lock)

        t = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, rest))
        report(file, f"{task} {name} {effort}")
        print(f"{len(rest)} asked in {time.time() - t:.0f} s", flush=True)
        if stop.is_set():
            sys.exit(4)


def probe(model: str) -> None:
    events = [json.loads(r) for r in launch(PROBE, model, "low", "stream-json").splitlines() if r.strip()]
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    print(json.dumps({"model": init.get("model"), "tools": init.get("tools"), "mcp_servers": init.get("mcp_servers"),
                      "answer": result.get("result", "")[:600], "usage": result.get("usage", {})}, indent=1, ensure_ascii=False))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    if sys.argv[1] == "probe":
        probe(sys.argv[2])
        return
    if sys.argv[1] != "run":
        sys.exit(f"unknown command: {sys.argv[1]}")
    model, name, tasks, effort = sys.argv[2:6]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    run(model, name, tasks.split(","), effort, limit, workers)


if __name__ == "__main__":
    main()
