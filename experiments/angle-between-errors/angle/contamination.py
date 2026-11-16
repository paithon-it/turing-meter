"""A11: contamination. Pre-registered in PROTOCOL.md, "After the critical review (v4): the new measures" (8ed4576).

Test 1, the options permuted. ChatBench: the options rotate so that the correct one lands on another letter, chosen
with SHA-256 of the seed 534 and the item id, as in the random anchor; the four Claude models answer with the settings
of the frontier run (closed/machines_claude.py, effort low). αNLI: the two hypotheses swap places; the open models
answer letter only on the L4 (gpu/machines_llm.py, gpu/run_contamination.sh).

The answers are mapped back to the original order of the options, so that the same code that measured the original
items (frontier._chatbench, frontier._alphanli) measures the permuted ones: the people did not change, only the
position of the options shown to the machine.

Measure: the machine's accuracy, original minus permuted, paired by item, with a bootstrap over items (2,000
resamples, seed 534). Criterion: the interval of the difference excludes zero. Then k, the angle and the margin on the
permuted items, next to the original ones.

Usage, from the experiment folder:
  python -m angle.contamination prepare           → data/items/chatbench-permuted.jsonl, data/items/alphanli-swapped.jsonl
  python -m angle.contamination analyse           → contamination-results/ (answers in the original order) and results/contamination.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEED = 534
REPETITIONS = 2000
BASE = Path("data")
CLOSED = Path("closed-results")
GPU = Path("gpu-results-contamination")
MAPPED = Path("contamination-results")
CLAUDE = ("haiku-5.5", "sonnet-5.5", "opus-5.5", "fable-5.2")


def target(item_id: str, k: int, correct: int, seed: int = SEED) -> int:
    """The new position of the correct option: drawn among the k − 1 other positions, fixed by the id and the seed."""
    others = [i for i in range(k) if i != correct]
    h = hashlib.sha256(f"{seed}|{item_id}".encode()).digest()
    return others[int.from_bytes(h[:8], "big") % len(others)]


def rotate(item: dict) -> dict:
    """The options rotated so that the correct one lands on its target position; `original` gives, for each new
    position, the index of the option in the original order."""
    k, c = len(item["options"]), item["correct_option"]
    shift = (target(item["id"], k, c) - c) % k
    original = [(j - shift) % k for j in range(k)]
    return {**item, "task": item["task"] + "-permuted", "options": [item["options"][i] for i in original],
            "correct_option": (c + shift) % k, "original": original}


def swap(item: dict) -> dict:
    """αNLI: the two hypotheses swap places."""
    assert len(item["options"]) == 2
    return {**item, "task": item["task"] + "-swapped", "options": item["options"][::-1],
            "correct_option": 1 - item["correct_option"], "original": [1, 0]}


def back(row: dict, item: dict) -> dict:
    """An answer to a permuted item, written in the original order of the options."""
    prob = [0.0] * len(row["prob"])
    for new, old in enumerate(item["original"]):
        prob[old] = row["prob"][new]
    return {**row, "prob": prob, "correct_option": item["original"][row["correct_option"]]}


def _items(name: str) -> list[dict]:
    return [json.loads(l) for l in open(BASE / "items" / f"{name}.jsonl", encoding="utf-8")]


def prepare() -> None:
    for source, f, name in (("chatbench", rotate, "chatbench-permuted"), ("alphanli", swap, "alphanli-swapped")):
        out = [f(c) for c in _items(source)]
        (BASE / "items" / f"{name}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
        moved = sum(c["correct_option"] != o["correct_option"] for c, o in zip(out, _items(source)))
        print(name, len(out), "items,", moved, "with the correct option on another letter")


def _map_back(folder: Path, task: str, permuted: str, name: str, mode: str) -> Path | None:
    src = folder / f"{permuted}__{name}__{mode}.jsonl"
    if not src.exists():
        return None
    items = {c["id"]: c for c in _items(permuted)}
    rows = [back(json.loads(l), items[json.loads(l)["id"]]) for l in open(src, encoding="utf-8")]
    MAPPED.mkdir(exist_ok=True)
    out = MAPPED / f"{task}__{name}__{mode}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return out


def paired(original: Path, permuted: Path, seed: int = SEED, repetitions: int = REPETITIONS) -> dict:
    """Accuracy original minus permuted, on the items both have, with a bootstrap over items."""
    ok = lambda p: {r["id"]: float(int(np.argmax(r["prob"])) == r["correct_option"]) for r in map(json.loads, open(p, encoding="utf-8"))}
    a, b = ok(original), ok(permuted)
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(repetitions)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"items": len(ids), "acc_original": float(np.mean([a[i] for i in ids])), "acc_permuted": float(np.mean([b[i] for i in ids])),
            "difference": float(d.mean()), "difference_ci": [float(lo), float(hi)], "changed": int((d != 0).sum()),
            "passes": bool(lo > 0 or hi < 0)}


def analyse() -> None:
    from . import frontier
    out = {"protocol": "PROTOCOL.md, A11 (8ed4576)", "seed": SEED, "repetitions": REPETITIONS, "chatbench": [], "alphanli": []}
    for name in CLAUDE:
        mapped = _map_back(CLOSED, "chatbench", "chatbench-permuted", name, "low")
        if mapped is None:
            continue
        r = {"machine": name, "mode": "low", **paired(CLOSED / f"chatbench__{name}__low.jsonl", mapped)}
        r["on_permuted"] = frontier._chatbench((MAPPED, name, "low", None))
        out["chatbench"].append(r)
    if GPU.exists():
        for src in sorted(GPU.glob("alphanli-swapped__*__letter.jsonl")):
            name = src.name.split("__")[1]
            mapped = _map_back(GPU, "alphanli", "alphanli-swapped", name, "letter")
            r = {"machine": name, "mode": "letter", **paired(Path("gpu-results") / f"alphanli__{name}__letter.jsonl", mapped)}
            r["on_permuted"] = frontier._alphanli(("original", MAPPED, name, "letter", False))
            out["alphanli"].append(r)
    Path("results/contamination.json").write_text(json.dumps(out, indent=1))
    for task in ("chatbench", "alphanli"):
        for r in out[task]:
            print(task, r["machine"], "orig %.3f perm %.3f diff %+.3f (%+.3f %+.3f) passes %s" % (
                r["acc_original"], r["acc_permuted"], r["difference"], *r["difference_ci"], r["passes"]))


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse}[sys.argv[1]]()
