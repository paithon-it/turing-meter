"""A19, what the person wrote first in the silent chats. Pre-registered in PROTOCOL.md, "After the fifth review (v4):
A19", before computing it.

On the GPT-4o trials of ChatBench: does the person's first message lack the options, or is it short, more often in
the chats that gave none of the four letters than in those that gave one? And how do the 7.6 points lost in the
former split by what the person wrote first?

Usage, from the experiment folder:
  python -m angle.first_message          → results/first-message.json
  python -m angle.first_message sample   → results/first-message-sample.jsonl, the 40 trials of the matching check
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import conversations, sources
from .mute_chats import normalize

BASE = Path("data")
RESULTS = Path("results")
SEED = 534
REPETITIONS = 2000
CHECK = 20
CATEGORIES = ("no message", "not classifiable", "missing", "present")
PARTS = ("outside", "silent_no_message", "silent_missing", "silent_present", "silent_not_classifiable")


# ---- one trial -------------------------------------------------------------------------------------------------

def first_person_message(history) -> str | None:
    """The first message with role User whose text is not empty."""
    for m in history or []:
        if m.get("role") == "User" and str(m.get("content", "") or "").strip():
            return str(m["content"])
    return None


def _token(needle: str, text: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", text) is not None


def usable_options(question: str, options: list[str]) -> list[str]:
    """Normalized options at least two characters long that the question itself does not contain."""
    q = normalize(question)
    out = [normalize(o) for o in options]
    return [o for o in out if len(o) >= 2 and not _token(o, q)]


def classify(message: str | None, question: str, options: list[str]) -> tuple[str, bool | None]:
    """The category of the first message, and whether it is short (None without a message)."""
    if message is None:
        return "no message", None
    m = normalize(message)
    short = len(m) < 0.5 * len(normalize(question))
    usable = usable_options(question, options)
    if len(usable) < 2:
        return "not classifiable", short
    return ("present" if any(_token(o, m) for o in usable) else "missing"), short


# ---- the trials ------------------------------------------------------------------------------------------------

def table(model: str) -> pd.DataFrame:
    x = conversations.trials(model)
    conv = sources.chatbench_conversations(BASE)
    conv = conv[conv.model == model].drop_duplicates(["worker_id", "question_id"])
    x = x.merge(conv[["worker_id", "question_id", "chat_history"]], on=["worker_id", "question_id"], how="left")
    q = pd.read_csv(BASE / "chatbench/questions.csv").set_index("question_id")
    x["question"] = x.question_id.map(q.question)
    x["options"] = [[q.loc[i, f"option_{L}"] for L in "ABCD"] for i in x.question_id]
    x["first_message"] = x.chat_history.map(first_person_message)
    c = [classify(m, qq, o) for m, qq, o in zip(x.first_message, x.question, x.options)]
    x["category"] = [a for a, _ in c]
    x["short"] = [b for _, b in c]
    silent = (x.conv_has_answer == False).to_numpy()  # noqa: E712
    x["group"] = np.where(x.letter, "letter", np.where(silent, "silent", "outside"))
    x["id"] = x.worker_id + "|" + x.question_id
    return x


# ---- the statistics, with weights on the trials ----------------------------------------------------------------

def _share(num: np.ndarray, den: np.ndarray, w: np.ndarray) -> float:
    d = w[den].sum()
    return float(w[num & den].sum() / d) if d > 0 else float("nan")


def _mean(v: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(v[s], weights=w[s])) if w[s].sum() > 0 else float("nan")


def stats(x: pd.DataFrame, w: np.ndarray) -> dict:
    g, cat = x.group.to_numpy(), x.category.to_numpy()
    msg = cat != "no message"
    cls = np.isin(cat, ("missing", "present"))
    missing = cat == "missing"
    short = np.array([bool(s) for s in x.short], dtype=bool) & msg
    out: dict = {}
    for name in ("silent", "letter", "outside"):
        s = g == name
        out[name] = {"missing": _share(missing, s & cls, w), "short": _share(short, s & msg, w),
                     "no_message": _share(~msg, s, w), "not_classifiable": _share(msg & ~cls, s, w)}
    out["diff_missing"] = out["silent"]["missing"] - out["letter"]["missing"]
    out["diff_short"] = out["silent"]["short"] - out["letter"]["short"]
    silent = g == "silent"
    out["silent_given_missing"] = _share(silent, cls & missing, w)
    out["silent_given_present"] = _share(silent, cls & ~missing, w)
    acc, alone = x.acc_ai.to_numpy(float), x.m_alone.to_numpy(float)
    masks = {"outside": g == "outside", "silent_no_message": silent & ~msg, "silent_missing": silent & missing,
             "silent_present": silent & cls & ~missing, "silent_not_classifiable": silent & msg & ~cls}
    parts = {}
    for k, s in masks.items():
        share = w[s].sum() / w.sum()
        parts[k] = {"points": float(share * (_mean(acc, s, w) - _mean(alone, s, w))) if share > 0 else 0.0,
                    "share": float(share), "people_with_chat": _mean(acc, s, w), "machine_alone": _mean(alone, s, w)}
    out["parts"] = parts
    out["chats_without_answer"] = float(sum(p["points"] for p in parts.values()))
    return out


def counts(x: pd.DataFrame) -> dict:
    return {gname: {c: int(((x.group == gname) & (x.category == c)).sum()) for c in CATEGORIES}
            | {"trials": int((x.group == gname).sum()), "short": int(((x.group == gname) & (x.short == True)).sum())}  # noqa: E712
            for gname in ("silent", "letter", "outside")}


def _ci(v) -> list[float]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def intervals(x: pd.DataFrame) -> dict:
    people, questions = x.worker_id.factorize()[0], x.question_id.factorize()[0]
    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(REPETITIONS):
        w = (rng.poisson(1.0, people.max() + 1)[people] * rng.poisson(1.0, questions.max() + 1)[questions]).astype(float)
        runs.append(stats(x, w))
    out = {k: _ci([r[k] for r in runs]) for k in ("diff_missing", "diff_short", "silent_given_missing",
                                                   "silent_given_present", "chats_without_answer")}
    for gname in ("silent", "letter", "outside"):
        for k in ("missing", "short"):
            out[f"{gname}_{k}"] = _ci([r[gname][k] for r in runs])
    for p in PARTS:
        out[f"part_{p}"] = _ci([r["parts"][p]["points"] for r in runs])
    return out


# ---- the matching check -----------------------------------------------------------------------------------------

def check_sample(x: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for c in ("missing", "present"):
        pool = x[x.category == c].sort_values("id").reset_index(drop=True)
        out.append(pool.iloc[rng.choice(len(pool), size=min(CHECK, len(pool)), replace=False)])
    return pd.concat(out, ignore_index=True)


def write_sample(x: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _, r in check_sample(x).iterrows():
            f.write(json.dumps({"id": r.id, "category": r.category, "question": r.question,
                                "options": dict(zip("ABCD", r.options)), "first_message": r.first_message},
                               ensure_ascii=False) + "\n")


# ---- run ---------------------------------------------------------------------------------------------------------

def run() -> dict:
    x = table("gpt-4o")
    n = counts(x)
    assert (n["letter"]["trials"], n["silent"]["trials"], n["outside"]["trials"]) == (1121, 330, 42), n
    point = stats(x, np.ones(len(x)))
    ref = conversations.where_points_go(x)["chats_without_answer"]
    assert abs(point["chats_without_answer"] - ref) < 1e-9, (point["chats_without_answer"], ref)
    ci = intervals(x)
    y = table("llama-3.1-8b")
    out = {"protocol": "PROTOCOL.md, After the fifth review (v4): A19", "seed": SEED, "replicas": REPETITIONS,
           "gpt4o": {"counts": n, **point, "ci": ci},
           "llama": {"counts": counts(y), **stats(y, np.ones(len(y)))}}
    out["a19a_passes"] = bool(ci["diff_missing"][0] > 0)
    out["a19b_passes"] = bool(ci["diff_short"][0] > 0)
    check = RESULTS / "first-message-check.json"
    if check.exists():
        marks = json.loads(check.read_text())
        out["matching_check"] = {"read": len(marks), "right": int(sum(v == "right" for v in marks.values())),
                                 "wrong": [k for k, v in marks.items() if v != "right"]}
    return out


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    if sys.argv[1:] == ["sample"]:
        write_sample(table("gpt-4o"), RESULTS / "first-message-sample.jsonl")
    elif sys.argv[1:] == []:
        o = run()
        (RESULTS / "first-message.json").write_text(json.dumps(o, indent=1, ensure_ascii=False) + "\n")
        g = o["gpt4o"]
        for k in ("silent", "letter", "outside"):
            print("%-7s missing %.3f short %.3f no message %.3f" % (k, g[k]["missing"], g[k]["short"], g[k]["no_message"]))
        print("A19a diff %.3f (%.3f %.3f) passes %s; A19b diff %.3f (%.3f %.3f) passes %s"
              % (g["diff_missing"], *g["ci"]["diff_missing"], o["a19a_passes"], g["diff_short"], *g["ci"]["diff_short"],
                 o["a19b_passes"]))
        print("silent given missing %.3f, given present %.3f" % (g["silent_given_missing"], g["silent_given_present"]))
        for p in PARTS:
            q = g["parts"][p]
            print("%-24s %+.4f (%+.4f %+.4f) share %.3f people %.3f machine %.3f"
                  % (p, q["points"], *g["ci"][f"part_{p}"], q["share"], q["people_with_chat"], q["machine_alone"]))
    else:
        raise SystemExit(__doc__)
