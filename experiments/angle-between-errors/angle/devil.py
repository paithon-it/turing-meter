"""The devil's advocate: the machine told the person's answer, and the machine asked for the arguments against it.

Usage, from the experiment folder:

  python3 -m angle.devil items      # data/items/chatbench-devil.jsonl, one item per (question, answer given alone)
  python3 -m angle.devil analyse    # results/devil.json

Three conditions (PROTOCOL.md, «The devil's advocate»): 0, the machine alone; A, told «I think the answer is X»;
B, told and asked first for the strongest arguments against X. The unit is the trial (person, question): in A and B
the machine's verdict depends on the person's answer, so it is assigned trial by trial.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from angle import sdt
from angle.devil_prompt import LETTERS, prompt  # noqa: F401  (re-exported)
from angle.measure import Trials, compare

BASE = Path("data")
RESULTS = Path("devil-results")
MODES = ("alone", "told", "devil")
REPETITIONS = 2000
SEED = 534


# ---- the items -------------------------------------------------------------------------------------------------

def solo_answers(base: Path = BASE) -> pd.DataFrame:
    """The 2,903 answers given alone: answer-first condition, phase 2, the questions of the study."""
    u = pd.read_csv(base / "chatbench/full_study/user_answers_filtered.csv")
    a = u[(u.condition == "answer-first") & (u.phase == 2) & u.question_id.str.contains("-redux-")
          & (u.answer_type == "user-alone")].copy()
    a["selected_answer"] = a.selected_answer.astype(str).str.strip().str.upper()
    return a[["worker_id", "question_id", "subject", "selected_answer", "acc"]].reset_index(drop=True)


def items(solo: pd.DataFrame, bank: list[dict]) -> list[dict]:
    """One item per distinct (question, letter chosen alone), in the order of the bank."""
    by_id = {c["id"]: c for c in bank}
    pairs = solo[["question_id", "selected_answer"]].drop_duplicates()
    wanted = {}
    for q, x in pairs.itertuples(index=False):
        if q in by_id and x in LETTERS[:len(by_id[q]["options"])]:
            wanted.setdefault(q, set()).add(x)
    out = []
    for c in bank:
        for x in sorted(wanted.get(c["id"], ())):
            out.append({**c, "id": f"{c['id']}|{x}", "question_id": c["id"], "hypothesis": x})
    return out


# ---- the analysis ----------------------------------------------------------------------------------------------

def verdicts(file: Path) -> dict[str, int]:
    """Item id → option chosen (the most probable)."""
    out = {}
    for line in open(file, encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = int(np.argmax(r["prob"]))
    return out


def trials(solo: pd.DataFrame, correct: dict[str, int], alone: dict[str, int], told: dict[str, int],
           devil: dict[str, int]) -> pd.DataFrame:
    """One row per answer given alone, with the machine's letter in the three conditions.

    `correct` maps a question to its key; `alone` is keyed by question, `told` and `devil` by «question|letter».
    A trial whose item is missing in any condition is dropped.
    """
    d = solo.copy()
    d["key"] = d.question_id.map(correct)
    d["item"] = d.question_id + "|" + d.selected_answer
    d["m0"] = d.question_id.map(alone)
    d["mA"] = d["item"].map(told)
    d["mB"] = d["item"].map(devil)
    d = d.dropna(subset=["key", "m0", "mA", "mB"]).copy()
    for c in ("key", "m0", "mA", "mB"):
        d[c] = d[c].astype(int)
    d["h"] = d.selected_answer.map(LETTERS.index).astype(int)
    d["person_ok"] = (d.h == d.key).astype(float)
    for c in ("m0", "mA", "mB"):
        d[c + "_ok"] = (d[c] == d.key).astype(float)
    return d.reset_index(drop=True)


def _mean(x: np.ndarray, w: np.ndarray) -> float | None:
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else None


def _d(a: float | None, m: int = 4) -> float | None:
    if a is None or a <= 1.0 / m + 1e-9:
        return None
    return sdt.d_from_accuracy(float(np.clip(a, 1e-4, 1 - 1e-4)), m)


def measures(d: pd.DataFrame, w: np.ndarray) -> dict:
    """The measures of the protocol on weighted trials."""
    h = d.person_ok.to_numpy()
    wrong, right = w * (1 - h), w * h
    out = {"acc_person": _mean(h, w)}
    for c, name in (("m0", "alone"), ("mA", "told"), ("mB", "devil")):
        ok = d[c + "_ok"].to_numpy()
        out[f"acc_{name}"] = _mean(ok, w)
        out[f"acc_{name}_person_wrong"] = _mean(ok, wrong)
        out[f"acc_{name}_person_right"] = _mean(ok, right)
        out[f"d_{name}"] = _d(out[f"acc_{name}"])
        both_wrong = wrong * (1 - ok)
        out[f"trap_{name}"] = _mean((d[c].to_numpy() == d.h.to_numpy()).astype(float), both_wrong)
    def diff(a, b):
        return None if out[a] is None or out[b] is None else out[a] - out[b]
    out["h1_told_minus_alone_person_wrong"] = diff("acc_told_person_wrong", "acc_alone_person_wrong")
    out["h2_devil_minus_told_person_wrong"] = diff("acc_devil_person_wrong", "acc_told_person_wrong")
    out["cost_devil_minus_alone_person_right"] = diff("acc_devil_person_right", "acc_alone_person_right")
    out["gain_devil_minus_alone"] = diff("acc_devil", "acc_alone")
    out["h3_yield_devil"] = None if not out["d_devil"] or not out["d_alone"] else out["d_devil"] / out["d_alone"]
    out["yield_told"] = None if not out["d_told"] or not out["d_alone"] else out["d_told"] / out["d_alone"]
    return out


def bootstrap(d: pd.DataFrame, repetitions: int = REPETITIONS, seed: int = SEED) -> dict:
    """Point estimates and 95% percentile intervals, Poisson weights over people and questions together."""
    p, _ = pd.factorize(d.worker_id, sort=True)
    q, _ = pd.factorize(d.question_id, sort=True)
    point = measures(d, np.ones(len(d)))
    gen = np.random.default_rng(seed)
    runs = [measures(d, gen.poisson(1.0, p.max() + 1)[p] * gen.poisson(1.0, q.max() + 1)[q].astype(float))
            for _ in range(repetitions)]
    out = {"n_trials": int(len(d)), "people": int(p.max() + 1), "questions": int(q.max() + 1),
           "person_wrong": int((d.person_ok == 0).sum()), "repetitions": repetitions}
    for key, v in point.items():
        out[key] = v
        sample = np.array([r[key] for r in runs if r.get(key) is not None], float)
        if v is not None and sample.size:
            out[key + "_ci"] = [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]
    out["H1_passes"] = bool(out.get("h1_told_minus_alone_person_wrong_ci", [0, 0])[1] < 0)
    out["H2_passes"] = bool(out.get("h2_devil_minus_told_person_wrong_ci", [0, 0])[0] > 0)
    out["H3_passes"] = bool(out.get("h3_yield_devil_ci", [0, 0])[0] > 1)
    return out


def angles(d: pd.DataFrame, repetitions: int = 1000) -> dict:
    """Descriptive: the angle between the person's errors and the machine's, per condition (machine per trial)."""
    p, _ = pd.factorize(d.worker_id, sort=True)
    q, _ = pd.factorize(d.question_id, sort=True)
    out = {}
    for c, name in (("m0", "alone"), ("mA", "told"), ("mB", "devil")):
        t = Trials(person=p, item=q, correct=d.person_ok.to_numpy(float), machine=d[c + "_ok"].to_numpy(float),
                   machine_per_trial=True)
        r = compare(t, 4, repetitions=repetitions)
        out[name] = {k: r.get(k) for k in ("angle", "angle_ci", "rho", "rho_ci", "excluded") if k in r}
    return out


MACHINES = {   # name → files of the three conditions
    "llama-3.1-8b": (RESULTS / "chatbench__llama-3.1-8b__alone.jsonl", RESULTS / "chatbench-devil__llama-3.1-8b__told.jsonl",
                     RESULTS / "chatbench-devil__llama-3.1-8b__devil.jsonl"),
    "haiku-5.5": (Path("closed-results/chatbench__haiku-5.5__low.jsonl"), RESULTS / "chatbench-devil__haiku-5.5__told.jsonl",
                  RESULTS / "chatbench-devil__haiku-5.5__devil.jsonl"),
    # a check decided after the results (PROTOCOL.md): the condition alone asked again on the same day
    "haiku-5.5, alone asked again": (Path("closed-results/chatbench__haiku-5.5-rerun__low.jsonl"),
                                     RESULTS / "chatbench-devil__haiku-5.5__told.jsonl", RESULTS / "chatbench-devil__haiku-5.5__devil.jsonl"),
}


def analyse(base: Path = BASE) -> dict:
    solo = solo_answers(base)
    bank = [json.loads(l) for l in open(base / "items/chatbench.jsonl", encoding="utf-8")]
    correct = {c["id"]: c["correct_option"] for c in bank}
    out = {}
    for name, files in MACHINES.items():
        if not all(f.exists() for f in files):
            out[name] = {"missing": [str(f) for f in files if not f.exists()]}
            continue
        d = trials(solo, correct, *(verdicts(f) for f in files))
        out[name] = {**bootstrap(d), "angles": angles(d)}
    return out


def main() -> None:
    if sys.argv[1] == "items":
        bank = [json.loads(l) for l in open(BASE / "items/chatbench.jsonl", encoding="utf-8")]
        its = items(solo_answers(BASE), bank)
        with open(BASE / "items/chatbench-devil.jsonl", "w", encoding="utf-8") as f:
            for c in its:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{len(its)} items from {len({c['question_id'] for c in its})} questions")
    elif sys.argv[1] == "analyse":
        r = analyse()
        Path("results/devil.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for name, v in r.items():
            print(name, {k: v.get(k) for k in ("acc_alone", "acc_told", "acc_devil", "H1_passes", "H2_passes", "H3_passes")})


if __name__ == "__main__":
    main()
