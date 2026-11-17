"""The random anchor: the machine told a letter drawn at random, instead of the person's answer.

Usage, from the experiment folder:

  python3 -m angle.anchor items      # data/items/chatbench-anchor.jsonl: the devil's items, each with a random letter
  python3 -m angle.anchor analyse    # results/anchor.json

A control for the told condition of the devil's advocate (PROTOCOL.md, «The random anchor»). Told «I think the
answer is X», with X the person's answer, Claude Haiku 5.5 answered better than alone. Is it the person's answer, or
any candidate written in the prompt? Same items, same prompt, same machine and settings; X replaced by a letter
drawn uniformly among the options, independently of the person's answer and of the key.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from angle import devil
from angle.devil_prompt import LETTERS

BASE = Path("data")
RESULTS = Path("devil-results")
ITEMS = BASE / "items/chatbench-anchor.jsonl"
SEED = 534
REPETITIONS = 2000
ALONE = Path("closed-results/chatbench__haiku-5.5__low.jsonl")               # the run named by the devil's protocol
ALONE_AGAIN = Path("closed-results/chatbench__haiku-5.5-rerun__low.jsonl")   # the same prompt, asked again
TOLD = RESULTS / "chatbench-devil__haiku-5.5__told.jsonl"
ANCHOR = RESULTS / "chatbench-anchor__haiku-5.5__told.jsonl"


# ---- the items -------------------------------------------------------------------------------------------------

def anchor_letter(item_id: str, k: int, seed: int = SEED) -> str:
    """A letter drawn uniformly among the k options, fixed by the item id and the seed (not by Python's hash)."""
    h = hashlib.sha256(f"{seed}|{item_id}".encode()).digest()
    return LETTERS[int.from_bytes(h[:8], "big") % k]


def items(devil_items: list[dict]) -> list[dict]:
    """The devil's items with the hypothesis replaced by the random letter; the id and the person's answer are kept."""
    return [{**c, "person_answer": c["hypothesis"], "hypothesis": anchor_letter(c["id"], len(c["options"]))}
            for c in devil_items]


# ---- the analysis ----------------------------------------------------------------------------------------------

def trials(solo: pd.DataFrame, correct: dict[str, int], alone: dict[str, int], told: dict[str, int],
           anchor: dict[str, int], letters: dict[str, str]) -> pd.DataFrame:
    """One row per answer given alone: the machine alone (m0), told the person's answer (mA), told the random letter (mR).

    `letters` maps an item («question|person's letter») to its random letter. A trial missing in any condition is dropped.
    """
    d = devil.trials(solo, correct, alone, told, anchor).rename(columns={"mB": "mR", "mB_ok": "mR_ok"})
    d["r"] = d["item"].map(letters).map(LETTERS.index).astype(int)
    d["anchor_ok"] = (d.r == d.key).astype(float)
    return d


def _mean(x, w) -> float | None:
    return devil._mean(np.asarray(x, float), np.asarray(w, float))


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if not a or not b else a / b


def measures(d: pd.DataFrame, w: np.ndarray) -> dict:
    """The measures of the protocol on weighted trials."""
    h, r = d.person_ok.to_numpy(), d.anchor_ok.to_numpy()
    out = {"acc_person": _mean(h, w), "acc_random_letter": _mean(r, w)}
    for c, name in (("m0", "alone"), ("mA", "told"), ("mR", "anchor")):
        out[f"acc_{name}"] = _mean(d[c + "_ok"], w)
        out[f"d_{name}"] = devil._d(out[f"acc_{name}"])
    a0, aA, aR = (d[c + "_ok"].to_numpy() for c in ("m0", "mA", "mR"))
    # accuracy when the letter in the prompt is right and when it is wrong; the machine alone on the same trials
    out["acc_told_hint_right"], out["acc_told_hint_wrong"] = _mean(aA, w * h), _mean(aA, w * (1 - h))
    out["acc_anchor_hint_right"], out["acc_anchor_hint_wrong"] = _mean(aR, w * r), _mean(aR, w * (1 - r))
    out["acc_alone_person_right"], out["acc_alone_person_wrong"] = _mean(a0, w * h), _mean(a0, w * (1 - h))
    out["acc_alone_anchor_right"], out["acc_alone_anchor_wrong"] = _mean(a0, w * r), _mean(a0, w * (1 - r))
    # the pull of a wrong letter: how often the machine gives it, minus how often it gives it alone
    m0, mA, mR, hh, rr = (d[c].to_numpy() for c in ("m0", "mA", "mR", "h", "r"))
    out["pull_told"] = None if out["acc_told_hint_wrong"] is None else \
        _mean(mA == hh, w * (1 - h)) - _mean(m0 == hh, w * (1 - h))
    out["pull_anchor"] = None if out["acc_anchor_hint_wrong"] is None else \
        _mean(mR == rr, w * (1 - r)) - _mean(m0 == rr, w * (1 - r))
    # told, as it would be if the machine reacted only to whether the letter is right, as it does to the random one
    if None not in (out["acc_anchor_hint_right"], out["acc_anchor_hint_wrong"], out["acc_person"]):
        p = out["acc_person"]
        out["told_predicted"] = out["acc_anchor_hint_right"] * p + out["acc_anchor_hint_wrong"] * (1 - p)
        out["told_minus_predicted"] = out["acc_told"] - out["told_predicted"]
    out["r1_anchor_over_alone"] = _ratio(out["d_anchor"], out["d_alone"])
    out["r2_told_over_anchor"] = _ratio(out["d_told"], out["d_anchor"])
    out["told_over_alone"] = _ratio(out["d_told"], out["d_alone"])
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
           "repetitions": repetitions}
    for key, v in point.items():
        out[key] = v
        sample = np.array([x[key] for x in runs if x.get(key) is not None], float)
        if v is not None and sample.size:
            out[key + "_ci"] = [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]
    out["R1_passes_this_run"] = bool(out.get("r1_anchor_over_alone_ci", [0, 0])[0] > 1)
    out["R2_passes"] = bool(out.get("r2_told_over_anchor_ci", [0, 0])[0] > 1)
    return out


def analyse(base: Path = BASE) -> dict:
    solo = devil.solo_answers(base)
    bank = [json.loads(l) for l in open(base / "items/chatbench.jsonl", encoding="utf-8")]
    correct = {c["id"]: c["correct_option"] for c in bank}
    letters = {c["id"]: c["hypothesis"] for c in map(json.loads, open(ITEMS, encoding="utf-8"))}
    told, anchor = devil.verdicts(TOLD), devil.verdicts(ANCHOR)
    out = {}
    for name, alone_file in (("alone, the protocol's run", ALONE), ("alone, asked again", ALONE_AGAIN)):
        d = trials(solo, correct, devil.verdicts(alone_file), told, anchor, letters)
        out[name] = bootstrap(d)
    out["R1_passes"] = all(v["R1_passes_this_run"] for v in out.values() if isinstance(v, dict))
    out["R2_passes"] = out["alone, the protocol's run"]["R2_passes"]
    return out


def main() -> None:
    if sys.argv[1] == "items":
        its = items([json.loads(l) for l in open(BASE / "items/chatbench-devil.jsonl", encoding="utf-8")])
        with open(ITEMS, "w", encoding="utf-8") as f:
            for c in its:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        same = sum(c["hypothesis"] == c["person_answer"] for c in its)
        print(f"{len(its)} items; the random letter equals the person's in {same}")
    elif sys.argv[1] == "analyse":
        r = analyse()
        Path("results/anchor.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for name, v in r.items():
            if isinstance(v, dict):
                print(name, {k: v.get(k) for k in ("acc_alone", "acc_told", "acc_anchor", "r1_anchor_over_alone",
                                                   "r1_anchor_over_alone_ci", "r2_told_over_anchor",
                                                   "r2_told_over_anchor_ci")})
        print("R1", r["R1_passes"], "R2", r["R2_passes"])


if __name__ == "__main__":
    main()
