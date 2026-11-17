"""A16, the third run of Haiku. Pre-registered in PROTOCOL.md, "After the fifth review (v4): A16, the third run of
Haiku" (ed771d6, on the server at 20:05:09Z), before the run.

Haiku 5.5 answered the 396 ChatBench items three times, unpermuted, with the same code and settings. The majority of
three is the letter given by at least two runs; when the three letters all differ the item counts by expectation (a tie
broken at random: 1/3 if one of the three is right). Measured margin of the majority, in sensitivity:
d(majority) / d(single) − 1, with d from the accuracy among four options and "single" the mean accuracy of the three runs.
Interval: 2,000 resamples of the items, seed 534, the three runs together.

Usage, from the experiment folder:  python -m angle.third_run   → results/third-run.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import sdt

SEED = 534
REPETITIONS = 2000
CLOSED = Path("closed-results")
RUNS = ("haiku-5.5", "haiku-5.5-rerun", "haiku-5.5-rerun2")
M = 4


def letters(name: str) -> dict[str, tuple[int, int]]:
    """id → (the letter given, the right one)."""
    out = {}
    for line in open(CLOSED / f"chatbench__{name}__low.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = (int(np.argmax(r["prob"])), int(r["correct_option"]))
    return out


def majority(given: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per item, the score of the majority of three: 1 or 0 when two runs agree, by expectation when all three differ."""
    a, b, c = given.T
    agree = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, -1)))
    return np.where(agree >= 0, (agree == right).astype(float), (given == right[:, None]).mean(axis=1))


def _d(acc: float) -> float:
    return sdt.d_from_accuracy(float(np.clip(acc, 1e-4, 1 - 1e-4)), M)


def margin(single: np.ndarray, maj: np.ndarray) -> float:
    return _d(maj.mean()) / _d(single.mean()) - 1.0


def analyse() -> dict:
    runs = [letters(n) for n in RUNS]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    given = np.array([[r[i][0] for r in runs] for i in ids])
    right = np.array([runs[0][i][1] for i in ids])
    ok = (given == right[:, None]).astype(float)
    single, maj = ok.mean(axis=1), majority(given, right)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(REPETITIONS):
        i = rng.integers(0, len(ids), len(ids))
        boots.append(margin(single[i], maj[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    distinct = np.array([len(set(g)) for g in given.tolist()])
    a13 = json.loads(Path("results/third-review.json").read_text())["a13_k1"]
    out = {"protocol": "PROTOCOL.md, After the fifth review (ed771d6)", "seed": SEED, "repetitions": REPETITIONS,
           "items": len(ids), "missing": {n: 396 - len(r) for n, r in zip(RUNS, runs)},
           "accuracy_runs": [float(x) for x in ok.mean(axis=0)], "accuracy_single": float(single.mean()),
           "accuracy_majority": float(maj.mean()),
           "d_single": _d(single.mean()), "d_majority": _d(maj.mean()),
           "margin": margin(single, maj), "margin_ci": [float(lo), float(hi)], "passes": bool(lo > 0),
           "split_3_0": int((distinct == 1).sum()), "split_2_1": int((distinct == 2).sum()), "split_1_1_1": int((distinct == 3).sum()),
           "outcome_changes": int((ok.min(axis=1) != ok.max(axis=1)).sum()),
           "majority_right_of_changes": float(maj[ok.min(axis=1) != ok.max(axis=1)].sum()),
           "ideal_pair": a13["margin"], "ideal_pair_ci": a13["margin_ci"],
           "ideal_three": math.sqrt(3 / (1 + 2 * a13["rho"])) - 1.0, "rho_a13": a13["rho"]}
    return out


def main() -> None:
    out = analyse()
    Path("results/third-run.json").write_text(json.dumps(out, indent=1))
    print("A16: items %d, runs %s, single %.4f, majority %.4f, margin %.4f (%.4f %.4f) passes %s; splits %d/%d/%d; ideal pair %.4f, three %.4f"
          % (out["items"], ["%.4f" % a for a in out["accuracy_runs"]], out["accuracy_single"], out["accuracy_majority"],
             out["margin"], *out["margin_ci"], out["passes"], out["split_3_0"], out["split_2_1"], out["split_1_1_1"],
             out["ideal_pair"], out["ideal_three"]))


if __name__ == "__main__":
    main()
