"""A6: the person as a verifier in the disagreements with the chat, and the premium over a fixed weight.

Pre-registered in PROTOCOL.md, "After the critical review (v4): the new measures". In the disagreements of
table 16 (the chat gives a letter among the options, different from the person's answer alone), following the
chat is the answer "the chat is right": d_v = z(follow | chat right) − z(follow | chat wrong). The observer
without gating (conversations.linear_observer, a fixed bonus calibrated on the pair's accuracy) gives d_v0.
Premium = d_v − d_v0; interval from a bootstrap of people and questions on d_v (Poisson weights, 2,000
replicas, seed 534); d_v0 is computed once at the point, its sampling error is not carried. Per declared
confidence of the answer alone, only where at least 30 disagreements have the chat wrong.

Usage, from the experiment folder:  python -m angle.verifier   → results/verifier.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import sdt
from .conversations import MODELS, SEED, _disagreements, linear_observer, trials

REPLICAS = 2000
MIN_WRONG = 30


def _rate(s: np.ndarray, w: np.ndarray) -> float:
    n = w.sum()
    r = float(np.sum(w * s) / n)
    return float(np.clip(r, 0.5 / n, 1 - 0.5 / n))       # the usual half-count correction at 0 and 1


def d_verifier(d: pd.DataFrame, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(d)) if w is None else w
    right = d.chat_ok.to_numpy() == 1
    s = d.followed.to_numpy(float)
    hit, fa = _rate(s[right], w[right]), _rate(s[~right], w[~right])
    return {"follow_right": hit, "follow_wrong": fa, "d_v": float(norm.ppf(hit) - norm.ppf(fa))}


def _interval(d: pd.DataFrame, gen) -> list[float]:
    people = pd.factorize(d.worker_id)[0]
    items = pd.factorize(d.question_id)[0]
    right = d.chat_ok.to_numpy() == 1
    reps = []
    for _ in range(REPLICAS):
        w = gen.poisson(1.0, people.max() + 1)[people] * gen.poisson(1.0, items.max() + 1)[items].astype(float)
        if w[right].sum() > 0 and w[~right].sum() > 0:
            reps.append(d_verifier(d, w)["d_v"])
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def one(model: str) -> dict:
    x = trials(model)
    d = _disagreements(x)
    gen = np.random.default_rng(SEED)
    a = x[x.letter]
    obs = linear_observer(sdt.d_from_accuracy(a.acc.mean(), 4), a.chat_ok.mean(), a.acc_ai.mean())
    d_v0 = float(norm.ppf(obs["follow_right"]) - norm.ppf(obs["follow_wrong"]))
    point = d_verifier(d)
    ci = _interval(d, gen)
    out = {"disagreements": int(len(d)), "chat_right": int((d.chat_ok == 1).sum()), "chat_wrong": int((d.chat_ok == 0).sum()),
           **point, "d_v_ci": ci, "fixed_weight": {**obs, "d_v0": d_v0},
           "premium": point["d_v"] - d_v0, "premium_ci": [ci[0] - d_v0, ci[1] - d_v0], "passes": bool(ci[0] - d_v0 > 0),
           "by_confidence": {}}
    for level, g in d.groupby("confidence"):
        wrong = int((g.chat_ok == 0).sum())
        row = {"disagreements": int(len(g)), "chat_wrong": wrong}
        if wrong >= MIN_WRONG and (g.chat_ok == 1).sum() > 0:
            p, c = d_verifier(g), _interval(g, gen)
            row.update(p, d_v_ci=c, premium=p["d_v"] - d_v0, premium_ci=[c[0] - d_v0, c[1] - d_v0], passes=bool(c[0] - d_v0 > 0))
        else:
            row["reported"] = False                        # fewer than 30 disagreements with the chat wrong
        out["by_confidence"][str(level)] = row
    return out


def main() -> None:
    out = {"replicas": REPLICAS, "seed": SEED, "min_chat_wrong": MIN_WRONG, **{m: one(m) for m in MODELS}}
    Path("results/verifier.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for m in MODELS:
        r = out[m]
        print(m, "d_v %.2f (%.2f-%.2f)" % (r["d_v"], *r["d_v_ci"]), "| fixed %.2f" % r["fixed_weight"]["d_v0"],
              "| premium %.2f (%.2f to %.2f)" % (r["premium"], *r["premium_ci"]), "| passes", r["passes"],
              "| by confidence", {k: v.get("premium") for k, v in r["by_confidence"].items()})


if __name__ == "__main__":
    main()
