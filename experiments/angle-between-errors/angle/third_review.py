"""A13 and A14, after the third review. Pre-registered in PROTOCOL.md, "After the third review (v4): two more
analyses" (cc0fc2d), before computing them.

A13, the machine asked again: Haiku 5.5's two unpermuted runs on the 396 ChatBench items as two opinions; the
correlation of their correctness, its angle, and the margin of the ideal pair "Haiku plus Haiku" next to the margin of
"people plus Haiku" (frontier.json, margins.json).

A14, an interval for where the points go with GPT-4o: the three terms of `conversations.where_points_go` and the
difference between the points lost in the silent chats and those lost in the decision, on 2,000 resamples with Poisson
weights over people and over questions, seed 534.

Usage, from the experiment folder:  python -m angle.third_review   → results/third-review.json
                                    python -m angle.third_review k1   → adds the correction of A13 (k = 1)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import conversations, sdt
from .analysis import load_machine

SEED = 534
REPETITIONS = 2000
CLOSED = Path("closed-results")


_TABLE: dict = {}


def _convert(r_correct: float, da: float, db: float) -> float:
    """sdt.convert_rho_mafc, with its simulated table built once at the point estimate and reused in the resamples
    (the table moves little with d; rebuilding it 2,000 times takes hours)."""
    if "t" not in _TABLE:
        g = np.linspace(-0.3, 0.95, 26)
        _TABLE["g"], _TABLE["t"] = g, np.array([sdt.rho_correct_from_rho_evidence(float(x), da, db, 4, n=120_000) for x in g])
    return float(np.interp(r_correct, _TABLE["t"], _TABLE["g"]))


def _pair(a: np.ndarray, b: np.ndarray) -> dict:
    da, db = sdt.d_from_accuracy(float(np.clip(a.mean(), 1e-4, 1 - 1e-4)), 4), sdt.d_from_accuracy(float(np.clip(b.mean(), 1e-4, 1 - 1e-4)), 4)
    r_correct = sdt.rho_machines(a, b)
    rho = _convert(r_correct, da, db)
    ideal = sdt.ideal(da, db, rho)
    return {"acc_a": float(a.mean()), "acc_b": float(b.mean()), "rho_correct": r_correct, "rho": rho,
            "phi": sdt.angle_degrees(rho), "margin": ideal / max(da, db) - 1.0, "margin_90": math.hypot(da, db) / max(da, db) - 1.0}


def a13() -> dict:
    a, b = load_machine(CLOSED, "chatbench", "haiku-5.5", "low").ok, load_machine(CLOSED, "chatbench", "haiku-5.5-rerun", "low").ok
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    point = _pair(a, b)
    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(REPETITIONS):
        i = rng.integers(0, a.size, a.size)
        try:
            runs.append(_pair(a[i], b[i]))
        except (ValueError, ZeroDivisionError):
            continue
    ci = lambda k: [float(np.nanpercentile([r[k] for r in runs], 2.5)), float(np.nanpercentile([r[k] for r in runs], 97.5))]
    fr = json.loads(Path("results/frontier.json").read_text())
    people = next(c for c in fr["comparisons"] if c.get("source") == "chatbench" and c.get("task") == "all" and c.get("machine") == "haiku-5.5")
    m7 = next(r for r in json.loads(Path("results/margins.json").read_text())["table7"] if r["machine"] == "haiku-5.5")
    return {"items": int(a.size), "changed": int((a != b).sum()), "errors_a": int((a == 0).sum()), "errors_b": int((b == 0).sum()),
            "errors_both": int(((a == 0) & (b == 0)).sum()), **point, "phi_ci": ci("phi"), "margin_ci": ci("margin"),
            "repetitions": len(runs),
            "people_plus_haiku": {"phi": people.get("phi"), "phi_ci": people.get("phi_ci"), "margin": people.get("margin"),
                                  "margin_90": m7["margin_90"], "margin_90_ci": m7["margin_90_ci"]}}


_TABLE_K1: dict = {}


def _pair_k1(a: np.ndarray, b: np.ndarray) -> dict:
    """A13 as the protocol wrote it, k = 1 (correction of the fourth review, PROTOCOL.md d75500f): the two runs are the
    same machine, so one sensitivity, from their mean accuracy; ρ converted at that sensitivity; margin of two equal
    observers, sqrt(2/(1+ρ)) − 1."""
    d = sdt.d_from_accuracy(float(np.clip((a.mean() + b.mean()) / 2, 1e-4, 1 - 1e-4)), 4)
    r_correct = sdt.rho_machines(a, b)
    if "t" not in _TABLE_K1:     # built once at the point, as in A13
        g = np.linspace(-0.3, 0.95, 26)
        _TABLE_K1["g"], _TABLE_K1["t"] = g, np.array([sdt.rho_correct_from_rho_evidence(float(x), d, d, 4, n=120_000) for x in g])
    rho = float(np.interp(r_correct, _TABLE_K1["t"], _TABLE_K1["g"]))
    return {"d": d, "rho_correct": r_correct, "rho": rho, "phi": sdt.angle_degrees(rho), "margin": sdt.ideal(d, d, rho) / d - 1.0}


def a13_k1() -> dict:
    a, b = load_machine(CLOSED, "chatbench", "haiku-5.5", "low").ok, load_machine(CLOSED, "chatbench", "haiku-5.5-rerun", "low").ok
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    point = _pair_k1(a, b)
    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(REPETITIONS):
        i = rng.integers(0, a.size, a.size)
        try:
            runs.append(_pair_k1(a[i], b[i]))
        except (ValueError, ZeroDivisionError):
            continue
    ci = lambda k: [float(np.nanpercentile([r[k] for r in runs], 2.5)), float(np.nanpercentile([r[k] for r in runs], 97.5))]
    return {"items": int(a.size), **point, "phi_ci": ci("phi"), "margin_ci": ci("margin"), "repetitions": len(runs),
            "identical_runs": 2, "note": "two identical runs: the majority of three would need a third"}


def _weighted(x, w) -> dict:
    """where_points_go with weights on the trials."""
    a = x.letter.to_numpy()
    wa, wb, wt = w[a].sum(), w[~a].sum(), w.sum()
    m = lambda col, s: float(np.average(x[col].to_numpy()[s], weights=w[s]))
    return {"conversation_with_answer": wa / wt * (m("chat_ok", a) - m("m_alone", a)),
            "decision_with_answer": wa / wt * (m("acc_ai", a) - m("chat_ok", a)),
            "chats_without_answer": wb / wt * (m("acc_ai", ~a) - m("m_alone", ~a))}


def a14() -> dict:
    x = conversations.trials("gpt-4o")
    point = conversations.where_points_go(x)
    people, questions = x.worker_id.factorize()[0], x.question_id.factorize()[0]
    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(REPETITIONS):
        w = rng.poisson(1.0, people.max() + 1)[people] * rng.poisson(1.0, questions.max() + 1)[questions]
        w = w.astype(float)
        if w[x.letter.to_numpy()].sum() == 0 or w[~x.letter.to_numpy()].sum() == 0:
            continue
        r = _weighted(x, w)
        r["difference"] = (-r["chats_without_answer"]) - (-r["decision_with_answer"])
        runs.append(r)
    ci = lambda k: [float(np.percentile([r[k] for r in runs], 2.5)), float(np.percentile([r[k] for r in runs], 97.5))]
    diff = (-point["chats_without_answer"]) - (-point["decision_with_answer"])
    out = {**point, "lost_silent_minus_lost_decision": diff, "repetitions": len(runs)}
    for k in ("conversation_with_answer", "decision_with_answer", "chats_without_answer"):
        out[k + "_ci"] = ci(k)
    out["lost_silent_minus_lost_decision_ci"] = ci("difference")
    out["passes"] = bool(out["lost_silent_minus_lost_decision_ci"][0] > 0)
    return out


def main() -> None:
    import sys
    if sys.argv[1:] == ["k1"]:      # only the correction, added to the results already there
        out = json.loads(Path("results/third-review.json").read_text())
        out["a13_k1"] = {"protocol": "PROTOCOL.md, After the fourth review (d75500f)", **a13_k1()}
        Path("results/third-review.json").write_text(json.dumps(out, indent=1))
        h = out["a13_k1"]
        print("A13 k=1: d %.3f rho_correct %.3f phi %.1f (%.1f-%.1f) margin %.4f (%.4f-%.4f)"
              % (h["d"], h["rho_correct"], h["phi"], *h["phi_ci"], h["margin"], *h["margin_ci"]))
        return
    out = {"protocol": "PROTOCOL.md, After the third review (cc0fc2d)", "seed": SEED, "a13": a13(), "a14": a14()}
    Path("results/third-review.json").write_text(json.dumps(out, indent=1))
    h, p = out["a13"], out["a14"]
    print("A13 Haiku vs Haiku: changed %d, rho_correct %.3f, phi %.1f (%.1f-%.1f), margin %.3f (%.3f-%.3f); people+Haiku margin %.3f, at 90 %.3f"
          % (h["changed"], h["rho_correct"], h["phi"], *h["phi_ci"], h["margin"], *h["margin_ci"], h["people_plus_haiku"]["margin"], h["people_plus_haiku"]["margin_90"]))
    print("A14 silent %.3f (%.3f %.3f), decision %.3f (%.3f %.3f), difference %.3f (%.3f %.3f) passes %s"
          % (p["chats_without_answer"], *p["chats_without_answer_ci"], p["decision_with_answer"], *p["decision_with_answer_ci"],
             p["lost_silent_minus_lost_decision"], *p["lost_silent_minus_lost_decision_ci"], p["passes"]))


if __name__ == "__main__":
    main()
