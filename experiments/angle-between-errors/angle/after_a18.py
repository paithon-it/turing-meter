"""Added after A18's outcome (17 November 2026, on two further reviews of the article; descriptive, no criterion).

First computed by hand while weighing a review, then written here so that the article reads the numbers from a file
(PROTOCOL.md, "Added after the outcome of A18").

1. GPT-4o in table 5, the raw frequencies against the model: errors in common per hundred trials and the machine's
   errors where the person errs, in the data (`conversations.hundred_questions`) and in the model at the in-house
   angle, at the mixed model's and at 90 degrees, with the sensitivities of the comparison.
2. The verdict-only ceiling with GPT-4o at the same angles (`verdict.pair`).
3. GPT-4o asked again: ChatBench gives about fifty answers of GPT-4o per question. Two of them, drawn without
   replacement, differ in correctness on a share of the questions; their correlation, converted at one sensitivity
   as A13 does for Haiku (k = 1), gives an angle and the margin of the ideal pair "GPT-4o plus GPT-4o". The
   conversion table of the article stops at an evidence correlation of 0.95 (18.2 degrees); here it goes to 0.99.
   Intervals: 2,000 resamples of the questions, seed 534, with the table built once at the point. Llama 3.1 8B too.

Usage, from the experiment folder:  python -m angle.after_a18   → results/after-a18.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import sdt, verdict

BASE = Path("data")
SEED = 534
REPETITIONS = 2000
SIMULATED = 2_000_000
M = 4


def model_frequencies(dh: float, dm: float, phi: float, n: int = SIMULATED) -> dict:
    a, b = sdt._simulate_correct(n, M, math.cos(math.radians(phi)), dh, dm, SEED)
    both = float(np.mean(~a & ~b))
    return {"phi": phi, "both_wrong": both, "machine_wrong_if_person_wrong": both / float(np.mean(~a))}


def raw_against_model() -> dict:
    g = json.loads(Path("results/glmm.json").read_text())
    c = next(r for r in g["comparisons"] if r["set"] == "table5" and r["machine"] == "gpt-4o")
    h = json.loads(Path("results/conversations.json").read_text())["gpt-4o"]["hundred_questions"]
    assert abs((1 - h["person_wrong"]) - c["acc_h"]) < 1e-9 and abs((1 - h["machine_wrong"]) - c["acc_m"]) < 1e-9
    data = {"both_wrong": h["both_wrong"], "both_if_independent": h["both_if_independent"],
            "machine_wrong_if_person_wrong": h["both_wrong"] / h["person_wrong"]}
    angles = {"in_house": c["phi"], "mixed_model": c["glmm_phi"], "ninety": 90.0}
    model = {k: model_frequencies(c["dh"], c["dm"], phi) for k, phi in angles.items()}
    ceiling = {}
    for k, phi in angles.items():
        r = verdict.pair(c["dh"], c["dm"], verdict.rho_of(phi), M)
        ceiling[k] = {"phi": phi, "over_machine": r["d_pair"] / c["dm"], "errors_removed": r["errors_removed"]}
    fit = c["glmm"]
    return {"n_trials": c["n_trials"], "dh": c["dh"], "dm": c["dm"], "data": data, "model": model,
            "verdict_ceiling": ceiling,
            "mixed_model_fit": {"warnings": fit["warnings"], "singular": fit["singular"], "sigma_m2": fit["sigma_m2"],
                                "sigma_p2": fit["sigma_p2"], "rho_c": c["glmm_rho_c"], "rho_c_in_house": c["rho_correct"]}}


# ---- the machine asked again --------------------------------------------------------------------------------------

def two_answers(model: str) -> pd.DataFrame:
    a = pd.read_csv(BASE / "chatbench/ai_alone_answers.csv")
    a = a[a.model == model].dropna(subset=["free_text_acc", "free_text_count"])
    n = a.free_text_count.to_numpy(float)
    k = np.rint(a.free_text_acc.to_numpy(float) * n)
    keep = n >= 2
    return pd.DataFrame({"n": n[keep], "k": k[keep]})


def _stats(q: pd.DataFrame, w: np.ndarray, grid, table) -> dict:
    n, k = q.n.to_numpy(), q.k.to_numpy()
    p1 = float(np.average(k / n, weights=w))
    p11 = float(np.average(k * (k - 1) / (n * (n - 1)), weights=w))
    differ = float(np.average(2 * k * (n - k) / (n * (n - 1)), weights=w))
    rc = sdt.tetrachoric(p1, p1, p11)
    rho = float(np.interp(rc, table, grid))
    d = sdt.d_from_accuracy(float(np.clip(p1, 1e-4, 1 - 1e-4)), M)
    return {"accuracy": p1, "differ": differ, "rho_correct": rc, "rho": rho, "phi": sdt.angle_degrees(rho),
            "margin": sdt.ideal(d, d, rho) / d - 1.0, "at_table_top": bool(rc >= table[-1])}


def asked_again(model: str) -> dict:
    q = two_answers(model)
    p1 = float((q.k / q.n).mean())
    d = sdt.d_from_accuracy(p1, M)
    grid = np.linspace(-0.3, 0.99, 44)
    table = np.array([sdt.rho_correct_from_rho_evidence(float(x), d, d, M, n=400_000) for x in grid])
    point = _stats(q, np.ones(len(q)), grid, table)
    rng = np.random.default_rng(SEED)
    runs = [_stats(q, rng.poisson(1.0, len(q)).astype(float), grid, table) for _ in range(REPETITIONS)]
    ci = lambda key: [float(np.percentile([r[key] for r in runs], 2.5)), float(np.percentile([r[key] for r in runs], 97.5))]
    return {"questions": int(len(q)), "answers_median": float(q.n.median()),
            "all_same": float(np.mean((q.k == 0) | (q.k == q.n))), **point,
            "differ_ci": ci("differ"), "phi_ci": ci("phi"), "margin_ci": ci("margin"),
            "replicas_at_table_top": int(sum(r["at_table_top"] for r in runs))}


def main() -> None:
    out = {"protocol": "PROTOCOL.md, Added after the outcome of A18 (descriptive, no criterion)", "seed": SEED,
           "gpt4o_table5": raw_against_model(),
           "asked_again": {m: asked_again(m) for m in ("gpt-4o", "llama-3.1-8b")}}
    Path("results/after-a18.json").write_text(json.dumps(out, indent=1))
    r = out["gpt4o_table5"]
    print("data: both wrong %.2f per 100, M|H %.1f%% (independent %.2f)" % (
        100 * r["data"]["both_wrong"], 100 * r["data"]["machine_wrong_if_person_wrong"], 100 * r["data"]["both_if_independent"]))
    for k, v in r["model"].items():
        c = r["verdict_ceiling"][k]
        print("%-12s phi %.1f: both wrong %.2f, M|H %.1f%% | verdict ceiling %.4f, errors removed %.3f" % (
            k, v["phi"], 100 * v["both_wrong"], 100 * v["machine_wrong_if_person_wrong"], c["over_machine"], c["errors_removed"]))
    print("mixed model fit:", r["mixed_model_fit"])
    for m, a in out["asked_again"].items():
        print("%s asked again: %d questions, all same %.3f, differ %.3f (%.3f-%.3f), rho_c %.3f, phi %.1f (%.1f-%.1f), "
              "margin %.3f (%.3f-%.3f), at table top %s (%d replicas)" % (
                  m, a["questions"], a["all_same"], a["differ"], *a["differ_ci"], a["rho_correct"], a["phi"], *a["phi_ci"],
                  a["margin"], *a["margin_ci"], a["at_table_top"], a["replicas_at_table_top"]))


if __name__ == "__main__":
    main()
