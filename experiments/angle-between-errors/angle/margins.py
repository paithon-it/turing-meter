"""A1, A2 and A4: intervals for the margin and the two ceilings, the margin in correct answers, the weight of evidence.
Pre-registered in PROTOCOL.md, "After the critical review (v4): the new measures". Runs after A8 (corrected estimator).

- Table 5 (the two ChatBench pairs, machine queried alone, all trials): the comparison is recomputed as in
  run_chatbench (2,000 resamples, seed 534) keeping its replicas. On each replica: the ceiling with shared cues
  |S|(φ)/d_M and its margin over the stronger; the verdict-only ceiling (the machine's verdict to a person with
  graded cues, verdict.pair) interpolated on a grid of k and φ at the point d_H; W = d′²/2 for H, M and S, ΔW_SH
  and ΔW_SM, in nats and decibans; the margin in correct answers.
- Table 7 (ChatBench, today's machines): the margin at 90 degrees depends on k only and decreases with it, so its
  interval is the transform of the interval of k (map.json for GPT-4o and Llama, frontier.json for Claude).
- Table 8 (αNLI): the comparisons of frontier.json recomputed with compare_votes keeping the replicas, with the key
  of the story and without the contested stories; margin at the measured angle and in correct answers.

Usage, from the experiment folder:  python -m angle.margins   → results/margins.json
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from . import frontier, measure, review, sdt, sources, verdict
from .analysis import BASE, REPETITIONS, load_machine

NAT_TO_DECIBAN = 10.0 / math.log(10.0)
GRIDS = (5, 7, 9)            # points per side of the verdict grid, refined in this order
TOLERANCE = 0.001            # 0.1 points of margin: the interpolation against the direct computation at the point


@contextmanager
def replicas():
    """Wraps measure._summarize, which both compare and compare_votes call with (point, runs, base)."""
    store, original = [], measure._summarize

    def wrap(point, runs, base):
        store.append((point, list(runs)))
        return original(point, runs, base)

    measure._summarize = wrap
    try:
        yield store
    finally:
        measure._summarize = original


def _ci(x) -> list[float]:
    x = np.asarray([v for v in x if v is not None and v == v], float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def accuracy(d: float, m: int) -> float:
    """m options, forced choice; m = 2 here is the yes/no of αNLI with d′ = 2z(a), unbiased: Φ(d/2)."""
    return float(norm.cdf(d / 2.0)) if m == 2 else sdt.pc_mafc(d, m)


def in_answers(stronger: float, margin: float, m: int) -> float:
    return accuracy(stronger * (1.0 + margin), m) - accuracy(stronger, m)


def margin_90(k: float) -> float:
    """Margin of the ideal pair at 90 degrees over the stronger, as a function of k = d_M/d_H alone."""
    return math.sqrt(1.0 + k * k) / max(k, 1.0) - 1.0


def _shared(r: dict) -> dict:
    stronger = max(r["dh"], r["dm"])
    out = {"over_machine": r["ideal_phi"] / r["dm"], "margin": r["ideal_phi"] / stronger - 1.0,
           "margin_90": r["ideal_90"] / stronger - 1.0, "stronger": stronger}
    for who in ("h", "m", "s"):
        d = r.get("d" + who)
        if d is not None:
            out["W_" + who] = d * d / 2.0
    if "W_s" in out:
        out["dW_sh"], out["dW_sm"] = out["W_s"] - out["W_h"], out["W_s"] - out["W_m"]
    return out


def table5() -> list[dict]:
    c = sources.chatbench_pairs(BASE)
    rows = []
    for model in verdict.MODELS:
        m_alone = sources.chatbench_machine_alone(BASE, model)
        with replicas() as store:
            r = measure.compare(sources.chatbench_trials(c[c.model == model], m_alone, None), m=4, repetitions=2000)
        point, runs = store[-1]
        sh = [_shared(x) for x in runs]
        row = {"machine": model, "point": _shared(point), "phi": point["phi"], "phi_ci": r["phi_ci"], "k": point["k"], "k_ci": r["k_ci"]}
        for key in sh[0]:
            row[key + "_ci"] = _ci([x.get(key) for x in sh])
        row["margin_in_answers"] = in_answers(row["point"]["stronger"], row["point"]["margin"], 4)
        row["margin_in_answers_ci"] = _ci([in_answers(x["stronger"], x["margin"], 4) for x in sh])
        # the verdict-only ceiling: a grid of k and φ at the point d_H, interpolated on each replica
        dh = point["dh"]
        ks = np.array([x["k"] for x in runs])
        ph = np.array([x["phi"] for x in runs])
        direct = verdict.pair(dh, point["dm"], verdict.rho_of(point["phi"]), 4)["d_pair"] / point["dm"]
        for size in GRIDS:                        # the protocol: refine the grid until the point is within tolerance
            kg = np.linspace(*np.percentile(ks, [0.5, 99.5]), size)
            pg = np.linspace(*np.percentile(ph, [0.5, 99.5]), size)
            table = np.array([[verdict.pair(dh, k * dh, verdict.rho_of(p), 4)["d_pair"] / (k * dh) for p in pg] for k in kg])
            f = RegularGridInterpolator((kg, pg), table, method="cubic", bounds_error=False, fill_value=None)
            at_point = float(f([[point["k"], point["phi"]]])[0])
            if abs(at_point - direct) < TOLERANCE:
                break
        row["verdict_over_machine"] = direct
        row["verdict_grid"] = size
        row["verdict_interpolated_at_point"] = at_point
        row["verdict_grid_ok"] = bool(abs(at_point - direct) < TOLERANCE)
        row["verdict_over_machine_ci"] = _ci(f(np.column_stack([ks, ph])))
        rows.append(row)
    return rows


def table7() -> list[dict]:
    rows = []
    # GPT-4o and Llama of ChatBench, queried alone, against all the people who answered alone (both arms): in
    # version 3 this k (2.1 and 1.2) was computed outside the code; here it comes from the data, with its interval
    c = sources.chatbench_pairs(BASE)
    for model in verdict.MODELS:
        p = sources.chatbench_trials(c, sources.chatbench_machine_alone(BASE, model), None)
        p.with_machine = None
        r = measure.compare(p, m=4, repetitions=REPETITIONS)
        rows.append({"machine": model, "mode": "chatbench", "k": r["k"], "k_ci": r["k_ci"], "dh": r["dh"], "acc_h": r["acc_h"], "acc_m": r["acc_m"]})
    fr = json.loads(Path("results/frontier.json").read_text())
    for c in fr["comparisons"]:
        if c.get("source") == "chatbench" and c.get("task") == "all" and "k_ci" in c:
            rows.append({"machine": c["machine"], "mode": c.get("mode"), "k": c["k"], "k_ci": c["k_ci"], "dh": c["dh"], "acc_m": c.get("acc_m")})
    n_q = 396
    for r in rows:
        if r.get("acc_m") is not None and r["acc_m"] >= 1.0:
            # no errors: k is not estimable; the rule of three gives the machine's accuracy at least 1 − 3/n
            r["k_lower_bound"] = sdt.d_from_accuracy(1.0 - 3.0 / n_q, 4) / r["dh"]
            r["margin_90_upper_bound"] = margin_90(r["k_lower_bound"])
            # the prudent one, as table 21 reads the others (the low end of k): the people's sensitivity at the
            # high end of its interval; with the machine clipped at no errors, k_ci[0]/k is d_H/d_H,high
            r["k_lower_bound_prudent"] = r["k_lower_bound"] * r["k_ci"][0] / r["k"]
            r["margin_90_upper_bound_prudent"] = margin_90(r["k_lower_bound_prudent"])
    for r in rows:
        r["margin_90"] = margin_90(r["k"])
        r["margin_90_ci"] = [margin_90(r["k_ci"][1]), margin_90(r["k_ci"][0])]
        stronger = r["k"] * r["dh"]
        r["margin_90_in_answers"] = in_answers(stronger, r["margin_90"], 4)
    return rows


def table8() -> list[dict]:
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    keys = review.alphanli_keys(items)
    rows = []
    for folder, closed in ((Path("closed-results"), True), (Path("gpu-results"), False)):
        if not folder.exists():
            continue
        for name, mode in sorted(frontier._tasks_in_folder(folder).get("alphanli", [])):
            for variant in ("original", "without_contested"):
                h, votes, ok, key, _ = review._votes_table(keys[variant], load_machine(folder, "alphanli", name, mode))
                with replicas() as store:
                    r = measure.compare_votes(h, votes, ok, m=2, repetitions=REPETITIONS, answer_class=key)
                if "excluded" in r:
                    continue
                point, runs = store[-1]
                sh = [_shared(x) for x in runs]
                p = _shared(point)
                rows.append({"machine": name, "mode": mode, "closed": closed, "key": variant, "k": point["k"], "phi": point["phi"],
                             "margin": p["margin"], "margin_ci": _ci([x["margin"] for x in sh]),
                             "margin_in_answers": in_answers(p["stronger"], p["margin"], 2),
                             "margin_in_answers_ci": _ci([in_answers(x["stronger"], x["margin"], 2) for x in sh])})
    return rows


def main() -> None:
    out = {"nat_to_deciban": NAT_TO_DECIBAN, "table5": table5(), "table7": table7(), "table8": table8()}
    Path("results/margins.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for r in out["table5"]:
        p = r["point"]
        print(r["machine"], "margin %.3f (%.3f-%.3f)" % (p["margin"], *r["margin_ci"]), "| shared/M %.3f (%.3f-%.3f)" % (p["over_machine"], *r["over_machine_ci"]),
              "| verdict/M %.3f (%.3f-%.3f) grid ok %s" % (r["verdict_over_machine"], *r["verdict_over_machine_ci"], r["verdict_grid_ok"]),
              "| dW_sh %.2f nat" % p["dW_sh"], "| in answers %.3f" % r["margin_in_answers"])
    for r in out["table7"]:
        print(r["machine"], "k %.2f (%.2f-%.2f)" % (r["k"], *r["k_ci"]), "margin90 %.3f (%.3f-%.3f)" % (r["margin_90"], *r["margin_90_ci"]),
              "| no errors: k >= %.2f, margin90 <= %.3f" % (r["k_lower_bound"], r["margin_90_upper_bound"]) if "k_lower_bound" in r else "")
    for r in out["table8"]:
        if r["closed"] or r["machine"] == "gemma-4-26b":
            print(r["machine"], r["mode"], r["key"], "margin %.3f (%.3f-%.3f)" % (r["margin"], *r["margin_ci"]))


if __name__ == "__main__":
    main()
