"""Meta-analysis by Vaccaro, Almaatouq and Malone (2024): the gain share as ρ varies.

There the angle cannot be measured: we report how the numbers change for each plausible ρ.

Usage:  python -m angle.vaccaro data/vaccaro/Data_Extraction.csv
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import norm

from . import sdt


def d_prime_from_rates(h, f):
    return float(norm.ppf(h) - norm.ppf(f))


def d_prime_from_accuracy(a):
    return float(2.0 * norm.ppf(a))


def d_prime_from_auc(a):
    return float(math.sqrt(2.0) * norm.ppf(a))


WITHIN = {"Within-Subjects", "Mixed, Within-Subjects"}
PAIRED = {"Sensitivity", "Specificity", "FPR", "FNR"}


MEANS = ("Avg_Perf_Human", "Avg_Perf_AI", "Avg_Perf_HumanAI")
SDS = ("Sd_Perf_Human", "Sd_Perf_AI", "Sd_Perf_HumanAI")


def _num(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _scaled(x: dict, keys: tuple[str, ...]) -> list[float | None]:
    v = [_num(x[k]) for k in keys]
    if None in v:
        return v
    scale = 100.0 if max(v) > 1.0001 else 1.0
    return [a / scale for a in v]


def load(path: str) -> tuple[list[dict], Counter]:
    """One comparison per experiment, treatment and task, on the d' scale."""
    groups: dict[tuple, dict] = defaultdict(dict)
    meta: dict[tuple, dict] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for x in csv.DictReader(fh):
            if x["Task_Output_Cleaned"] != "Binary":
                continue
            key = (x["Paper_ID"], x["Exp_ID"], x["Treatment_ID"], x["Task_Desc"])
            groups[key].setdefault(x["Perf_Metric"], []).append(x)
            meta.setdefault(key, x)

    rows, skipped = [], Counter()
    for key, metrics in groups.items():
        x = meta[key]
        d, source, se = None, None, None
        rates = {m: _scaled(metrics[m][0], MEANS) for m in PAIRED if m in metrics}
        sens = rates.get("Sensitivity") or (
            [1 - a for a in rates["FNR"]] if "FNR" in rates else None)
        spec = rates.get("Specificity") or (
            [1 - a for a in rates["FPR"]] if "FPR" in rates else None)
        if sens and spec and None not in sens + spec:
            d = [d_prime_from_rates(s, 1 - f) for s, f in zip(sens, spec)]
            source = "sensitivity+specificity"
        elif "Sensitivity (d')" in metrics:
            d = [_num(metrics["Sensitivity (d')"][0][k]) for k in MEANS]
            source = "d' as reported"
            if None in d:
                d = None
        elif "Accuracy" in metrics:
            r = metrics["Accuracy"][0]
            a = _scaled(r, MEANS)
            if None not in a:
                d = [d_prime_from_accuracy(v) for v in a]
                source = "accuracy"
                sd = [(_num(r[k]) or 0.0) / (100.0 if max(a) > 1.0001 else 1.0) for k in SDS]
                n_h, n_s = _num(r["N_Human"]), _num(r["N_HumanAI"])
                se = (a, (sd[0] / math.sqrt(n_h) if n_h else 0.0,
                          sd[2] / math.sqrt(n_s) if n_s else 0.0))
        elif "AUC" in metrics:
            a = _scaled(metrics["AUC"][0], MEANS)
            if None not in a:
                d = [d_prime_from_auc(v) for v in a]
                source = "AUC"
        if d is None:
            for m in metrics:
                skipped[f"metric {m}: no d'"] += 1
            continue
        if min(d) <= 0:
            skipped[f"{source}: a condition at or below chance"] += 1
            continue
        rows.append({
            "paper": x["Paper_ID"], "name": x["Paper_Name"], "source": source,
            "within": x["Exp_Design"] in WITHIN,
            "confidence": x["AI_Conf_Incl"] == "Yes",
            "dh": d[0], "dm": d[1], "ds": d[2], "noise": se,
            "year": int(x["Year"]) if x["Year"].strip().isdigit() else None, "ai_type": x["AI_Type_Cleaned"],
        })
    return rows, skipped




def turing_of_comparisons(rows: list[dict], repetitions: int = 3000, seed: int = 5341) -> dict:
    """The turing of each comparison, dS/dH: the same unit on machines of different years and types.

    Descriptive, added after seeing the data. The interval of the median resamples the studies.
    """
    tu = np.array([r["ds"] / r["dh"] for r in rows])
    k = np.array([r["dm"] / r["dh"] for r in rows])
    per_study: dict[str, list] = defaultdict(list)
    for r, t in zip(rows, tu):
        per_study[r["paper"]].append(float(t))
    names = sorted(per_study)
    gen = np.random.default_rng(seed)
    boot = []
    for _ in range(repetitions):
        sc = gen.choice(len(names), size=len(names), replace=True)
        boot.append(float(np.median([v for i in sc for v in per_study[names[i]]])))
    years = sorted({r["year"] for r in rows if r.get("year")})
    per_type: dict[str, list] = defaultdict(list)
    for r, t in zip(rows, tu):
        per_type[r.get("ai_type") or "undeclared"].append(float(t))
    return {
        "comparisons": len(rows), "studies": len(names),
        "median": float(np.median(tu)), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "q1": float(np.percentile(tu, 25)), "q3": float(np.percentile(tu, 75)),
        "min": float(tu.min()), "max": float(tu.max()),
        "below_one": int((tu < 1.0).sum()),
        "studies_below_one": len({r["paper"] for r, t in zip(rows, tu) if t < 1.0}),
        "at_least_two": int((tu >= 2.0).sum()),
        "above_the_best": int((tu > np.maximum(1.0, k)).sum()),
        "years": [years[0], years[-1]] if years else [],
        # where the people alone and with the machine are the same: elsewhere the turing is a ratio between groups
        "same_people": {"comparisons": int(sum(1 for r in rows if r.get("within"))),
                        "below_one": int(sum(1 for r, t in zip(rows, tu) if r.get("within") and t < 1.0)),
                        "median": (float(np.median([t for r, t in zip(rows, tu) if r.get("within")]))
                                   if any(r.get("within") for r in rows) else None)},
        "per_type": {t: {"comparisons": len(v), "median": float(np.median(v))} for t, v in sorted(per_type.items())},
    }


def run(path: str) -> dict:
    rows, _ = load(path)
    studies = sorted({r["paper"] for r in rows})
    gen = np.random.default_rng(534)
    output = {"rows": len(rows), "studies": len(studies), "per_rho": []}
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        g, eta, above = {}, {}, 0
        for i, r in enumerate(rows):
            id_ = sdt.ideal(r["dh"], r["dm"], rho)
            q = sdt.gain_share(r["ds"], r["dh"], id_)
            eta[i] = r["ds"] / id_
            if q is not None and r["dm"] > r["dh"] * rho:      # the person has room: the ideal leaves them a gain
                g[i] = q
            above += r["ds"] > id_ + 1e-9
        per_study: dict[str, list] = defaultdict(list)
        for i, v in g.items():
            per_study[rows[i]["paper"]].append(v)
        boot = []
        names = sorted(per_study)
        for _ in range(3000):
            sc = gen.choice(len(names), size=len(names), replace=True)
            boot.append(float(np.median([v for k in sc for v in per_study[names[k]]])))
        output["per_rho"].append({"rho": rho, "phi": sdt.angle_degrees(rho), "n_with_gain": len(g), "g": float(np.median(list(g.values()))),
                                  "g_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                  "eta": float(np.median(list(eta.values()))), "beyond_ideal": above})
    output["turing"] = turing_of_comparisons(rows)
    return output


if __name__ == "__main__":
    r = run(sys.argv[1])
    Path("results").mkdir(exist_ok=True)
    Path("results/vaccaro-sensitivity.json").write_text(json.dumps(r, indent=1))
    print(r["rows"], r["studies"])
    for x in r["per_rho"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k != "g_ci"}, [round(v, 3) for v in x["g_ci"]])
