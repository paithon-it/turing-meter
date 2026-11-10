"""Apparent intelligence on the reference meta-analysis (docs/SPEC.md, section 6, test 5).

Data: Vaccaro, Almaatouq and Malone (2024), "When combinations of humans and AI
are useful", Nature Human Behaviour 8, 2293-2303. Each of its 370 effect sizes
reports the performance of humans alone, AI alone and humans with AI on the same
task. The data are not redistributed here: download them from the authors' OSF
repository (https://osf.io/wrq7c/?view_only=b9e1e86079c048b4bfb03bee6966e560,
the share link published in the paper) as Data_Extraction.csv.

Usage:
    python composition_test.py Data_Extraction.csv [--draws 2000] [--seed 534]

What it does, on the binary decision tasks:
  1. puts the three conditions on the sensitivity scale d' of signal detection
     theory (SPEC section 3), from the raw metric the paper reported, in this
     order of preference: sensitivity and specificity together, d' itself,
     accuracy, the area under the ROC curve. Other
     metrics (precision, recall, F1, scores, response bias) do not determine d'
     and are left out, with the reason printed;
  2. computes the apparent intelligence of the pair, the ideal composition of two
     observers who err independently, |S| = sqrt(dH^2 + dM^2), and the
     intelligence factor eta = dS / |S| (SPEC section 2);
  3. computes, as a second reference, the ideal for a person who receives only
     the machine's verdict, not its confidence;
  4. reports medians with intervals from a bootstrap over papers, by stronger
     partner, by design (same people or different people) and by whether the AI
     showed its confidence;
  5. counts the pairs that beat the better of the two alone and the pairs above
     the ideal, with the probability that sampling noise explains them.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turing_meter.apparent import (  # noqa: E402
    confidence_sharing,
    d_prime_from_accuracy,
    d_prime_from_auc,
    d_prime_from_rates,
    pair,
    verdict_ideal,
)

WITHIN = {"Within-Subjects", "Mixed, Within-Subjects"}
PAIRED = {"Sensitivity", "Specificity", "FPR", "FNR"}


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


MEANS = ("Avg_Perf_Human", "Avg_Perf_AI", "Avg_Perf_HumanAI")
SDS = ("Sd_Perf_Human", "Sd_Perf_AI", "Sd_Perf_HumanAI")


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
        })
    return rows, skipped


def describe(rows: list[dict]) -> dict:
    out = []
    for r in rows:
        p = pair(r["dh"], r["dm"], r["ds"])
        v = verdict_ideal(r["dh"], r["dm"])
        out.append({**r, **p._asdict(), "eta_verdict": r["ds"] / v})
    return out


def boot_median(rows: list[dict], field: str, draws: int, rng) -> tuple[float, float, float]:
    """Median with a 95% interval from resampling papers, not comparisons."""
    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r[field])
    papers = list(by_paper)
    meds = []
    for _ in range(draws):
        pick = rng.choice(len(papers), size=len(papers), replace=True)
        vals = [v for i in pick for v in by_paper[papers[i]]]
        meds.append(float(np.median(vals)))
    return median(r[field] for r in rows), *np.percentile(meds, [2.5, 97.5])


def noise_probability(r: dict, test, draws: int, rng, ai_items: int | None) -> float | None:
    """Share of noisy redraws in which `test(pair)` still holds (accuracy rows only)."""
    if not r["noise"]:
        return None
    (ah, am, as_), (se_h, se_s) = r["noise"]
    se_m = math.sqrt(am * (1 - am) / ai_items) if ai_items else 0.0
    hits = 0
    for _ in range(draws):
        h = float(np.clip(rng.normal(ah, se_h), 1e-4, 1 - 1e-4))
        m = float(np.clip(rng.normal(am, se_m), 1e-4, 1 - 1e-4))
        s = float(np.clip(rng.normal(as_, se_s), 1e-4, 1 - 1e-4))
        if min(h, m, s) <= 0.5:
            continue
        p = pair(d_prime_from_accuracy(h), d_prime_from_accuracy(m), d_prime_from_accuracy(s))
        hits += bool(test(p))
    return hits / draws


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=534)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    raw, skipped = load(args.csv)
    rows = describe(raw)
    papers = {r["paper"] for r in rows}
    print(f"Comparisons on the d' scale: {len(rows)} from {len(papers)} papers")
    for k, n in Counter(r["source"] for r in rows).most_common():
        print(f"  from {k}: {n}")
    print("Left out:")
    for k, n in sorted(skipped.items()):
        print(f"  {k}: {n}")

    def per_paper(sub: list[dict], field: str) -> float:
        by_paper = defaultdict(list)
        for r in sub:
            if r[field] is not None:
                by_paper[r["paper"]].append(r[field])
        return median(median(v) for v in by_paper.values())

    def block(title: str, sub: list[dict]) -> None:
        if not sub:
            return
        n_p = len({r["paper"] for r in sub})
        m, lo, hi = boot_median(sub, "eta", args.draws, rng)
        gs = [r for r in sub if r["gain_share"] is not None]
        g, glo, ghi = boot_median(gs, "gain_share", args.draws, rng)
        t, tlo, thi = boot_median(sub, "turing", args.draws, rng)
        v = median(r["eta_verdict"] for r in sub)
        best = sum(r["ds"] > max(r["dh"], r["dm"]) for r in sub)
        below = sum(r["turing"] < 1 for r in sub)
        few = "  (fewer than 10 papers: the interval is optimistic)" if n_p < 10 else ""
        print(f"\n{title}: n = {len(sub)} from {n_p} papers{few}")
        print(f"  share of the ideal, eta      median {m:.2f}  [{lo:.2f}, {hi:.2f}]"
              f"  (each paper once: {per_paper(sub, 'eta'):.2f})")
        print(f"  share of the possible gain, g median {g:.2f}  [{glo:.2f}, {ghi:.2f}]"
              f"  (each paper once: {per_paper(gs, 'gain_share'):.2f})")
        print(f"  eta against the verdict ideal       {v:.2f}")
        print(f"  turing                      median {t:.2f}  [{tlo:.2f}, {thi:.2f}]")
        print(f"  pair beats the better of the two: {best}; pair below the person alone: {below}")

    strong_m = lambda r: r["theta_deg"] >= 45
    block("All", rows)
    block("Person stronger (theta < 45)", [r for r in rows if not strong_m(r)])
    block("Machine stronger (theta >= 45)", [r for r in rows if strong_m(r)])
    block("Same people in both conditions", [r for r in rows if r["within"]])
    block("Different people", [r for r in rows if not r["within"]])
    block("AI showed its confidence", [r for r in rows if r["confidence"]])
    block("AI showed only its verdict", [r for r in rows if not r["confidence"]])
    block("Machine stronger, AI showed its confidence",
          [r for r in rows if strong_m(r) and r["confidence"]])
    block("Machine stronger, AI showed only its verdict",
          [r for r in rows if strong_m(r) and not r["confidence"]])

    big = Counter(r["name"] for r in rows).most_common(1)[0][0]
    rest = [r for r in rows if r["name"] != big]
    print(f"\nWithout the largest study ({big}, "
          f"{sum(r['name'] == big for r in rows)} comparisons, "
          f"{len({r['dh'] for r in rows if r['name'] == big})} distinct groups of people alone):")
    for title, sel in (("person stronger", lambda r: not strong_m(r)),
                       ("machine stronger", strong_m),
                       ("AI showed its confidence", lambda r: r["confidence"]),
                       ("AI showed only its verdict", lambda r: not r["confidence"])):
        sub = [r for r in rest if sel(r)]
        print(f"  {title}: n = {len(sub)}, eta median {median(r['eta'] for r in sub):.2f}, "
              f"g median {median(r['gain_share'] for r in sub if r['gain_share'] is not None):.2f}")

    rank = lambda a: np.argsort(np.argsort(np.asarray(a)))
    spear = lambda x, y: float(np.corrcoef(rank(x), rank(y))[0, 1])
    th = [r["theta_deg"] for r in rows]
    ct = np.array([r["cos_theta"] for r in rows])
    gg = np.array([r["gain_share"] for r in rows])
    perm = [spear(th, ct + rng.permutation(gg) * (1 - ct)) for _ in range(args.draws)]
    print("\nIs the fall of eta with theta behaviour or arithmetic?")
    print(f"  rank correlation theta-eta: {spear(th, [r['eta'] for r in rows]):.2f}; "
          f"with g shuffled across comparisons: {np.median(perm):.2f} "
          f"[{np.percentile(perm, 2.5):.2f}, {np.percentile(perm, 97.5):.2f}]")
    print(f"  rank correlation theta-g: {spear(th, gg):.2f}; same people only: "
          f"{spear([r['theta_deg'] for r in rows if r['within']], [r['gain_share'] for r in rows if r['within']]):.2f}")

    print("\nAnchors:")
    n = len(rows)
    wcs = [r["ds"] / confidence_sharing(r["dh"], r["dm"]) for r in rows]
    print(f"  below the person alone (Tu < 1): {sum(r['turing'] < 1 for r in rows)}/{n}")
    print(f"  above the better of the two: {sum(r['ds'] > max(r['dh'], r['dm']) for r in rows)}/{n}")
    print(f"  above confidence sharing (Bahrami 2010): {sum(w > 1 for w in wcs)}/{n}; "
          f"its median share of the ideal: "
          f"{median(confidence_sharing(r['dh'], r['dm']) / r['apparent'] for r in rows):.2f}")
    print(f"  above the ideal (eta > 1): {sum(r['eta'] > 1 for r in rows)}/{n}")
    weak = [r for r in rows if r["dh"] < 0.35]
    print(f"  person alone barely above chance (dH < 0.35, accuracy < 0.57): {len(weak)}/{n}, "
          f"from {dict(Counter(r['name'] for r in weak))}")

    print("\nNoise (accuracy rows): comparisons that hold in at least 95% of redraws,")
    print("by the number of items assumed behind each AI accuracy")
    for label, test, sel in (
        ("above the ideal", lambda p: p.eta > 1, lambda r: r["eta"] > 1),
        ("beats the better of the two", lambda p: p.eta > max(p.cos_theta, p.eta_follow),
         lambda r: r["ds"] > max(r["dh"], r["dm"])),
    ):
        cand = [r for r in rows if sel(r)]
        known = [r for r in cand if r["noise"]]
        counts = []
        for items in (20, 50, 100, 200, 1000, None):
            probs = [noise_probability(r, test, args.draws, rng, items) for r in known]
            counts.append(f"{items or 'exact'}: {sum(q >= 0.95 for q in probs)}")
        print(f"  {label}: {len(cand)} observed, {len(known)} with known noise; " + ", ".join(counts))

    print("\nAbove the ideal:")
    for r in rows:
        if r["eta"] > 1:
            print(f"  {r['name']}  {r['source']}  dH {r['dh']:.2f}  dM {r['dm']:.2f}  "
                  f"dS {r['ds']:.2f}  eta {r['eta']:.2f}")

if __name__ == "__main__":
    main()
