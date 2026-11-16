"""A12: hypotheses H1-H3 with the criterion as the protocol writes it ("passes" = the 95 percent interval
meets the criterion). Pre-registered in PROTOCOL.md, "After the critical review (v4): the audit".

Usage, from the experiment folder:  python -m angle.criteria [OUT.json]   (default results/criteria.json)
Reads results/map.json. Bootstrap over the comparisons, stratified by data family, 2,000 replicas, seed 534.
It does not carry each comparison's own uncertainty: the intervals are too narrow, not too wide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .summary import _valid, family, independent, real_machine

R = Path("results")
FAMILIES = ("chatbench", "alphanli", "haiid-text", "haiid-images")
REPLICAS, SEED = 2000, 534


def _ci(x) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _resample(gen, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One stratified replica: within each group, as many draws with replacement as its size."""
    return {g: v[gen.integers(0, v.size, v.size)] for g, v in groups.items()}


def h1(comparisons, gen) -> dict:
    out = {}
    for f in FAMILIES:
        s = [c for c in comparisons if family(c) == f and independent(c) and real_machine(c)]
        if not s:
            continue
        phi = np.array([c["phi"] for c in s])
        reps = [np.median(phi[gen.integers(0, phi.size, phi.size)]) for _ in range(REPLICAS)]
        ci = _ci(reps)
        out[f] = {"comparisons": len(s), "interval_below_90": int(sum(c["phi_ci"][1] < 90 for c in s)),
                  "phi_median": float(np.median(phi)), "phi_median_ci": ci, "family_passes": bool(ci[1] < 90)}
    return out


def h2(map_data, comparisons, gen) -> dict:
    per_task = {"chatbench": "chatbench", "alphanli": "alphanli", "haiid_sarcasm": "haiid-text", "haiid_census": "haiid-text",
                "haiid_art": "haiid-images", "haiid_cities": "haiid-images", "haiid_dermatology": "haiid-images"}
    out = {}
    for f in FAMILIES:
        mm = np.array([x["rho_correct"] for x in map_data["machine_vs_machine"]
                       if per_task.get(x["task"]) == f and x["a"].split("/")[0] != x["b"].split("/")[0]
                       and x["rho_correct"] == x["rho_correct"]])
        pm = np.array([c["rho_correct"] for c in comparisons if family(c) == f and independent(c) and real_machine(c)])
        if not (mm.size and pm.size):
            continue
        reps = []
        for _ in range(REPLICAS):
            r = _resample(gen, {"mm": mm, "pm": pm})
            reps.append(np.median(r["mm"]) - np.median(r["pm"]))
        ci = _ci(reps)
        out[f] = {"difference": float(np.median(mm) - np.median(pm)), "difference_ci": ci,
                  "pairs": int(mm.size), "comparisons": int(pm.size), "passes": bool(ci[0] > 0)}
    return out


def _errors(s):
    dc = np.array([c["law"]["d_combo"] for c in s])
    pf = np.array([c["law"]["pred_phi"] for c in s])
    p9 = np.array([c["law"]["pred_90"] for c in s])
    lo, hi = np.minimum(pf, p9), np.maximum(pf, p9)
    return np.abs(dc - pf) - np.abs(dc - p9), int(((dc > lo) & (dc < hi)).sum())


def h3(comparisons, gen) -> dict:
    everything = [c for c in comparisons if c.get("law") and independent(c) and real_machine(c)]
    sets = {"all": everything, "without_haiid": [c for c in everything if not family(c).startswith("haiid")]}
    sets.update({f: [c for c in everything if family(c) == f] for f in FAMILIES})
    out = {}
    for name, s in sets.items():
        if not s:
            continue
        diff, between = _errors(s)
        strata = {}
        for c, d in zip(s, diff):
            strata.setdefault(family(c), []).append(d)
        strata = {g: np.array(v) for g, v in strata.items()}
        reps = []
        for _ in range(REPLICAS):
            r = _resample(gen, strata)
            reps.append(np.concatenate(list(r.values())).mean())
        ci = _ci(reps)
        out[name] = {"comparisons": len(s), "mean_difference": float(diff.mean()), "mean_difference_ci": ci,
                     "better_phi": int((diff < 0).sum()), "between_the_two": between, "passes": bool(ci[1] < 0)}
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    target = Path(argv[0]) if argv else R / "criteria.json"
    map_data = json.loads((R / "map.json").read_text())
    conf = _valid(map_data["comparisons"])
    gen = np.random.default_rng(SEED)
    out = {"replicas": REPLICAS, "seed": SEED, "h1": h1(conf, gen), "h2": h2(map_data, conf, gen), "h3": h3(conf, gen)}
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k in ("h1", "h2", "h3"):
        for f, v in out[k].items():
            print(k, f, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
