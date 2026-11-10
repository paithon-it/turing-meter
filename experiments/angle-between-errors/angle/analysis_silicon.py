"""Silicon people: measure A of the protocol on real networks. Writes results/silicon.json.

Usage:  python -m angle.analysis_silicon OUTCOMES_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

from . import sdt

STAGES = ("T0", "T1", "T3")
ARMS = ("control", "assisted", "defect")
TOUCHED_CLASSES = (3, 5)                 # cats and dogs: the only classes the injected defect acts on


def load(folder: Path):
    mac = np.load(folder / "machines.npz")
    y = mac["y"]
    outcomes = {b: {t: [] for t in STAGES} for b in ARMS}
    for b in ARMS:
        for f in sorted(folder.glob(f"{b}-*.npz")):
            d = np.load(f)
            for t in STAGES:
                outcomes[b][t].append(d[t])
    return y, mac["m"], mac["d"], {b: {t: np.stack(v) for t, v in x.items()} for b, x in outcomes.items()}


def rho_hm(gi, m_ok):
    P, I = gi.shape
    p = np.repeat(np.arange(P), I)
    c = np.tile(np.arange(I), P)
    return sdt.rho_personal_thresholds(p, c, gi.ravel().astype(float), m_ok.astype(float))


def rho_hh(gi, source_idx=None):
    """source_idx: the original indices of the networks, if they were resampled. A network and its copy do not form a pair."""
    weights = None
    if source_idx is not None:
        _, first, times = np.unique(source_idx, return_index=True, return_counts=True)
        gi, weights = gi[first], times.astype(float)
    P, I = gi.shape
    p = np.repeat(np.arange(P), I)
    c = np.tile(np.arange(I), P)
    return sdt.rho_person_pairs(p, c, gi.ravel().astype(float), person_weights=weights)


def per_class(fn, gi, y, *extra):
    total = 0.0
    for cl in np.unique(y):
        s = y == cl
        if gi[:, s].mean() >= 0.999 or gi[:, s].mean() <= 0.001:
            continue
        total += fn(gi[:, s], *(e[s] for e in extra)) * s.sum()
    return total / y.size


def series(outcomes, y, m_ok, by_class=True, source_idx=None):
    """rhoHM, rhoHH, accuracy per stage; within the true class, as the protocol requires (by_class=False: on all together)."""
    out = {}
    for t in STAGES:
        gi = outcomes[t] == y
        if by_class:
            out[t] = {"rho_hm": per_class(rho_hm, gi, y, m_ok), "rho_hh": per_class(lambda g: rho_hh(g, source_idx), gi, y), "acc": float(gi.mean())}
        else:
            out[t] = {"rho_hm": float(rho_hm(gi, m_ok)), "rho_hh": float(rho_hh(gi, source_idx)), "acc": float(gi.mean())}
    q = max(out["T0"]["rho_hm"], 1e-3) ** 2 / max(out["T0"]["rho_hh"], 1e-3)
    for t in STAGES:
        out[t]["A"] = out[t]["rho_hm"] - np.sqrt(q * max(out[t]["rho_hh"], 0.0))
    out["q"] = float(q)
    return out


def _transmission(outcomes, y, d, different, trans):
    """On the items where the defect changes the answer: how often the network gives the injected wrong answer."""
    out = {}
    for b in ARMS:
        x = outcomes[b]["T3"][:, different]
        out[b] = {"copies_the_defect": float((x == d[different]).mean()), "correct": float((x == y[different]).mean())}
    v = np.array(trans["defect"]) - np.array(trans["control"])
    w = np.array(trans["assisted"]) - np.array(trans["control"])
    out["items"] = int(different.size)
    out["defect_minus_control"] = {"point": out["defect"]["copies_the_defect"] - out["control"]["copies_the_defect"], "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    out["assisted_minus_control"] = {"point": out["assisted"]["copies_the_defect"] - out["control"]["copies_the_defect"], "ci": [float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]}
    return out


def main(folder: Path) -> None:
    y, m, d, outcomes = load(folder)
    mok = {"control": m == y, "assisted": m == y, "defect": d == y}
    point = {b: series(outcomes[b], y, mok[b]) for b in ARMS}
    point["control_with_defective"] = series(outcomes["control"], y, d == y)    # the same comparator as the defect arm
    without = {b: series(outcomes[b], y, mok[b], by_class=False) for b in ARMS}
    without["control_with_defective"] = series(outcomes["control"], y, d == y, by_class=False)
    gen = np.random.default_rng(534)
    N = 300
    boot = {b: {t: [] for t in STAGES} for b in ARMS}
    did = {"assisted": [], "defect": [], "defect_same_comparator": [], "assisted_without_class": [], "defect_without_class": [],
           "assisted_touched_classes": [], "defect_touched_classes": []}
    trans = {b: [] for b in ARMS}
    different = np.flatnonzero(d != m)                    # the items where the defect really changes the answer
    P = outcomes["control"]["T0"].shape[0]
    I = y.size
    for _ in range(N):
        ip = gen.integers(0, I, I)
        pe = {b: gen.integers(0, P, P) for b in ARMS}
        try:
            r = {b: series({t: outcomes[b][t][pe[b]][:, ip] for t in STAGES}, y[ip], mok[b][ip], source_idx=pe[b]) for b in ARMS}
        except Exception:
            continue
        dd = {b: r[b]["T3"]["A"] - r[b]["T0"]["A"] for b in ARMS}
        for b in ("assisted", "defect"):
            did[b].append(dd[b] - dd["control"])
        cd = series({t: outcomes["control"][t][pe["control"]][:, ip] for t in STAGES}, y[ip], (d == y)[ip], source_idx=pe["control"])
        did["defect_same_comparator"].append(dd["defect"] - (cd["T3"]["A"] - cd["T0"]["A"]))
        # Exploratory analysis, decided after the second run: the same difference without conditioning on the class.
        sc = {b: series({t: outcomes[b][t][pe[b]][:, ip] for t in STAGES}, y[ip], mok[b][ip], by_class=False, source_idx=pe[b]) for b in ("control", "assisted", "defect")}
        cs = series({t: outcomes["control"][t][pe["control"]][:, ip] for t in STAGES}, y[ip], (d == y)[ip], by_class=False, source_idx=pe["control"])
        ds = {b: sc[b]["T3"]["A"] - sc[b]["T0"]["A"] for b in sc}
        did["assisted_without_class"].append(ds["assisted"] - ds["control"])
        did["defect_without_class"].append(ds["defect"] - (cs["T3"]["A"] - cs["T0"]["A"]))
        # Exploratory too: only the classes the defect acts on, to see whether A is blind or diluted.
        z = np.flatnonzero(np.isin(y[ip], TOUCHED_CLASSES))
        ipz = ip[z]
        rz = {b: series({t: outcomes[b][t][pe[b]][:, ipz] for t in STAGES}, y[ipz], mok[b][ipz], source_idx=pe[b]) for b in ("control", "assisted", "defect")}
        cz = series({t: outcomes["control"][t][pe["control"]][:, ipz] for t in STAGES}, y[ipz], (d == y)[ipz], source_idx=pe["control"])
        dz = {b: rz[b]["T3"]["A"] - rz[b]["T0"]["A"] for b in rz}
        did["assisted_touched_classes"].append(dz["assisted"] - dz["control"])
        did["defect_touched_classes"].append(dz["defect"] - (cz["T3"]["A"] - cz["T0"]["A"]))
        ic = gen.integers(0, different.size, different.size)
        for b in ARMS:
            x = outcomes[b]["T3"][pe[b]][:, different[ic]]
            trans[b].append(float((x == d[different[ic]]).mean()))
        for b in ARMS:
            for t in STAGES:
                boot[b][t].append(r[b][t]["A"])
    output = {"point": point, "without_class": without, "repetitions": len(did["assisted"]),
              "did": {b: {"mean": float(np.mean(v)), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for b, v in did.items()},
              "transmission": _transmission(outcomes, y, d, different, trans),
              "rho_hm_naive": {b: {t: float(rho_hm(outcomes[b][t] == y, mok[b])) for t in STAGES} for b in ARMS}}
    Path("results").mkdir(exist_ok=True)
    Path("results/silicon.json").write_text(json.dumps(output, indent=1))
    for b in ARMS:
        print(b, {t: {k: round(v, 3) for k, v in point[b][t].items()} for t in STAGES}, "q", round(point[b]["q"], 3))
    print(output["did"]); print(output["rho_hm_naive"])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
