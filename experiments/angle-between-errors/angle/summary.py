"""From the maps to the verdicts: one row per hypothesis of the protocol. Writes results/summary.json.

Usage, from the experiment folder:  python -m angle.summary
The per-subject ChatBench columns are subsets of "all": they are not counted twice in the averages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path("results")


def _valid(comparisons):
    return [c for c in comparisons if "error" not in c and "excluded" not in c]


def family(c) -> str:
    if c["source"] == "chatbench":
        return "chatbench"
    if c["source"] == "alphanli":
        return "alphanli"
    if c["task"] in ("sarcasm", "census"):
        return "haiid-text"
    return "haiid-images"


def independent(c) -> bool:
    return not (c["source"] == "chatbench" and c["task"] != "all")


def real_machine(c) -> bool:
    return c.get("machine") != "advice"            # the HAIID crowd is a control, not a machine


def h1(comparisons):
    out = {}
    for f in ("chatbench", "alphanli", "haiid-text", "haiid-images"):
        s = [c for c in comparisons if family(c) == f and independent(c) and real_machine(c)]
        if not s:
            continue
        phi = np.array([c["phi"] for c in s])
        below = sum(c["phi_ci"][1] < 90 for c in s)
        out[f] = {"comparisons": len(s), "interval_below_90": below, "phi_median": float(np.median(phi)),
                  "phi_min": float(phi.min()), "phi_max": float(phi.max())}
    return out


def h2(map_data, comparisons):
    """rho between the correct answers, machine with machine and person with machine, same quantity."""
    out = {}
    per_task = {"chatbench": "chatbench", "alphanli": "alphanli", "haiid_sarcasm": "haiid-text", "haiid_census": "haiid-text",
                "haiid_art": "haiid-images", "haiid_cities": "haiid-images", "haiid_dermatology": "haiid-images"}
    for f in dict.fromkeys(per_task.values()):     # in a fixed order, not the order of a set
        mm = [x["rho_correct"] for x in map_data["machine_vs_machine"]
              if per_task.get(x["task"]) == f and x["a"].split("/")[0] != x["b"].split("/")[0] and x["rho_correct"] == x["rho_correct"]]
        pm = [c["rho_correct"] for c in comparisons if family(c) == f and independent(c) and real_machine(c)]
        if mm and pm:
            out[f] = {"rho_machine_machine": float(np.median(mm)), "pairs": len(mm),
                      "rho_person_machine": float(np.median(pm)), "comparisons": len(pm),
                      "passes": bool(np.median(mm) > np.median(pm))}
    return out


def _law_row(s):
    """With rho > 0 the prediction at 90 degrees lies above the one with the angle: the real test is on the rows between the two."""
    dc = np.array([c["law"]["d_combo"] for c in s])
    pf = np.array([c["law"]["pred_phi"] for c in s])
    p9 = np.array([c["law"]["pred_90"] for c in s])
    dx = np.array([c["law"]["dmax"] for c in s])
    e_phi, e_90 = np.abs(dc - pf), np.abs(dc - p9)
    lo, hi = np.minimum(pf, p9), np.maximum(pf, p9)
    between = (dc > lo) & (dc < hi)
    beyond = (dc - hi)[dc > hi]
    return {"above_median": float(np.median(beyond)) if beyond.size else 0.0, "above_max": float(beyond.max()) if beyond.size else 0.0,
            "comparisons": len(s), "mae_phi": float(e_phi.mean()), "mae_90": float(e_90.mean()),
            "better_phi": int((e_phi < e_90).sum()), "passes": bool(e_phi.mean() < e_90.mean()),
            "between_the_two": int(between.sum()), "below_both": int((dc < lo).sum()), "above_both": int((dc > hi).sum()),
            "ties": int(((dc == lo) | (dc == hi)).sum()),
            "better_phi_between": int((e_phi[between] < e_90[between]).sum()),
            "gain_over_dmax_mean": float((dc - dx).mean()), "predicted_over_dmax_mean": float((pf - dx).mean())}


def h3(comparisons):
    """Mean absolute error between the combiner's d' and the two predictions."""
    out = {}
    everything = [c for c in comparisons if c.get("law") and independent(c) and real_machine(c)]
    row = _law_row
    out["all"] = row(everything)
    out["without_haiid"] = row([c for c in everything if not family(c).startswith("haiid")])
    for f in ("chatbench", "alphanli", "haiid-text", "haiid-images"):
        s = [c for c in everything if family(c) == f]
        if s:
            out[f] = row(s)
    return out


def h4(source, map_comparisons):
    rows = []
    for c in source["comparisons"]:
        if "g_phi" in c and c.get("subject") == "all":
            rows.append({k: c.get(k) for k in ("machine", "mode", "g_90", "g_phi", "phi", "phi_ci", "eta_phi", "turing") if k in c})
    for c in map_comparisons:
        if c.get("machine") == "advice":
            rows.append({"machine": "crowd", "task": c["task"], "g_90": c.get("g_90"), "g_phi": c.get("g_phi"), "phi": c.get("phi")})
    return rows


def h5(comparisons):
    s = [c for c in comparisons if c.get("law") and independent(c) and real_machine(c) and "over_best_ci" in (c.get("combo") or {})]
    tab = {"useful_and_gains": 0, "useful_and_not": 0, "not_useful_and_gains": 0, "not_useful_and_not": 0}
    for c in s:
        u = c["phi"] > c["useful_angle"]
        g = c["combo"]["over_best_ci"][0] > 0
        tab[("useful" if u else "not_useful") + ("_and_gains" if g else "_and_not")] += 1
    tab["comparisons"] = len(s)
    return tab


def h7(comparisons):
    """The scale: the same people and the same questions, ever stronger machines."""
    out = {}
    choice = {"chatbench": lambda c: c["source"] == "chatbench" and c["task"] == "all",
              "alphanli": lambda c: c["source"] == "alphanli"}
    for name, f in choice.items():
        for mode in ("letter", "reason"):
            s = sorted([c for c in comparisons if f(c) and c["mode"] == mode], key=lambda c: c["dm"])
            if len(s) < 3:
                continue
            rows = [{"machine": c["machine"], "dm": c["dm"], "acc_m": c["acc_m"], "phi": c["phi"], "phi_ci": c["phi_ci"],
                     "useful": c["useful_angle"], "cos_theta": c["dh"] / c["ideal_phi"], "ideal": c["ideal_phi"]} for c in s]
            dm = [r["dm"] for r in rows]
            sp = spearmanr(dm, [r["phi"] for r in rows])
            out[f"{name}/{mode}"] = {"rows": rows, "spearman_dm_phi": float(sp.statistic), "n": len(rows)}
    return out


def vision_summary(rows):
    """model-vs-human: one row per network (median over conditions) and the law put to the test."""
    ok = [r for r in rows if "phi" in r and "error" not in r and "excluded" not in r]
    networks = []
    for name in sorted({r["machine"] for r in ok}):
        s = [r for r in ok if r["machine"] == name]
        networks.append({"machine": name, "conditions": len(s), "phi_median": float(np.median([r["phi"] for r in s])),
                         "dm_median": float(np.median([r["dm"] for r in s])), "acc_m_median": float(np.median([r["acc_m"] for r in s])),
                         "below_90": int(sum(r["phi_ci"][1] < 90 for r in s))})
    networks.sort(key=lambda r: r["dm_median"])
    sets = {d: float(np.median([r["phi"] for r in ok if r["dataset"] == d])) for d in sorted({r["dataset"] for r in ok})}
    le = [r for r in ok if r.get("law") and r["law"]["d_combo"] == r["law"]["d_combo"]]
    return {"comparisons": len(ok), "phi_median": float(np.median([r["phi"] for r in ok])), "below_90": int(sum(r["phi_ci"][1] < 90 for r in ok)),
            "networks": networks, "sets": sets, "spearman_dm_phi": float(spearmanr([r["dm_median"] for r in networks], [r["phi_median"] for r in networks]).statistic),
            "law": _law_row(le),
            "at_limit": int(sum(r["rho"] <= -0.89 or r["rho"] >= 0.989 for r in ok))}


def main() -> None:
    map_data = json.loads((R / "map.json").read_text())
    conf = _valid(map_data["comparisons"])
    source = json.loads((R / "chatbench-source.json").read_text())
    out = {"h1": h1(conf), "h2": h2(map_data, conf), "h3": h3(conf), "h4": h4(source, conf), "h5": h5(conf), "h7": h7(conf)}
    if (R / "vision.json").exists():
        out["vision"] = vision_summary(json.loads((R / "vision.json").read_text()))
    if (R / "silicon.json").exists():
        out["silicon"] = json.loads((R / "silicon.json").read_text())
    (R / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k in ("h1", "h2", "h3", "h5"):
        print(k, json.dumps(out[k], ensure_ascii=False))
    for k, v in out["h7"].items():
        print(k, "spearman", round(v["spearman_dm_phi"], 2), [(r["machine"][:10], round(r["dm"], 2), round(r["phi"]), round(r["cos_theta"], 2)) for r in v["rows"]])


if __name__ == "__main__":
    main()
