"""A15: H1, H2 and H3 resampling people and questions, with every comparison recomputed in each replica.
Pre-registered in PROTOCOL.md, "After the fourth review (v4)" (d75500f), before computing it.

A12 (angle/criteria.py) resampled the comparisons, which share people and items. Here each replica gives a
Poisson(1) weight to every person and every item of every task (ChatBench: workers and questions; HAIID:
participants and items, task by task; αNLI: items only), and the same weights serve every comparison and every
pair of machines on that task. The comparisons are recomputed with the code of the map (measure._one_pass with
the comparison's conversion table, fixed at the point as in measure.compare; the step of compare_votes for αNLI).

H3: the combiner is not refitted in the replica, its choices are reweighted (as in combiner.margin); the two
predictions follow the replica's d_H, d_M and ρ through their first-order expansion around the point, with
derivatives by central differences on the same simulation (combiner.predict, fixed seed).

Usage, from the experiment folder:  python -m angle.joint GPU_RESULTS_DIR [REPLICAS]   → results/joint.json
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import combiner, sdt, sources
from .analysis import BASE, LETTERS, _d, _d_combo_classes, _haiid_table, load_machine, logit
from .criteria import FAMILIES
from .measure import _one_pass, _sd_sensitivity
from .summary import _valid, family, independent, real_machine

SEED, REPLICAS, STEP, N_SIM = 534, 2000, 0.05, 250_000
R = Path("results")
PER_TASK = {"chatbench": "chatbench", "alphanli": "alphanli", "haiid_sarcasm": "haiid-text", "haiid_census": "haiid-text",
            "haiid_art": "haiid-images", "haiid_cities": "haiid-images", "haiid_dermatology": "haiid-images"}
JOBS: list = []
PAIRS: list = []
UNIVERSE: dict = {}


def _conversion(dh, dm, m, sd_dh=0.0, per_person=0):
    """measure.conversion_table, kept as arrays."""
    if m == 2:
        return None
    g = np.r_[np.linspace(-0.6, 0.9, 31), 0.93, 0.96, 0.98, 0.99]
    t = np.maximum.accumulate(np.array([sdt.estimate_from_evidence(float(x), dh, dm, m, sd_dh=sd_dh, per_person=per_person) for x in g]))
    return t, g


def _conv(arrays):
    if arrays is None:
        return lambda r, a=None, b=None: r
    t, g = arrays
    return lambda r, a=None, b=None: float(np.interp(r, t, g))


def _md(a, m):
    """measure._d"""
    if a <= 1.0 / m + 1e-9:
        return float("nan")
    return sdt.d_from_accuracy(float(np.clip(a, 1e-4, 1 - 1e-4)), m)


def _votes_step(h, machine, m, answer_class, w):
    """The step of measure.compare_votes."""
    ah = float(np.average(h, weights=w))
    pm = float(np.average(machine, weights=w))
    p11 = float(np.average(h * machine, weights=w))
    r_correct = sdt.tetrachoric(np.clip(ah, 1e-4, 1 - 1e-4), np.clip(pm, 1e-4, 1 - 1e-4), p11)
    dh, dm = _md(ah, m), _md(pm, m)
    if math.isnan(dh) or math.isnan(dm):
        return None
    c1, c0 = answer_class == 1, answer_class == 0
    if w[c1].sum() > 0 and w[c0].sum() > 0:
        dh = sdt.d_from_classes(float(np.average(h[c1], weights=w[c1])), float(np.average(h[c0], weights=w[c0])), n1=float(w[c1].sum()), n0=float(w[c0].sum()))
        dm = sdt.d_from_classes(float(np.average(machine[c1], weights=w[c1])), float(np.average(machine[c0], weights=w[c0])), n1=float(w[c1].sum()), n0=float(w[c0].sum()))
    return {"dh": dh, "dm": dm, "rho_correct": r_correct, "rho": r_correct, "phi": sdt.angle_degrees(r_correct)}


def _slopes(dh, dm, rho, m, levels) -> dict:
    """pred_φ and pred_90 at the point, and their derivatives by central differences (same seed)."""
    f = lambda a, b, r: _d(combiner.predict(a, b, r, m, levels, n=N_SIM), m)
    s = STEP
    return {"pf": f(dh, dm, rho), "p9": f(dh, dm, 0.0),
            "pf_g": [(f(dh + s, dm, rho) - f(dh - s, dm, rho)) / (2 * s), (f(dh, dm + s, rho) - f(dh, dm - s, rho)) / (2 * s),
                     (f(dh, dm, rho + s) - f(dh, dm, rho - s)) / (2 * s)],
            "p9_g": [(f(dh + s, dm, 0.0) - f(dh - s, dm, 0.0)) / (2 * s), (f(dh, dm + s, 0.0) - f(dh, dm - s, 0.0)) / (2 * s)]}


def _chatbench(folder, c, pairs):
    t = load_machine(folder, "chatbench", c["machine"], c["mode"])
    d = pairs[pairs.question_id.isin(t.index)].dropna(subset=["acc", "acc_ai"]).reset_index(drop=True)
    p = sources.chatbench_trials(d, t.ok, None)
    p.with_machine = None
    _, people = sources._index(d.worker_id)
    _, items = sources._index(d.question_id)
    y, mp = p.correct, p.machine[p.item]
    per_person = int(np.median(np.bincount(p.person)[np.bincount(p.person) > 0]))
    conv = _conversion(_md(float(y.mean()), 4), _md(float(mp.mean()), 4), 4, _sd_sensitivity(p, 4), per_person)
    t_all = t.reindex(d.question_id)
    conf = d.confidence.map({"not-confident": 0.0, "somewhat-confident": 1.0, "very-confident": 2.0}).fillna(1.0).to_numpy()
    op_h = d.selected_answer.map(lambda x: LETTERS.index(x) if x in LETTERS else -1).to_numpy()
    lm = logit(t_all.ptop.to_numpy())
    combo = combiner.combine(d.acc.to_numpy(float), t_all.ok.to_numpy(float), op_h, t_all.top.to_numpy(), np.column_stack([conf, lm, conf * lm]),
                             pd.factorize(d.question_id)[0])
    levels = list(np.bincount(conf.astype(int), minlength=3) / conf.size)
    return {"kind": "trials", "task": "chatbench", "m": 4, "p": p, "conv": conv, "people": np.array(people), "items": np.array(items),
            "levels": levels, "outcome": combo["outcome"], "row_people": d.worker_id.to_numpy(), "row_items": d.question_id.to_numpy(),
            "row_base": np.ones(len(d)), "combo": "accuracy"}


def _alphanli(folder, c):
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    t = load_machine(folder, "alphanli", c["machine"], c["mode"])
    items = items.join(t[["ok", "ptop", "top"]], how="inner")
    n_correct, n_total = items.votes_correct.to_numpy(int), items.votes.to_numpy(int)
    correct_option, top = items.correct_option.to_numpy(int), items.top.to_numpy(int)
    lm = logit(items.ptop.to_numpy())
    y_h = np.r_[np.ones(n_correct.size), np.zeros(n_correct.size)]
    y_m = np.r_[items.ok.to_numpy(float), items.ok.to_numpy(float)]
    weights = np.r_[n_correct, n_total - n_correct].astype(float)
    keep = weights > 0
    groups = np.r_[np.arange(n_correct.size), np.arange(n_correct.size)]
    sign = 2.0 * np.r_[top, top] - 1.0
    X = np.column_stack([np.r_[lm, lm], sign, np.r_[lm, lm] * sign])
    combo = combiner.combine(y_h[keep], y_m[keep], np.r_[correct_option, 1 - correct_option][keep], np.r_[top, top][keep], X[keep],
                             groups[keep], weights=weights[keep])
    ids = items.index.to_numpy()
    return {"kind": "votes", "task": "alphanli", "m": 2, "h": (items.votes_correct / items.votes).to_numpy(float),
            "machine": items.ok.to_numpy(float), "answer_class": correct_option, "items": ids, "levels": None,
            "outcome": combo["outcome"], "row_items": np.r_[ids, ids][keep], "row_people": None, "row_base": weights[keep],
            "classes": np.r_[correct_option, correct_option][keep], "groups": groups[keep], "combo": "classes"}


def _haiid(folder, c):
    task = c["task"]
    t = load_machine(folder, f"haiid_{task}", c["machine"], c["mode"])
    p = sources.haiid_trials(BASE, task, t.ok)
    d = _haiid_table(task)
    pc, up = sources._index(d.participant_id)
    cc, ui = sources._index(d.id)
    keep = t.ok.reindex(ui).notna().to_numpy()[cc]
    _, pu = pd.factorize(pc[keep], sort=True)
    _, iu = pd.factorize(cc[keep], sort=True)
    dl = d[d.id.isin(t.index)]
    t_r = t.reindex(dl.id)
    y_h = (dl.response_1 > 0).to_numpy(float)
    correct_option = t_r.correct_option.to_numpy(int)
    op_m = t_r.top.to_numpy(int)
    conf = np.abs(dl.response_1.to_numpy(float))
    lm = logit(t_r.ptop.to_numpy())
    sign = 2.0 * op_m - 1.0
    combo = combiner.combine(y_h, t_r.ok.to_numpy(float), np.where(y_h > 0, correct_option, 1 - correct_option), op_m,
                             np.column_stack([conf, lm, conf * lm, sign, lm * sign]), pd.factorize(dl.id)[0])
    return {"kind": "trials", "task": f"haiid_{task}", "m": 2, "p": p, "conv": None, "people": np.array(up, dtype=object)[pu],
            "items": np.array(ui, dtype=object)[iu], "levels": None, "outcome": combo["outcome"],
            "row_people": dl.participant_id.to_numpy(), "row_items": dl.id.to_numpy(), "row_base": np.ones(len(dl)),
            "classes": correct_option, "groups": pd.factorize(dl.id)[0], "combo": "classes"}


def _index_in(universe: np.ndarray, values) -> np.ndarray:
    idx = pd.Index(universe).get_indexer(pd.Index(values))
    assert (idx >= 0).all()
    return idx


def build(folder: Path) -> None:
    map_data = json.loads((R / "map.json").read_text())
    comps = [c for c in _valid(map_data["comparisons"]) if independent(c) and real_machine(c)]
    pairs = sources.chatbench_pairs(BASE)
    for c in comps:
        j = _chatbench(folder, c, pairs) if c["source"] == "chatbench" else _alphanli(folder, c) if c["source"] == "alphanli" else _haiid(folder, c)
        j.update(family=family(c), label=f'{c["task"]}/{c["machine"]}/{c["mode"]}', ref=c)
        JOBS.append(j)
    ids: dict = {}
    for j in JOBS:
        if j["kind"] == "trials":
            ids.setdefault((j["task"], "people"), set()).update(j["people"].tolist())
        ids.setdefault((j["task"], "items"), set()).update(j["items"].tolist())
    for x in map_data["machine_vs_machine"]:
        f = PER_TASK.get(x["task"])
        if f is None or x["a"].split("/")[0] == x["b"].split("/")[0] or x["rho_correct"] != x["rho_correct"]:
            continue
        a = load_machine(folder, x["task"], *x["a"].split("/")).ok
        b = load_machine(folder, x["task"], *x["b"].split("/")).ok
        idx = a.index.intersection(b.index)
        key = "chatbench" if x["task"] == "chatbench" else x["task"]
        ids.setdefault((key, "items"), set()).update(idx.tolist())
        PAIRS.append({"family": f, "task": key, "a": a.loc[idx].to_numpy(), "b": b.loc[idx].to_numpy(), "items": idx.to_numpy(), "ref": x["rho_correct"]})
    for k in sorted(ids):
        UNIVERSE[k] = np.array(sorted(ids[k]), dtype=object)
    for j in JOBS:
        j["pi"] = _index_in(UNIVERSE[(j["task"], "people")], j["people"]) if j["kind"] == "trials" else None
        j["ii"] = _index_in(UNIVERSE[(j["task"], "items")], j["items"])
        j["rp"] = _index_in(UNIVERSE[(j["task"], "people")], j["row_people"]) if j["row_people"] is not None else None
        j["ri"] = _index_in(UNIVERSE[(j["task"], "items")], j["row_items"])
    for q in PAIRS:
        q["ii"] = _index_in(UNIVERSE[(q["task"], "items")], q["items"])


def _weights(r: int) -> dict:
    gen = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(r,)))      # the r-th child of seed 534
    return {k: gen.poisson(1.0, u.size).astype(float) for k, u in UNIVERSE.items()}


def _one(j, W) -> dict | None:
    """One comparison in one replica. A comparison that falls to or below chance in a replica is left out of that
    replica only, and counted (`failed_cells`), as measure.compare already does with its own resamples (`_one_pass`
    returns None). A negative sensitivity raises in `sdt.ideal`, and here counts the same way."""
    try:
        return _one_cell(j, W)
    except ValueError as e:
        if "sensitivit" in str(e):
            return None
        raise


def _one_cell(j, W) -> dict | None:
    if j["kind"] == "trials":
        out = _one_pass(j["p"], j["m"], W[(j["task"], "people")][j["pi"]], W[(j["task"], "items")][j["ii"]], _conv(j["conv"]))
        if out is None:
            return None
        out = {k: out[k] for k in ("dh", "dm", "rho", "rho_correct", "phi")}
        row_w = j["row_base"] * W[(j["task"], "people")][j["rp"]] * W[(j["task"], "items")][j["ri"]]
    else:
        w = W[(j["task"], "items")][j["ii"]]
        if w.sum() == 0:
            return None
        out = _votes_step(j["h"], j["machine"], j["m"], j["answer_class"], w)
        if out is None:
            return None
        row_w = j["row_base"] * W[(j["task"], "items")][j["ri"]]
    if row_w.sum() == 0:
        return None
    if j["combo"] == "accuracy":
        out["d_combo"] = _d(float(np.average(j["outcome"], weights=row_w)), j["m"])
    else:
        c1, c0 = j["classes"] == 1, j["classes"] == 0
        if row_w[c1].sum() == 0 or row_w[c0].sum() == 0:
            return None
        out["d_combo"] = _d_combo_classes(j["outcome"], j["classes"], j["groups"], row_w)
    return out


def _pair_rho(q, W) -> float:
    w = W[(q["task"], "items")][q["ii"]]
    if w.sum() == 0:
        return float("nan")
    p1, p2 = np.average(q["a"], weights=w), np.average(q["b"], weights=w)
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return float("nan")
    return sdt.tetrachoric(float(p1), float(p2), float(np.average(q["a"] * q["b"], weights=w)))


def _replica(r: int) -> dict:
    W = _weights(r)
    rows = [_one(j, W) for j in JOBS]
    keys = ("dh", "dm", "rho", "rho_correct", "phi", "d_combo")
    return {"r": r, **{k: [np.nan if x is None else x[k] for x in rows] for k in keys}, "pairs": [_pair_rho(q, W) for q in PAIRS]}


def _ci(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main(argv=None) -> None:
    global REPLICAS
    argv = sys.argv[1:] if argv is None else argv
    folder = Path(argv[0])
    if len(argv) > 1:
        REPLICAS = int(argv[1])
    t0 = time.time()
    build(folder)
    print(len(JOBS), "comparisons,", len(PAIRS), "pairs of machines, built in %.0f s" % (time.time() - t0), flush=True)
    # the point must give the map back
    W1 = {k: np.ones(u.size) for k, u in UNIVERSE.items()}
    for j in JOBS:
        pt = _one(j, W1)
        ref = j["ref"]
        for k in ("phi", "dh", "dm", "rho"):
            assert abs(pt[k] - ref[k]) < 1e-6, (j["label"], k, pt[k], ref[k])
        assert abs(pt["d_combo"] - ref["law"]["d_combo"]) < 1e-6, (j["label"], pt["d_combo"], ref["law"]["d_combo"])
        j["point"] = pt
        j["slopes"] = _slopes(pt["dh"], pt["dm"], pt["rho"], j["m"], j["levels"])
        assert abs(j["slopes"]["pf"] - ref["law"]["pred_phi"]) < 1e-6 and abs(j["slopes"]["p9"] - ref["law"]["pred_90"]) < 1e-6, j["label"]
    for q in PAIRS:
        assert abs(_pair_rho(q, W1) - q["ref"]) < 1e-9
    print("point checked against map.json, slopes in %.0f s" % (time.time() - t0), flush=True)
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("fork")) as ex:   # the workers inherit JOBS
        reps = []
        for i, x in enumerate(ex.map(_replica, range(REPLICAS), chunksize=10)):
            reps.append(x)
            if (i + 1) % 100 == 0:
                print(i + 1, "replicas, %.0f s" % (time.time() - t0), flush=True)
    A = {k: np.array([x[k] for x in reps], float) for k in ("dh", "dm", "rho", "rho_correct", "phi", "d_combo", "pairs")}
    fam = np.array([j["family"] for j in JOBS])
    pfam = np.array([q["family"] for q in PAIRS])
    point = {k: np.array([j["point"][k] for j in JOBS]) for k in ("phi", "rho_correct", "d_combo", "dh", "dm", "rho")}
    out = {"protocol": "PROTOCOL.md, After the fourth review (d75500f)", "seed": SEED, "replicas": REPLICAS, "comparisons": len(JOBS),
           "machine_pairs": len(PAIRS), "failed_cells": int(np.isnan(A["phi"]).sum()), "h1": {}, "h2": {}, "h3": {}}
    for f in FAMILIES:
        s = fam == f
        if not s.any():
            continue
        med = np.nanmedian(A["phi"][:, s], axis=1)
        ci = _ci(med)
        out["h1"][f] = {"comparisons": int(s.sum()), "phi_median": float(np.median(point["phi"][s])), "phi_median_ci": ci, "family_passes": bool(ci[1] < 90)}
        ps = pfam == f
        if ps.any():
            diff = np.nanmedian(A["pairs"][:, ps], axis=1) - np.nanmedian(A["rho_correct"][:, s], axis=1)
            ref = np.array([q["ref"] for q in PAIRS])[ps]
            ci2 = _ci(diff)
            out["h2"][f] = {"difference": float(np.median(ref) - np.median(point["rho_correct"][s])), "difference_ci": ci2,
                            "machine_machine_median": float(np.median(ref)), "person_machine_median": float(np.median(point["rho_correct"][s])),
                            "pairs": int(ps.sum()), "comparisons": int(s.sum()), "passes": bool(ci2[0] > 0)}
    # H3: the predictions follow d_H, d_M and ρ of the replica, to first order
    pf = np.array([j["slopes"]["pf"] for j in JOBS])
    p9 = np.array([j["slopes"]["p9"] for j in JOBS])
    gf = np.array([j["slopes"]["pf_g"] for j in JOBS])
    g9 = np.array([j["slopes"]["p9_g"] for j in JOBS])
    ddh, ddm, drho = A["dh"] - point["dh"], A["dm"] - point["dm"], A["rho"] - point["rho"]
    pf_r = pf + gf[:, 0] * ddh + gf[:, 1] * ddm + gf[:, 2] * drho
    p9_r = p9 + g9[:, 0] * ddh + g9[:, 1] * ddm
    err_r = np.abs(A["d_combo"] - pf_r) - np.abs(A["d_combo"] - p9_r)
    err_0 = np.abs(point["d_combo"] - pf) - np.abs(point["d_combo"] - p9)
    sets = {"all": np.ones(len(JOBS), bool), "without_haiid": np.array([not f.startswith("haiid") for f in fam])}
    sets.update({f: fam == f for f in FAMILIES if (fam == f).any()})
    for name, s in sets.items():
        ci = _ci(np.nanmean(err_r[:, s], axis=1))
        out["h3"][name] = {"comparisons": int(s.sum()), "mean_difference": float(err_0[s].mean()), "mean_difference_ci": ci,
                           "better_phi": int((err_0[s] < 0).sum()), "passes": bool(ci[1] < 0)}
    failed = np.isnan(A["phi"]).sum(axis=0)
    out["per_comparison"] = [{"label": j["label"], "family": j["family"], "phi": float(point["phi"][i]), "phi_ci": _ci(A["phi"][:, i]),
                              "failed_replicas": int(failed[i]), "slopes": j["slopes"]} for i, j in enumerate(JOBS)]
    out["failed_by_family"] = {f: {"cells": int(failed[fam == f].sum()), "comparisons_that_fail": int((failed[fam == f] > 0).sum()),
                                   "comparisons": int((fam == f).sum())} for f in FAMILIES if (fam == f).any()}
    # added after the protocol: the same verdicts with only the comparisons that never fall below chance in a replica
    # (all of HAIID falls there sometimes, with 32 items per task: the check covers ChatBench and αNLI)
    keep = failed == 0
    out["never_failing"] = {"h1": {}, "h3": {}}
    for f in FAMILIES:
        s = (fam == f) & keep
        if s.any():
            ci = _ci(np.nanmedian(A["phi"][:, s], axis=1))
            out["never_failing"]["h1"][f] = {"comparisons": int(s.sum()), "phi_median": float(np.median(point["phi"][s])), "phi_median_ci": ci,
                                             "family_passes": bool(ci[1] < 90)}
    out["never_failing"]["h2"] = {}
    for f in FAMILIES:
        s, ps = (fam == f) & keep, pfam == f
        if s.any() and ps.any():
            ci2 = _ci(np.nanmedian(A["pairs"][:, ps], axis=1) - np.nanmedian(A["rho_correct"][:, s], axis=1))
            out["never_failing"]["h2"][f] = {"comparisons": int(s.sum()), "difference_ci": ci2, "passes": bool(ci2[0] > 0)}
    for name, s0 in sets.items():
        s = s0 & keep
        if s.any():
            ci = _ci(np.nanmean(err_r[:, s], axis=1))
            out["never_failing"]["h3"][name] = {"comparisons": int(s.sum()), "mean_difference": float(err_0[s].mean()), "mean_difference_ci": ci,
                                                "passes": bool(ci[1] < 0)}
    out["seconds"] = round(time.time() - t0)
    (R / "joint.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k in ("h1", "h2", "h3"):
        for f, v in out[k].items():
            print(k, f, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
