"""One row per person-machine comparison: all the quantities and the intervals.

The bootstrap resamples at two levels (items and people), with Poisson(1) weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from . import sdt

_Z_GRID = np.linspace(-3.6, 3.6, 361)


@dataclass
class Trials:
    """Aligned trials of people alone, one row per answer."""
    person: np.ndarray    # index 0..P-1
    item: np.ndarray      # index 0..C-1
    correct: np.ndarray   # 0/1
    machine: np.ndarray   # per item (0..C-1), fraction of correct answers; or per trial if machine_per_trial
    with_machine: np.ndarray | None = None   # same order: the answer after seeing the machine
    machine_per_trial: bool = False
    answer_class: np.ndarray | None = None   # index of the class of the correct answer (ρ is computed per class)


def _d(a: float, m: int) -> float:
    if a <= 1.0 / m + 1e-9:
        return float("nan")
    return sdt.d_from_accuracy(float(np.clip(a, 1e-4, 1 - 1e-4)), m)


def _fast_rho(y, zp, zm, m_c, w):
    """Like sdt.rho_personal_thresholds, with the thresholds already computed and a grid of z."""
    observed = np.sum(w * y * m_c) / np.sum(w)
    idx = np.clip(np.round((zp - _Z_GRID[0]) / (_Z_GRID[1] - _Z_GRID[0])).astype(int), 0, _Z_GRID.size - 1)
    weight = np.bincount(idx, weights=w, minlength=_Z_GRID.size)
    alive = weight > 0
    expected = (sdt.bvn(_Z_GRID[alive][None, :], zm, sdt.RHO_GRID[:, None]) * weight[alive]).sum(1) / weight.sum()
    return float(np.interp(observed, expected, sdt.RHO_GRID))


def _thresholds(p: Trials, w):
    k = np.bincount(p.person, weights=w * p.correct)
    n = np.bincount(p.person, weights=w)
    a = np.zeros(k.size)
    alive = n > 0
    a[alive] = sdt.shrink(k[alive], n[alive])
    return norm.ppf(np.clip(a[p.person], 1e-4, 1 - 1e-4))


def _machine_per_trial(p: Trials):
    return p.machine if p.machine_per_trial else p.machine[p.item]


def _one_pass(p: Trials, m: int, wp: np.ndarray, wc: np.ndarray, conv):
    w = wp[p.person] * wc[p.item]
    if w.sum() == 0:
        return None
    y = p.correct
    mp = _machine_per_trial(p)
    acc_h = float(np.sum(w * y) / np.sum(w))
    pm = float(np.sum(w * mp) / np.sum(w))
    zp = _thresholds(p, w)
    if p.answer_class is None:
        zm = norm.ppf(np.clip(pm, 1e-4, 1 - 1e-4))
        r_correct = _fast_rho(y, zp, zm, mp, w)
    else:
        total, weight = 0.0, 0.0
        for cl in np.unique(p.answer_class):
            s = p.answer_class == cl
            ws = w[s]
            if ws.sum() <= 0:
                continue
            q = float(np.sum(ws * mp[s]) / ws.sum())
            if not 0.0 < q < 1.0:
                continue
            r = _fast_rho(y[s], _thresholds_in_class(p, w, s), norm.ppf(q), mp[s], ws)
            total += r * ws.sum()
            weight += ws.sum()
        if weight == 0:
            return None
        r_correct = total / weight
    dh, dm = _d(acc_h, m), _d(pm, m)
    if math.isnan(dh) or math.isnan(dm):
        return None
    two_classes = m == 2 and p.answer_class is not None and np.unique(p.answer_class).size == 2
    if two_classes:
        dh, dm = _d_two_classes(p, y, mp, w)
    rho = float(conv(r_correct, dh, dm))
    out = {"acc_h": acc_h, "acc_m": pm, "dh": dh, "dm": dm, "rho_correct": r_correct, "rho": rho,
           "ideal_90": sdt.ideal(dh, dm, 0.0), "ideal_phi": sdt.ideal(dh, dm, rho),
           "phi": sdt.angle_degrees(rho), "useful_angle": sdt.useful_angle(dh, dm),
           "k": dm / dh}
    if p.with_machine is not None:
        acc_s = float(np.sum(w * p.with_machine) / np.sum(w))
        ds = _d(acc_s, m)
        if not math.isnan(ds) and two_classes:
            # the same convention as person and machine: free criterion, class by class
            c1, c0 = p.answer_class == 1, p.answer_class == 0
            ds = sdt.d_from_classes(float(np.sum(w[c1] * p.with_machine[c1]) / w[c1].sum()),
                                    float(np.sum(w[c0] * p.with_machine[c0]) / w[c0].sum()),
                                    n1=float(w[c1].sum()), n0=float(w[c0].sum()))
        if not math.isnan(ds):
            out.update({"acc_s": acc_s, "ds": ds, "turing": ds / dh, "yield_on_machine": ds / dm,
                        "g_90": sdt.gain_share(ds, dh, out["ideal_90"]), "g_phi": sdt.gain_share(ds, dh, out["ideal_phi"]),
                        "eta_90": ds / out["ideal_90"], "eta_phi": ds / out["ideal_phi"]})
    return out


def _d_two_classes(p: Trials, y, mp, w):
    """d′ of person and machine with the free criterion, class by class."""
    c1, c0 = p.answer_class == 1, p.answer_class == 0
    def rate(x, s):
        return float(np.sum(w[s] * x[s]) / w[s].sum())
    dh = sdt.d_from_classes(rate(y, c1), rate(y, c0), n1=float(w[c1].sum()), n0=float(w[c0].sum()))
    if p.machine_per_trial:               # the advice changes from trial to trial: counted per trial
        dm = sdt.d_from_classes(rate(mp, c1), rate(mp, c0), n1=float(w[c1].sum()), n0=float(w[c0].sum()))
    else:
        dm = sdt.d_from_classes(rate(mp, c1), rate(mp, c0), n1=float(np.unique(p.item[c1]).size), n0=float(np.unique(p.item[c0]).size))
    return dh, dm


def _sd_sensitivity(p: Trials, m: int) -> float:
    """How much the sensitivities vary between people, net of the binomial noise."""
    k = np.bincount(p.person, weights=p.correct).astype(float)
    n = np.bincount(p.person).astype(float)
    keep = n > 0
    a = k[keep] / n[keep]
    mean = np.average(a, weights=n[keep])
    var = np.average((a - mean) ** 2, weights=n[keep]) - np.average(mean * (1 - mean) / n[keep], weights=n[keep])
    sd = math.sqrt(max(var, 0.0))
    if sd == 0.0:
        return 0.0
    lo = max(mean - sd, 1.0 / m + 0.005)
    hi = min(mean + sd, 0.999)
    return (sdt.d_from_accuracy(hi, m) - sdt.d_from_accuracy(lo, m)) / 2.0


def _thresholds_in_class(p: Trials, w, s):
    """The personal threshold within a class: the person's accuracy on that class."""
    people_s = p.person[s]
    k = np.bincount(people_s, weights=w[s] * p.correct[s], minlength=p.person.max() + 1)
    n = np.bincount(people_s, weights=w[s], minlength=p.person.max() + 1)
    a = np.zeros(k.size)
    alive = n > 0
    a[alive] = sdt.shrink(k[alive], n[alive])
    return norm.ppf(np.clip(a[people_s], 1e-4, 1 - 1e-4))


def conversion_table(dh, dm, m, sd_dh=0.0, per_person=0):
    """The conversion estimated ρ -> evidence ρ, fixed for the comparison's sensitivities and their dispersion."""
    if m == 2:
        return lambda r, a=None, b=None: r
    g = np.r_[np.linspace(-0.6, 0.9, 31), 0.93, 0.96, 0.98, 0.99]
    t = np.array([sdt.estimate_from_evidence(float(x), dh, dm, m, sd_dh=sd_dh, per_person=per_person) for x in g])
    t = np.maximum.accumulate(t)
    return lambda r, a=None, b=None: float(np.interp(r, t, g))


def _summarize(point: dict, runs: list, base: dict) -> dict:
    out = dict(base)
    for key, v in point.items():
        out[key] = v
        if v is None:
            continue
        sample = np.array([x[key] for x in runs if x.get(key) is not None], float)
        if sample.size:
            out[key + "_ci"] = [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]
    return out


def compare(p: Trials, m: int, repetitions: int = 1000, seed: int = 534) -> dict:
    P = int(p.person.max()) + 1
    C = int(p.item.max()) + 1
    y = p.correct
    mp = _machine_per_trial(p)
    if y.mean() <= 1 / m or mp.mean() <= 1 / m:
        return {"excluded": "one of the two at or below chance level", "n_trials": int(y.size),
                "acc_h": float(y.mean()), "acc_m": float(mp.mean())}
    per_person = int(np.median(np.bincount(p.person)[np.bincount(p.person) > 0]))
    conv = conversion_table(_d(float(y.mean()), m), _d(float(mp.mean()), m), m, _sd_sensitivity(p, m) if m > 2 else 0.0, per_person)
    point = _one_pass(p, m, np.ones(P), np.ones(C), conv)
    if point is None:
        return {"excluded": "estimate failed", "n_trials": int(y.size)}
    gen = np.random.default_rng(seed)
    runs = []
    for _ in range(repetitions):
        r = _one_pass(p, m, gen.poisson(1.0, P).astype(float), gen.poisson(1.0, C).astype(float), conv)
        if r is not None:
            runs.append(r)
    return _summarize(point, runs, {"n_trials": int(y.size), "people": P, "items": C, "m": m, "repetitions": len(runs)})


def compare_votes(h: np.ndarray, n: np.ndarray, machine: np.ndarray, m: int,
                  repetitions: int = 1000, seed: int = 534, answer_class: np.ndarray | None = None) -> dict:
    """When only the votes per item are known (αNLI): no personal threshold, the items are resampled."""
    def step(w):
        ah = float(np.average(h, weights=w))
        pm = float(np.average(machine, weights=w))
        p11 = float(np.average(h * machine, weights=w))
        r_correct = sdt.tetrachoric(np.clip(ah, 1e-4, 1 - 1e-4), np.clip(pm, 1e-4, 1 - 1e-4), p11)
        dh, dm = _d(ah, m), _d(pm, m)
        if math.isnan(dh) or math.isnan(dm):
            return None
        if m == 2 and answer_class is not None:      # the machine's criterion may lean toward one of the two options
            c1, c0 = answer_class == 1, answer_class == 0
            if w[c1].sum() > 0 and w[c0].sum() > 0:
                dh = sdt.d_from_classes(float(np.average(h[c1], weights=w[c1])), float(np.average(h[c0], weights=w[c0])), n1=float(w[c1].sum()), n0=float(w[c0].sum()))
                dm = sdt.d_from_classes(float(np.average(machine[c1], weights=w[c1])), float(np.average(machine[c0], weights=w[c0])), n1=float(w[c1].sum()), n0=float(w[c0].sum()))
        return {"acc_h": ah, "acc_m": pm, "dh": dh, "dm": dm, "rho": r_correct, "rho_correct": r_correct,
                "ideal_90": sdt.ideal(dh, dm, 0.0), "ideal_phi": sdt.ideal(dh, dm, r_correct),
                "phi": sdt.angle_degrees(r_correct), "useful_angle": sdt.useful_angle(dh, dm), "k": dm / dh}
    point = step(np.ones(h.size))
    if point is None:
        return {"excluded": "one of the two at or below chance level", "acc_h": float(h.mean()), "acc_m": float(machine.mean())}
    gen = np.random.default_rng(seed)
    runs = [r for r in (step(gen.poisson(1.0, h.size).astype(float)) for _ in range(repetitions)) if r is not None]
    return _summarize(point, runs, {"n_trials": int(n.sum()), "items": h.size, "m": m, "repetitions": len(runs)})
