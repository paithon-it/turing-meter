"""The test of the law: a combiner that uses only what the pair has, and the simulated prediction.

On the items where person and machine give different answers, a logistic regression
(cross-validation by item) decides whom to believe, using the person's confidence and the
machine's probability. From there the sensitivity reached is computed. The prediction does
the same on data simulated from the model, with the measured ρ or with ρ = 0: the
difference says whether φ improves the prediction.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import sdt


def _fit(X, y, weights=None):
    if len(np.unique(y)) < 2:
        return None
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    if weights is None:
        m.fit(X, y)
    else:
        m.fit(X, y, logisticregression__sample_weight=weights)
    return m


def choose_with_machine(X: np.ndarray, follow_true: np.ndarray, groups: np.ndarray, use: np.ndarray, weights=None) -> np.ndarray:
    """Out-of-sample probability that following the machine pays off, only on the rows `use`."""
    out_of_sample = np.zeros(X.shape[0])
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, max(2, n_groups)))
    indices = np.arange(X.shape[0])
    for tr, te in gkf.split(indices, groups=groups):
        train = tr[use[tr]]
        m = _fit(X[train], follow_true[train], None if weights is None else weights[train]) if train.size else None
        if m is None:
            out_of_sample[te] = follow_true[train].mean() if train.size else 0.5
        else:
            out_of_sample[te] = m.predict_proba(X[te])[:, 1]
    return out_of_sample


def combine(y_h, y_m, opt_h, opt_m, X_decide, items, weights=None) -> dict:
    """y_* = correct answer; opt_* = the option chosen. What is left is the accuracy of the pair."""
    y_h, y_m = np.asarray(y_h, float), np.asarray(y_m, float)
    same = np.asarray(opt_h) == np.asarray(opt_m)
    different = ~same
    follow = y_m.copy()                         # following the machine pays off if it is the one that is right
    informative = different & (y_h != y_m)
    p = choose_with_machine(X_decide, follow, items, informative, weights)
    machine_choice = p > 0.5
    outcome = np.where(same, y_h, np.where(machine_choice, y_m, y_h))
    w = np.ones(outcome.size) if weights is None else np.asarray(weights, float)
    return {"acc_pair": float(np.average(outcome, weights=w)), "acc_h": float(np.average(y_h, weights=w)),
            "acc_m": float(np.average(y_m, weights=w)),
            "follow_share": float(np.average(machine_choice[different], weights=w[different])) if different.any() else float("nan"),
            "disagreement": float(np.average(different, weights=w)), "outcome": outcome}


def margin(combo: dict, y_h, y_m, items, weights=None, repetitions: int = 1000, seed: int = 534, people=None) -> dict:
    """By how much the pair beats the better of the two alone, with the interval resampling the items and, if present, the people."""
    outcome, y_h, y_m = np.asarray(combo["outcome"], float), np.asarray(y_h, float), np.asarray(y_m, float)
    w = np.ones(outcome.size) if weights is None else np.asarray(weights, float)
    idx = np.unique(items, return_inverse=True)[1]
    k = idx.max() + 1
    sums = [np.bincount(idx, weights=w * v, minlength=k) for v in (outcome, y_h, y_m, np.ones(outcome.size))]
    gen = np.random.default_rng(seed)
    d = np.empty(repetitions)
    if people is None:
        for i in range(repetitions):
            p = gen.poisson(1.0, k)
            tot = (p * sums[3]).sum()
            d[i] = (p * sums[0]).sum() / tot - max((p * sums[1]).sum(), (p * sums[2]).sum()) / tot
    else:
        person_idx = np.unique(people, return_inverse=True)[1]
        for i in range(repetitions):
            v = w * gen.poisson(1.0, k)[idx] * gen.poisson(1.0, person_idx.max() + 1)[person_idx]
            tot = v.sum()
            d[i] = ((v * outcome).sum() - max((v * y_h).sum(), (v * y_m).sum())) / tot if tot > 0 else np.nan
        d = d[~np.isnan(d)]
    point = float(np.average(outcome, weights=w) - max(np.average(y_h, weights=w), np.average(y_m, weights=w)))
    return {"over_best": point, "over_best_ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}


def predict(dh: float, dm: float, rho: float, m: int, confidence_levels: list[float] | None,
            n: int = 300_000, seed: int = 534) -> float:
    """Accuracy an identical combiner would reach, under the model, with the given ρ and sensitivities."""
    gen = np.random.default_rng(seed)
    if m == 2:                 # d′ here is in the yes/no convention (PC = Φ(d/2)): the two-option simulation needs d/√2
        dh, dm = dh / np.sqrt(2.0), dm / np.sqrt(2.0)
    k = max(-0.95, min(0.95, rho))
    common = gen.standard_normal((n, m))
    w = np.sqrt(abs(k))
    sign = 1.0 if k >= 0 else -1.0
    a = w * common + np.sqrt(1 - abs(k)) * gen.standard_normal((n, m))
    b = sign * w * common + np.sqrt(1 - abs(k)) * gen.standard_normal((n, m))
    if abs(k) < 1e-6:
        a, b = gen.standard_normal((n, m)), gen.standard_normal((n, m))
    a[:, 0] += dh
    b[:, 0] += dm
    oh = a.argmax(1)
    om = b.argmax(1)
    y_h = (oh == 0).astype(float)
    y_m = (om == 0).astype(float)
    pa = np.sort(a, 1)
    marg_h = pa[:, -1] - pa[:, -2]
    pb = np.sort(b, 1)
    marg_m = pb[:, -1] - pb[:, -2]
    if confidence_levels:                     # the person gives only a few levels of confidence
        thresholds = np.quantile(marg_h, np.cumsum(confidence_levels)[:-1])
        s = np.digitize(marg_h, thresholds).astype(float)
    else:
        s = np.zeros(n)
    X = np.column_stack([s, marg_m, marg_m * s])
    items = np.arange(n) // 50
    r = combine(y_h, y_m, oh, om, X, items)
    return r["acc_pair"]
