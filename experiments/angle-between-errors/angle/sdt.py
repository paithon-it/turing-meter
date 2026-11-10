"""The article's quantities on the d′ scale, and the estimators of the angle between errors.

Pure functions. Model: signal detection, Gaussian noise with equal variance. The
definitions also hold for choices among several options; the estimates of the angle from
many people with few items each are new.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import owens_t
from scipy.stats import norm

RHO_GRID = np.r_[np.arange(-0.9, 0.9, 0.01), np.arange(0.9, 0.9951, 0.005)]  # dense where the curve is steep
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(80)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


# --- sensitivity ----------------------------------------------------------

def d_from_accuracy(a: float, m: int = 2) -> float:
    """d′ from a proportion of correct answers with m options.

    m = 2 uses the yes/no convention, 2 z(a); from m ≥ 3 the forced choice. Ratios
    between sensitivities do not change if all sit on the same convention.
    """
    if not 0.0 < a < 1.0:
        raise ValueError(f"accuracy outside (0, 1): {a!r}")
    if m == 2:
        return float(2.0 * norm.ppf(a))
    if a <= 1.0 / m:
        raise ValueError("accuracy at chance level or below")
    return float(brentq(lambda d: pc_mafc(d, m) - a, 1e-6, 12.0))


def d_from_classes(p1: float, p0: float, n1: float | None = None, n0: float | None = None) -> float:
    """Yes/no d′ with free criterion: z(hit) − z(fa), that is z(p1) + z(p0).

    p1 and p0 are the correct answers on the two classes. With n1 and n0 the log-linear
    correction of ½ is applied, which keeps the case 0 or 1 finite.
    """
    if n1 is not None:
        p1 = (p1 * n1 + 0.5) / (n1 + 1.0)
    if n0 is not None:
        p0 = (p0 * n0 + 0.5) / (n0 + 1.0)
    return float(norm.ppf(np.clip(p1, 1e-4, 1 - 1e-4)) + norm.ppf(np.clip(p0, 1e-4, 1 - 1e-4)))


def pc_mafc(d: float, m: int) -> float:
    """Proportion of correct answers of an ideal observer among m options."""
    return float(np.sum(_WEIGHTS * norm.cdf(_NODES + d) ** (m - 1)))


# --- ideal, useful angle ----------------------------------------------------

def ideal(dh: float, dm: float, rho: float = 0.0) -> float:
    """Ideal of the pair at full evidence, with correlated errors ρ; never negative weights."""
    if dh < 0 or dm < 0:
        raise ValueError("negative sensitivities")
    if abs(rho) >= 1.0 - 1e-12:
        return max(dh, dm)
    if dh - rho * dm < 0 or dm - rho * dh < 0:
        return max(dh, dm)
    return math.sqrt((dh * dh + dm * dm - 2.0 * rho * dh * dm) / (1.0 - rho * rho))


def useful_angle(dh: float, dm: float) -> float:
    """Minimum angle between the errors for the weaker one to add something to the ideal."""
    if dh <= 0 or dm <= 0:
        raise ValueError("two sensitivities above chance needed")
    return math.degrees(math.acos(min(dh, dm) / max(dh, dm)))


def gain_share(ds: float, dh: float, pair_ideal: float) -> float | None:
    """g: the share of the possible gain that was realized."""
    space = pair_ideal - dh
    return None if space <= 1e-9 else (ds - dh) / space


def apparent_turings(dh: float, dm: float, rho: float = 0.0) -> float:
    """|S| / dH: what the two readings promise, in the unit of the person.

    It depends only on k = dM/dH and on ρ: sqrt(1 + k² − 2ρk) / sqrt(1 − ρ²), stopped at the
    stronger of the two below the useful angle. The measured turing is η times this.
    """
    if dh <= 0:
        raise ValueError("a person above chance is needed")
    return ideal(dh, dm, rho) / dh


def angle_degrees(rho: float) -> float:
    return math.degrees(math.acos(float(np.clip(rho, -1.0, 1.0))))


# --- bivariate normal --------------------------------------------------------

def bvn(h, k, r):
    """P(X < h, Y < k) for standard normals with correlation r (Owen 1956)."""
    h, k, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (h, k, r)))
    h = np.where(h == 0, 1e-12, h)
    k = np.where(k == 0, 1e-12, k)
    s = np.sqrt((1.0 - r) * (1.0 + r))
    beta = np.where(h * k > 0, 0.0, 0.5)
    return (0.5 * norm.cdf(h) + 0.5 * norm.cdf(k)
            - owens_t(h, (k - r * h) / (h * s)) - owens_t(k, (h - r * k) / (k * s)) - beta)


def tetrachoric(p1: float, p2: float, p11: float) -> float:
    """Latent correlation of two binary variables with proportions p1, p2 and joint p11."""
    z1, z2 = norm.ppf(p1), norm.ppf(p2)
    expected = bvn(z1, z2, RHO_GRID)
    return float(np.interp(p11, expected, RHO_GRID))


# --- personal thresholds ---------------------------------------------------------

def shrink(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Accuracy of each one, shrunk toward the mean with a beta prior estimated from the data.

    With ten answers each, the raw accuracy is 0 or 1 for many, and its dispersion is
    largely noise: using it as a threshold pushes ρ toward zero. The prior is obtained
    with the method of moments on the variance between people, net of the binomial.
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    mean = np.average(p, weights=n)
    var_tot = np.average((p - mean) ** 2, weights=n)
    noise = np.average(mean * (1 - mean) / n, weights=n)
    var_true = max(var_tot - noise, 1e-5)
    strength = max(mean * (1 - mean) / var_true - 1.0, 1.0)
    return (k + strength * mean) / (n + strength)


def rho_personal_thresholds(people, items, correct, machine_correct,
                            person_weights=None, item_weights=None) -> float:
    """ρ between the evidence of the person and of the machine, each with its own threshold.

    `people`, `items`, `correct` are aligned trials (each row one answer).
    `machine_correct` is indexed by item, with values between 0 and 1 (fraction of correct
    samples if the machine gives more than one). The weights serve the bootstrap:
    they multiply each trial by the weight of the person and of the item.
    """
    p = np.asarray(people)
    c = np.asarray(items)
    y = np.asarray(correct, float)
    m = np.asarray(machine_correct, float)
    n_p, n_c = int(p.max()) + 1, int(m.size)
    wp = np.ones(n_p) if person_weights is None else np.asarray(person_weights, float)
    wc = np.ones(n_c) if item_weights is None else np.asarray(item_weights, float)
    w = wp[p] * wc[c]
    k = np.bincount(p, weights=w * y, minlength=n_p)
    n = np.bincount(p, weights=w, minlength=n_p)
    alive = n > 0
    acc = np.zeros(n_p)
    acc[alive] = shrink(k[alive], n[alive])
    pm = np.clip(np.average(m, weights=wc), 1e-4, 1 - 1e-4)
    zp = norm.ppf(np.clip(acc[p], 1e-4, 1 - 1e-4))
    zm = norm.ppf(pm)
    observed = float(np.sum(w * y * m[c]) / np.sum(w))
    # the mean of Φ2 over the trials: group by person, Φ2 depends only on the threshold
    zu = np.unique(np.round(zp, 4), return_inverse=True)
    weight_z = np.bincount(zu[1], weights=w)
    expected = (bvn(zu[0][None, :], zm, RHO_GRID[:, None]) * weight_z).sum(1) / np.sum(w)
    return float(np.interp(observed, expected, RHO_GRID))


def rho_person_pairs(people, items, correct, item_weights=None, person_weights=None) -> float:
    """ρ between two people on the same item, with the same personal thresholds.

    `item_weights` serves the bootstrap: an item with weight 2 counts twice, but the pairs stay between
    different trials. Duplicating the rows, instead, would pair a trial with its copy.
    `person_weights` does the same for people: a person with weight 2 counts double in the pairs with
    the others, without pairing with themselves.
    """
    p = np.asarray(people)
    c = np.asarray(items)
    y = np.asarray(correct, float)
    n_p = int(p.max()) + 1
    wc = np.ones(int(c.max()) + 1) if item_weights is None else np.asarray(item_weights, float)
    wp = np.ones(n_p) if person_weights is None else np.asarray(person_weights, float)
    k = np.bincount(p, weights=y * wc[c], minlength=n_p)
    n = np.bincount(p, weights=wc[c], minlength=n_p).astype(float)
    alive = n > 0
    acc = np.zeros(n_p)
    acc[alive] = shrink(k[alive], n[alive])
    zp = norm.ppf(np.clip(acc[p], 1e-4, 1 - 1e-4))
    order = np.argsort(c, kind="stable")
    c_ord, y_ord, z_ord = c[order], y[order], zp[order]
    w_ord = wc[c_ord]
    v_ord = wp[p[order]]
    ends = np.flatnonzero(np.diff(c_ord)) + 1
    starts = np.r_[0, ends]
    stops = np.r_[ends, c_ord.size]
    sum_obs, pairs = 0.0, 0.0
    sum_exp = np.zeros(RHO_GRID.size)
    zu, inv = np.unique(np.round(z_ord, 3), return_inverse=True)
    tab = np.stack([bvn(zu[:, None], zu[None, :], r) for r in RHO_GRID])
    for a, b in zip(starts, stops):
        weight = w_ord[a]
        v = v_ord[a:b]
        between_two = v.sum() ** 2 - (v * v).sum()
        if between_two <= 0 or weight == 0:
            continue
        yv = y_ord[a:b]
        sum_obs += weight * ((v * yv).sum() ** 2 - (v * v * yv).sum())
        pairs += weight * between_two
        count = np.bincount(inv[a:b], weights=v, minlength=zu.size)
        count2 = np.bincount(inv[a:b], weights=v * v, minlength=zu.size)
        form = (count @ tab @ count) - (tab[:, np.arange(zu.size), np.arange(zu.size)] @ count2)
        sum_exp += weight * form
    return float(np.interp(sum_obs / pairs, sum_exp / pairs, RHO_GRID))


# --- more options: from the correlation of the correct answers to that of the evidence --

def _simulate_correct(n, m, rho, dh, dm, seed):
    gen = np.random.default_rng(seed)
    common = gen.standard_normal((n, m))
    if abs(rho) < 1e-9:
        a, b = gen.standard_normal((n, m)), gen.standard_normal((n, m))
    else:
        w = math.sqrt(abs(rho))
        sign = 1.0 if rho >= 0 else -1.0
        a = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
        b = sign * w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
    a[:, 0] += dh
    b[:, 0] += dm
    return a.argmax(1) == 0, b.argmax(1) == 0


def rho_correct_from_rho_evidence(rho: float, dh: float, dm: float, m: int,
                                  n: int = 400_000, seed: int = 534) -> float:
    """The tetrachoric of the correct answers, if the evidence of each option has correlation ρ."""
    a, b = _simulate_correct(n, m, rho, dh, dm, seed)
    return tetrachoric(float(a.mean()), float(b.mean()), float((a & b).mean()))


def estimate_from_evidence(rho: float, dh: float, dm: float, m: int, sd_dh: float = 0.0,
                           n: int = 100_000, seed: int = 534, per_person: int = 0) -> float:
    """What the personal-threshold estimator would return, if the evidence had correlation ρ.

    People have different sensitivities (standard deviation sd_dh). With `per_person` > 0 each
    enters with the threshold estimated from their `per_person` answers and shrunk toward the
    mean, as in the real estimator; otherwise with the threshold of their exact accuracy.
    """
    gen = np.random.default_rng(seed)
    if per_person > 0:
        n_people = max(n // per_person, 2)
        n = n_people * per_person
        source_idx = np.repeat(np.arange(n_people), per_person)
        d_i = np.clip(dh + sd_dh * gen.standard_normal(n_people), 0.02, None)[source_idx]
    else:
        source_idx = None
        d_i = np.clip(dh + sd_dh * gen.standard_normal(n), 0.02, None)
    common = gen.standard_normal((n, m))
    if abs(rho) < 1e-9:
        a, b = gen.standard_normal((n, m)), gen.standard_normal((n, m))
    else:
        w = math.sqrt(abs(rho))
        sign = 1.0 if rho >= 0 else -1.0
        a = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
        b = sign * w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
    a[:, 0] += d_i
    b[:, 0] += dm
    ok_a, ok_b = a.argmax(1) == 0, b.argmax(1) == 0
    if source_idx is not None:
        k = np.bincount(source_idx, weights=ok_a.astype(float))
        nn = np.bincount(source_idx).astype(float)
        zi = norm.ppf(np.clip(shrink(k, nn)[source_idx], 1e-4, 1 - 1e-4))
    else:
        axis = np.linspace(d_i.min(), d_i.max(), 200)
        pc = np.sum(_WEIGHTS * norm.cdf(_NODES[None, :] + axis[:, None]) ** (m - 1), axis=1)
        zi = norm.ppf(np.clip(np.interp(d_i, axis, pc), 1e-4, 1 - 1e-4))
    zm = norm.ppf(np.clip(ok_b.mean(), 1e-4, 1 - 1e-4))
    zu = np.linspace(-3.6, 3.6, 361)
    idx = np.clip(np.round((zi - zu[0]) / (zu[1] - zu[0])).astype(int), 0, zu.size - 1)
    weight = np.bincount(idx, minlength=zu.size).astype(float)
    alive = weight > 0
    expected = (bvn(zu[alive][None, :], zm, RHO_GRID[:, None]) * weight[alive]).sum(1) / weight.sum()
    return float(np.interp((ok_a & ok_b).mean(), expected, RHO_GRID))


def convert_rho_mafc(r_correct: float, dh: float, dm: float, m: int) -> float:
    """Numerical inverse: from ρ of the correct answers to ρ of the evidence, in a choice among m options."""
    if m == 2:
        return r_correct
    g = np.linspace(-0.3, 0.95, 26)
    t = np.array([rho_correct_from_rho_evidence(float(x), dh, dm, m, n=120_000) for x in g])
    return float(np.interp(r_correct, t, g))


def rho_machines(a, b) -> float:
    """ρ between two machines on the same items: tetrachoric of the correct answers."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    p1, p2 = a.mean(), b.mean()
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return float("nan")
    return tetrachoric(float(p1), float(p2), float((a * b).mean()))
