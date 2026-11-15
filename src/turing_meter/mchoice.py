"""Sensitivity among m options, and the machinery that goes with it.

`apparent.py` has the two-alternative case. Here: d' from the proportion of correct answers
among m options, the tetrachoric correlation of two binary variables, the conversion from the
correlation of the correct answers to that of the evidence, the ideal of a pair and the useful
angle. The functions are ported unchanged from `angle/sdt.py` of
`experiments/angle-between-errors/`, which keeps its own frozen copy: that folder is the record of
the paper, and the estimators here may evolve without touching it.

Model: signal detection, Gaussian evidence with equal variance, one draw per option, the right
option shifted by d'.
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

def pc_mafc(d: float, m: int) -> float:
    """Proportion of correct answers of an ideal observer among m options."""
    return float(np.sum(_WEIGHTS * norm.cdf(_NODES + d) ** (m - 1)))


def d_from_accuracy(a: float, m: int = 2) -> float:
    """d' from a proportion of correct answers with m options.

    m = 2 uses the yes/no convention, 2 z(a); from m >= 3 the forced choice. Ratios
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
    """Yes/no d' with free criterion: z(hit) - z(fa), that is z(p1) + z(p0).

    p1 and p0 are the correct answers on the two classes. With n1 and n0 the log-linear
    correction of 1/2 is applied, which keeps the case 0 or 1 finite.
    """
    if n1 is not None:
        p1 = (p1 * n1 + 0.5) / (n1 + 1.0)
    if n0 is not None:
        p0 = (p0 * n0 + 0.5) / (n0 + 1.0)
    return float(norm.ppf(np.clip(p1, 1e-4, 1 - 1e-4)) + norm.ppf(np.clip(p0, 1e-4, 1 - 1e-4)))


# --- ideal, useful angle ----------------------------------------------------

def ideal(dh: float, dm: float, rho: float = 0.0) -> float:
    """Ideal of the pair at full evidence, with correlated errors rho; never negative weights."""
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
    """The tetrachoric of the correct answers, if the evidence of each option has correlation rho."""
    a, b = _simulate_correct(n, m, rho, dh, dm, seed)
    return tetrachoric(float(a.mean()), float(b.mean()), float((a & b).mean()))


def convert_rho_mafc(r_correct: float, dh: float, dm: float, m: int) -> float:
    """Numerical inverse: from rho of the correct answers to rho of the evidence, in a choice among m options."""
    if m == 2:
        return r_correct
    g = np.linspace(-0.3, 0.95, 26)
    t = np.array([rho_correct_from_rho_evidence(float(x), dh, dm, m, n=120_000) for x in g])
    return float(np.interp(r_correct, t, g))
