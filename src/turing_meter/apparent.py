"""Apparent intelligence on the sensitivity scale d' (docs/SPEC.md, sections 2, 3, 5).

Pure functions. The model is signal detection theory with equal-variance
Gaussian noise: each mind, on each decision, holds evidence whose mean is d'/2
on the side of the right answer and whose noise has unit variance.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np
from scipy.special import owens_t
from scipy.stats import norm


def _rate(p: float, name: str) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {p!r}")
    return p


def d_prime_from_rates(hit: float, false_alarm: float) -> float:
    """d' from the hit rate (sensitivity) and the false-alarm rate (1 - specificity)."""
    return float(norm.ppf(_rate(hit, "hit")) - norm.ppf(_rate(false_alarm, "false_alarm")))


def d_prime_from_accuracy(accuracy: float) -> float:
    """d' of an unbiased observer on a balanced binary task: 2 z(accuracy)."""
    return float(2.0 * norm.ppf(_rate(accuracy, "accuracy")))


def d_prime_from_auc(auc: float) -> float:
    """d' from the area under the ROC curve, equal-variance model: sqrt(2) z(AUC)."""
    return float(math.sqrt(2.0) * norm.ppf(_rate(auc, "auc")))


def apparent(dh: float, dm: float, phi_deg: float = 90.0) -> float:
    """Ideal sensitivity of the pair: the apparent intelligence |S|.

    Two observers on the same decisions, whose errors meet at angle phi
    (cos phi = correlation of their noise), combining all their evidence with
    non-negative weights (Sorkin and Dai 1994 give the unconstrained ideal). At
    90 degrees the errors are independent and |S| = sqrt(dh^2 + dm^2). As the
    errors align, the ideal falls to the stronger partner, reached at
    rho = weaker / stronger; beyond that point the unconstrained optimum would
    rise again by giving the weaker partner a negative weight, using it only to
    subtract the noise the two share. Negative weights are excluded by choice, a
    constraint and not a property of the model: the ideal stays at the stronger
    partner. Undefined when the errors coincide or are exactly opposed.
    """
    if dh < 0 or dm < 0:
        raise ValueError("sensitivities must be non-negative")
    rho = math.cos(math.radians(phi_deg))
    if abs(rho) >= 1.0 - 1e-12:
        raise ValueError("phi must be strictly between 0 and 180 degrees")
    if dh - rho * dm < 0 or dm - rho * dh < 0:
        return max(dh, dm)
    return math.sqrt((dh * dh + dm * dm - 2.0 * rho * dh * dm) / (1.0 - rho * rho))


def useful_angle(dh: float, dm: float) -> float:
    """Smallest angle between the errors at which the weaker mind still adds to the ideal.

    The weaker partner's optimal weight is positive only while
    rho < weaker / stronger. Next to a partner k times stronger, the errors must
    meet at more than arccos(1 / k): 60 degrees for k = 2, about 78 for k = 5,
    84 for k = 10. Below it the ideal of the pair is the stronger partner alone.
    """
    if dh <= 0 or dm <= 0:
        raise ValueError("person and machine alone must both be above chance")
    return math.degrees(math.acos(min(dh, dm) / max(dh, dm)))


def confidence_sharing(dh: float, dm: float) -> float:
    """Sensitivity of a pair that shares only how confident each one is.

    The weighted-confidence-sharing model of Bahrami et al. (2010), which fitted
    pairs of people better than the ideal: (dh + dm) / sqrt(2). Below the better
    partner when the weaker one is under about 0.41 of the stronger.
    """
    if dh < 0 or dm < 0:
        raise ValueError("sensitivities must be non-negative")
    return (dh + dm) / math.sqrt(2.0)


def turing(h: float, s: float) -> float:
    """The turing: what the person shows with the machine over what they show alone."""
    if h <= 0:
        raise ValueError("the person alone must be above chance")
    return s / h


def apparent_turings(dh: float, dm: float, phi_deg: float = 90.0) -> float:
    """The apparent turings, |S| / dH: the ideal of the pair in the person's unit.

    What the two separate readings promise, as volt-amperes do for a circuit, with
    the turing the pair shows as the watts; unlike volt-amperes, which are read
    without the angle, |S| already contains it. Tu = eta * Tu_app, and
    Tu_app = 1 / cos(theta). It depends only on k = dM / dH and the angle between
    the errors (`apparent_turings_from_ratio`).
    """
    if dh <= 0:
        raise ValueError("the person alone must be above chance")
    return apparent(dh, dm, phi_deg) / dh


def apparent_turings_from_ratio(k: float, phi_deg: float = 90.0) -> float:
    """The apparent turings in closed form, from k = dM / dH and the angle phi between the errors.

    sqrt(1 + k^2 - 2 k cos phi) / sin phi, which is sqrt(1 + k^2) at 90 degrees. Below
    the useful angle the ideal, with its weights non-negative by choice, is the
    stronger partner alone, so the value stops at max(1, k).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    rho = math.cos(math.radians(phi_deg))
    if abs(rho) >= 1.0 - 1e-12:
        raise ValueError("phi must be strictly between 0 and 180 degrees")
    if rho * k > 1.0 or rho > k:
        return max(1.0, k)
    return math.sqrt(1.0 + k * k - 2.0 * k * rho) / math.sin(math.radians(phi_deg))


def expected_weight_of_evidence(d: float) -> float:
    """Weight of evidence one observation carries on average, in natural units: d'^2 / 2.

    On the yes/no convention (evidence at +-d'/2, unit variance), the log-likelihood
    ratio of an observation is normal with variance d'^2, twice its mean: what
    Turing found for a normal weight of evidence when the hypothesis is true, as
    Good (1985) reports it. Weights of independent evidence add, which is why
    squared sensitivities add in the ideal of the pair.
    """
    if d < 0:
        raise ValueError("a sensitivity at or above chance is needed")
    return d * d / 2.0


def evidence_ratio(h: float, s: float) -> float:
    """Tu^2: the evidence an ideal observer needs to match the pair, over the person alone.

    The weight of evidence an ideal observer would need to do as well as the pair,
    over what it would need to do as well as the person alone. Equivalently, the
    number of hypothetical copies of the person, with independent errors and their
    evidence pooled, that would decide as well. Real people are not such copies:
    n observers whose errors correlate at rho reach at most 1 / sqrt(rho) times one
    of them.
    """
    return turing(h, s) ** 2


class Pair(NamedTuple):
    apparent: float            # |S|, the ideal of the pair
    eta: float                 # share of the ideal that shows: dS / |S|
    gain_share: float | None   # share of the possible gain: (dS - dH) / (|S| - dH)
    turing: float              # dS / dH
    theta_deg: float           # arccos(cos_theta); at phi = 90, atan(dM / dH)
    cos_theta: float           # the person's share of the ideal, dH / |S|: eta when dS = dH
    eta_follow: float          # eta of a pair that always follows the machine, dM / |S|
    adoption: float | None     # (dS - dH) / (dM - dH), only when the machine is stronger
    apparent_turings: float    # |S| / dH = 1 / cos_theta: turing = eta * apparent_turings
    k: float                   # dM / dH: the turing of a person who copies the machine
    yield_on_machine: float    # dS / dM: the turing read from the machine's side; < 1, the person subtracts


def pair(dh: float, dm: float, ds: float, phi_deg: float = 90.0) -> Pair:
    """Everything the SPEC reports for one person, one machine, one protocol.

    By construction turing = eta * apparent_turings = eta / cos_theta and
    eta = cos_theta + gain_share * (1 - cos_theta). So eta falls as the machine
    gets stronger even when people realize the same share of the possible gain:
    report gain_share for behaviour, eta for how far the pair is from its ideal.
    Neither tells a person who copies the machine (turing = k) from one who adds
    to it: that shows only where turing > k, that is yield_on_machine > 1.
    """
    if dh <= 0 or dm <= 0:
        raise ValueError("person and machine alone must both be above chance")
    s = apparent(dh, dm, phi_deg)
    room = s - dh
    cos_t = min(1.0, dh / s)
    return Pair(
        apparent=s,
        eta=ds / s,
        gain_share=(ds - dh) / room if room > 1e-9 else None,
        turing=turing(dh, ds),
        theta_deg=math.degrees(math.acos(cos_t)),
        cos_theta=cos_t,
        eta_follow=dm / s,
        adoption=(ds - dh) / (dm - dh) if dm - dh > 1e-9 else None,
        apparent_turings=s / dh,
        k=dm / dh,
        yield_on_machine=ds / dm,
    )


def verdict_ideal(dh: float, dm: float) -> float:
    """Ideal d' of a person who receives only the machine's verdict, not its confidence.

    The person weighs their own evidence (log-likelihood ratio dh * x) against the
    verdict (log-likelihood ratio +-logit(accuracy of the machine)), errors
    independent. Never below max(dh, dm), never above apparent(dh, dm).
    """
    if dh <= 0 or dm <= 0:
        raise ValueError("person and machine alone must both be above chance")
    am = norm.cdf(dm / 2.0)
    lm = math.log(am / (1.0 - am))
    acc = am * norm.cdf(dh / 2.0 + lm / dh) + (1.0 - am) * norm.cdf(dh / 2.0 - lm / dh)
    return float(2.0 * norm.ppf(acc))


class Alignment(NamedTuple):
    rho: float       # correlation of the two minds' errors
    phi_deg: float   # the angle between them: 90 = independent errors
    items: int


def error_alignment(person_correct: Sequence[bool], machine_correct: Sequence[bool]) -> Alignment:
    """Angle between the errors of the person alone and the machine alone, item by item.

    Under the model, a mind is right on an item when its noise stays above -d'/2,
    so the share of items both get right fixes the correlation of their noise
    (the tetrachoric correlation with thresholds set by the two accuracies).
    """
    h = np.asarray(person_correct, dtype=bool)
    m = np.asarray(machine_correct, dtype=bool)
    if h.shape != m.shape or h.ndim != 1:
        raise ValueError("one answer per item from each, on the same items")
    ah, am, both = h.mean(), m.mean(), (h & m).mean()
    if not (0 < ah < 1 and 0 < am < 1):
        raise ValueError("each mind must get some items right and some wrong")
    rho = float(tetrachoric(ah, am, both))
    return Alignment(rho=rho, phi_deg=math.degrees(math.acos(rho)), items=int(h.size))


def _bvn_cdf(h, k, r):
    """P(X < h, Y < k) for standard normals with correlation r (Owen 1956, with his T function)."""
    h, k, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (h, k, r)))
    h = np.where(h == 0, 1e-12, h)  # the formula is continuous across zero; it divides by h and k
    k = np.where(k == 0, 1e-12, k)
    s = np.sqrt((1.0 - r) * (1.0 + r))
    beta = np.where(h * k > 0, 0.0, 0.5)
    return (0.5 * norm.cdf(h) + 0.5 * norm.cdf(k)
            - owens_t(h, (k - r * h) / (h * s)) - owens_t(k, (h - r * k) / (k * s)) - beta)


def tetrachoric(p1, p2, p11):
    """The correlation of two latent normals cut at shares p1 and p2 that land together with share p11.

    Works on arrays (a bootstrap solves thousands of tables at once). Bounded at +-0.999.
    """
    z1, z2 = norm.ppf(np.asarray(p1, dtype=float)), norm.ppf(np.asarray(p2, dtype=float))
    target = np.asarray(p11, dtype=float)
    z1, z2, target = np.broadcast_arrays(z1, z2, target)
    lo, hi = np.full(target.shape, -0.999), np.full(target.shape, 0.999)
    for _ in range(45):  # bisection: the joint share grows with the correlation
        mid = 0.5 * (lo + hi)
        below = _bvn_cdf(z1, z2, mid) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    out = 0.5 * (lo + hi)
    return float(out) if out.ndim == 0 else out


class ClassAlignment(NamedTuple):
    rho: float                     # correlation of the evidence, the criterion left out
    by_class: tuple[float, float]  # within the cases whose answer is yes, and no
    phi_deg: float
    items: int


def _class_rho(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = a.mean(), b.mean()
    if not (0 < pa < 1 and 0 < pb < 1):
        raise ValueError("within each class, each mind must answer both ways sometimes")
    return float(tetrachoric(pa, pb, (a & b).mean()))


def class_alignment(person_yes: Sequence[bool], machine_yes: Sequence[bool],
                    truth: Sequence[int]) -> ClassAlignment:
    """Alignment of the evidence of person and machine, within each class of the right answer.

    Within a class, being right and answering yes are the same thing, so the table of
    the two minds' answers gives the correlation of their evidence whatever their
    criteria. Pooled over classes, a lean toward the same answer makes the two err
    together on one class and look aligned (review of 2 November 2026, P2): this is
    the measure the useful angle needs. The classes are combined weighted by size.
    """
    h = np.asarray(person_yes, dtype=bool)
    m = np.asarray(machine_yes, dtype=bool)
    y = np.asarray(truth) > 0
    if not (h.shape == m.shape == y.shape) or h.ndim != 1:
        raise ValueError("one answer per item from each, and the right answer, on the same items")
    if y.all() or not y.any():
        raise ValueError("both classes of the right answer are needed")
    pos = _class_rho(h[y], m[y])
    neg = _class_rho(h[~y], m[~y])
    rho = (pos * y.sum() + neg * (~y).sum()) / y.size
    return ClassAlignment(rho=rho, by_class=(pos, neg), phi_deg=math.degrees(math.acos(rho)),
                          items=int(y.size))


_GRID = np.linspace(-0.5, 0.95, 74)  # steps of 0.02: linear interpolation is exact to 1e-4


def _own_thresholds(block: np.ndarray) -> np.ndarray:
    """Each person's cut point within a class, from their own share of yes answers."""
    n = block.shape[1]
    share = np.clip(block.mean(axis=1), 0.5 / n, 1 - 0.5 / n)
    return norm.ppf(share)


def _solve(expected: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Per row, the grid value where an increasing curve reaches the observed share."""
    out = np.empty(observed.shape)
    for b in range(observed.size):
        out[b] = np.interp(observed[b], expected[b], _GRID)
    return out


def _weights(weights, people: int) -> np.ndarray:
    w = np.ones((1, people)) if weights is None else np.atleast_2d(np.asarray(weights, dtype=float))
    if w.shape[1] != people:
        raise ValueError("one weight per person")
    return w


def _by_class(yes, truth, per_class, weights):
    yes = np.asarray(yes, dtype=bool)
    y = np.asarray(truth) > 0
    if yes.ndim != 2 or yes.shape[1] != y.size or yes.shape[0] < 2:
        raise ValueError("one row per person (at least two), one column per item")
    if y.all() or not y.any():
        raise ValueError("both classes of the right answer are needed")
    w = _weights(weights, yes.shape[0])
    rho = sum(per_class(yes[:, cls], cls, w) * cls.sum() for cls in (y, ~y)) / y.size
    return float(rho[0]) if weights is None else rho


def pooled_alignment(yes, machine_yes, truth, weights=None):
    """Person-machine alignment within class, many people pooled, each with their own criterion.

    `yes` has one row per person, all on the same items as the machine. The estimate
    matches the share of items where person and machine both say yes, summed over
    people, to what the model gives with each person's own cut point: pooling with
    one cut point for all would mix criteria and pull the correlation toward zero.
    `weights` (one row per bootstrap draw, one column per person) gives one estimate
    per row.
    """
    m = np.asarray(machine_yes, dtype=bool)

    def per_class(block, cls, w):
        mc = m[cls]
        if not 0 < mc.mean() < 1:
            raise ValueError("within each class the machine must answer both ways sometimes")
        zp = _own_thresholds(block)
        zm = norm.ppf(mc.mean())
        observed = (w @ (block & mc).mean(axis=1)) / w.sum(axis=1)
        model = _bvn_cdf(zp[None, :], zm, _GRID[:, None])          # grid x people
        expected = (w @ model.T) / w.sum(axis=1, keepdims=True)    # draws x grid
        return _solve(expected, observed)

    return _by_class(yes, truth, per_class, weights)


def people_alignment(yes, truth, weights=None):
    """Alignment between people on the same items, within class, over all pairs of people.

    Each person keeps their own cut point, as in `pooled_alignment`. Under the model
    with a shared difficulty this is what people share with each other: the baseline
    that practice alone moves. With weights, a person drawn twice is not paired with
    themself.
    """
    def per_class(block, cls, w):
        zp = _own_thresholds(block)
        b = block.astype(float)
        both = b @ b.T / block.shape[1]                              # people x people
        norm_ = w.sum(axis=1) ** 2 - (w * w).sum(axis=1)
        observed = (np.einsum("bp,pq,bq->b", w, both, w) - (w * w) @ np.diag(both)) / norm_
        model = _bvn_cdf(zp[None, :, None], zp[None, None, :], _GRID[:, None, None])
        diag = np.einsum("gpp->gp", model)
        expected = (np.einsum("bp,gpq,bq->bg", w, model, w) - (w * w) @ diag.T) / norm_[:, None]
        return _solve(expected, observed)

    return _by_class(yes, truth, per_class, weights)


def practice_adjusted(rho_hm, rho_hh, q):
    """Alignment with the machine beyond what practice alone explains.

    With a difficulty shared by person and machine, and nothing else shared,
    rho(person, machine)^2 = q * rho(person, person), where q is the share of the
    machine's evidence that is the shared difficulty: fixed, since the machine is
    frozen. Less noise, or more weight on the shared difficulty, moves both sides
    together; only a quirk of the machine taken up by the person breaks the
    identity. q is estimated before, where nobody has used the machine.
    """
    return np.asarray(rho_hm) - np.sqrt(q * np.asarray(rho_hh))
