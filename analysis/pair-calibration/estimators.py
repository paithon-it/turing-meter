"""Shared functions for the calculations of the usage protocol.

The estimators are those of the experiment repository (angle/sdt.py, angle/measure.py):
d' from accuracy with m options, tetrachoric correlation, conversion from the correlation of
the correct answers to that of the evidence, the pair's ideal, the useful angle. This module
holds only what is needed to run them tens of thousands of times on a single person: the same
formulas in vectorised form, with the conversion table computed once on a grid of (dH, dM).
`sim_sample.py --verify` compares this fast path with `measure.compare`.

The experiment is found by path relative to this repository, or through the environment
variable TURING_METER_EXPERIMENT. The repository is left untouched: no bytecode is written
next to its files.
"""
from __future__ import annotations

import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get(
    "TURING_METER_EXPERIMENT",
    os.path.normpath(os.path.join(HERE, "..", "..", "experiments", "angle-between-errors")),
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from angle import sdt  # noqa: E402

G = np.r_[np.linspace(-0.6, 0.9, 31), 0.93, 0.96, 0.98, 0.99]   # the grid of measure.conversion_table
RHO_MIN, RHO_MAX = float(sdt.RHO_GRID[0]), float(sdt.RHO_GRID[-1])

# --- d' from accuracy, vectorised -------------------------------------------------

_DGRID = np.linspace(0.0, 7.0, 1401)
_PC: dict[int, np.ndarray] = {}


def _pc(m: int) -> np.ndarray:
    if m not in _PC:
        _PC[m] = np.array([sdt.pc_mafc(float(d), m) for d in _DGRID])
    return _PC[m]


def d_da_acc(a, m: int):
    """As measure._d: accuracy clipped between 0.0001 and 0.9999; NaN at chance level or below."""
    a = np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4)
    if m == 2:
        return np.where(a > 0.5 + 1e-9, 2.0 * norm.ppf(a), np.nan)
    return np.where(a > 1.0 / m + 1e-9, np.interp(a, _pc(m), _DGRID), np.nan)


def acc_da_d(d, m: int):
    d = np.asarray(d, float)
    if m == 2:
        return norm.cdf(d / 2.0)
    return np.interp(d, _DGRID, _pc(m))


# --- the generative process (that of sdt._simulate_correct, keeping the chosen option) ----

def scelte(n: int, m: int, rho: float, dh: float, dm: float, gen: np.random.Generator):
    """Gaussian evidence for each of the m options, with correlation rho between person and machine.

    This is the process of `sdt._simulate_correct`; it returns the chosen option (0 is the correct
    one), because agreement needs to know which wrong option was taken, not only that it is wrong.
    """
    common = gen.standard_normal((n, m))
    w = math.sqrt(abs(rho))
    sign = 1.0 if rho >= 0 else -1.0
    a = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
    b = sign * w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
    a[:, 0] += dh
    b[:, 0] += dm
    return a.argmax(1), b.argmax(1), a, b


def celle(dh: float, dm: float, rho: float, m: int, n: int = 4_000_000, seed: int = 1, blocco: int = 250_000):
    """Probabilities of: both correct, only the person, only the machine, neither; and agreement on a wrong option."""
    gen = np.random.default_rng(seed)
    c = np.zeros(5)
    giri = max(n // blocco, 1)
    for _ in range(giri):
        ca, cb, _, _ = scelte(blocco, m, rho, dh, dm, gen)
        ha, hb = ca == 0, cb == 0
        c += [np.sum(ha & hb), np.sum(ha & ~hb), np.sum(~ha & hb), np.sum(~ha & ~hb), np.sum(~ha & ~hb & (ca == cb))]
    return c / (giri * blocco)


# --- the conversion, on a grid ------------------------------------------------

def _una_tabella(arg):
    """The tetrachoric of the correct answers that a pair with evidence correlation x would give.

    For a single person the threshold is their accuracy, with no grid: this is `sdt.rho_correct_from_rho_evidence`,
    the same function that `sdt.convert_rho_mafc` rests on. `measure.conversion_table` instead uses
    `sdt.estimate_from_evidence`, which rounds thresholds to steps of 0.02 (the flaw declared in
    appendix D): with a machine above 95% that table stops being invertible.
    """
    dh, dm, m, n = arg
    t = np.array([sdt.rho_correct_from_rho_evidence(float(x), dh, dm, m, n=n) for x in G])
    return np.maximum.accumulate(t)


class Conversione:
    """From rho of the correct answers to rho of the evidence, for m options, interpolated on (dH, dM).

    Each node is the table of `sdt.convert_rho_mafc` for those sensitivities (one person, exact
    threshold), with more trials and the rho grid of `measure.conversion_table`. Between nodes
    the interpolation is bilinear.
    """

    def __init__(self, m: int, n: int = 250_000, processi: int = 3):
        self.m = m
        self.dhs = np.round(np.arange(0.1, 3.31, 0.2), 3)
        self.dms = np.round(np.arange(0.1, 6.11, 0.25), 3)
        file = os.path.join(HERE, f"conversion-m{m}.npz")
        if os.path.exists(file):
            z = np.load(file)
            if z["dhs"].shape == self.dhs.shape and z["dms"].shape == self.dms.shape and int(z["n"]) == n:
                self.T = z["T"]
                return
        from concurrent.futures import ProcessPoolExecutor
        args = [(float(a), float(b), m, n) for a in self.dhs for b in self.dms]
        with ProcessPoolExecutor(max_workers=processi) as ex:
            tab = list(ex.map(_una_tabella, args, chunksize=4))
        self.T = np.array(tab).reshape(self.dhs.size, self.dms.size, G.size)
        np.savez_compressed(file, T=self.T, dhs=self.dhs, dms=self.dms, n=n)

    def tabella(self, dh, dm) -> np.ndarray:
        dh = np.clip(np.atleast_1d(np.asarray(dh, float)), self.dhs[0], self.dhs[-1])
        dm = np.clip(np.atleast_1d(np.asarray(dm, float)), self.dms[0], self.dms[-1])
        i = np.clip(np.searchsorted(self.dhs, dh) - 1, 0, self.dhs.size - 2)
        j = np.clip(np.searchsorted(self.dms, dm) - 1, 0, self.dms.size - 2)
        u = ((dh - self.dhs[i]) / (self.dhs[i + 1] - self.dhs[i]))[:, None]
        v = ((dm - self.dms[j]) / (self.dms[j + 1] - self.dms[j]))[:, None]
        return ((1 - u) * (1 - v) * self.T[i, j] + u * (1 - v) * self.T[i + 1, j]
                + (1 - u) * v * self.T[i, j + 1] + u * v * self.T[i + 1, j + 1])

    def rho(self, r_giuste, dh, dm) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r_giuste, float))
        return interp_righe(r, self.tabella(dh, dm))


def interp_righe(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """np.interp(x[i], T[i], G) for each row, with T non-decreasing along the columns."""
    n = T.shape[1]
    j = np.clip((T < x[:, None]).sum(1), 1, n - 1)
    righe = np.arange(T.shape[0])
    t0, t1 = T[righe, j - 1], T[righe, j]
    passo = np.where(t1 > t0, (x - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
    out = G[j - 1] + np.clip(passo, 0.0, 1.0) * (G[j] - G[j - 1])
    return np.where(x <= T[:, 0], G[0], np.where(x >= T[:, -1], G[-1], out))


# --- the tetrachoric, vectorised ----------------------------------------------------

def tetracorica(p1, p2, p11, giri: int = 32) -> np.ndarray:
    """As sdt.tetrachoric (same bounds as its grid), by bisection on vectors."""
    z1 = norm.ppf(np.clip(np.asarray(p1, float), 1e-4, 1 - 1e-4))
    z2 = norm.ppf(np.clip(np.asarray(p2, float), 1e-4, 1 - 1e-4))
    p11 = np.asarray(p11, float)
    z1, z2, p11 = np.broadcast_arrays(z1, z2, p11)
    lo, hi = np.full(p11.shape, RHO_MIN), np.full(p11.shape, RHO_MAX)
    for _ in range(giri):
        mid = 0.5 * (lo + hi)
        sotto = sdt.bvn(z1, z2, mid) < p11
        lo, hi = np.where(sotto, mid, lo), np.where(sotto, hi, mid)
    return 0.5 * (lo + hi)


# --- the quantities of the article, from four counts --------------------------------

def ideale(dh, dm, rho):
    """sdt.ideal on vectors: the linear optimum, stopped at the stronger one below the useful angle."""
    dh, dm, rho = np.broadcast_arrays(np.asarray(dh, float), np.asarray(dm, float), np.asarray(rho, float))
    forte = np.maximum(dh, dm)
    libero = np.sqrt(np.maximum(dh * dh + dm * dm - 2 * rho * dh * dm, 0.0) / np.maximum(1 - rho * rho, 1e-12))
    fermo = (dh - rho * dm < 0) | (dm - rho * dh < 0) | (np.abs(rho) >= 1 - 1e-12)
    return np.where(fermo, forte, libero)


def margine(k, phi_gradi):
    """Formula (7) of the article: how much the ideal pair exceeds the stronger one; zero below the useful angle."""
    k = np.asarray(k, float)
    k = np.where(k < 1, 1 / k, k)
    c, s = np.cos(np.radians(phi_gradi)), np.sin(np.radians(phi_gradi))
    m = np.sqrt(np.maximum(1 + k * k - 2 * k * c, 0.0)) / (k * s) - 1
    return np.where(c >= 1 / k, 0.0, m)


def rapporto_pesi(k: float, phi_gradi: float) -> float:
    """conversations.weight_ratio: how many times the machine's evidence weighs the person's."""
    c = math.cos(math.radians(phi_gradi))
    return (k - c) / (1 - k * c)


def stima(n11, n10, n01, n00, m: int, conv: Conversione | None = None, tabella: np.ndarray | None = None) -> dict:
    """The quantities from a 2x2 table (also weighted): person correct/wrong by machine correct/wrong.

    n11 both correct, n10 only the person, n01 only the machine, n00 neither.
    With `tabella` the conversion is fixed (as in the bootstrap of measure.compare); with `conv`
    it is interpolated at the sensitivity of each row.
    """
    n11, n10, n01, n00 = (np.atleast_1d(np.asarray(x, float)) for x in (n11, n10, n01, n00))
    n = n11 + n10 + n01 + n00
    with np.errstate(invalid="ignore", divide="ignore"):
        ah, am, p11 = (n11 + n10) / n, (n11 + n01) / n, n11 / n
        dh, dm = d_da_acc(ah, m), d_da_acc(am, m)
        valido = (n > 0) & ~np.isnan(dh) & ~np.isnan(dm)
        rc = tetracorica(np.where(valido, ah, 0.5), np.where(valido, am, 0.5), np.where(valido, p11, 0.25))
        if m == 2:
            rho = rc
        elif tabella is not None:
            rho = np.interp(rc, tabella, G)
        else:
            rho = conv.rho(rc, np.where(valido, dh, 1.0), np.where(valido, dm, 1.0))
        # the angle makes sense only if each of the two is wrong and also correct at least once
        angolo_ok = valido & (n10 + n00 > 0) & (n01 + n00 > 0) & (n11 + n10 > 0) & (n11 + n01 > 0)
        phi = np.degrees(np.arccos(np.clip(rho, -1, 1)))
        k = dm / dh
        utile = np.degrees(np.arccos(np.minimum(dh, dm) / np.maximum(dh, dm)))
        forte = np.maximum(dh, dm)
        s = ideale(np.where(valido, dh, 1.0), np.where(valido, dm, 1.0), rho)
        s90 = np.sqrt(dh * dh + dm * dm)
    out = {"acc_h": ah, "acc_m": am, "dh": dh, "dm": dm, "k": k, "rho_giuste": rc, "rho": rho, "phi": phi,
           "utile": utile, "margine": s / forte - 1, "margine90": s90 / forte - 1, "ideale": s,
           "sopra_utile": phi - utile, "valido": valido, "angolo_ok": angolo_ok,
           "errori_m": n10 + n00, "errori_h": n01 + n00, "errori_comuni": n00}
    for key in ("k", "utile", "margine", "margine90", "sopra_utile", "phi", "ideale"):
        out[key] = np.where(valido, out[key], np.nan)
    out["phi"] = np.where(angolo_ok, out["phi"], np.nan)
    out["sopra_utile"] = np.where(angolo_ok, out["sopra_utile"], np.nan)
    return out


def quantili(x, q=(2.5, 50, 97.5)):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(v) for v in np.percentile(x, q)] if x.size else [float("nan")] * len(q)


def clopper_pearson(k: int, n: int, livello: float = 0.95) -> tuple[float, float]:
    """Exact interval of a proportion: needed where errors are few and resampling does not move them."""
    from scipy.stats import beta
    a = (1 - livello) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi
