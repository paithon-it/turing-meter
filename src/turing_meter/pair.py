"""One person and one machine, measured on the person's own cases (docs/PAIR-PROTOCOL.md).

Input: a sheet with one row per case, holding the right answer (checked beforehand), the
answer of the person alone and the answer of the machine asked alone; optionally the answer
given with the machine, the confidence, the chat, times and weights. Output: the two
sensitivities, k = dM/dH, the angle between the errors, the useful angle, the margin, the
alarm of disagreement, the turing and the yield, and at the end the zone and the rule of use.

Intervals: resampling of the cases (Poisson weights), and exact intervals (Clopper-Pearson)
where the machine errs so rarely that resampling cannot move its errors. The angle is
reported only when each side has enough errors: below `MIN_ANGLE_ERRORS` it is "not
estimable", below `FEW_ERRORS` it is "indicative".

Limits: one person per sheet, the same number of options in every case; with two options
(yes/no) the sensitivities are computed per class.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import norm

from turing_meter import mchoice as sdt

# Column names: the canonical English one first, then the aliases. The Italian names are
# the ones of the first prototype of the protocol and stay accepted.
ALIASES = {
    "case": ("case", "caso", "id"),
    "key": ("key", "chiave", "answer"),
    "person": ("person", "persona", "human"),
    "machine": ("machine", "macchina", "model"),
    "confidence": ("confidence", "sicurezza"),
    "chat": ("chat",),
    "pair": ("pair", "coppia", "with_machine"),
    "stratum": ("stratum", "strato", "materia", "subject"),
    "weight": ("weight", "peso"),
    "seconds_alone": ("seconds_alone", "secondi_solo"),
    "seconds_pair": ("seconds_pair", "secondi_coppia"),
}
REQUIRED = ("key", "person", "machine")

MIN_ANGLE_ERRORS = 10   # below, the angle is not reported: resampling itself does not hold (analysis/pair-calibration)
FEW_ERRORS = 40         # below, the angle is indicative: the interval exceeds 15 degrees per side
MIN_RESAMPLING_ERRORS = 5   # below, the exact interval is used for k
K_THRESHOLD = 1.5       # the whole interval of k above: the machine decides (ceiling at 90 degrees: 20 per cent)


class SheetError(ValueError):
    """The sheet cannot be read or measured."""


# --- reading the sheet ----------------------------------------------------------------

def parse_sheet(rows: Sequence[dict]) -> dict:
    """From the rows of a CSV (dicts) to columns under the canonical names.

    The order of the columns does not matter, names are case-insensitive, rows missing the key,
    the person's answer or the machine's answer are dropped and counted in `dropped`.
    """
    rows = list(rows)
    if not rows:
        raise SheetError("the sheet is empty")
    names = {str(c).strip().lower(): c for c in rows[0].keys() if c}
    col: dict[str, list[str]] = {}
    for field, aliases in ALIASES.items():
        for a in aliases:
            if a in names:
                col[field] = [(r[names[a]] or "").strip() for r in rows]
                break
    for field in REQUIRED:
        if field not in col:
            raise SheetError(f"missing column '{field}' (accepted: {', '.join(ALIASES[field])})")
    keep = [i for i in range(len(rows)) if col["key"][i] and col["person"][i] and col["machine"][i]]
    out: dict = {field: [v[i] for i in keep] for field, v in col.items()}
    out["dropped"] = len(rows) - len(keep)
    return out


def read_sheet(path: str | Path) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return parse_sheet(list(csv.DictReader(f)))


def _norm(v):
    return np.array([x.strip().upper() for x in v])


def _float(x: str) -> float:
    return float(x.replace(",", "."))


# --- vectorised estimators ------------------------------------------------------------

G = np.r_[np.linspace(-0.6, 0.9, 31), 0.93, 0.96, 0.98, 0.99]   # the grid of the conversion table
RHO_MIN, RHO_MAX = float(sdt.RHO_GRID[0]), float(sdt.RHO_GRID[-1])

_DGRID = np.linspace(0.0, 7.0, 1401)
_PC: dict[int, np.ndarray] = {}


def _pc(m: int) -> np.ndarray:
    if m not in _PC:
        _PC[m] = np.array([sdt.pc_mafc(float(d), m) for d in _DGRID])
    return _PC[m]


def d_from_acc(a, m: int):
    """d' from accuracy, vectorised: accuracy clipped to [0.0001, 0.9999]; NaN at chance or below."""
    a = np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4)
    if m == 2:
        return np.where(a > 0.5 + 1e-9, 2.0 * norm.ppf(a), np.nan)
    return np.where(a > 1.0 / m + 1e-9, np.interp(a, _pc(m), _DGRID), np.nan)


def acc_from_d(d, m: int):
    d = np.asarray(d, float)
    if m == 2:
        return norm.cdf(d / 2.0)
    return np.interp(d, _DGRID, _pc(m))


def tetrachoric(p1, p2, p11, rounds: int = 32) -> np.ndarray:
    """Like `mchoice.tetrachoric` (same bounds of its grid), by bisection on vectors."""
    z1 = norm.ppf(np.clip(np.asarray(p1, float), 1e-4, 1 - 1e-4))
    z2 = norm.ppf(np.clip(np.asarray(p2, float), 1e-4, 1 - 1e-4))
    p11 = np.asarray(p11, float)
    z1, z2, p11 = np.broadcast_arrays(z1, z2, p11)
    lo, hi = np.full(p11.shape, RHO_MIN), np.full(p11.shape, RHO_MAX)
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        below = sdt.bvn(z1, z2, mid) < p11
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def ideal_pair(dh, dm, rho):
    """`mchoice.ideal` on vectors: the linear optimum, stopped at the stronger below the useful angle."""
    dh, dm, rho = np.broadcast_arrays(np.asarray(dh, float), np.asarray(dm, float), np.asarray(rho, float))
    strong = np.maximum(dh, dm)
    free = np.sqrt(np.maximum(dh * dh + dm * dm - 2 * rho * dh * dm, 0.0) / np.maximum(1 - rho * rho, 1e-12))
    stop = (dh - rho * dm < 0) | (dm - rho * dh < 0) | (np.abs(rho) >= 1 - 1e-12)
    return np.where(stop, strong, free)


def margin(k, phi_degrees):
    """The margin of the ideal pair over the stronger partner at angle phi; zero below the useful angle."""
    k = np.asarray(k, float)
    k = np.where(k < 1, 1 / k, k)
    c, s = np.cos(np.radians(phi_degrees)), np.sin(np.radians(phi_degrees))
    m = np.sqrt(np.maximum(1 + k * k - 2 * k * c, 0.0)) / (k * s) - 1
    return np.where(c >= 1 / k, 0.0, m)


def weight_ratio(k: float, phi_degrees: float) -> float:
    """How many times the evidence of the machine weighs that of the person in the ideal pair."""
    c = math.cos(math.radians(phi_degrees))
    return (k - c) / (1 - k * c)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact interval of a proportion: for where errors are few and resampling does not move them."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def estimate(n11, n10, n01, n00, m: int, table: np.ndarray | None = None) -> dict:
    """The quantities from a 2x2 table (also weighted): person right/wrong by machine right/wrong.

    n11 both right, n10 only the person, n01 only the machine, n00 neither. `table` is the
    conversion from the tetrachoric of the correct answers to the correlation of the evidence
    (same grid `G`); with two options no conversion is needed.
    """
    n11, n10, n01, n00 = (np.atleast_1d(np.asarray(x, float)) for x in (n11, n10, n01, n00))
    n = n11 + n10 + n01 + n00
    with np.errstate(invalid="ignore", divide="ignore"):
        ah, am, p11 = (n11 + n10) / n, (n11 + n01) / n, n11 / n
        dh, dm = d_from_acc(ah, m), d_from_acc(am, m)
        valid = (n > 0) & ~np.isnan(dh) & ~np.isnan(dm)
        rc = tetrachoric(np.where(valid, ah, 0.5), np.where(valid, am, 0.5), np.where(valid, p11, 0.25))
        if m == 2:
            rho = rc
        else:
            if table is None:
                raise ValueError("more than two options need the conversion table")
            rho = np.interp(rc, table, G)
        # the angle makes sense only if each one errs and answers right at least once
        angle_ok = valid & (n10 + n00 > 0) & (n01 + n00 > 0) & (n11 + n10 > 0) & (n11 + n01 > 0)
        phi = np.degrees(np.arccos(np.clip(rho, -1, 1)))
        k = dm / dh
        useful = np.degrees(np.arccos(np.minimum(dh, dm) / np.maximum(dh, dm)))
        strong = np.maximum(dh, dm)
        s = ideal_pair(np.where(valid, dh, 1.0), np.where(valid, dm, 1.0), rho)
        s90 = np.sqrt(dh * dh + dm * dm)
    out = {"acc_h": ah, "acc_m": am, "dh": dh, "dm": dm, "k": k, "rho_correct": rc, "rho": rho, "phi": phi,
           "useful": useful, "margin": s / strong - 1, "margin90": s90 / strong - 1, "ideal": s,
           "above_useful": phi - useful, "valid": valid, "angle_ok": angle_ok,
           "errors_m": n10 + n00, "errors_h": n01 + n00, "errors_common": n00}
    for key in ("k", "useful", "margin", "margin90", "above_useful", "phi", "ideal"):
        out[key] = np.where(valid, out[key], np.nan)
    out["phi"] = np.where(angle_ok, out["phi"], np.nan)
    out["above_useful"] = np.where(angle_ok, out["above_useful"], np.nan)
    return out


# --- the quantities for a matrix of weights (first row: the true weights; others: resamples) ---

def _quantities(W: np.ndarray, h: np.ndarray, mm: np.ndarray, s, klass, m: int, tab) -> dict:
    tot = W.sum(1)
    if m > 2:
        e = estimate(W @ (h * mm), W @ (h * (1 - mm)), W @ ((1 - h) * mm), W @ ((1 - h) * (1 - mm)), m, table=tab)
        out = {k: e[k] for k in ("acc_h", "acc_m", "dh", "dm", "k", "rho", "phi", "useful", "margin", "margin90", "valid", "angle_ok")}
        if s is not None:
            out["acc_s"] = (W @ s) / tot
            out["ds"] = d_from_acc(out["acc_s"], m)
    else:
        out = _two_classes(W, h, mm, s, klass)
    if s is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["turing"], out["yield"] = out["ds"] / out["dh"], out["ds"] / out["dm"]
    return out


def _two_classes(W, h, mm, s, klass) -> dict:
    """Yes/no: d' = z(right on yes) + z(right on no), with the correction of half a case
    (`mchoice.d_from_classes`); rho within each class, weighted average over the classes."""
    def rate(x, sel):
        n = W[:, sel].sum(1)
        return np.clip((W[:, sel] @ x[sel] + 0.5) / (n + 1.0), 1e-4, 1 - 1e-4), n
    c1, c0 = klass == 1, klass == 0
    out, d = {}, {}
    for name, x in (("h", h), ("m", mm)) + ((("s", s),) if s is not None else ()):
        p1, _ = rate(x, c1)
        p0, _ = rate(x, c0)
        d[name] = norm.ppf(p1) + norm.ppf(p0)
        out["acc_" + name] = (W @ x) / W.sum(1)
    out["dh"], out["dm"] = d["h"], d["m"]
    if s is not None:
        out["ds"] = d["s"]
    num, den = np.zeros(W.shape[0]), np.zeros(W.shape[0])
    for sel in (c1, c0):
        n = W[:, sel].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ph, pm, p11 = W[:, sel] @ h[sel] / n, W[:, sel] @ mm[sel] / n, W[:, sel] @ (h[sel] * mm[sel]) / n
        ok = (n > 0) & (pm > 0) & (pm < 1) & (ph > 0) & (ph < 1)
        r = tetrachoric(np.where(ok, ph, 0.5), np.where(ok, pm, 0.5), np.where(ok, p11, 0.25))
        num += np.where(ok, r * n, 0.0)
        den += np.where(ok, n, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
        valid = (out["dh"] > 0) & (out["dm"] > 0)
        out["k"] = np.where(valid, out["dm"] / out["dh"], np.nan)
        strong = np.maximum(out["dh"], out["dm"])
        out["useful"] = np.degrees(np.arccos(np.minimum(out["dh"], out["dm"]) / strong))
        out["rho"] = rho
        out["phi"] = np.degrees(np.arccos(np.clip(rho, -1, 1)))
        out["margin"] = ideal_pair(np.where(valid, out["dh"], 1.0), np.where(valid, out["dm"], 1.0), np.nan_to_num(rho)) / strong - 1
        out["margin90"] = np.sqrt(out["dh"] ** 2 + out["dm"] ** 2) / strong - 1
    out["valid"], out["angle_ok"] = valid, valid & (den > 0)
    out["phi"] = np.where(out["angle_ok"], out["phi"], np.nan)
    return out


def _ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 50:
        return None
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _errors_per_hundred(d: float, m: int) -> float:
    return 100 * (1 - float(acc_from_d(d, m)))


def zone_from_k(k_lo: float, k_hi: float) -> tuple[str, float]:
    """The zone and the ceiling of the margin at 90 degrees, from the interval of k.

    A: the whole interval above 1.5. B: above 1, not all above 1.5. C: it contains 1.
    D and E: the mirror, with the person stronger. The ceiling is the margin at 90 degrees at
    the least favourable k of the interval (the closest to 1); if the interval contains 1 it
    is that of two equals, 41 per cent.
    """
    if k_lo > 1:
        return ("A" if k_lo >= K_THRESHOLD else "B"), float(math.sqrt(1 + k_lo ** 2) / k_lo - 1)
    if k_hi < 1:
        return ("E" if 1 / k_hi >= K_THRESHOLD else "D"), float(math.sqrt(1 + (1 / k_hi) ** 2) * k_hi - 1)
    return "C", math.sqrt(2) - 1


def k_exact(right_h: int, right_m: int, n: int, m: int) -> list:
    """Cautious interval of k from two exact intervals of the accuracies; holds with zero errors."""
    lo_h, hi_h = clopper_pearson(right_h, n)
    lo_m, hi_m = clopper_pearson(right_m, n)
    with np.errstate(invalid="ignore"):
        k_lo = float(d_from_acc(lo_m, m) / d_from_acc(hi_h, m))
        d_h_lo, d_m_hi = float(d_from_acc(lo_h, m)), float(d_from_acc(hi_m, m))
    k_hi = d_m_hi / d_h_lo if d_h_lo == d_h_lo and d_h_lo > 0 and d_m_hi == d_m_hi else None
    return [k_lo if k_lo == k_lo else 0.0, k_hi]


# --- the report -----------------------------------------------------------------------

def analyze(data: dict, m: int, B: int = 2000, seed: int = 534, delegate_seconds: float | None = None) -> dict:
    """From the columns of a sheet (`parse_sheet`) to the full measurement, as nested dicts."""
    key, person, machine = _norm(data["key"]), _norm(data["person"]), _norm(data["machine"])
    n = key.size
    h, mm = (person == key).astype(float), (machine == key).astype(float)
    weight = np.array([_float(x) if x else 1.0 for x in data["weight"]]) if "weight" in data else np.ones(n)
    weighted = bool(np.any(weight != 1.0))
    s = None
    if "pair" in data and all(data["pair"]):
        s = (_norm(data["pair"]) == key).astype(float)
    klass = None
    if m == 2:
        values = sorted(set(key))
        if len(values) != 2:
            raise SheetError("with two options the key must have two values (for example YES and NO)")
        klass = (key == values[0]).astype(int)
    gen = np.random.default_rng(seed)
    ref = {"cases": int(n), "dropped_rows": data.get("dropped", 0), "options": m, "weighted": weighted}

    # the 2x2 table and the errors
    def total(sel):
        return float(weight[sel].sum())
    n11, n10, n01, n00 = total((h == 1) & (mm == 1)), total((h == 1) & (mm == 0)), total((h == 0) & (mm == 1)), total((h == 0) & (mm == 0))
    err_m, err_h = int((mm == 0).sum()), int((h == 0).sum())
    ref["table"] = {"both_right": n11, "only_person": n10, "only_machine": n01, "neither": n00}
    ref["errors_seen"] = {"machine": err_m, "person": err_h, "common": int(((mm == 0) & (h == 0)).sum())}
    ref["overlap"] = {"person_wrong_where_machine_wrong": n00 / (n10 + n00) if n10 + n00 > 0 else None,
                      "person_wrong_where_machine_right": n01 / (n11 + n01) if n11 + n01 > 0 else None}

    # the point and the resampling of the cases
    point = _quantities(weight[None, :], h, mm, s, klass, m, None if m == 2 else np.zeros(G.size))
    if not bool(point["valid"][0]):
        raise SheetError("the person or the machine is at chance level: the sensitivities cannot be computed")
    tab = None
    if m > 2:   # the conversion table for these sensitivities, like convert_rho_mafc but with more draws
        tab = np.maximum.accumulate([sdt.rho_correct_from_rho_evidence(float(x), float(point["dh"][0]), float(point["dm"][0]), m, n=200_000) for x in G])
        point = _quantities(weight[None, :], h, mm, s, klass, m, tab)
    Wb = gen.poisson(1.0, size=(B, n)).astype(float) * weight[None, :]
    Wb = Wb[Wb.sum(1) > 0]
    boot = _quantities(Wb, h, mm, s, klass, m, tab)
    for name in ("acc_h", "acc_m", "dh", "dm", "k", "phi", "useful", "margin", "margin90"):
        ref[name] = {"estimate": float(point[name][0]), "ci95": _ci(boot[name])}
    ref["rho"] = float(point["rho"][0])

    # exact intervals for the accuracies and for k (unweighted): they hold even with zero errors
    if not weighted and m > 2:
        ref["acc_h"]["ci95_exact"] = list(clopper_pearson(int(h.sum()), n))
        ref["acc_m"]["ci95_exact"] = list(clopper_pearson(int(mm.sum()), n))
        ref["k"]["ci95_exact"] = k_exact(int(h.sum()), int(mm.sum()), n, m)
        if err_m == 0:
            ref["k"]["note"] = "the machine made no errors: k is a lower bound; its low end comes from the rule of three"
    # the angle is reported only if there are enough errors on both sides
    few = min(err_m, err_h)
    ref["phi"]["status"] = "not estimable" if few < MIN_ANGLE_ERRORS else ("indicative" if few < FEW_ERRORS else "estimated")
    ref["phi"]["above_useful_ci95"] = _ci(boot["phi"] - boot["useful"])

    # the ceiling: what the ideal pair can promise at most, with independent errors, at the least favourable k.
    # Resampling does not move errors that are not there: below five machine errors its interval of k
    # contains the truth less than nine times in ten, and the exact one is used
    use_exact = "ci95_exact" in ref["k"] and (err_m < MIN_RESAMPLING_ERRORS or ref["k"]["ci95"] is None)
    k_ci = ref["k"]["ci95_exact"] if use_exact else ref["k"]["ci95"]
    ref["k"]["interval_used"] = "exact" if use_exact else "resampling"
    zone, ceiling = zone_from_k(k_ci[0], k_ci[1] if k_ci[1] is not None else math.inf)
    ref["ceiling_at_90deg"] = ceiling
    ref["zone"] = zone
    strong_d = max(ref["dh"]["estimate"], ref["dm"]["estimate"])
    ref["errors_per_hundred"] = {"strongest_alone": _errors_per_hundred(strong_d, m),
                                 "ideal_pair_at_measured_angle": _errors_per_hundred(strong_d * (1 + ref["margin"]["estimate"]), m),
                                 "ideal_pair_at_ceiling": _errors_per_hundred(strong_d * (1 + ceiling), m)}
    k = ref["k"]["estimate"]
    if ref["phi"]["status"] != "not estimable" and ref["phi"]["estimate"] > ref["useful"]["estimate"]:
        kk = k if k >= 1 else 1 / k
        ref["weight_ratio"] = weight_ratio(kk, ref["phi"]["estimate"])

    # the alarm: agreement and disagreement between the two answers given alone
    agree = person == machine

    def share(num, den):
        return float(weight[num].sum() / weight[den].sum()) if weight[den].sum() > 0 else None
    ref["alarm"] = {"disagreements": share(~agree, np.ones(n, bool)),
                    "machine_error_in_agreement": share(agree & (mm == 0), agree),
                    "machine_error_in_disagreement": share(~agree & (mm == 0), ~agree),
                    "share_of_machine_errors_in_disagreement": share(~agree & (mm == 0), mm == 0),
                    "in_disagreement_machine_right": share(~agree & (mm == 1), ~agree),
                    "in_disagreement_person_right": share(~agree & (h == 1), ~agree),
                    "disagreements_seen": int((~agree).sum()), "agreements_on_a_wrong_one": int((agree & (mm == 0)).sum())}
    if "confidence" in data and any(data["confidence"]):
        level = np.array(data["confidence"])
        try:
            num = np.array([_float(x) for x in level])
            if np.unique(num).size > 5:
                t1, t2 = np.percentile(num, [100 / 3, 200 / 3])
                level = np.where(num <= t1, "1 low", np.where(num <= t2, "2 medium", "3 high"))
        except ValueError:
            pass
        ref["confidence"] = []
        for lv in sorted(set(level)):
            sel = level == lv
            d = sel & ~agree
            ref["confidence"].append({"level": str(lv), "cases": int(sel.sum()), "person_right": float(h[sel].mean()), "disagreements": int(d.sum()),
                                      "in_disagreement_person_right": int(h[d].sum()), "in_disagreement_machine_right": int(mm[d].sum())})

    # the pair
    if s is not None:
        for name in ("acc_s", "ds", "turing", "yield"):
            ref[name] = {"estimate": float(point[name][0]), "ci95": _ci(boot[name])}
        tu, yd = ref["turing"], ref["yield"]
        if yd["ci95"] and yd["ci95"][0] > 1:
            ref["pair_zone"] = "the pair beats the machine"
        elif yd["ci95"] and yd["ci95"][1] < 1 and tu["ci95"] and tu["ci95"][0] > 1:
            ref["pair_zone"] = "the machine helps the person, the person holds the machine back"
        elif tu["ci95"] and tu["ci95"][1] < 1:
            ref["pair_zone"] = "the machine weighs the person down"
        else:
            ref["pair_zone"] = "cannot be told apart"
        if "chat" in data:
            chat = _norm(data["chat"])
            speaks = chat != ""
            chat_right = (chat == key).astype(float)
            sa, sb = share(speaks, np.ones(n, bool)), share(~speaks, np.ones(n, bool))

            def mean(x, sel):
                return float(np.average(x[sel], weights=weight[sel])) if sel.any() else 0.0
            ref["where_the_points_go"] = {"chat_without_answer": sb,
                                          "conversation": sa * (mean(chat_right, speaks) - mean(mm, speaks)),
                                          "decision": sa * (mean(s, speaks) - mean(chat_right, speaks)),
                                          "silent_chat": sb * (mean(s, ~speaks) - mean(mm, ~speaks)),
                                          "total": mean(s, np.ones(n, bool)) - mean(mm, np.ones(n, bool))}
            dis = speaks & (chat != person)
            if dis.any():
                follows = (_norm(data["pair"]) == chat)
                ref["in_disagreements_with_the_chat"] = {"disagreements": int(dis.sum()), "chat_right": float(chat_right[dis].mean()),
                                                         "person_follows": float(follows[dis].mean()), "pair_right": float(s[dis].mean())}
        if "seconds_alone" in data and "seconds_pair" in data:
            ts = np.array([_float(x) for x in data["seconds_alone"]])
            tc = np.array([_float(x) for x in data["seconds_pair"]])
            ref["time"] = {"seconds_alone": float(ts.mean()), "seconds_pair": float(tc.mean()),
                           "power_over_person": tu["estimate"] ** 2 * float(ts.mean() / tc.mean())}
            if delegate_seconds:
                ref["time"]["seconds_delegate"] = delegate_seconds
                ref["time"]["power_over_machine"] = yd["estimate"] ** 2 * delegate_seconds / float(tc.mean())
                ref["time"]["yield_that_repays_checking"] = math.sqrt(float(tc.mean()) / delegate_seconds)

    if "stratum" in data and any(data["stratum"]):
        ref["strata"] = []
        stratum = np.array(data["stratum"])
        for lv in sorted(set(stratum)):
            sel = stratum == lv
            if sel.sum() < 30 or m == 2:
                ref["strata"].append({"stratum": str(lv), "cases": int(sel.sum()), "note": "fewer than 30 cases: counts only",
                                      "person_right": float(h[sel].mean()), "machine_right": float(mm[sel].mean())})
                continue
            ah, am = float(np.average(h[sel], weights=weight[sel])), float(np.average(mm[sel], weights=weight[sel]))
            with np.errstate(invalid="ignore"):
                kk = float(d_from_acc(am, m) / d_from_acc(ah, m))
            entry = {"stratum": str(lv), "cases": int(sel.sum()), "person_right": ah, "machine_right": am, "k": kk if kk == kk else None}
            if not weighted:
                ic = k_exact(int(h[sel].sum()), int(mm[sel].sum()), int(sel.sum()), m)
                entry["k_ci95_exact"] = ic
                entry["zone"] = zone_from_k(ic[0], ic[1] if ic[1] is not None else math.inf)[0]
            ref["strata"].append(entry)
        zones = {x["zone"] for x in ref["strata"] if "zone" in x}
        ref["strata_diverge"] = len(zones) > 1
    return ref


def _pc_(x, digits=0):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _pt(x):
    return f"{100 * x:+.1f}".replace("-", "−")


def _num(x, digits=2):
    return "n/a" if x is None or x != x else ("unbounded" if math.isinf(x) else f"{x:.{digits}f}")


def _range(ic, digits=2, pc=False):
    if not ic:
        return "interval not computable"
    f = (lambda v: _pc_(v, digits)) if pc else (lambda v: _num(v, digits))
    return f"from {f(ic[0])} to {f(ic[1])}"


RULES = {
    "A": ("The machine is clearly stronger than you on these cases (the whole interval of k is above 1.5).",
          ["The machine's answer stands, even when you disagree and even when you are very sure. In the data of the paper, with a machine "
           "twice as sensitive, in a disagreement the machine was right 86 times in a hundred and the person 10; those who did not follow it were right one time in four.",
           "The disagreement is an alarm: that is where most of the machine's errors fall. It serves to choose what a third source should check "
           "(a colleague, another method, the documents), not to decide on your own.",
           "Let the answer come out: give the machine the whole case, and if it does not answer, ask again, do not fall back on your own. In the data the biggest loss "
           "came from chats that stayed without an answer.",
           "Your own check, by itself, adds no sensitivity that can be measured: if it stays, it is for responsibility or because the two errors do not cost the same."]),
    "B": ("The machine is stronger, but not by much: you are in the middle zone (k above 1, not all above 1.5).",
          ["Here the model promises the pair a margin that may be worth having, and the data do not say how much of it you get.",
           "Do not draw a fixed rule from this calibration: add cases, and decide on the test of the pair.",
           "Meanwhile: where you agree, go ahead; a disagreement is checked with a third source."]),
    "C": ("You and the machine cannot be told apart, on these cases (the interval of k contains 1).",
          ["Neither of you is in charge. Where you agree the error is rarer; where you do not, neither opinion is enough.",
           "A disagreement is to be decided with something else: more time, a third source. In the data of the paper, at par, who was right split almost in half.",
           "The model says that a pair like this can gain a lot; in the data that gain was not clearly seen. The test of the pair is what counts."]),
    "D": ("You are stronger than the machine, but not by much (k below 1, not all below 0.67).",
          ["You decide. The machine is a second opinion: its disagreement marks the cases to look at again.",
           "Be careful not to be dragged along: with a weaker machine the people in the paper followed its wrong answers more than was good for them."]),
    "E": ("You are clearly stronger than the machine on these cases (the whole interval of k is below 0.67).",
          ["You decide: the machine adds no sensitivity that can be measured. Use it for what saves time, not for deciding.",
           "The risk is the opposite of the usual one: being persuaded by a wrong answer well put."]),
}


def report(ref: dict) -> str:
    """The report as text, in English."""
    m = ref["options"]
    r = []
    r.append(f"CALIBRATION: {ref['cases']} cases, {m} options" + (" (weighted)" if ref["weighted"] else "")
             + (f"; {ref['dropped_rows']} incomplete rows dropped" if ref["dropped_rows"] else ""))
    t, e = ref["table"], ref["errors_seen"]
    r.append(f"  both right {t['both_right']:.0f} | only you {t['only_person']:.0f} | only the machine {t['only_machine']:.0f} | neither {t['neither']:.0f}")
    so = ref["overlap"]
    r.append(f"  Where the machine errs, you err {_pc_(so['person_wrong_where_machine_wrong'])} of the time; where it is right, {_pc_(so['person_wrong_where_machine_right'])}.")
    r.append("")
    r.append("THE TWO READINGS, SEPARATE")
    for name, ka, kd in (("You, alone", "acc_h", "dh"), ("The machine, alone", "acc_m", "dm")):
        ic = ref[ka].get("ci95_exact") or ref[ka]["ci95"]
        r.append(f"  {name:<22} right answers {_pc_(ref[ka]['estimate'], 1)} ({_range(ic, 0, True)}); sensitivity {_num(ref[kd]['estimate'])} ({_range(ref[kd]['ci95'])})")
    k = ref["k"]
    r.append(f"  k, how many times the machine is worth you: {_num(k['estimate'])} ({_range(k['ci95'])}"
             + (f"; exact interval {_range(k['ci95_exact'])}" if "ci95_exact" in k else "") + ")")
    if "note" in k:
        r.append(f"    note: {k['note']}")
    r.append("")
    r.append("THE ANGLE AND THE MARGIN")
    phi = ref["phi"]
    if phi["status"] == "not estimable":
        r.append(f"  Angle between the errors: not estimable (machine errors {e['machine']}, yours {e['person']}; at least {MIN_ANGLE_ERRORS} per side are needed, and {FEW_ERRORS} for an interval within 15 degrees).")
    else:
        r.append(f"  Angle between the errors: {_num(phi['estimate'], 0)} degrees ({_range(phi['ci95'], 0)})"
                 + (f". INDICATIVE: only {min(e['machine'], e['person'])} errors on the side that errs less, {FEW_ERRORS} are needed" if phi["status"] == "indicative" else ""))
    r.append(f"  Useful angle: {_num(ref['useful']['estimate'], 0)} degrees ({_range(ref['useful']['ci95'], 0)})")
    if phi["status"] != "not estimable":
        su = phi["above_useful_ci95"]
        if su and su[0] > 0:
            r.append("  The angle is above the useful angle: the weaker one adds something, in principle.")
        elif su and su[1] < 0:
            r.append("  The angle is below the useful angle: the weaker one adds nothing to the ideal.")
        else:
            r.append("  Whether the angle is above or below the useful angle, these cases do not say.")
        r.append(f"  Margin of the ideal pair at the measured angle: {_pc_(ref['margin']['estimate'], 1)} of sensitivity ({_range(ref['margin']['ci95'], 0, True)})")
        if "weight_ratio" in ref:
            r.append(f"  In the ideal pair the evidence of the stronger weighs {_num(ref['weight_ratio'], 1)} times that of the weaker (it changes a lot in a few degrees).")
    r.append(f"  Ceiling: with independent errors (90 degrees) and the least favourable k of the interval ({ref['k']['interval_used']}), the ideal pair promises at most {_pc_(ref['ceiling_at_90deg'])} of sensitivity above the stronger.")
    ec = ref["errors_per_hundred"]
    r.append(f"  In errors per hundred cases: the stronger alone {_num(ec['strongest_alone'], 1)}; the ideal pair at the measured angle {_num(ec['ideal_pair_at_measured_angle'], 1)}; at the ceiling {_num(ec['ideal_pair_at_ceiling'], 1)}.")
    r.append("")
    r.append("THE ALARM (the two answers given alone)")
    a = ref["alarm"]
    r.append(f"  You disagree in {_pc_(a['disagreements'])} of the cases ({a['disagreements_seen']} cases).")
    r.append(f"  The machine errs in {_pc_(a['machine_error_in_agreement'], 1)} of the cases where you agree and in {_pc_(a['machine_error_in_disagreement'], 1)} where you do not; "
             f"{_pc_(a['share_of_machine_errors_in_disagreement'])} of its errors fall in the disagreements.")
    r.append(f"  In a disagreement the machine is right {_pc_(a['in_disagreement_machine_right'])} of the time, you {_pc_(a['in_disagreement_person_right'])}. "
             f"Agreements on a wrong answer: {a['agreements_on_a_wrong_one']} (the alarm does not sound there).")
    if "confidence" in ref:
        r.append("  By declared confidence:")
        for x in ref["confidence"]:
            few = " (few cases: do not make a rule of it)" if x["disagreements"] < 20 else ""
            r.append(f"    {x['level']:<10} cases {x['cases']:>4}, right {_pc_(x['person_right'])}; disagreements {x['disagreements']:>3}: you right {x['in_disagreement_person_right']}, the machine {x['in_disagreement_machine_right']}{few}")
    if "turing" in ref:
        r.append("")
        r.append("THE PAIR")
        r.append(f"  With the machine: right answers {_pc_(ref['acc_s']['estimate'], 1)}; sensitivity {_num(ref['ds']['estimate'])}")
        r.append(f"  Turing: {_num(ref['turing']['estimate'])} ({_range(ref['turing']['ci95'])}). Yield over the machine: {_num(ref['yield']['estimate'])} ({_range(ref['yield']['ci95'])}).")
        r.append(f"  Zone: {ref['pair_zone']}.")
        if "where_the_points_go" in ref:
            d = ref["where_the_points_go"]
            r.append(f"  Where the points go between the machine alone and the pair ({_pt(d['total'])} in all): conversation {_pt(d['conversation'])}, "
                     f"decision {_pt(d['decision'])}, chat without answer {_pt(d['silent_chat'])} ({_pc_(d['chat_without_answer'])} of the cases).")
        if "in_disagreements_with_the_chat" in ref:
            d = ref["in_disagreements_with_the_chat"]
            r.append(f"  In the {d['disagreements']} disagreements with the chat: the chat was right {_pc_(d['chat_right'])} of the time, you followed it {_pc_(d['person_follows'])}, "
                     f"the pair answered right {_pc_(d['pair_right'])}.")
        if "time" in ref:
            t = ref["time"]
            r.append(f"  Time per case: {_num(t['seconds_alone'], 0)} s alone, {_num(t['seconds_pair'], 0)} s with the machine. "
                     f"Power over you alone: {_num(t['power_over_person'])}.")
            if "power_over_machine" in t:
                r.append(f"  Against delegating ({_num(t['seconds_delegate'], 0)} s per case): power {_num(t['power_over_machine'])}; "
                         f"checking pays only with a yield above {_num(t['yield_that_repays_checking'])}.")
    if "strata" in ref:
        r.append("")
        r.append("BY KIND OF CASE (k changes with the task: do not use the overall k where the strata diverge)")
        for x in ref["strata"]:
            r.append(f"  {x['stratum']:<16} cases {x['cases']:>4}: you {_pc_(x['person_right'])}, machine {_pc_(x['machine_right'])}"
                     + (f", k {_num(x['k'])}" if x.get("k") else "")
                     + (f" (exact interval {_range(x['k_ci95_exact'])}), zone {x['zone']}" if "zone" in x else "") + (f" ({x['note']})" if "note" in x else ""))
        if ref.get("strata_diverge"):
            r.append("  WARNING: the zones change with the kind of case. The rule must be given kind by kind, not on the total.")
    r.append("")
    title, points = RULES[ref["zone"]]
    r.append(f"ZONE {ref['zone']}" + (" (on the total: see the warning on the kinds of case)" if ref.get("strata_diverge") else "") + f". {title}")
    for p in points:
        r.append(f"  - {p}")
    r.append("  This holds for these cases, this version of the model and this way of asking it. Inside the chat the machine is not the same: only the test of the pair says.")
    return "\n".join(r)
