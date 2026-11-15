#!/usr/bin/env python3
"""The pair test and repetition over time: what is visible with N cases.

1. Turing and yield of a single person. On each case the person answers alone, the machine
   alone, and then the person with the machine: three answers, correct or wrong, eight cells.
   The simulated pair sums the evidence with a weight w (as in `calibration._raw`); w is chosen
   so that the pair yields as much as in the data, or as much as the ideal.
2. Repetition: two calibrations of the same person (or of a group) on two modules of different
   cases, some time apart. How far the sensitivity alone must fall for the drop to be visible.

Usage:  python3 sim_pair.py     # writes results/pair.json (and prints the tables)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.stats import norm

import estimators as st
from estimators import sdt

SEME = 20261115
ENNE = (50, 100, 200, 400, 800)

COPPIE = {
    "gpt4o-come-nei-dati": dict(nome="GPT-4o, pair as in the data (77% with the machine)", dh=1.080, dm=2.338, rho=0.267, w=0.45, m=4),
    "gpt4o-ideale": dict(nome="GPT-4o, ideal pair", dh=1.080, dm=2.338, rho=0.267, w=None, m=4),
    "pari-ideale": dict(nome="On par (ethics), ideal pair", dh=1.536, dm=1.663, rho=0.346, w=None, m=4),
    "pari-pesi-meta": dict(nome="On par (ethics), the person listens little to the machine", dh=1.536, dm=1.663, rho=0.346, w=0.3, m=4),
    "llama-ideale": dict(nome="Llama 3.1 8B, ideal pair", dh=1.170, dm=1.378, rho=0.258, w=None, m=4),
}


def otto_celle(sc: dict, n: int = 4_000_000, blocco: int = 250_000) -> np.ndarray:
    gen = np.random.default_rng(SEME)
    w = sc["w"]
    if w is None:
        w = (sc["dm"] - sc["rho"] * sc["dh"]) / (sc["dh"] - sc["rho"] * sc["dm"])
    c = np.zeros(8)
    for _ in range(n // blocco):
        ca, cb, a, b = st.scelte(blocco, sc["m"], sc["rho"], sc["dh"], sc["dm"], gen)
        s = (a + w * b).argmax(1) == 0
        codice = (ca == 0) * 4 + (cb == 0) * 2 + s * 1
        c += np.bincount(codice, minlength=8)
    return c / c.sum()


def grandezze(c: np.ndarray, m: int) -> dict:
    """c: (..., 8) counts of the cells coded as H*4 + M*2 + S."""
    n = c.sum(-1)
    idx = np.arange(8)
    ah = c[..., (idx & 4) > 0].sum(-1) / n
    am = c[..., (idx & 2) > 0].sum(-1) / n
    as_ = c[..., (idx & 1) > 0].sum(-1) / n
    dh, dm, ds = st.d_da_acc(ah, m), st.d_da_acc(am, m), st.d_da_acc(as_, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"turing": ds / dh, "resa": ds / dm, "k": dm / dh, "acc_s": as_}


def coppia(sc: dict, gen, R: int = 4000, Rb: int = 600, B: int = 400) -> dict:
    p = otto_celle(sc)
    vero = grandezze(p[None, :] * 1e9, sc["m"])
    vero = {k: float(v[0]) for k, v in vero.items()}
    out = {"scenario": sc, "vero": vero, "per_N": {}}
    for n in ENNE:
        c = gen.multinomial(n, p, size=R).astype(float)
        g = grandezze(c, sc["m"])
        cella = {"turing": st.quantili(g["turing"]), "resa": st.quantili(g["resa"]),
                 "turing_sd_log": float(np.nanstd(np.log(g["turing"]))), "resa_sd_log": float(np.nanstd(np.log(g["resa"])))}
        lo_r, hi_r, lo_t, hi_t = [], [], [], []
        for r in range(Rb):
            gb = grandezze(gen.poisson(c[r], size=(B, 8)).astype(float), sc["m"])
            a, b = np.nanpercentile(gb["resa"], [2.5, 97.5])
            lo_r.append(a); hi_r.append(b)
            a, b = np.nanpercentile(gb["turing"], [2.5, 97.5])
            lo_t.append(a); hi_t.append(b)
        lo_r, hi_r, lo_t, hi_t = (np.array(x) for x in (lo_r, hi_r, lo_t, hi_t))
        cella.update({"resa_ci_mediano": [float(np.median(lo_r)), float(np.median(hi_r))],
                      "turing_ci_mediano": [float(np.median(lo_t)), float(np.median(hi_t))],
                      "resa_copertura": float(np.mean((lo_r <= vero["resa"]) & (vero["resa"] <= hi_r))),
                      "turing_copertura": float(np.mean((lo_t <= vero["turing"]) & (vero["turing"] <= hi_t))),
                      "resa_tutta_sotto_uno": float(np.mean(hi_r < 1)), "resa_tutta_sopra_uno": float(np.mean(lo_r > 1)),
                      "turing_tutto_sopra_uno": float(np.mean(lo_t > 1))})
        out["per_N"][str(n)] = cella
    return out


def ripetizione(m: int = 4, acc0: float = 0.577, R: int = 20000) -> list:
    """Two calibrations on different modules: the power to see a drop in the sensitivity alone.

    The person (or the group, with answers pooled) goes from d0 to d0 * (1 - drop). The
    comparison is the one anyone would make: two proportions, two-sided test at 5%.
    For a group everyone is assumed to drop in the same way: the most favourable case.
    """
    gen = np.random.default_rng(SEME + 5)
    d0 = float(st.d_da_acc(acc0, m))
    righe = []
    for calo in (0.10, 0.20, 0.30, 0.40):
        acc1 = float(st.acc_da_d(d0 * (1 - calo), m))
        for risposte in (100, 200, 400, 1000, 3000, 10000):
            x0 = gen.binomial(risposte, acc0, R)
            x1 = gen.binomial(risposte, acc1, R)
            p0, p1 = x0 / risposte, x1 / risposte
            pm = (x0 + x1) / (2 * risposte)
            z = (p0 - p1) / np.sqrt(np.maximum(2 * pm * (1 - pm) / risposte, 1e-12))
            rapporto = st.d_da_acc(p1, m) / st.d_da_acc(p0, m)
            righe.append({"calo_di_d": calo, "acc_prima": acc0, "acc_dopo": acc1, "risposte_per_ondata": risposte,
                          "potenza": float(np.mean(np.abs(z) > norm.ppf(0.975))),
                          "rapporto_q": st.quantili(rapporto)})
    return righe


def main():
    gen = np.random.default_rng(SEME + 4)
    out = {"coppie": {}, "ripetizione": ripetizione()}
    for nome, sc in COPPIE.items():
        out["coppie"][nome] = coppia(sc, gen)
        v = out["coppie"][nome]["vero"]
        print(f"\n## {sc['nome']}: true turing {v['turing']:.2f}, true yield {v['resa']:.2f}, k {v['k']:.2f}, with the machine {100 * v['acc_s']:.1f}%")
        print("    N | turing 2.5-97.5 | yield 2.5-97.5 | CI95 yield (median) | yield cov. | yield CI below 1 | above 1 | turing CI above 1")
        for n, c in out["coppie"][nome]["per_N"].items():
            print(f"{int(n):5d} | {c['turing'][0]:.2f}-{c['turing'][2]:.2f} | {c['resa'][0]:.2f}-{c['resa'][2]:.2f} | "
                  f"{c['resa_ci_mediano'][0]:.2f}-{c['resa_ci_mediano'][1]:.2f} | {100 * c['resa_copertura']:.0f}% | "
                  f"{100 * c['resa_tutta_sotto_uno']:.0f}% | {100 * c['resa_tutta_sopra_uno']:.0f}% | {100 * c['turing_tutto_sopra_uno']:.0f}%")
    print("\n## Repetition: power to see a drop in the sensitivity alone (58% at the start, four options)")
    print("drop in d'  acc. after  answers per wave  power   ratio d'(t)/d'(0): 2.5-97.5")
    for r in out["ripetizione"]:
        print(f"{100 * r['calo_di_d']:8.0f}%  {100 * r['acc_dopo']:8.1f}%  {r['risposte_per_ondata']:18d}  {100 * r['potenza']:6.0f}%   "
              f"{r['rapporto_q'][0]:.2f}-{r['rapporto_q'][2]:.2f}")
    os.makedirs(os.path.join(st.HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(st.HERE, "results", "pair.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
