#!/usr/bin/env python3
"""The two-phase design: the machine answers the whole archive, the person a chosen sample.

The person's hours are costly, the machine's are not. If there is an archive of cases with the key,
the machine does them all; the person does a few, drawn at random within two strata: the cases the
machine got wrong and those it got right (without knowing which are which). Each answer weighs as
many archive cases as it represents. The angle lives on the machine's errors: with this design
the person sees many more of them for the same hours.

Comparison, for the same number of the person's answers:
  random             the person does n cases taken at random from the archive;
  half and half      half of their cases among the machine's errors (or all of them, if fewer);
  all errors         all the machine's errors, the rest among the correct ones (at least 30).
In all three the machine's sensitivity comes from the whole archive.

Usage:  python3 sim_two_phase.py     # writes results/two-phase.json (and prints the table)
"""
from __future__ import annotations

import json
import os

import numpy as np

import estimators as st
from sim_sample import SCENARI, SEME, vero


def una(sc: dict, p: np.ndarray, archivio: int, n: int, modo: str, conv, gen, R: int = 4000) -> dict:
    m = sc["m"]
    am = p[0] + p[2]
    h_se_m_ok, h_se_m_err = p[0] / (p[0] + p[2]), p[1] / (p[1] + p[3])
    n_err_arch = gen.binomial(archivio, 1 - am, R)
    n_ok_arch = archivio - n_err_arch
    if modo == "random":
        ne = gen.hypergeometric(n_err_arch, n_ok_arch, n)
    elif modo == "half and half":
        ne = np.minimum(n // 2, n_err_arch)
    else:
        ne = np.minimum(n - 30, n_err_arch)
    nr = n - ne
    xe, xr = gen.binomial(ne, h_se_m_err), gen.binomial(nr, h_se_m_ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        we = np.where(ne > 0, n_err_arch / np.maximum(ne, 1), 0.0)
        wr = np.where(nr > 0, n_ok_arch / np.maximum(nr, 1), 0.0)
    e = st.stima(xr * wr, xe * we, (nr - xr) * wr, (ne - xe) * we, m, conv=conv)
    e["phi"] = np.where(ne >= 2, e["phi"], np.nan)
    return {"archivio": archivio, "risposte_persona": n, "modo": modo, "errori_macchina_visti_dalla_persona": float(ne.mean()),
            "k": st.quantili(e["k"]), "k_sd_log": float(np.nanstd(np.log(e["k"]))), "phi": st.quantili(e["phi"]), "phi_sd": float(np.nanstd(e["phi"])),
            "angolo_non_stimabile": float(np.isnan(e["phi"]).mean()), "margine": st.quantili(e["margine"])}


def main():
    conv = st.Conversione(4)
    gen = np.random.default_rng(SEME + 6)
    out = {}
    for nome in ("gpt4o", "mezzo", "haiku", "forte"):
        sc = SCENARI[nome]
        p = st.celle(sc["dh"], sc["dm"], sc["rho"], sc["m"], seed=SEME)
        v = vero(sc)
        out[nome] = {"vero": v, "disegni": []}
        print(f"\n## {sc['nome']}: k {v['k']:.2f}, angle {v['phi']:.1f}, useful {v['utile']:.1f}")
        print(f"{'archive':>9}{'answers':>9}  {'mode':<17}{'err.M seen':>12}{'k 2.5-97.5':>14}{'sd log k':>10}{'angle 2.5-97.5':>18}{'sd':>6}{'n.e.':>7}")
        for archivio in (500, 1000, 3000):
            for n in (100, 200):
                for modo in ("random", "half and half", "all errors"):
                    r = una(sc, p, archivio, n, modo, conv, gen)
                    out[nome]["disegni"].append(r)
                    print(f"{archivio:>9}{n:>9}  {modo:<17}{r['errori_macchina_visti_dalla_persona']:>12.1f}"
                          f"{r['k'][0]:>8.2f}-{r['k'][2]:<5.2f}{r['k_sd_log']:>10.3f}{r['phi'][0]:>11.0f}-{r['phi'][2]:<6.0f}{r['phi_sd']:>6.1f}"
                          f"{100 * r['angolo_non_stimabile']:>6.1f}%", flush=True)
    os.makedirs(os.path.join(st.HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(st.HERE, "results", "two-phase.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
