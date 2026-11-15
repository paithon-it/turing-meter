#!/usr/bin/env python3
"""The pitfalls, in figures: how much they shift the estimates, with endless cases (systematic error).

1. Wrong keys: a share of cases has a wrong option as its key.
2. Contamination: a share of cases the machine already knows, and there it answers correctly anyway.
3. Cases too easy or too hard: the relative error of d' with 100 cases, by accuracy.
4. Yes/no against four options: precision for the same sensitivity, and what a shared bias
   towards the same answer does if the two classes are not separated.
5. Open answers: k for the same accuracies according to how many alternatives are assumed.

Usage:  python3 sim_pitfalls.py     # writes results/pitfalls.json (and prints the tables)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.stats import norm

import estimators as st
from estimators import sdt
from sim_sample import SCENARI, SEME, d4

N_TOT, BLOCCO = 2_000_000, 250_000


# --- 1. wrong keys ---------------------------------------------------------------

def chiavi_sbagliate(sc: dict, quota: float, modo: str, conv, seed: int) -> dict:
    """mode 'random': the wrong key is one of the wrong options, at random.
    mode 'trap': it is the wrong option that attracts person and machine together the most
    (the distractor with the highest common part): the case of a questionable key."""
    m, dh, dm, rho = sc["m"], sc["dh"], sc["dm"], sc["rho"]
    gen = np.random.default_rng(seed)
    c = np.zeros(4)
    err_chiave = 0.0
    w = math.sqrt(abs(rho))
    for _ in range(N_TOT // BLOCCO):
        n = BLOCCO
        common = gen.standard_normal((n, m))
        a = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
        b = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
        a[:, 0] += dh
        b[:, 0] += dm
        ca, cb = a.argmax(1), b.argmax(1)
        chiave = np.zeros(n, int)
        guasta = gen.random(n) < quota
        if modo == "random" or m == 2:
            chiave[guasta] = gen.integers(1, m, guasta.sum())
        else:
            chiave[guasta] = 1 + common[guasta, 1:].argmax(1)
        ha, hb = ca == chiave, cb == chiave
        c += [np.sum(ha & hb), np.sum(ha & ~hb), np.sum(~ha & hb), np.sum(~ha & ~hb)]
        err_chiave += np.sum(guasta & ~hb)
    e = st.stima(c[0], c[1], c[2], c[3], m, conv=conv)
    return {"quota": quota, "modo": modo, "acc_h": float(e["acc_h"][0]), "acc_m": float(e["acc_m"][0]), "k": float(e["k"][0]),
            "phi": float(e["phi"][0]), "utile": float(e["utile"][0]), "margine": float(e["margine"][0]),
            "quota_errori_macchina_dovuti_alla_chiave": float(err_chiave / max(c[1] + c[3], 1.0))}


# --- 2. contamination -----------------------------------------------------------------

def contaminazione(sc: dict, quota: float, conv) -> dict:
    p = st.celle(sc["dh"], sc["dm"], sc["rho"], sc["m"], n=N_TOT, seed=SEME)
    ah = p[0] + p[1]
    q = np.array([(1 - quota) * p[0] + quota * ah, (1 - quota) * p[1], (1 - quota) * p[2] + quota * (1 - ah), (1 - quota) * p[3]])
    e = st.stima(q[0] * 1e9, q[1] * 1e9, q[2] * 1e9, q[3] * 1e9, sc["m"], conv=conv)
    return {"quota": quota, "acc_m_misurata": float(e["acc_m"][0]), "k": float(e["k"][0]), "phi": float(e["phi"][0]),
            "utile": float(e["utile"][0]), "margine": float(e["margine"][0]), "margine90": float(e["margine90"][0])}


# --- 3. too easy, too hard -------------------------------------------------

def errore_relativo_d(m: int, n: int = 100) -> list:
    righe = []
    griglia = (0.30, 0.35, 0.40, 0.50, 0.58, 0.70, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99) if m == 4 else (0.55, 0.60, 0.65, 0.70, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99)
    gen = np.random.default_rng(SEME + 20)
    for a in griglia:
        x = gen.binomial(n, a, 200_000) / n
        d = st.d_da_acc(x, m)
        d0 = float(st.d_da_acc(a, m))
        ok = ~np.isnan(d) & (x < 1)
        q = np.percentile(d[ok], [2.5, 97.5])
        righe.append({"m": m, "accuratezza": a, "d": d0, "non_misurabile": float(1 - ok.mean()),
                      "errore_relativo": float(np.std(d[ok]) / d0), "intervallo_95": [float(q[0] / d0), float(q[1] / d0)]})
    return righe


# --- 4. yes/no against four options -----------------------------------------------------

def si_no(dh: float, dm: float, rho: float, n: int, ch: float, cm: float, gen, R: int = 3000) -> dict:
    """One person, balanced yes/no cases. Evidence at ±d/2, criterion c (0 is neutral), noises with correlation rho.

    The estimate is that of the tool for two classes: d' = z(yes on yes) + z(no on no), and rho within
    each class (`measure._one_pass` with `answer_class`). Alongside, the estimate that ignores the classes.
    """
    from angle import measure
    from angle.measure import Trials
    ident = lambda r, a=None, b=None: r
    ks, phis, phis_insieme = [], [], []
    classe = np.repeat([1, 0], n // 2)
    segno = np.where(classe == 1, 1.0, -1.0)
    cov = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    for _ in range(R):
        e = gen.standard_normal((classe.size, 2)) @ cov.T
        si_h = segno * dh / 2 + e[:, 0] > ch
        si_m = segno * dm / 2 + e[:, 1] > cm
        ok_h, ok_m = (si_h == (classe == 1)).astype(float), (si_m == (classe == 1)).astype(float)
        if ok_m.mean() >= 1 or ok_h.mean() <= 0.5 or ok_m.mean() <= 0.5:
            continue
        p = Trials(person=np.zeros(classe.size, int), item=np.arange(classe.size), correct=ok_h, machine=ok_m, answer_class=classe)
        r = measure._one_pass(p, 2, np.ones(1), np.ones(classe.size), ident)
        if r is None:
            continue
        ks.append(r["k"])
        phis.append(r["phi"])
        rc = st.tetracorica(ok_h.mean(), ok_m.mean(), (ok_h * ok_m).mean())
        phis_insieme.append(float(np.degrees(np.arccos(rc))))
    ks, phis, phis_insieme = np.array(ks), np.array(phis), np.array(phis_insieme)
    return {"N": n, "criterio_persona": ch, "criterio_macchina": cm, "k": st.quantili(ks), "k_sd_log": float(np.std(np.log(ks[ks > 0]))),
            "phi": st.quantili(phis), "phi_sd": float(np.std(phis)), "phi_senza_classi": st.quantili(phis_insieme), "repliche": int(ks.size)}


# --- 5. open answers ----------------------------------------------------------------

def aperte() -> list:
    righe = []
    for ah, am in ((0.58, 0.88), (0.71, 0.745), (0.58, 0.98)):
        for m in (2, 3, 4, 5, 10, 100, 1000):
            dh, dm = sdt.d_from_accuracy(ah, m), sdt.d_from_accuracy(am, m)
            righe.append({"acc_h": ah, "acc_m": am, "alternative_supposte": m, "dh": dh, "dm": dm, "k": dm / dh,
                          "utile": sdt.useful_angle(dh, dm), "margine90": math.hypot(dh, dm) / max(dh, dm) - 1})
    return righe


def main():
    conv = st.Conversione(4)
    out = {}
    print("## 1. Wrong keys")
    out["chiavi"] = {}
    # two options: in the forced-choice process accuracy is Phi(d / square root of 2); 1.812 and 2.476 give 90% and 96%
    storie = dict(nome="Two-option stories: person 90%, machine 96%, angle 68", dh=1.812, dm=2.476, rho=0.375, m=2)
    for nome, sc in (("gpt4o", SCENARI["gpt4o"]), ("forte", SCENARI["forte"]), ("pari", SCENARI["pari"]), ("storie", storie)):
        out["chiavi"][nome] = []
        for modo in (("random", "trap") if sc["m"] > 2 else ("random",)):
            for quota in (0.0, 0.01, 0.02, 0.05, 0.10):
                r = chiavi_sbagliate(sc, quota, modo, conv, SEME + 30)
                out["chiavi"][nome].append(r)
                print(f"{nome:<8}{modo:<10} wrong keys {100 * quota:4.0f}% | acc {100 * r['acc_h']:.1f} and {100 * r['acc_m']:.1f} | k {r['k']:.2f} | "
                      f"angle {r['phi']:.1f} | useful {r['utile']:.1f} | margin {100 * r['margine']:.1f}% | M errors due to the key "
                      f"{100 * r['quota_errori_macchina_dovuti_alla_chiave']:.0f}%", flush=True)
    print("\n## 2. Contamination")
    out["contaminazione"] = {}
    vera80 = dict(nome="True machine at 80%", dh=1.080, dm=d4(0.80), rho=0.267, m=4)
    for nome, sc in (("vera-80", vera80), ("gpt4o", SCENARI["gpt4o"]), ("pari", SCENARI["pari"])):
        out["contaminazione"][nome] = [contaminazione(sc, q, conv) for q in (0.0, 0.1, 0.2, 0.4, 0.6)]
        for r in out["contaminazione"][nome]:
            print(f"{nome:<8} cases already seen {100 * r['quota']:3.0f}% | measured machine {100 * r['acc_m_misurata']:.1f}% | k {r['k']:.2f} | "
                  f"angle {r['phi']:.1f} | useful {r['utile']:.1f} | margin {100 * r['margine']:.1f}% | margin at 90 {100 * r['margine90']:.1f}%")
    print("\n## 3. Relative error of d' with 100 cases")
    out["errore_relativo_d"] = errore_relativo_d(4) + errore_relativo_d(2)
    for r in out["errore_relativo_d"]:
        print(f"m={r['m']} acc {100 * r['accuratezza']:5.1f}% d' {r['d']:.2f} relative error {100 * r['errore_relativo']:5.1f}% "
              f"(95%: {r['intervallo_95'][0]:.2f}-{r['intervallo_95'][1]:.2f} times the true value) not measurable {100 * r['non_misurabile']:.1f}%")
    print("\n## 4. Yes/no against four options (same sensitivities as GPT-4o on ChatBench: 1.08 and 2.34; angle 74.5)")
    gen = np.random.default_rng(SEME + 40)
    out["si_no"] = []
    for n in (100, 200, 400, 800):
        for ch, cm in ((0.0, 0.0), (0.4, 0.4)):
            r = si_no(1.080, 2.338, 0.267, n, ch, cm, gen)
            out["si_no"].append(r)
            print(f"N={n:4d} criteria {ch:.1f}/{cm:.1f} | k {r['k'][0]:.2f}-{r['k'][2]:.2f} (sd log {r['k_sd_log']:.3f}) | "
                  f"angle by class {r['phi'][0]:.0f}-{r['phi'][2]:.0f} (med {r['phi'][1]:.0f}, sd {r['phi_sd']:.1f}) | "
                  f"angle without classes: median {r['phi_senza_classi'][1]:.0f}", flush=True)
    print("\n## 5. Open answers: k according to the assumed alternatives")
    out["aperte"] = aperte()
    for r in out["aperte"]:
        print(f"acc {100 * r['acc_h']:.0f}% and {100 * r['acc_m']:.1f}% | alternatives {r['alternative_supposte']:5d} | dH {r['dh']:.2f} dM {r['dm']:.2f} | "
              f"k {r['k']:.2f} | useful {r['utile']:.0f} | margin at 90 {100 * r['margine90']:.0f}%")
    os.makedirs(os.path.join(st.HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(st.HERE, "results", "pitfalls.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
