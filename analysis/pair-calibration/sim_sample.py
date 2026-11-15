#!/usr/bin/env python3
"""How many cases a single person needs to estimate k, the angle and the margin.

For each scenario (true sensitivities of person and machine, true angle, m options) and each
number of cases N:
  1. the probabilities of the four cells (correct or wrong, for person and machine) from the
     generative process of the repository, over 4 million trials;
  2. R simulated calibrations, one multinomial each, and where their estimates fall;
  3. on a subset of the calibrations, resampling of the cases with Poisson weights (that of
     `measure.compare`, which with a single person resamples only the cases): width and coverage
     of the 95% interval, and how often it allows the readings on which a usage rule rests.

Usage:
    python3 sim_sample.py                 # all scenarios, writes results/sample.json and results/sample.txt
    python3 sim_sample.py --verify        # the fast path against measure.compare of the repository (results/verification.json)
    python3 sim_sample.py --errors        # angle precision as a function of machine errors (results/machine-errors.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from scipy.stats import beta

import estimators as st
from estimators import sdt

SEME = 20261115
ENNE = (50, 100, 200, 400, 800, 1600)


def d4(a: float) -> float:
    return sdt.d_from_accuracy(a, 4)


SCENARI = {
    "gpt4o": dict(nome="Person at 58%, machine at 88% (GPT-4o on ChatBench)", dh=1.080, dm=2.338, rho=0.267, m=4),
    "pari": dict(nome="On par (GPT-4o on ethics: 71% and 74.5%)", dh=1.536, dm=1.663, rho=0.346, m=4),
    "llama": dict(nome="Machine slightly above the person (Llama 3.1 8B: 60.5% and 67%)", dh=1.170, dm=1.378, rho=0.258, m=4),
    "mezzo": dict(nome="Middle zone (k = 1.5; 65% and 82%)", dh=1.30, dm=1.95, rho=0.26, m=4),
    "haiku": dict(nome="Machine at 93% (like Claude Haiku 5.5), independent errors", dh=1.080, dm=d4(0.93), rho=0.0, m=4),
    "forte": dict(nome="Machine at 98% (like Claude Sonnet 5.5), independent errors", dh=1.080, dm=d4(0.98), rho=0.0, m=4),
}


def vero(sc: dict) -> dict:
    k = sc["dm"] / sc["dh"]
    phi = math.degrees(math.acos(sc["rho"]))
    return {"k": k, "phi": phi, "utile": sdt.useful_angle(sc["dh"], sc["dm"]),
            "margine": sdt.ideal(sc["dh"], sc["dm"], sc["rho"]) / max(sc["dh"], sc["dm"]) - 1,
            "margine90": math.hypot(sc["dh"], sc["dm"]) / max(sc["dh"], sc["dm"]) - 1,
            "acc_h": float(st.acc_da_d(sc["dh"], sc["m"])), "acc_m": float(st.acc_da_d(sc["dm"], sc["m"]))}


def k_esatto(nh_ok, nm_ok, n, m):
    """Bounds of k from two exact intervals (Clopper-Pearson), one for each accuracy.

    Each bound is at a 2.5% tail; together they give a conservative interval for k, which does
    not depend on resampling and stays defined with zero machine errors.
    """
    def cp(x, lato):
        x = np.asarray(x, float)
        with np.errstate(invalid="ignore"):
            lo = np.where(x > 0, beta.ppf(0.025, np.maximum(x, 1e-9), n - x + 1), 0.0)
            hi = np.where(x < n, beta.ppf(0.975, x + 1, np.maximum(n - x, 1e-9)), 1.0)
        return lo if lato == "lo" else hi
    with np.errstate(invalid="ignore", divide="ignore"):
        k_lo = st.d_da_acc(cp(nm_ok, "lo"), m) / st.d_da_acc(cp(nh_ok, "hi"), m)
        k_hi = st.d_da_acc(cp(nm_ok, "hi"), m) / st.d_da_acc(cp(nh_ok, "lo"), m)
    return k_lo, k_hi


def margine90(k):
    k = np.asarray(k, float)
    k = np.where(k < 1, 1 / k, k)
    return np.sqrt(1 + k * k) / k - 1


def una_cella(sc: dict, p: np.ndarray, n: int, conv, gen, R: int, Rb: int, B: int) -> dict:
    m, v = sc["m"], vero(sc)
    conteggi = gen.multinomial(n, p[:4] / p[:4].sum(), size=R).astype(float)
    e = st.stima(conteggi[:, 0], conteggi[:, 1], conteggi[:, 2], conteggi[:, 3], m, conv=conv)
    ok, ang = e["valido"], e["angolo_ok"]
    out = {"N": n, "errori_macchina_attesi": n * (p[1] + p[3]), "errori_persona_attesi": n * (p[2] + p[3]),
           "errori_comuni_attesi": n * p[3],
           "non_valida": float(1 - ok.mean()), "angolo_non_stimabile": float(1 - ang.mean()),
           "zero_errori_macchina": float((e["errori_m"] == 0).mean()),
           "angolo_al_limite": float(np.nanmean((e["phi"] < 8.5) | (e["phi"] > 126.5)))}
    for key in ("k", "phi", "utile", "margine", "sopra_utile"):
        out[key] = st.quantili(e[key])
        out[key + "_sd"] = float(np.nanstd(e[key]))
    out["k_sd_log"] = float(np.nanstd(np.log(e["k"][ok & (e["k"] > 0)])))
    out["punto_sopra_utile"] = float(np.nanmean(e["sopra_utile"] > 0))

    # --- intervals: resampling of the cases, replicate by replicate
    righe = np.flatnonzero(ok)[:Rb]
    ci = {key: [] for key in ("k", "phi", "utile", "margine", "sopra_utile")}
    for r in righe:
        c = conteggi[r]
        tab = conv.tabella(e["dh"][r], e["dm"][r])[0] if m > 2 else None
        b = gen.poisson(c, size=(B, 4)).astype(float)
        eb = st.stima(b[:, 0], b[:, 1], b[:, 2], b[:, 3], m, conv=conv, tabella=tab)
        for key in ci:
            x = eb[key][~np.isnan(eb[key])]
            ci[key].append([np.percentile(x, 2.5), np.percentile(x, 97.5)] if x.size >= B // 2 else [np.nan, np.nan])
    ci = {key: np.array(val) for key, val in ci.items()}
    for key in ("k", "phi", "utile", "margine"):
        lo, hi = ci[key][:, 0], ci[key][:, 1]
        out[key + "_ci_mediano"] = [float(np.nanmedian(lo)), float(np.nanmedian(hi))]
        out[key + "_copertura"] = float(np.nanmean((lo <= v[key] + 1e-9) & (v[key] - 1e-9 <= hi)))
    out["ci_non_calcolabile_angolo"] = float(np.isnan(ci["phi"][:, 0]).mean())
    k_lo, k_hi = ci["k"][:, 0], ci["k"][:, 1]
    ke_lo, ke_hi = k_esatto(conteggi[righe, 0] + conteggi[righe, 1], conteggi[righe, 0] + conteggi[righe, 2], n, m)
    ke_lo = np.where(np.isnan(ke_lo), 0.0, ke_lo)
    ke_hi = np.where(np.isnan(ke_hi), np.inf, ke_hi)
    out["k_esatto_mediano"] = [float(np.median(ke_lo)), float(np.median(ke_hi[np.isfinite(ke_hi)])) if np.isfinite(ke_hi).any() else None]
    out["k_esatto_copertura"] = float(np.mean((ke_lo <= v["k"]) & (v["k"] <= ke_hi)))
    letture = {
        "macchina_piu_forte": k_lo > 1, "persona_piu_forte": k_hi < 1, "k_comprende_uno": (k_lo <= 1) & (k_hi >= 1),
        "k_almeno_1.5": k_lo >= 1.5, "k_almeno_2": k_lo >= 2.0,
        "tetto90_entro_10": (k_lo > 1) & (margine90(k_lo) <= 0.10), "tetto90_entro_15": (k_lo > 1) & (margine90(k_lo) <= 0.15),
        "tetto90_entro_20": (k_lo > 1) & (margine90(k_lo) <= 0.20),
        "esatto_macchina_piu_forte": ke_lo > 1, "esatto_k_comprende_uno": (ke_lo <= 1) & (ke_hi >= 1),
        "esatto_tetto90_entro_10": (ke_lo > 1) & (margine90(np.maximum(ke_lo, 1.0001)) <= 0.10),
        "esatto_tetto90_entro_15": (ke_lo > 1) & (margine90(np.maximum(ke_lo, 1.0001)) <= 0.15),
        "esatto_tetto90_entro_20": (ke_lo > 1) & (margine90(np.maximum(ke_lo, 1.0001)) <= 0.20),
        "margine_misurato_entro_10": ci["margine"][:, 1] <= 0.10, "margine_misurato_entro_20": ci["margine"][:, 1] <= 0.20,
        "margine_misurato_sopra_zero": ci["margine"][:, 0] > 0.005,
        "angolo_sopra_utile": ci["sopra_utile"][:, 0] > 0, "angolo_sotto_utile": ci["sopra_utile"][:, 1] < 0,
        "angolo_sotto_90": ci["phi"][:, 1] < 90,
    }
    out["letture"] = {key: float(np.mean(val)) for key, val in letture.items()}
    out["repliche"], out["repliche_con_intervallo"], out["ricampionamenti"] = R, int(righe.size), B
    return out


def corri(scenari, enne, R, Rb, B) -> dict:
    conv = st.Conversione(4)
    gen = np.random.default_rng(SEME)
    out = {"seme": SEME, "scenari": {}}
    for nome in scenari:
        sc = SCENARI[nome]
        t0 = time.time()
        p = st.celle(sc["dh"], sc["dm"], sc["rho"], sc["m"], seed=SEME)
        voce = {"scenario": sc, "vero": vero(sc), "celle": p.tolist(), "per_N": {}}
        for n in enne:
            voce["per_N"][str(n)] = una_cella(sc, p, n, conv, gen, R, Rb, B)
        out["scenari"][nome] = voce
        print(f"{nome}: {time.time() - t0:.0f} s", flush=True)
    return out


# English column labels for the readings table (the dictionary keys stay as they are)
LETTURE_EN = {"macchina_piu_forte": "machine_stronger", "k_comprende_uno": "k_includes_one", "k_almeno_1.5": "k_at_least_1.5", "k_almeno_2": "k_at_least_2",
              "tetto90_entro_10": "ceiling90_within_10", "tetto90_entro_15": "ceiling90_within_15", "tetto90_entro_20": "ceiling90_within_20",
              "esatto_macchina_piu_forte": "exact_machine_stronger", "esatto_tetto90_entro_10": "exact_ceiling90_in_10", "esatto_tetto90_entro_20": "exact_ceiling90_in_20",
              "margine_misurato_entro_10": "margin_within_10", "margine_misurato_sopra_zero": "margin_above_zero", "angolo_sopra_utile": "angle_above_useful",
              "angolo_sotto_utile": "angle_below_useful", "angolo_sotto_90": "angle_below_90"}


def tabella(res: dict) -> str:
    righe = []
    for nome, voce in res["scenari"].items():
        v = voce["vero"]
        righe.append(f"\n## {voce['scenario']['nome']}")
        righe.append(f"true: k {v['k']:.2f}, angle {v['phi']:.1f}, useful angle {v['utile']:.1f}, margin {100 * v['margine']:.1f}%, "
                     f"margin at 90 degrees {100 * v['margine90']:.1f}%; accuracies {100 * v['acc_h']:.1f} and {100 * v['acc_m']:.1f}")
        righe.append("    N  err.M  common | k: 2.5-97.5 of the estimates | angle: 2.5-97.5  (sd) | margin %: 2.5-97.5 | CI95 k | CI95 angle | cov. k/ang/marg | angle not est.")
        for n, c in voce["per_N"].items():
            righe.append(
                f"{int(n):5d} {c['errori_macchina_attesi']:6.1f} {c['errori_comuni_attesi']:6.1f} | "
                f"{c['k'][0]:.2f}-{c['k'][2]:.2f} (med {c['k'][1]:.2f}) | {c['phi'][0]:.0f}-{c['phi'][2]:.0f} (sd {c['phi_sd']:.1f}) | "
                f"{100 * c['margine'][0]:.1f}-{100 * c['margine'][2]:.1f} (med {100 * c['margine'][1]:.1f}) | "
                f"{c['k_ci_mediano'][0]:.2f}-{c['k_ci_mediano'][1]:.2f} | {c['phi_ci_mediano'][0]:.0f}-{c['phi_ci_mediano'][1]:.0f} | "
                f"{100 * c['k_copertura']:.0f}/{100 * c['phi_copertura']:.0f}/{100 * c['margine_copertura']:.0f} | {100 * c['angolo_non_stimabile']:.1f}%")
        righe.append("  readings allowed by the interval (share of calibrations):")
        chiavi = ("macchina_piu_forte", "k_comprende_uno", "k_almeno_1.5", "k_almeno_2", "tetto90_entro_10", "tetto90_entro_15", "tetto90_entro_20",
                  "esatto_macchina_piu_forte", "esatto_tetto90_entro_10", "esatto_tetto90_entro_20",
                  "margine_misurato_entro_10", "margine_misurato_sopra_zero", "angolo_sopra_utile", "angolo_sotto_utile", "angolo_sotto_90")
        righe.append("    N    " + "  ".join(f"{LETTURE_EN[x][:22]:>22}" for x in chiavi))
        for n, c in voce["per_N"].items():
            righe.append(f"{int(n):5d}    " + "  ".join(f"{100 * c['letture'][x]:21.0f}%" for x in chiavi))
        righe.append("  coverage of the exact interval of k: " + ", ".join(f"N={n}: {100 * c['k_esatto_copertura']:.0f}%" for n, c in voce["per_N"].items()))
    return "\n".join(righe)


def verifica(repliche: int = 30) -> dict:
    """The fast path against the repository tool, on the same simulated calibrations."""
    from angle import measure
    from angle.measure import Trials
    conv = st.Conversione(4)
    gen = np.random.default_rng(SEME + 1)
    out = {}
    for nome, n in (("gpt4o", 100), ("gpt4o", 400), ("pari", 100), ("pari", 400), ("haiku", 400)):
        sc = SCENARI[nome]
        diff_phi, diff_k, larg_r, larg_v = [], [], [], []
        for _ in range(repliche):
            ca, cb, _, _ = st.scelte(n, sc["m"], sc["rho"], sc["dh"], sc["dm"], gen)
            a, b = (ca == 0).astype(float), (cb == 0).astype(float)
            if a.mean() <= 0.26 or b.mean() <= 0.26 or b.mean() == 1 or a.mean() == 1:
                continue
            r = measure.compare(Trials(person=np.zeros(n, int), item=np.arange(n), correct=a, machine=b), sc["m"], repetitions=300,
                                seed=int(gen.integers(1 << 30)))
            c = [np.sum(a * b), np.sum(a * (1 - b)), np.sum((1 - a) * b), np.sum((1 - a) * (1 - b))]
            e = st.stima(*c, sc["m"], conv=conv)
            tab = conv.tabella(e["dh"][0], e["dm"][0])[0]
            bb = gen.poisson(c, size=(400, 4)).astype(float)
            eb = st.stima(bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3], sc["m"], tabella=tab)
            diff_phi.append(e["phi"][0] - r["phi"])
            diff_k.append(e["k"][0] / r["k"] - 1)
            larg_r.append(r["phi_ci"][1] - r["phi_ci"][0])
            larg_v.append(np.nanpercentile(eb["phi"], 97.5) - np.nanpercentile(eb["phi"], 2.5))
        out[f"{nome}-{n}"] = {"repliche": len(diff_phi), "angolo_differenza_media": float(np.mean(diff_phi)),
                              "angolo_differenza_assoluta_mediana": float(np.median(np.abs(diff_phi))),
                              "angolo_differenza_assoluta_massima": float(np.max(np.abs(diff_phi))),
                              "k_differenza_relativa_massima": float(np.max(np.abs(diff_k))),
                              "larghezza_ic_angolo_repository": float(np.median(larg_r)), "larghezza_ic_angolo_rapida": float(np.median(larg_v))}
        print(f"{nome}-{n}", out[f"{nome}-{n}"], flush=True)
    return out


def errori(R: int = 4000) -> dict:
    """Angle precision as a function of the expected machine errors, at several accuracies."""
    conv = st.Conversione(4)
    gen = np.random.default_rng(SEME + 2)
    out = []
    for dh, etichetta_h in ((1.080, "person at 58%"), (1.95, "person at 82%")):
        for acc_m in (0.70, 0.80, 0.88, 0.93, 0.96, 0.98):
            for rho in (0.0, 0.267):
                dm = d4(acc_m)
                p = st.celle(dh, dm, rho, 4, n=2_000_000, seed=SEME)
                for attesi in (10, 20, 40, 80, 160, 320):
                    n = int(round(attesi / (p[1] + p[3])))
                    c = gen.multinomial(n, p[:4] / p[:4].sum(), size=R).astype(float)
                    e = st.stima(c[:, 0], c[:, 1], c[:, 2], c[:, 3], 4, conv=conv)
                    q = st.quantili(e["phi"])
                    out.append({"persona": etichetta_h, "acc_m": acc_m, "rho": rho, "phi_vero": math.degrees(math.acos(rho)), "errori_attesi": attesi, "N": n,
                                "errori_comuni_attesi": n * p[3], "phi_sd": float(np.nanstd(e["phi"])), "phi_q": q,
                                "mezza_larghezza_95": (q[2] - q[0]) / 2, "non_stimabile": float(np.isnan(e["phi"]).mean()),
                                "al_limite": float(np.nanmean((e["phi"] < 8.5) | (e["phi"] > 126.5)))})
        print(etichetta_h, "done", flush=True)
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--scenari", default=",".join(SCENARI))
    a.add_argument("--enne", default=",".join(str(n) for n in ENNE))
    a.add_argument("--repliche", type=int, default=4000)
    a.add_argument("--con-intervallo", type=int, default=500)
    a.add_argument("--ricampionamenti", type=int, default=400)
    a.add_argument("--verify", action="store_true")
    a.add_argument("--errors", action="store_true")
    x = a.parse_args()
    os.makedirs(os.path.join(st.HERE, "results"), exist_ok=True)
    if x.verify:
        json.dump(verifica(), open(os.path.join(st.HERE, "results", "verification.json"), "w"), indent=1)
        return
    if x.errors:
        res = errori()
        json.dump(res, open(os.path.join(st.HERE, "results", "machine-errors.json"), "w"), indent=1)
        print(f"{'person':<18}{'acc M':>6}{'rho':>6}{'err.M':>7}{'N':>7}{'common':>8}{'sd ang':>8}{'±95%':>7}{'limit':>8}{'n.e.':>7}")
        for r in res:
            print(f"{r['persona']:<18}{r['acc_m']:>6.2f}{r['rho']:>6.2f}{r['errori_attesi']:>7d}{r['N']:>7d}{r['errori_comuni_attesi']:>8.1f}"
                  f"{r['phi_sd']:>8.1f}{r['mezza_larghezza_95']:>7.0f}{100 * r['al_limite']:>7.1f}%{100 * r['non_stimabile']:>6.1f}%")
        return
    res = corri(x.scenari.split(","), [int(n) for n in x.enne.split(",")], x.repliche, x.con_intervallo, x.ricampionamenti)
    json.dump(res, open(os.path.join(st.HERE, "results", "sample.json"), "w"), indent=1)
    testo = tabella(res)
    open(os.path.join(st.HERE, "results", "sample.txt"), "w").write(testo + "\n")
    print(testo)


if __name__ == "__main__":
    main()
