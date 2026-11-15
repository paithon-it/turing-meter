#!/usr/bin/env python3
"""Two checks on the real ChatBench data (read from the experiment repository, without touching it).

The data are read from <experiment>/data (gitignored; fetch them with
`python -m angle.download data chatbench ...`, see the experiment repository).

1. Small calibrations on real data. From the ChatBench trials, "calibrations" of N cases are drawn
   at random (N questions with replacement, and for each question the answer of a random person):
   a fake person, made of many real people, with real, non-Gaussian questions. We look at how
   much k, angle, turing and yield fluctuate, and compare with what the model simulation predicts
   (results/sample.json, written by sim_sample.py). If the real data fluctuated much more, the
   figures on sample size would be optimistic.

2. The machine that changes its answer. ChatBench queried each model fifty times per question:
   the share of correct answers per question says how much of the errors is stable (the same
   question always wrong) and how much is noise between one call and the next.

Usage:  python3 verify_chatbench.py     # writes results/chatbench-check.json (and prints the tables)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import estimators as st
from estimators import sdt
from angle import sources

BASE = Path(st.REPO) / "data"
SEME = 20261115


def prove(model: str) -> pd.DataFrame:
    pairs = sources.chatbench_pairs(BASE)
    machine = sources.chatbench_machine_alone(BASE, model)
    x = pairs[(pairs.model == model) & pairs.question_id.isin(machine.index)].dropna(subset=["acc", "acc_ai"]).copy()
    x["m_alone"] = x.question_id.map(machine)
    return x.reset_index(drop=True)


def tarature(x: pd.DataFrame, n: int, conv, gen, R: int = 3000) -> dict:
    q, inv = np.unique(x.question_id.to_numpy(), return_inverse=True)
    per_domanda = [np.flatnonzero(inv == i) for i in range(q.size)]
    h, s, pm = x.acc.to_numpy(float), x.acc_ai.to_numpy(float), x.m_alone.to_numpy(float)
    c = np.zeros((R, 4))
    acc_s = np.zeros(R)
    for r in range(R):
        dom = gen.integers(0, q.size, n)
        righe = np.array([per_domanda[d][gen.integers(per_domanda[d].size)] for d in dom])
        mm = (gen.random(n) < pm[righe]).astype(float)        # a single machine call per case
        hh = h[righe]
        c[r] = [np.sum(hh * mm), np.sum(hh * (1 - mm)), np.sum((1 - hh) * mm), np.sum((1 - hh) * (1 - mm))]
        acc_s[r] = s[righe].mean()
    e = st.stima(c[:, 0], c[:, 1], c[:, 2], c[:, 3], 4, conv=conv)
    ds = st.d_da_acc(acc_s, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        tu, resa = ds / e["dh"], ds / e["dm"]
    return {"N": n, "k": st.quantili(e["k"]), "k_sd_log": float(np.nanstd(np.log(e["k"]))), "phi": st.quantili(e["phi"]), "phi_sd": float(np.nanstd(e["phi"])),
            "turing": st.quantili(tu), "turing_sd_log": float(np.nanstd(np.log(tu))), "resa": st.quantili(resa), "resa_sd_log": float(np.nanstd(np.log(resa)))}


def macchina_instabile(model: str) -> dict:
    a = pd.read_csv(BASE / "chatbench/ai_alone_answers.csv")
    p = a[a.model == model].free_text_acc.to_numpy(float)
    err = 1 - p
    return {"domande": int(p.size), "accuratezza_di_una_chiamata": float(p.mean()),
            "sempre_giuste": int((p == 1).sum()), "sempre_sbagliate": int((p == 0).sum()), "variabili": int(((p > 0) & (p < 1)).sum()),
            "errore_di_una_chiamata": float(err.mean()),
            # two independent calls on the same question: both wrong
            "errore_in_due_chiamate_su_due": float((err ** 2).mean()),
            "quota_dell_errore_che_si_ripete": float((err ** 2).mean() / err.mean()),
            # the two calls give different outcomes (one correct, one wrong): a lower bound on their disagreement
            "esiti_diversi_fra_due_chiamate": float((2 * p * err).mean()),
            # the majority of five calls
            "accuratezza_maggioranza_di_5": float(np.mean(sum(_binom(5, j) * p ** j * err ** (5 - j) for j in (3, 4, 5)))),
            "errori_su_domande_quasi_sempre_sbagliate": float(err[p <= 0.1].sum() / err.sum()),
            "errori_su_domande_quasi_sempre_giuste": float(err[p >= 0.9].sum() / err.sum())}


def _binom(n, k):
    from math import comb
    return comb(n, k)


def main():
    conv = st.Conversione(4)
    gen = np.random.default_rng(SEME + 7)
    out = {"tarature": {}, "macchina": {}}
    modello = json.load(open(os.path.join(st.HERE, "results", "sample.json")))["scenari"]
    for model, scen in (("gpt-4o", "gpt4o"), ("llama-3.1-8b", "llama")):
        x = prove(model)
        print(f"\n## {model}: {len(x)} trials, {x.question_id.nunique()} questions, {x.worker_id.nunique()} people; "
              f"people {100 * x.acc.mean():.1f}%, machine {100 * x.m_alone.mean():.1f}%, pair {100 * x.acc_ai.mean():.1f}%")
        out["tarature"][model] = []
        for n in (50, 100, 200, 400):
            r = tarature(x, n, conv, gen)
            mod = modello[scen]["per_N"][str(n)]
            r["modello_k_sd_log"], r["modello_phi_sd"] = mod["k_sd_log"], mod["phi_sd"]
            out["tarature"][model].append(r)
            print(f"N={n:4d} | k {r['k'][0]:.2f}-{r['k'][2]:.2f} (med {r['k'][1]:.2f}; sd log data {r['k_sd_log']:.3f}, model {mod['k_sd_log']:.3f}) | "
                  f"angle {r['phi'][0]:.0f}-{r['phi'][2]:.0f} (med {r['phi'][1]:.0f}; sd data {r['phi_sd']:.1f}, model {mod['phi_sd']:.1f}) | "
                  f"turing {r['turing'][0]:.2f}-{r['turing'][2]:.2f} | yield {r['resa'][0]:.2f}-{r['resa'][2]:.2f}", flush=True)
        out["macchina"][model] = macchina_instabile(model)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["macchina"][model].items()})
    json.dump(out, open(os.path.join(st.HERE, "results", "chatbench-check.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
