#!/usr/bin/env python3
"""A single person against a group that pools the cases.

The tool is that of the article: `measure._one_pass` (personal thresholds, tetrachoric, conversion
to four options), without resampling, on simulated calibrations. The process is that of
`calibration._raw`, with two additions: the cases can be the same for everyone or different for
each person, and people share a part of the noise that the machine does not have (on ChatBench
two people are at 66 degrees, the person and GPT-4o at 75).

Designs compared, with the average person at 58% and the machine at 88%:
  one person with N cases; P people on the same cases; P people on different cases.
We look at how much the group's k and the angle fluctuate from one calibration to another.

Usage:  python3 sim_group.py     # writes results/group.json (and prints the table)
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import estimators as st
from estimators import sdt
from angle import measure
from angle.measure import Trials

SEME = 20261115
DH, SD, DM, RHO, RHO_HH, M = 1.08, 0.40, 2.338, 0.267, 0.41, 4
D_MIN = 0.02


def genera(P: int, n: int, stessi: bool, gen: np.random.Generator, sd: float = SD) -> Trials:
    items = n if stessi else P * n
    sr, st_, sn = math.sqrt(RHO), math.sqrt(RHO_HH - RHO), math.sqrt(1.0 - RHO_HH)
    comune = gen.standard_normal((items, M))            # what people and machine have in common
    umano = gen.standard_normal((items, M))             # what only the people have in common
    di = np.clip(DH + sd * gen.standard_normal(P), D_MIN, None)
    person = np.repeat(np.arange(P), n)
    item = np.tile(np.arange(n), P) if stessi else np.arange(P * n)
    a = sr * comune[item] + st_ * umano[item] + sn * gen.standard_normal((item.size, M))
    a[:, 0] += di[person]
    b = sr * comune + math.sqrt(1.0 - RHO) * gen.standard_normal((items, M))
    b[:, 0] += DM
    return Trials(person=person, item=item, correct=(a.argmax(1) == 0).astype(float), machine=(b.argmax(1) == 0).astype(float))


def riferimento() -> dict:
    """The group value: the same process on a very large number of people."""
    gen = np.random.default_rng(SEME)
    p = genera(40_000, 25, False, gen)
    acc_h, acc_m = float(p.correct.mean()), float(p.machine.mean())
    dh, dm = sdt.d_from_accuracy(acc_h, M), sdt.d_from_accuracy(acc_m, M)
    return {"acc_h": acc_h, "acc_m": acc_m, "dh": dh, "dm": dm, "k": dm / dh, "phi": math.degrees(math.acos(RHO))}


def disegno(nome: str, P: int, n: int, stessi: bool, repliche: int, gen, rif: dict, sd: float = SD) -> dict:
    conv = measure.conversion_table(rif["dh"], rif["dm"], M, sd if P > 1 else 0.0, n)
    ks, phis, esclusi = [], [], 0
    for _ in range(repliche):
        p = genera(P, n, stessi, gen, sd if P > 1 else 0.0)
        if p.machine.mean() >= 1.0 or p.correct.mean() <= 0.26:
            esclusi += 1
            continue
        r = measure._one_pass(p, M, np.ones(P), np.ones(int(p.item.max()) + 1), conv)
        if r is None:
            esclusi += 1
            continue
        ks.append(r["k"])
        phis.append(r["phi"])
    ks, phis = np.array(ks), np.array(phis)
    return {"disegno": nome, "persone": P, "casi_a_testa": n, "stessi_casi": stessi, "risposte_umane": P * n,
            "casi_con_chiave": n if stessi else P * n, "errori_macchina_distinti_attesi": (n if stessi else P * n) * (1 - rif["acc_m"]),
            "k": st.quantili(ks), "k_sd_log": float(np.std(np.log(ks))), "phi": st.quantili(phis), "phi_sd": float(np.std(phis)),
            "repliche": int(ks.size), "escluse": esclusi}


def main():
    gen = np.random.default_rng(SEME + 3)
    rif = riferimento()
    print("group reference:", {k: round(v, 3) for k, v in rif.items()}, flush=True)
    disegni = [
        ("1 person, 100 cases", 1, 100, True), ("1 person, 200 cases", 1, 200, True), ("1 person, 500 cases", 1, 500, True),
        ("5 people, the same 100 cases", 5, 100, True), ("5 people, 100 different cases each", 5, 100, False),
        ("10 people, the same 50 cases", 10, 50, True), ("10 people, 50 different cases each", 10, 50, False),
        ("20 people, the same 100 cases", 20, 100, True), ("20 people, the same 200 cases", 20, 200, True),
        ("20 people, 25 different cases each", 20, 25, False), ("20 people, 100 different cases each", 20, 100, False),
        ("50 people, the same 100 cases", 50, 100, True), ("50 people, 40 different cases each", 50, 40, False),
    ]
    out = {"riferimento": rif, "parametri": {"dh": DH, "sd_fra_persone": SD, "dm": DM, "rho": RHO, "rho_fra_persone": RHO_HH, "m": M}, "disegni": []}
    for nome, P, n, stessi in disegni:
        t0 = time.time()
        d = disegno(nome, P, n, stessi, 300, gen, rif)
        out["disegni"].append(d)
        print(f"{nome:<40} answers {d['risposte_umane']:>5} keys {d['casi_con_chiave']:>5} err.M {d['errori_macchina_distinti_attesi']:>6.0f} | "
              f"k {d['k'][0]:.2f}-{d['k'][2]:.2f} (sd log {d['k_sd_log']:.3f}) | angle {d['phi'][0]:.0f}-{d['phi'][2]:.0f} (sd {d['phi_sd']:.1f}) "
              f"[{time.time() - t0:.0f} s]", flush=True)
    os.makedirs(os.path.join(st.HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(st.HERE, "results", "group.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
