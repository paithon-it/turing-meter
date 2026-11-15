#!/usr/bin/env python3
"""A made-up calibration sheet, to try `turing-meter pair`: nobody's data.

A person at 58 per cent and a machine at 88 on four-option cases (the values of GPT-4o on
ChatBench), with the true angle at 74.5 degrees; a second, rarer kind of case where the two are
almost level. The chat stays silent in one case out of five; where it answers, the person adds
its evidence to their own with a low weight, as in the data.

Usage:  python3 make_example.py [N] [seed]     # writes docs/examples/pair-sheet-example.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

import estimators as st

LETTERE = np.array(list("ABCD"))


def genera(n: int, seme: int) -> list[dict]:
    gen = np.random.default_rng(seme)
    righe = []
    blocchi = (("ordinary", int(round(n * 0.7)), 1.15, 2.60, 0.267), ("atypical", n - int(round(n * 0.7)), 1.45, 1.60, 0.30))
    for strato, quanti, dh, dm, rho in blocchi:
        ca, cb, a, b = st.scelte(quanti, 4, rho, dh, dm, gen)
        ordinati = np.sort(a, axis=1)
        distanza = ordinati[:, -1] - ordinati[:, -2]
        for i in range(quanti):
            perm = gen.permutation(4)                       # perm[option] = position of the letter
            lettera = lambda opzione: LETTERE[perm[opzione]]
            muta = gen.random() < 0.21
            in_chat = cb[i] if gen.random() < 0.9 else int((b[i] + 0.8 * gen.standard_normal(4)).argmax())
            finale = ca[i] if muta else int((a[i] + 0.45 * b[i] + 0.9 * (np.arange(4) == in_chat)).argmax())
            righe.append({"case": "", "stratum": strato, "key": lettera(0), "person": lettera(ca[i]),
                          "confidence": 1 + int(distanza[i] > 0.55) + int(distanza[i] > 1.35), "machine": lettera(cb[i]),
                          "chat": "" if muta else lettera(in_chat), "pair": lettera(finale),
                          "seconds_alone": int(round(float(np.exp(gen.normal(np.log(70), 0.45))))),
                          "seconds_pair": int(round(float(np.exp(gen.normal(np.log(110), 0.45)))))})
    ordine = gen.permutation(len(righe))
    righe = [righe[i] for i in ordine]
    for i, r in enumerate(righe, 1):
        r["case"] = f"C{i:03d}"
    return righe


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seme = int(sys.argv[2]) if len(sys.argv) > 2 else 20261115
    righe = genera(n, seme)
    file = os.path.join(st.HERE, "..", "..", "docs", "examples", "pair-sheet-example.csv")
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"{len(righe)} cases in {os.path.normpath(file)}")


if __name__ == "__main__":
    main()
