#!/usr/bin/env python3
"""Minor figures cited in the protocol: costs of the two errors, sentinel, memory of the cases, reliability of k per person.

Reads results/pair.json (written by sim_pair.py).

Usage:  python3 minor_figures.py > results/minor-figures.txt
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, norm

import estimators as st
from estimators import sdt


def costo(d, c, prevalenza, costo_omissione, costo_falso_allarme):
    """Expected cost over a hundred yes/no cases: evidence at ±d/2, criterion c (positive: more cautious about saying yes)."""
    colpiti, falsi = norm.cdf(d / 2 - c), norm.cdf(-d / 2 - c)
    return 100 * (prevalenza * (1 - colpiti) * costo_omissione + (1 - prevalenza) * falsi * costo_falso_allarme), colpiti, falsi


def criterio_ottimo(d, prevalenza, costo_omissione, costo_falso_allarme):
    beta = (1 - prevalenza) / prevalenza * costo_falso_allarme / costo_omissione
    return math.log(beta) / d


print("## The two errors do not cost the same (worked example, yes/no, 10 positives in 100, one omission costs 20 false alarms)")
prev, c_om, c_fa = 0.10, 20.0, 1.0
dm, ds = 2.338, 1.778
for nome, d, c in (("machine, neutral criterion", dm, 0.0), ("machine, cautious criterion (+0.5)", dm, 0.5),
                   ("machine, best criterion for these costs", dm, criterio_ottimo(dm, prev, c_om, c_fa)),
                   ("pair (yield 0.76), best criterion for these costs", ds, criterio_ottimo(ds, prev, c_om, c_fa)),
                   ("pair (yield 0.76), neutral criterion", ds, 0.0)):
    x, h, f = costo(d, c, prev, c_om, c_fa)
    print(f"  {nome:<55} correct yes {100 * h:5.1f}%  false alarms {100 * f:5.1f}%  cost per hundred cases {x:6.1f}")

print("\n## The sentinel: how many answers change from noise alone between two calls")
for p, nome in ((0.053, "GPT-4o on ChatBench"), (0.201, "Llama 3.1 8B on ChatBench")):
    for n in (30, 50):
        soglia = int(binom.ppf(0.99, n, p)) + 1
        print(f"  {nome}: over {n} cases {n * p:.1f} changed answers expected; {soglia} or more happen less than once in a hundred ({100 * binom.sf(soglia - 1, n, p):.2f}%)")

print("\n## The person who remembers the cases: share of cases whose key they remember, and the resulting k (true 2.16)")
dh, dm = 1.080, 2.338
ah = float(st.acc_da_d(dh, 4))
for q in (0.0, 0.1, 0.2, 0.3, 0.5):
    a = ah + q * (1 - ah)
    print(f"  remembered {100 * q:3.0f}%: correct answers {100 * a:.1f}%, k {dm / sdt.d_from_accuracy(a, 4):.2f}")

print("\n## k per person: how much of what distinguishes people is real and how much is sampling noise")
sd_vera = 0.150           # standard deviation of the true accuracies across people, ChatBench with GPT-4o (conversations.json)
for n in (9, 25, 50, 100, 200):
    rumore = math.sqrt(0.577 * 0.423 / n)
    print(f"  {n:3d} cases each: noise {100 * rumore:.1f} points, true spread {100 * sd_vera:.0f} points, reliability {sd_vera ** 2 / (sd_vera ** 2 + rumore ** 2):.2f}")

print("\n## The margin in errors: by how much the ideal pair's errors fall, with four options")
for am, marg in ((0.883, 0.02), (0.883, 0.10), (0.883, 0.20), (0.93, 0.077), (0.98, 0.05), (1 - 3 / 396, 0.05), (0.745, 0.174)):
    d = sdt.d_from_accuracy(am, 4)
    a2 = float(st.acc_da_d(d * (1 + marg), 4))
    print(f"  machine {100 * am:.1f}%, margin {100 * marg:.1f}%: errors per hundred from {100 * (1 - am):.2f} to {100 * (1 - a2):.2f} ({100 * ((1 - a2) / (1 - am) - 1):+.0f}%)")

print("\n## The same margin, read in errors: the ideal pair next to ever stronger machines (person at 58%, four options)")
print("   k   machine   | at 90 degrees: margin, errors per hundred, cut | at 75 degrees: margin, errors per hundred, cut")
dh = 1.080
for k in (1.2, 1.5, 2.0, 2.16, 3.0, 5.0):
    dm = k * dh
    am = float(st.acc_da_d(dm, 4))
    riga = f"  {k:4.2f}  {100 * am:6.2f}%  |"
    for phi in (90.0, 75.0):
        mg = float(st.margine(k, phi))
        a2 = float(st.acc_da_d(dm * (1 + mg), 4))
        riga += f" {100 * mg:5.1f}%  {100 * (1 - am):5.2f} -> {100 * (1 - a2):5.2f}  ({100 * ((1 - a2) / (1 - am) - 1):+4.0f}%) |"
    print(riga)
print("  (check by simulation, results/rules.json: the ideal pair goes from 88.3 to 89.0 correct answers at 74.5 degrees with k 2.16,")
print("   from 93.0 to 94.9 at 90 degrees with k 2.5, from 98.0 to 98.5 at 90 degrees with k 3.19)")

print("\n## Two modules, two yields: how far apart they must be for the difference to show (power 80%, two-sided 5%)")
import json, os
coppie = json.load(open(os.path.join(st.HERE, "results", "pair.json")))["coppie"]
for nome in ("gpt4o-come-nei-dati", "pari-ideale"):
    for n in ("100", "200", "400"):
        q = coppie[nome]["per_N"][n]["resa"]
        sd = (q[2] - q[0]) / 3.92
        print(f"  {nome}, {n} cases per module: standard deviation of the yield {sd:.3f}, visible difference {2.8 * sd * math.sqrt(2):.2f}")
