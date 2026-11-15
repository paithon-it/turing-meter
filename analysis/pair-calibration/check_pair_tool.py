#!/usr/bin/env python3
"""`turing-meter pair` on sheets whose truth is known.

1. Four options, many cases: it finds k, the angle and the margin of the scenario.
2. Yes/no with a shared bias toward "no": the sensitivities and the angle are found because the
   computation is per class; over six large sheets the mean shows no systematic error.
3. Two-phase design: an archive of 2,000 cases, the person does 200 (half among the machine's
   errors), with the weights: it finds k and the angle, with a narrower interval than 200 cases
   taken at random from the same archive.
4. A machine with no errors on 100 cases: k is a lower bound, the angle is not reported, the zone
   comes from the exact interval.

The quick checks of the same kind that run with the test suite are in tests/test_pair.py.

Usage:  python3 check_pair_tool.py > results/tool-check.txt
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import estimators as st  # noqa: E402
from turing_meter import pair as mc  # noqa: E402

L = np.array(list("ABCD"))


def sheet(ca, cb, extra=None):
    d = {"key": ["A"] * ca.size, "person": list(L[ca]), "machine": list(L[cb])}
    if extra:
        d.update(extra)
    return d


def line(name, ref, truth):
    phi = ref["phi"]
    print(f"{name}\n  k {ref['k']['estimate']:.2f} {ref['k']['ci95']} (true {truth['k']:.2f}) | "
          f"angle {phi['estimate']:.1f} {None if phi['ci95'] is None else [round(x, 1) for x in phi['ci95']]} [{phi['status']}] (true {truth['phi']:.1f}) | "
          f"margin {100 * ref['margin']['estimate']:.1f}% (true {100 * truth['margin']:.1f}%) | zone {ref['zone']} | ceiling {100 * ref['ceiling_at_90deg']:.0f}%")


def main():
    gen = np.random.default_rng(7)
    dh, dm, rho = 1.080, 2.338, 0.267
    truth = {"k": dm / dh, "phi": math.degrees(math.acos(rho)), "margin": st.sdt.ideal(dh, dm, rho) / dm - 1}

    ca, cb, _, _ = st.scelte(6000, 4, rho, dh, dm, gen)
    line("1. four options, 6,000 cases", mc.analyze(sheet(ca, cb), 4, B=400), truth)

    n = 6000
    klass = np.repeat([1, 0], n // 2)
    e = gen.standard_normal((n, 2)) @ np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]])).T
    sign = np.where(klass == 1, 1.0, -1.0)
    yes_h, yes_m = sign * dh / 2 + e[:, 0] > 0.4, sign * dm / 2 + e[:, 1] > 0.4
    d = {"key": list(np.where(klass == 1, "YES", "NO")), "person": list(np.where(yes_h, "YES", "NO")), "machine": list(np.where(yes_m, "YES", "NO"))}
    line("2. yes/no, 6,000 cases, both leaning toward no", mc.analyze(d, 2, B=400), truth)

    means = []
    for seed in range(6):
        g2 = np.random.default_rng(100 + seed)
        n2 = 40000
        cl = np.repeat([1, 0], n2 // 2)
        e2 = g2.standard_normal((n2, 2)) @ np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]])).T
        sg = np.where(cl == 1, 1.0, -1.0)
        h2 = ((sg * dh / 2 + e2[:, 0] > 0.4) == (cl == 1)).astype(float)
        m2 = ((sg * dm / 2 + e2[:, 1] > 0.4) == (cl == 1)).astype(float)
        g = mc._two_classes(np.ones((1, n2)), h2, m2, None, cl)
        means.append((float(g["k"][0]), float(g["phi"][0])))
    means = np.mean(means, axis=0)
    print(f"   six yes/no sheets of 40,000 cases: mean k {means[0]:.3f} (true {truth['k']:.3f}), mean angle {means[1]:.1f} (true {truth['phi']:.1f})")

    ca, cb, _, _ = st.scelte(2000, 4, rho, dh, dm, gen)
    err, ok = np.flatnonzero(cb != 0), np.flatnonzero(cb == 0)
    ne = min(100, err.size)
    chosen_err, chosen_ok = gen.choice(err, ne, replace=False), gen.choice(ok, 200 - ne, replace=False)
    idx = np.r_[chosen_err, chosen_ok]
    weight = np.r_[np.full(ne, err.size / ne), np.full(200 - ne, ok.size / (200 - ne))]
    ref = mc.analyze(sheet(ca[idx], cb[idx], {"weight": [str(x) for x in weight]}), 4, B=1000)
    line(f"3. two phases: archive of 2,000, the person does 200 ({ne} among the {err.size} machine errors)", ref, truth)
    simple = gen.choice(2000, 200, replace=False)
    line("   comparison: 200 cases drawn at random from the same archive", mc.analyze(sheet(ca[simple], cb[simple]), 4, B=1000), truth)

    ca, _, _, _ = st.scelte(100, 4, 0.0, dh, 6.0, gen)
    ref = mc.analyze(sheet(ca, np.zeros(100, int)), 4, B=400)
    print(f"4. a machine with no errors on 100 cases\n  k {ref['k']['estimate']:.2f}, exact interval {ref['k']['ci95_exact']} ({ref['k'].get('note')}) | "
          f"angle: {ref['phi']['status']} | zone {ref['zone']} | ceiling {100 * ref['ceiling_at_90deg']:.0f}%")


if __name__ == "__main__":
    main()
