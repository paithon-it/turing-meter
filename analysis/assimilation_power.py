"""How large the assimilation study must be, and whether its analysis holds (docs/STUDY.md v0.2).

The study asks whether using a machine moves a person's unassisted errors onto
the machine's: the correlation rho between the evidence of the person alone and
of a frozen machine, on the same items, rising with use. Version 0.1 of this
script showed one trap (practice alone raises rho when items share a
difficulty). The review of 2 November 2026 found more, and this version
simulates them:

1. the two arms may learn by different amounts (P1);
2. a lean toward the machine's answer inflates rho pooled over classes (P2);
3. the bootstrap must match the statistic (P3);
4. people may come to weigh the shared difficulty more, learning the truth
   rather than the machine (P5).

Evidence toward the right answer on item i (y_i = +1 or -1 its class):
    machine  e_M = mu_M + delta_i + u_i                says yes if y_i e_M > c_M
    person   e_H = mu_p + load * delta_i + v_pi         says yes if y_i e_H > c_p
    v_pi = a * (sigma_p / sigma_M) * u_i + sqrt(1 - a^2) * sigma_p * z_pi
delta_i is the shared difficulty, u_i the machine's own quirk, a the share of the
person's noise that has become the machine's quirk: the assimilation. People
differ in mean evidence, in noise and in criterion, and the same people answer
before and after.

Three statistics, each a difference in differences (three months minus before,
machine arm minus control):
    pooled    rho on right/wrong pooled over classes (version 0.1)
    class     rho within class of the right answer, each person with their own
              criterion (apparent.pooled_alignment)
    adjusted  class rho minus what practice alone predicts from the alignment
              between people (apparent.practice_adjusted): the primary
Intervals: percentile bootstrap over people within arm, the same draw at both
waves, forms fixed. Decision at 90% two-sided: assimilation if the interval is
above 0; an effect of +0.10 ruled out if it is below +0.10.

    python assimilation_power.py               # the tables in RESULTS.md, section 4
    python assimilation_power.py --reps 100    # quicker, rougher
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm

from turing_meter.apparent import (
    error_alignment,
    people_alignment,
    pooled_alignment,
    practice_adjusted,
    useful_angle,
)

SESOI = 0.10   # the smallest effect the study is built to see, and to rule out


@dataclass(frozen=True)
class World:
    dm: float = 2.0           # machine d'
    dh_pre: float = 1.0       # person d' before, both arms
    dh_post: float = 1.22     # after three months, control arm
    dh_post_m: float = 1.22   # after three months, machine arm
    rho0: float = 0.20        # rho(person, machine) before
    load_m: float = 1.0       # weight on the shared difficulty, machine arm after
    c_m: float = 0.0          # the machine's criterion
    c_shift_m: float = 0.0    # how far the machine arm's criterion moves, by T2
    did: float = 0.0          # assimilation, as the rise of class rho it causes
    spread: float = 0.15      # sd of mean evidence across people
    noise_het: float = 0.20   # sd of log noise across people
    crit_sd: float = 0.20     # sd of criteria across people
    people: int = 50          # per arm, present at both waves
    items: int = 200          # per test, half per class

    # Latent scale: the machine's evidence has variance 1, the person's 2.25 before.
    @property
    def tau2(self) -> float:
        return self.rho0 * 1.5

    @property
    def sm2(self) -> float:
        return 1.0 - self.tau2

    @property
    def mu_h(self) -> float:
        return self.dh_pre * 1.5 / 2

    def sh2(self, dh: float, load: float = 1.0) -> float:
        out = (2 * self.mu_h / dh) ** 2 - load * load * self.tau2
        if out <= 0:
            raise ValueError("no noise left for the person: lower d' or the load")
        return out

    def a(self) -> float:
        sh2 = self.sh2(self.dh_post_m, self.load_m)
        total = self.load_m ** 2 * self.tau2 + sh2
        out = self.did * math.sqrt(total) / math.sqrt(sh2 * self.sm2)
        if not 0 <= out < 1:
            raise ValueError(f"no assimilation share gives {self.did}")
        return out


def _form(rng, w: World):
    """A test form: balanced classes; the machine must err sometimes in each."""
    y = np.where(np.arange(w.items) % 2 == 0, 1, -1)
    while True:
        d = rng.normal(0, math.sqrt(w.tau2), w.items)
        u = rng.normal(0, math.sqrt(w.sm2), w.items)
        m_yes = y * (w.dm / 2 + d + u) > w.c_m
        if all(0 < m_yes[y == s].mean() < 1 for s in (1, -1)):
            return y, d, u, m_yes


def _answers(rng, w: World, form, people, sh2, load, a, shift):
    y, d, u, _ = form
    mu, het, crit = people
    sig = np.sqrt(sh2) * het
    z = rng.standard_normal((mu.size, y.size))
    v = a * sig / math.sqrt(w.sm2) * u + math.sqrt(1 - a * a) * sig * z
    return y * (mu + load * d + v) > crit + shift


def _d_prime(yes, y):
    n = (y > 0).sum()
    hit = np.clip(yes[:, y > 0].mean(axis=1), 0.5 / n, 1 - 0.5 / n)
    fa = np.clip(yes[:, y < 0].mean(axis=1), 0.5 / n, 1 - 0.5 / n)
    return float((norm.ppf(hit) - norm.ppf(fa)).mean())


def study(args):
    """One simulated study: point estimates, bootstrap intervals, decisions."""
    w, seed, boots = args
    rng = np.random.default_rng(seed)
    forms = [_form(rng, w) for _ in range(2)]                  # T0 and T2, shared by the arms
    k = w.people
    arms = []
    for machine_arm in (True, False):
        people = (w.mu_h + w.spread * rng.standard_normal((k, 1)),
                  np.exp(w.noise_het * rng.standard_normal((k, 1))),
                  w.crit_sd * rng.standard_normal((k, 1)))
        pre = _answers(rng, w, forms[0], people, w.sh2(w.dh_pre), 1.0, 0.0, 0.0)
        if machine_arm:
            post = _answers(rng, w, forms[1], people, w.sh2(w.dh_post_m, w.load_m), w.load_m,
                            w.a(), w.c_shift_m)
        else:
            post = _answers(rng, w, forms[1], people, w.sh2(w.dh_post), 1.0, 0.0, 0.0)
        arms.append((pre, post))

    # Row 0 of each weight matrix is the data as observed, the others bootstrap draws.
    draws = [np.vstack([np.ones(k), rng.multinomial(k, np.full(k, 1 / k), boots)]) for _ in arms]
    hm, hh, pooled = {}, {}, {}
    for i, (pre, post) in enumerate(arms):
        for t, (ans, form) in enumerate(((pre, forms[0]), (post, forms[1]))):
            y, _, _, m_yes = form
            hm[i, t] = pooled_alignment(ans, m_yes, y, draws[i])
            hh[i, t] = people_alignment(ans, y, draws[i])
            right_h = (ans == (y > 0)).ravel()
            right_m = np.broadcast_to(m_yes == (y > 0), ans.shape).ravel()
            pooled[i, t] = error_alignment(right_h, right_m).rho
    q = np.mean([hm[i, 0] ** 2 / hh[i, 0] for i in (0, 1)], axis=0)  # before: nobody has used it
    adj = {key: practice_adjusted(hm[key], hh[key], q) for key in hm}

    def did(s):
        return (s[0, 1] - s[0, 0]) - (s[1, 1] - s[1, 0])

    out = {"pooled": did(pooled)}
    for name, s in (("class", hm), ("adjusted", adj)):
        d = did(s)
        lo, hi = np.quantile(d[1:], [0.05, 0.95])
        out[name] = d[0]
        out[name + "_se"] = d[1:].std()
        out[name + "_yes"] = lo > 0
        out[name + "_no"] = hi < SESOI
    out["dh_gap"] = _d_prime(arms[0][1], forms[1][0]) - _d_prime(arms[1][1], forms[1][0])
    return out


def run(w: World, reps: int, boots: int, seed: int, pool) -> dict:
    rows = pool.map(study, [(w, seed + r, boots) for r in range(reps)])
    agg = {key: np.array([r[key] for r in rows], dtype=float) for key in rows[0]}
    return {key: (v.mean(), v.std()) for key, v in agg.items()}


SCENARIOS = [
    ("null: equal learning", {}),
    ("null: machine arm learns more (dH 1.43)", {"dh_post_m": 1.43}),
    ("null: machine arm takes the machine's lean", {"c_m": 0.5, "c_shift_m": 0.5}),
    ("null: machine arm weighs shared difficulty x1.3", {"load_m": 1.3}),
    ("effect +0.10: equal learning", {"did": 0.10}),
    ("effect +0.10: machine arm learns more", {"did": 0.10, "dh_post_m": 1.43}),
]

EDGES = [  # corners of the pilot gate of STUDY v0.2
    ("corner dH 0.8, k = 3, rho0 0.13",
     {"dh_pre": 0.8, "dh_post": 0.98, "dh_post_m": 0.98, "dm": 2.4, "rho0": 0.13}),
    ("corner dH 1.5, k = 1.5, rho0 0.30",
     {"dh_pre": 1.5, "dh_post": 1.83, "dh_post_m": 1.83, "dm": 2.25, "rho0": 0.30}),
]

HEAD = (f"{'scenario':48s} {'pooled':>7s} {'class':>7s} {'adj':>7s} {'sd':>6s} {'se':>6s} "
        f"{'P(yes)':>10s} {'P(no)':>5s} {'cls':>5s} {'dHgap':>6s}")


def _line(name, r, reps):
    def mc(p):
        return math.sqrt(max(p * (1 - p), 1e-4) / reps)
    yes, no = r["adjusted_yes"][0], r["adjusted_no"][0]
    print(f"{name:48s} {r['pooled'][0]:+7.3f} {r['class'][0]:+7.3f} {r['adjusted'][0]:+7.3f} "
          f"{r['adjusted'][1]:6.3f} {r['adjusted_se'][0]:6.3f} {yes:5.2f}±{mc(yes):.2f} "
          f"{no:5.2f} {r['class_yes'][0]:5.2f} {r['dh_gap'][0]:+6.2f}")
    sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--boots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    base = World()
    print(f"World: dM {base.dm}, dH {base.dh_pre} -> {base.dh_post}, rho0 {base.rho0}, "
          f"useful angle {useful_angle(base.dh_pre, base.dm):.0f} degrees; "
          f"{base.people} people per arm at both waves, {base.items} items per test; "
          f"{args.reps} studies, {args.boots} bootstrap draws each.")
    print("Columns: means of the three differences in differences; sd over studies and mean")
    print("bootstrap se of the adjusted one; P(yes) = 90% interval above 0, P(no) = below +0.10,")
    print("both for the adjusted; cls = P(yes) for the class rho; dHgap = d'H machine minus")
    print("control at T2.\n")
    with Pool(args.workers) as pool:
        print(HEAD)
        for i, (name, kw) in enumerate(SCENARIOS):
            _line(name, run(replace(base, **kw), args.reps, args.boots, args.seed + 10_000 * i, pool),
                  args.reps)
        print("\nPeople per arm present at both waves")
        print(HEAD)
        for k in (35, 70):
            for j, eff in enumerate((0.10, 0.0)):
                w = replace(base, people=k, did=eff)
                _line(f"{k} people, {'effect +0.10' if eff else 'null'}",
                      run(w, args.reps, args.boots, args.seed + 100_000 + 1000 * k + j, pool), args.reps)
        print("\nCorners of the pilot gate, 50 people per arm")
        print(HEAD)
        for i, (name, kw) in enumerate(EDGES):
            for j, eff in enumerate((0.10, 0.0)):
                w = replace(base, **kw, did=eff)
                _line(f"{name}, {'effect' if eff else 'null'}",
                      run(w, args.reps, args.boots, args.seed + 200_000 + 1000 * i + j, pool), args.reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
