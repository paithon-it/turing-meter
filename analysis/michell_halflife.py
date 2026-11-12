"""Is the all-or-nothing of long tasks the explanation? Half-life models against the buckets.

Toby Ord's half-life model gives every agent a constant hazard per minute, so success
falls exponentially with the task's human time t: p = exp(-t/h_a), the same as
2^(-t/h) with h rescaled. With a shape k common to all agents (Weibull),
p = exp(-(t/h_a)^k), and k < 1 means the hazard per minute falls as tasks get longer.
Both are additive on the right scale, log(-log p) = k log t - k log h_a: if they
generated the data, double cancellation would hold. With the noise of the real runs and
the same buckets, how many violations would they show?

Usage: python michell_halflife.py task_results_1_1.yaml [--bins 6] [--sims 200] [--seed 77]
"""
import argparse

import numpy as np

import michell_models as mm
import michell_test as m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_results")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    D = m.Data(args.task_results)
    nb = args.bins
    bucket, _ = mm.buckets(D, nb)
    order = mm.agent_order(D)
    f_obs = mm.dc_rate(D, D.S, bucket, order, nb)
    print(f"{D.nT} tasks, {nb} buckets: observed {100*f_obs:.2f}%")
    n_runs = np.log(D.N.sum())

    M = m.models(D)
    _, ll1 = m.fit(D, *M["M1"][:3])
    k1 = M["M1"][3]
    print(f"METR's model (M1): LL {ll1:.1f}, {k1} parameters, BIC {-2*ll1 + k1*n_runs:.1f}")

    for name, free in (("Ord, constant hazard", False), ("Weibull, common shape", True)):
        mu, k, _ = mm.fit_halflife(D, D.S, D.N, free)
        ll = mm.loglik(D.S, D.N, mu) + D.const
        npar = D.nA + (1 if free else 0)
        fr = mm.expected_rates(D, mu, bucket, order, nb, args.sims, rng)
        print(f"{name}: shape {k:.2f}, LL {ll:.1f}, {npar} parameters, BIC {-2*ll + npar*n_runs:.1f}; "
              f"expected {100*fr.mean():.2f}% (p={(fr >= f_obs).mean():.3f})", flush=True)


if __name__ == "__main__":
    main()
