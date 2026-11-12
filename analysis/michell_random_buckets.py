"""The buckets are a choice: the test on random divisions of the tasks.

For every division, cuts at random among the tasks sorted by human time, with at least
`--min-tasks` tasks per bucket. Observed violations of double cancellation against those
expected from noise alone under one of the test's models, by default Rasch with
difficulties estimated from the machines (M2), the most demanding of its additive models.

Usage: python michell_random_buckets.py task_results_1_1.yaml [--divisions 100] [--sims 100] [--bins 6]
"""
import argparse

import numpy as np
from scipy.special import expit

import michell_models as mm
import michell_test as m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_results")
    ap.add_argument("--divisions", type=int, default=100)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--min-tasks", type=int, default=20)
    ap.add_argument("--model", default="M2", choices=["M0", "M1", "M2"])
    ap.add_argument("--seed", type=int, default=912)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    D = m.Data(args.task_results)
    M = m.models(D)
    p, _ = m.fit(D, *M[args.model][:3])
    mu = expit(M[args.model][0](p))
    order = mm.agent_order(D)
    xs = np.sort(D.x)
    nb = args.bins
    results = []
    while len(results) < args.divisions:
        cuts = np.sort(rng.choice(np.arange(1, D.nT), nb - 1, replace=False))
        sizes = np.diff(np.concatenate([[0], cuts, [D.nT]]))
        if sizes.min() < args.min_tasks:
            continue
        edges = xs[cuts]
        if len(np.unique(edges)) < nb - 1:
            continue
        bucket = np.searchsorted(edges, D.x, side="right")
        f_obs = mm.dc_rate(D, D.S, bucket, order, nb)
        sims = mm.expected_rates(D, mu, bucket, order, nb, args.sims, rng)
        results.append((f_obs, sims.mean(), (sims >= f_obs).mean(), sizes.min(), sizes.max()))
        print(f"{len(results):3d}: observed {100*f_obs:.2f}%, expected {100*sims.mean():.2f}%, p {results[-1][2]:.2f}, "
              f"buckets of {sizes.min()} to {sizes.max()} tasks", flush=True)
    r = np.array(results)
    ratio = r[:, 0] / r[:, 1]
    print(f"\n{len(r)} random divisions into {nb} buckets, model {args.model}, {args.sims} simulations each:")
    print(f"  observed above expected in {int((r[:, 0] > r[:, 1]).sum())} of {len(r)}")
    print(f"  observed/expected: median {np.median(ratio):.2f}, "
          f"from {np.quantile(ratio, 0.1):.2f} to {np.quantile(ratio, 0.9):.2f} (10th-90th percentile)")
    print(f"  p: median {np.median(r[:, 2]):.3f}; p < 0.05 in {int((r[:, 2] < 0.05).sum())} of {len(r)}; "
          f"p < 0.10 in {int((r[:, 2] < 0.10).sum())}")


if __name__ == "__main__":
    main()
