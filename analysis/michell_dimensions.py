"""Do tasks weigh differently, or does autonomy need more than one dimension?

Factor models are logistic: logit p = d_t + sum_j u_aj v_tj, with d_t the task's
easiness, u_aj the agent's abilities and v_tj the weight the task gives each. With one
dimension it is the two-parameter logistic model: a discrimination per task, not the
per-agent slope of the test. An L2 penalty shrinks u and v.

1. Cross-validation, which uses no buckets: half of the runs of every agent-task cell
   fit the model, the other half judge it (log-likelihood of the held-out runs), over
   three random splits. The penalty and the number of dimensions (1 to 5) are chosen
   this way, and Rasch (M2 of the test) is judged on the same splits with its own
   penalty tuned too.
2. Double cancellation: for the best model, and for the best one-dimensional model, the
   violations expected from noise alone with 6 and 8 buckets.

Usage:
    python michell_dimensions.py task_results_1_1.yaml
    python michell_dimensions.py hcast_only.yaml --only-violations 5 1.0 --bins 6
"""
import argparse

import numpy as np

import michell_models as mm
import michell_test as m

RASCH_PENALTIES = (1e-3, 0.01, 0.03, 0.1, 0.3, 1.0)
FACTOR_PENALTIES = (0.3, 1.0, 3.0, 10.0)
DIMENSIONS = (1, 2, 3, 4, 5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_results")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--bins", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only-violations", nargs=2, type=float, metavar=("K", "L2"),
                    help="skip the cross-validation: expected violations for k dimensions and penalty L2")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    D = m.Data(args.task_results)
    print(f"{D.nT} tasks, {int(D.N.sum())} runs")

    if args.only_violations:
        choices = [(int(args.only_violations[0]), args.only_violations[1])]
    else:
        splits = [mm.half_split(D, rng) for _ in range(args.splits)]

        def held_out(fit):
            return [mm.loglik(D.S - S_fit, D.N - N_fit, fit(S_fit, N_fit, i)) for i, (S_fit, N_fit) in enumerate(splits)]

        print(f"\nlog-likelihood of the held-out runs (mean over {args.splits} splits; per split in brackets)")
        rasch = {}
        for l2 in RASCH_PENALTIES:
            v = held_out(lambda S, N, i: mm.fit_rasch(D, S, N, l2=l2))
            rasch[l2] = np.mean(v)
            label = f"Rasch (M2), penalty {l2:g}:"
            print(f"  {label:<30} {np.mean(v):9.1f}  ({', '.join(f'{x:.1f}' for x in v)})", flush=True)
        factors = {}
        for l2 in FACTOR_PENALTIES:
            for k in DIMENSIONS:
                v = held_out(lambda S, N, i: mm.fit_factors(D, S, N, k, seed=i, l2=l2))
                factors[(k, l2)] = np.mean(v)
                label = f"factors, k={k}, penalty {l2:g}:"
                print(f"  {label:<30} {np.mean(v):9.1f}  ({', '.join(f'{x:.1f}' for x in v)})", flush=True)
        best_rasch = max(rasch, key=rasch.get)
        best = max(factors, key=factors.get)
        best_one = max((c for c in factors if c[0] == 1), key=factors.get)
        print(f"best Rasch: penalty {best_rasch:g} ({rasch[best_rasch]:.1f}); "
              f"best factor model: k={best[0]}, penalty {best[1]} ({factors[best]:.1f}); "
              f"best with k=1: penalty {best_one[1]} ({factors[best_one]:.1f})")
        choices = [best, best_one]

    order = mm.agent_order(D)
    print("\ndouble cancellation, observed and expected from noise alone if the data came from the model")
    for k, l2 in choices:
        mu = mm.fit_factors(D, D.S, D.N, k, seed=99, l2=l2)
        for nb in args.bins:
            bucket, _ = mm.buckets(D, nb)
            f_obs = mm.dc_rate(D, D.S, bucket, order, nb)
            fr = mm.expected_rates(D, mu, bucket, order, nb, args.sims, rng)
            print(f"  k={k}, penalty {l2:g}, {nb} buckets: observed {100*f_obs:.2f}%, "
                  f"expected {100*fr.mean():.2f}% (p={(fr >= f_obs).mean():.3f})", flush=True)


if __name__ == "__main__":
    main()
