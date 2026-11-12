"""Details behind the Michell test, read from the same data. It does not change the test: it queries it.

  1. pairs of agents whose order flips between the 50% and the 80% horizon, with METR's
     published fits, and the horizons themselves;
  2. runs per agent-task cell (why the test uses buckets), tasks and runs per bucket,
     the mean success rate per bucket across agents and the pooled one;
  3. the stopwatch: how much human time and machine difficulty disagree, read both ways;
  4. a worked example of a violation of double cancellation (nine cells), and two agents'
     success at fixed durations from the published fits.

Usage: python michell_details.py task_results_1_1.yaml [--bins 6]
"""
import argparse
import itertools

import numpy as np
from scipy.special import expit

import michell_models as mm
import michell_test as m

EXAMPLE = ("GPT-5 (Inspect)", "GPT-5.4", "GPT-4o (Inspect)")
EXAMPLE_BUCKETS = (2, 3, 5)  # 0-based: buckets 3, 4 and 6 of six
CROSSING = ("Claude Opus 4.6 (Inspect)", "Gemini 3.1 Pro")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_results")
    ap.add_argument("--bins", type=int, default=6)
    args = ap.parse_args()
    D = m.Data(args.task_results)
    M = m.models(D)

    print("== 1. Horizons from METR's published fits ==")
    a = np.array([D.published[x][0] for x in D.agents])
    b = np.array([D.published[x][1] for x in D.agents])
    h50, h80 = m.horizon(a, b, .5), m.horizon(a, b, .8)
    print(f"published slopes: {b.min():.3f} to {b.max():.3f}")
    print("pairs whose order flips between 50% and 80%:")
    for i, j in itertools.combinations(range(D.nA), 2):
        if (h50[i] - h50[j]) * (h80[i] - h80[j]) < 0:
            r50 = max(h50[i], h50[j]) / min(h50[i], h50[j])
            r80 = max(h80[i], h80[j]) / min(h80[i], h80[j])
            print(f"  {D.agents[i]:30s} | {D.agents[j]:30s} | h50 {h50[i]:8.1f} {h50[j]:8.1f} (x{r50:.2f}) "
                  f"| h80 {h80[i]:7.1f} {h80[j]:7.1f} (x{r80:.2f})")
    print("horizons (minutes):")
    for k, x in enumerate(D.agents):
        print(f"  {x:30s} p50 {h50[k]:9.2f} min = {h50[k]/60:6.2f} h | p80 {h80[k]:8.2f} min = {h80[k]/60:5.2f} h")

    print("\n== 2. Cells and buckets ==")
    print(f"runs per agent-task cell: mean {D.N.mean():.2f}, median {np.median(D.N):.0f}, "
          f"min {D.N.min():.0f}, max {D.N.max():.0f} ({len(D.N)} cells)")
    nb = args.bins
    bucket, _ = mm.buckets(D, nb)
    print("tasks per bucket:", np.bincount(bucket, minlength=nb))
    P, W = m.bucket_matrix(D, D.S, bucket, np.arange(D.nA), nb)
    print("agents with >= 20 runs per bucket:", (W >= mm.MIN_RUNS).sum(axis=0), " fewest runs in a cell:", W.min(axis=0))
    Pv = np.where(W >= mm.MIN_RUNS, P, np.nan)
    print("mean across agents:", np.round(np.nanmean(Pv, axis=0), 3))
    pooled = [D.S[bucket[D.T] == k].sum() / D.N[bucket[D.T] == k].sum() for k in range(nb)]
    print("pooled rate (all runs):", np.round(np.array(pooled), 3))
    d = Pv[:, 3] - Pv[:, 2]
    print(f"bucket 4 above bucket 3 for {int((d > 0).sum())} agents; equal {int((d == 0).sum())}; below {int((d < 0).sum())}")
    print("  not above:", [D.agents[i] for i in np.where(~(d > 0))[0]])

    print("\n== 3. The stopwatch, read both ways ==")
    p2, _ = m.fit(D, *M["M2"][:3])
    difficulty = p2[D.nA:]
    tot_s, tot_n = np.bincount(D.T, D.S, D.nT), np.bincount(D.T, D.N, D.nT)
    ok = (tot_s > 0) & (tot_s < tot_n)
    x, y = D.x[ok], difficulty[ok]
    r = np.corrcoef(x, y)[0, 1]
    s1, i1 = np.polyfit(x, y, 1)
    s2, i2 = np.polyfit(y, x, 1)
    res1, res2 = y - (s1 * x + i1), x - (s2 * y + i2)
    print(f"r={r:.3f} R2={r*r:.3f}, {ok.sum()} identified tasks ({(~ok).sum()} of {D.nT} solved by all or by none)")
    print(f"human minutes at equal machine difficulty (regression of time on difficulty): "
          f"residual SD {res2.std():.2f} doublings, a factor {2**res2.std():.1f}")
    print(f"machine difficulty at equal human time, in time units (difficulty on time): "
          f"residual SD {res1.std():.2f} logits / slope {abs(s1):.3f} per doubling, a factor {2**(res1.std()/abs(s1)):.1f}")

    print("\n== 4. Worked example ==")
    for name in EXAMPLE:
        i = D.agents.index(name)
        print(f"  {name:20s}", " | ".join(f"bucket {k+1} {P[i, k]*100:5.1f}% ({int(W[i, k])} runs)" for k in EXAMPLE_BUCKETS))
    for name in CROSSING:
        a, b = D.published[name]
        print(f"  {name}: intercept {a}, slope {b} | success at 30 min %.2f, 2 h %.2f, 8 h %.2f, 24 h %.2f"
              % tuple(expit(a + b * np.log2(t)) for t in (30, 120, 480, 1440)))


if __name__ == "__main__":
    main()
