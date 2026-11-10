"""Does measured AI autonomy behave like a quantity? A Michell test on METR data.

Joel Michell argued that psychometrics never tested whether the attributes it
measures are quantitative. Luce and Tukey's conjoint measurement (1964) gives a
way to test it. This script applies it to METR's public per-task results for
Time Horizon 1.1 (agents x tasks, with human completion times).

Three questions:
  1. Slopes. Do all agents share one slope on log2(human minutes) (a Rasch-like,
     specifically objective scale) or does each agent need its own (METR's
     model)? Likelihood-ratio test, and how many pairs of agents swap order
     between the 50% and the 80% horizon.
  2. The stopwatch. How much of the difficulty that tasks have for machines
     (a Rasch difficulty estimated from the runs alone) does human time explain?
  3. Conjoint measurement. On the agents x duration-bucket matrix of success
     rates, how many violations of independence and double cancellation, against
     the number expected from sampling noise alone under additive models
     (parametric bootstrap)?

Usage:
    curl -sLO https://metr.org/assets/task_results_1_1.yaml
    python michell_test.py task_results_1_1.yaml [--bins 6] [--min-runs 20] [--sims 200]
"""
import argparse
import itertools

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import chi2, pearsonr


def load(path):
    d = yaml.safe_load(open(path))
    agents = list(d["agents"])
    tasks = sorted({t["task_id"] for v in d["agents"].values() for t in v["tasks"]})
    ia = {a: i for i, a in enumerate(agents)}
    it = {t: j for j, t in enumerate(tasks)}
    minutes, rows = {}, []
    for a, v in d["agents"].items():
        for t in v["tasks"]:
            minutes[t["task_id"]] = t["human_minutes"]
            rows.append((ia[a], it[t["task_id"]], t["n_successes"], t["n_runs"]))
    rows = np.array(rows, float)
    published = {a: (v["intercept"], v["coefficient"]) for a, v in d["agents"].items()}
    x = np.log2(np.array([minutes[t] for t in tasks]))
    return agents, tasks, rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2], rows[:, 3], x, published


class Data:
    def __init__(self, path):
        (self.agents, self.tasks, self.A, self.T, self.S, self.N, self.x,
         self.published) = load(path)
        self.nA, self.nT = len(self.agents), len(self.tasks)
        self.const = (gammaln(self.N + 1) - gammaln(self.S + 1) - gammaln(self.N - self.S + 1)).sum()

    def loglik(self, eta):
        return (self.S * eta - self.N * np.logaddexp(0, eta)).sum() + self.const


# Each model: linear predictor eta(p) and gradient of the negative log-likelihood.
def models(D):
    A, T, x, nA, nT = D.A, D.T, D.x, D.nA, D.nT

    def resid(eta):
        return D.S - D.N * expit(eta)

    def m0(p):  # common slope on human time: Rasch with time as difficulty
        return p[:nA][A] + p[nA] * x[T]

    def g0(p):
        r = resid(m0(p))
        return np.concatenate([-np.bincount(A, r, nA), [-(r * x[T]).sum()]])

    def m1(p):  # METR: intercept and slope per agent
        return p[:nA][A] + p[nA:][A] * x[T]

    def g1(p):
        r = resid(m1(p))
        return np.concatenate([-np.bincount(A, r, nA), -np.bincount(A, r * x[T], nA)])

    def m2(p):  # Rasch with free task difficulties estimated from the runs
        return p[:nA][A] - p[nA:][T]

    def g2(p):
        r = resid(m2(p))
        return np.concatenate([-np.bincount(A, r, nA), np.bincount(T, r, nT)])

    def m3(p):  # free difficulties and a discrimination per agent (2PL)
        th, al, b = p[:nA], p[nA:2 * nA], p[2 * nA:]
        return th[A] - np.exp(al[A]) * b[T]

    def g3(p):
        th, al, b = p[:nA], p[nA:2 * nA], p[2 * nA:]
        r = resid(m3(p))
        e = np.exp(al[A])
        return np.concatenate([-np.bincount(A, r, nA), np.bincount(A, r * e * b[T], nA), np.bincount(T, r * e, nT)])

    return {"M0": (m0, g0, nA + 1, nA + 1), "M1": (m1, g1, 2 * nA, 2 * nA),
            "M2": (m2, g2, nA + nT, nA + nT - 1), "M3": (m3, g3, 2 * nA + nT, 2 * nA + nT - 2)}


def fit(D, eta, grad, npar, l2=1e-3, x0=None):
    """Penalized maximum likelihood; the small L2 keeps always-solved or never-solved tasks finite."""
    def f(p):
        e = eta(p)
        return -(D.S * e - D.N * np.logaddexp(0, e)).sum() + l2 * (p ** 2).sum()
    res = minimize(f, np.zeros(npar) if x0 is None else x0, method="L-BFGS-B",
                   jac=lambda p: grad(p) + 2 * l2 * p, options={"maxiter": 20000})
    return res.x, D.loglik(eta(res.x))


def horizon(a, b, p):
    """logit p = a + b log2 t  ->  t."""
    return 2 ** ((np.log(p / (1 - p)) - a) / b)


def flips(h50, h80):
    return sum(1 for i, j in itertools.combinations(range(len(h50)), 2) if (h50[i] - h50[j]) * (h80[i] - h80[j]) < 0)


def bucket_matrix(D, S, bucket, order, nb):
    M = np.zeros((D.nA, nb))
    W = np.zeros((D.nA, nb))
    np.add.at(M, (D.A, bucket[D.T]), S)
    np.add.at(W, (D.A, bucket[D.T]), D.N)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = M / W
    return P[order], W[order]


def violations(P, W, min_runs):
    """Independence (rows, columns) and double cancellation over all distinct triples."""
    nA, nb = P.shape
    Pm = np.where(W >= min_runs, P, np.nan)
    v_row = n_row = 0
    for i, k in itertools.combinations(range(nA), 2):
        d = Pm[i] - Pm[k]
        d = d[~np.isnan(d) & (d != 0)]
        if len(d) >= 2:
            n_row += 1
            v_row += int(len(set(np.sign(d))) > 1)
    v_col = n_col = 0
    for c1, c2 in itertools.combinations(range(nb), 2):
        d = Pm[:, c1] - Pm[:, c2]
        d = d[~np.isnan(d) & (d != 0)]
        if len(d) >= 2:
            n_col += 1
            v_col += int(len(set(np.sign(d))) > 1)
    # if P(a,y) >= P(b,x) and P(b,z) >= P(c,y) then P(a,z) >= P(c,x)
    a = np.arange(nA)[:, None, None, None, None, None]
    b = np.arange(nA)[None, :, None, None, None, None]
    c = np.arange(nA)[None, None, :, None, None, None]
    x = np.arange(nb)[None, None, None, :, None, None]
    y = np.arange(nb)[None, None, None, None, :, None]
    z = np.arange(nb)[None, None, None, None, None, :]
    distinct = (a != b) & (b != c) & (a != c) & (x != y) & (y != z) & (x != z)
    ay, bx, bz, cy, az, cx = Pm[a, y], Pm[b, x], Pm[b, z], Pm[c, y], Pm[a, z], Pm[c, x]
    ok = distinct & ~np.isnan(ay) & ~np.isnan(bx) & ~np.isnan(bz) & ~np.isnan(cy) & ~np.isnan(az) & ~np.isnan(cx)
    with np.errstate(invalid="ignore"):
        antecedent = ok & (ay >= bx) & (bz >= cy)
        violated = antecedent & (az < cx)
    return v_row, n_row, v_col, n_col, int(violated.sum()), int(antecedent.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_results")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--min-runs", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=534)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    D = Data(args.task_results)
    print(f"{D.nA} agents, {D.nT} tasks, {len(D.S)} cells, {int(D.N.sum())} runs, {int(D.S.sum())} successes")

    M = models(D)
    fits = {}
    for name in ["M0", "M1", "M2"]:
        eta, grad, npar, _ = M[name]
        fits[name] = fit(D, eta, grad, npar)
    p2 = fits["M2"][0]
    eta3, grad3, npar3, _ = M["M3"]
    fits["M3"] = fit(D, eta3, grad3, npar3, x0=np.concatenate([p2[:D.nA], np.zeros(D.nA), p2[D.nA:]]))

    print("\n== Models (binomial log-likelihood, AIC, BIC) ==")
    for name in ["M0", "M1", "M2", "M3"]:
        ll, k = fits[name][1], M[name][3]
        print(f"{name}: LL={ll:10.1f}  k={k:3d}  AIC={-2*ll+2*k:9.1f}  BIC={-2*ll+k*np.log(D.N.sum()):9.1f}")

    (p0, ll0), (p1, ll1) = fits["M0"], fits["M1"]
    lr, df = 2 * (ll1 - ll0), M["M1"][3] - M["M0"][3]
    print(f"\n[1] Per-agent slopes (M1) vs common slope (M0): LR={lr:.1f}, df={df}, p={chi2.sf(lr, df):.2e}")
    slopes = p1[D.nA:]
    print(f"    slopes (logit per doubling of human minutes): {slopes.min():.2f} to {slopes.max():.2f}, median {np.median(slopes):.2f}; common {p0[D.nA]:.2f}")
    lr3, df3 = 2 * (fits["M3"][1] - fits["M2"][1]), M["M3"][3] - M["M2"][3]
    print(f"    same question with free difficulties (M3 vs M2): LR={lr3:.1f}, df={df3}, p={chi2.sf(lr3, df3):.2e}")
    a_pub = np.array([D.published[a][0] for a in D.agents])
    b_pub = np.array([D.published[a][1] for a in D.agents])
    n_pairs = D.nA * (D.nA - 1) // 2
    print(f"    pairs whose order flips between the 50% and 80% horizon: {flips(horizon(a_pub, b_pub, .5), horizon(a_pub, b_pub, .8))} of {n_pairs} "
          f"with METR's published fits ({flips(horizon(p1[:D.nA], slopes, .5), horizon(p1[:D.nA], slopes, .8))} with ours)")

    b = p2[D.nA:]
    tot_s, tot_n = np.bincount(D.T, D.S, D.nT), np.bincount(D.T, D.N, D.nT)
    ok = (tot_s > 0) & (tot_s < tot_n)
    r, _ = pearsonr(D.x[ok], b[ok])
    slope, inter = np.polyfit(D.x[ok], b[ok], 1)
    resid = b[ok] - (slope * D.x[ok] + inter)
    print(f"\n[2] Rasch difficulty (from machine runs only) vs log2(human minutes), {ok.sum()} identified tasks:")
    print(f"    r={r:.3f}, R^2={r*r:.3f}")
    print(f"    residual SD {resid.std():.2f} logits, i.e. a factor {2**(resid.std()/abs(slope)):.1f} in human minutes at equal machine difficulty")
    print(f"    tasks failed by every agent: {(tot_s == 0).sum()}, solved by every agent: {(tot_s == tot_n).sum()}")

    nb = args.bins
    qs = np.quantile(D.x, np.linspace(0, 1, nb + 1))
    qs[-1] += 1e-9
    bucket = np.clip(np.searchsorted(qs, D.x, side="right") - 1, 0, nb - 1)
    order = np.argsort(np.bincount(D.A, D.S, D.nA) / np.bincount(D.A, D.N, D.nA))
    P, W = bucket_matrix(D, D.S, bucket, order, nb)
    obs = violations(P, W, args.min_runs)
    print(f"\n[3] Conjoint measurement, {D.nA} agents x {nb} duration buckets (cells with >= {args.min_runs} runs):")
    print("    buckets (human minutes):", [f"{2**qs[i]:.2f}-{2**qs[i+1]:.2f}" for i in range(nb)])
    print(f"    row independence: {obs[0]} of {obs[1]} agent pairs change order across buckets")
    print(f"    column independence: {obs[2]} of {obs[3]} bucket pairs change order across agents")
    print(f"    double cancellation: {obs[4]} violations in {obs[5]} cases with a true antecedent ({100*obs[4]/max(obs[5], 1):.2f}%)")
    f_obs = obs[4] / max(obs[5], 1)
    labels = {"M0": "common slope on human time (additive)", "M1": "per-agent slope (METR)",
              "M2": "Rasch, machine-estimated difficulty (additive)", "M3": "machine difficulty, per-agent slope"}
    for name in ["M0", "M1", "M2", "M3"]:
        mu = expit(M[name][0](fits[name][0]))
        sims = np.array([violations(*bucket_matrix(D, rng.binomial(D.N.astype(int), mu), bucket, order, nb), args.min_runs)
                         for _ in range(args.sims)])
        frac = sims[:, 4] / np.maximum(sims[:, 5], 1)
        print(f"    expected under {name} ({labels[name]}): rows {sims[:,0].mean():.1f} (p={(sims[:,0] >= obs[0]).mean():.3f}); "
              f"columns {sims[:,2].mean():.1f} (p={(sims[:,2] >= obs[2]).mean():.3f}); "
              f"double cancellation {100*frac.mean():.2f}% (p={(frac >= f_obs).mean():.3f})")

    Pv = np.where(W >= args.min_runs, P, np.nan)
    print("    mean success rate per bucket:", np.round(np.nanmean(Pv, axis=0), 2))
    for c1, c2 in itertools.combinations(range(nb), 2):
        d = Pv[:, c1] - Pv[:, c2]
        m = ~np.isnan(d) & (d != 0)
        if len(set(np.sign(d[m]))) > 1:
            print(f"    bucket {c1+1} vs {c2+1}: the longer bucket does better for {int((m & (d < 0)).sum())} agents")


if __name__ == "__main__":
    main()
