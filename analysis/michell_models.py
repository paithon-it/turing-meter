"""Models and helpers shared by the follow-ups to michell_test.py.

Each follow-up puts the same question to another model: if the data came from it,
with the noise of the real runs and the same duration buckets, how many violations of
double cancellation would the buckets show?
"""
import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import michell_test as m

MIN_RUNS = 20


def buckets(D, nb):
    """Equal-count buckets of log2(human minutes), as in michell_test.py. Returns (bucket per task, edges)."""
    qs = np.quantile(D.x, np.linspace(0, 1, nb + 1))
    qs[-1] += 1e-9
    return np.clip(np.searchsorted(qs, D.x, side="right") - 1, 0, nb - 1), qs


def agent_order(D):
    """Agents sorted by overall success rate, as in michell_test.py."""
    return np.argsort(np.bincount(D.A, D.S, D.nA) / np.bincount(D.A, D.N, D.nA))


def dc_rate(D, S, bucket, order, nb, min_runs=MIN_RUNS):
    """Share of double-cancellation cases with a true antecedent whose consequent fails."""
    v = m.violations(*m.bucket_matrix(D, S, bucket, order, nb), min_runs)
    return v[4] / max(v[5], 1)


def expected_rates(D, mu, bucket, order, nb, sims, rng, min_runs=MIN_RUNS):
    """Violation rates over `sims` binomial redraws of every agent-task cell with success probability mu."""
    return np.array([dc_rate(D, rng.binomial(D.N.astype(int), mu), bucket, order, nb, min_runs)
                     for _ in range(sims)])


def loglik(S, N, mu):
    """Binomial log-likelihood without the constant."""
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    return (S * np.log(mu) + (N - S) * np.log1p(-mu)).sum()


def half_split(D, rng):
    """Half of each cell's runs (rounded down) to fit, the rest to judge: a hypergeometric draw of the successes."""
    N_fit = np.floor(D.N / 2)
    S_fit = rng.hypergeometric(D.S.astype(int), (D.N - D.S).astype(int), N_fit.astype(int)).astype(float)
    return S_fit, N_fit


def fit_rasch(D, S, N, l2=1e-3):
    """M2 of the test (Rasch, free task difficulties) on other counts, with its L2 penalty. Returns mu per cell."""
    C = copy.copy(D)
    C.S, C.N = S, N
    M = m.models(C)
    p, _ = m.fit(C, *M["M2"][:3], l2=l2)
    return expit(M["M2"][0](p))


def fit_halflife(D, S, N, free_shape):
    """p = exp(-(t/h_a)^k), t the task's human minutes, h_a a time per agent; k = 1 (Ord) or common to all agents.

    Parameters: c_a = ln h_a, and ln k when the shape is free. Returns (mu per cell, k, c).
    """
    lt = np.log(2.0 ** D.x)

    def unpack(p):
        c = p[:D.nA]
        k = np.exp(p[D.nA]) if free_shape else 1.0
        return c, k

    def f(p):
        c, k = unpack(p)
        z = np.clip(np.exp(k * (lt[D.T] - c[D.A])), 1e-10, 50)  # cumulative hazard (t/h)^k
        ll = -S * z + (N - S) * np.log(-np.expm1(-z))
        return -ll.sum() + 1e-3 * (p ** 2).sum()

    x0 = np.concatenate([np.full(D.nA, np.log(60.0)), [0.0] if free_shape else []])
    res = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 20000})
    c, k = unpack(res.x)
    z = np.clip(np.exp(k * (lt[D.T] - c[D.A])), 1e-10, 50)
    return np.exp(-z), k, c


def fit_factors(D, S, N, k, seed, l2):
    """logit p = d_t + sum_j u_aj v_tj with k dimensions; L2 penalty l2 on u and v. Returns mu per cell.

    d_t is the task's easiness, u_aj the agent's abilities, v_tj the weight the task gives each.
    k = 1 is the two-parameter logistic model (a discrimination per task). `seed` sets the start.
    """
    rng = np.random.default_rng(seed)
    nA, nT = D.nA, D.nT

    def unpack(p):
        return p[:nT], p[nT:nT + nA * k].reshape(nA, k), p[nT + nA * k:].reshape(nT, k)

    def f(p):
        d, U, V = unpack(p)
        eta = d[D.T] + (U[D.A] * V[D.T]).sum(1)
        r = S - N * expit(eta)
        ll = (S * eta - N * np.logaddexp(0, eta)).sum()
        gd = -np.bincount(D.T, r, nT)
        gU = -np.stack([np.bincount(D.A, r * V[D.T, j], nA) for j in range(k)], 1)
        gV = -np.stack([np.bincount(D.T, r * U[D.A, j], nT) for j in range(k)], 1)
        pen = l2 * ((U ** 2).sum() + (V ** 2).sum()) + 1e-3 * (d ** 2).sum()
        g = np.concatenate([gd + 2e-3 * d, (gU + 2 * l2 * U).ravel(), (gV + 2 * l2 * V).ravel()])
        return -ll + pen, g

    x0 = np.concatenate([np.zeros(nT), 0.5 * rng.standard_normal(nA * k), 0.5 * rng.standard_normal(nT * k)])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 30000})
    d, U, V = unpack(res.x)
    return expit(d[D.T] + (U[D.A] * V[D.T]).sum(1))
