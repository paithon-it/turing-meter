"""The checks of the review of 2026-11-10 (`angle/review.py`), on hand-built or simulated data."""
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from angle import full_evidence as fe  # noqa: E402
from angle import review as rv  # noqa: E402
from angle import sdt  # noqa: E402
from angle.measure import compare_votes  # noqa: E402

M = 16


# --- 4. the exact permutation -------------------------------------------------------------------------

@pytest.mark.parametrize("x,y", [
    ([0.3, 1.2, 0.8, 2.5, 1.9, 0.1], [70, 61, 66, 52, 60, 64]),
    ([1, 2, 2, 3, 4, 5], [9, 7, 8, 8, 3, 1]),                  # with ties: average ranks, as scipy
])
def test_spearman_exact_matches_brute_force(x, y):
    obs = spearmanr(x, y).statistic
    everything = np.array([spearmanr(x, [y[i] for i in p]).statistic for p in permutations(range(len(y)))])
    r = rv.spearman_exact(x, y)
    assert r["rho"] == pytest.approx(obs)
    assert r["method"] == "exact" and r["permutations"] == math.factorial(len(x))
    assert r["p_two_sided"] == pytest.approx(np.mean(np.abs(everything) >= abs(obs) - 1e-9))
    assert r["p_negative"] == pytest.approx(np.mean(everything <= obs + 1e-9))
    assert r["p_positive"] == pytest.approx(np.mean(everything >= obs - 1e-9))


def test_spearman_exact_reversed_order():
    r = rv.spearman_exact(range(5), [5, 4, 3, 2, 1])
    assert r["rho"] == pytest.approx(-1.0)
    assert r["p_negative"] == pytest.approx(1 / 120)
    assert r["p_two_sided"] == pytest.approx(2 / 120)


def test_bootstrap_and_leave_one_out():
    x = np.arange(8.0)
    y = -x + np.array([0.5, -0.2, 0.1, 0.0, 0.3, -0.4, 0.2, 0.1])
    b = rv.spearman_bootstrap(x, y, repetitions=500)
    assert b["ci95"][1] <= 0 and b["share_below_zero"] == pytest.approx(1.0)
    s = rv.leave_one_out([str(i) for i in range(8)], x, y)
    assert len(s["per_machine_removed"]) == 8 and s["max"] < 0


# --- 1. the αNLI keys ---------------------------------------------------------------------------------

def test_alphanli_keys():
    items = pd.DataFrame({"correct_option": [0, 1, 0, 1], "votes_correct": [80, 30, 50, 49], "votes": [100] * 4}, index=list("abcd"))
    v = rv.alphanli_keys(items)
    assert list(v["original"].index) == list("abcd")
    assert list(v["without_contested"].index) == ["a", "c"]         # b and d against the key; c tied, stays
    m = v["majority"]
    assert list(m.index) == ["a", "b", "d"]                         # c tied: no majority
    assert list(m.key) == [0, 0, 0]
    assert list(m.key_votes) == [80, 70, 51]


def test_dm_phi_votes_like_compare_votes():
    g = np.random.default_rng(3)
    n = 600
    answer_class = g.integers(0, 2, n)
    z = g.standard_normal(n)
    h = np.clip(0.8 + 0.12 * z + 0.05 * g.standard_normal(n), 0.01, 0.99).round(2)
    ok = (0.9 * z + 0.45 * g.standard_normal(n) > -1.0 + 0.3 * answer_class).astype(float)
    r = compare_votes(h, np.full(n, 100.0), ok, m=2, repetitions=0, answer_class=answer_class)
    dm, phi = rv.dm_phi_votes(h, ok, answer_class, np.ones(n))
    assert dm == pytest.approx(r["dm"])
    assert phi == pytest.approx(r["phi"])


# --- 2. the free weights ----------------------------------------------------------------------------------

def _softmax(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def simulate(n, d_a, d_b, rho, seed=0):
    """As in test_full_evidence: two calibrated networks, cues correlated ρ for each option, 5 decimals."""
    g = np.random.default_rng(seed)
    y = g.integers(0, M, n)
    r = np.arange(n)
    common = g.standard_normal((n, M))
    xa = math.sqrt(rho) * common + math.sqrt(1 - rho) * g.standard_normal((n, M))
    xb = math.sqrt(rho) * common + math.sqrt(1 - rho) * g.standard_normal((n, M))
    xa[r, y] += d_a
    xb[r, y] += d_b
    return np.round(_softmax(d_a * xa), 5), np.round(_softmax(d_b * xb), 5), y


def test_below_the_useful_angle_free_weights_beat_the_strongest():
    """cos φ = 0.8 > dA/dB = 0.5: with the constraint the pair stays at the strongest, without it reaches the free |S|."""
    d_a, d_b, rho = 1.0, 2.0, 0.8
    pa, pb, y = simulate(8000, d_a, d_b, rho, seed=5)
    d_b_meas = sdt.d_from_accuracy(float((pb.argmax(1) == y).mean()), M)
    free = rv.cv_combiner_free(pa, pb, y)
    constrained = fe.cv_combiner(pa, pb, y)
    d_free = sdt.d_from_accuracy(free["acc"], M)
    d_constrained = sdt.d_from_accuracy(constrained["acc"], M)
    assert free["a_negative"] == fe.FOLDS
    assert d_constrained < d_b_meas + 0.05
    assert d_free > d_b_meas + 0.1
    assert abs(d_free - rv.ideal_unclipped(d_a, d_b, rho)) < 0.15


def test_above_the_useful_angle_free_weights_match_the_constraint():
    pa, pb, y = simulate(4000, 1.2, 2.0, 0.0, seed=6)
    free = rv.cv_combiner_free(pa, pb, y)
    constrained = fe.cv_combiner(pa, pb, y)
    assert free["a_negative"] == 0 and free["b_negative"] == 0
    assert abs(free["acc"] - constrained["acc"]) < 2e-3


def test_ideal_unclipped():
    assert rv.ideal_unclipped(1.2, 2.0, 0.3) == pytest.approx(sdt.ideal(1.2, 2.0, 0.3))   # above the useful angle
    assert sdt.ideal(1.0, 2.0, 0.8) == pytest.approx(2.0)                                 # stopped at the stronger
    assert rv.ideal_unclipped(1.0, 2.0, 0.8) == pytest.approx(math.sqrt(5.0))


# --- 3. the complementary ChatBench trials -----------------------------------------------------------------

def test_complement():
    pairs = pd.DataFrame({"worker_id": ["w1", "w1", "w2", "w3"], "question_id": ["q1", "q2", "q1", "q3"], "acc": [1, 0, 1, 1]})
    keys = pd.DataFrame({"worker_id": ["w1", "w1", "w3"], "question_id": ["q1", "q1", "q9"], "m_ok": [1.0, 1.0, 0.0]})
    c = rv.complement(pairs, keys)
    assert list(zip(c.worker_id, c.question_id)) == [("w1", "q2"), ("w2", "q1"), ("w3", "q3")]
    assert "_merge" not in c


def test_by_effort():
    trials = pd.DataFrame({"worker_id": ["w1", "w1", "w2", "w3"], "question_id": ["q1", "q2", "q1", "q3"],
                           "acc": [1.0, 0.0, 0.0, 1.0], "acc_ai": [1.0, 1.0, 0.0, 1.0]})
    conv = pd.DataFrame({"worker_id": ["w1", "w1", "w2"], "question_id": ["q1", "q2", "q1"],
                         "user_effort": ["tried", "tried", "low effort"]})
    machine = pd.Series({"q1": 1.0, "q2": 0.0, "q3": 1.0})
    p = rv.by_effort(trials, conv, machine)
    assert p["tried"] == {"n_trials": 2, "acc_h": 0.5, "acc_m": 0.5, "acc_s": 1.0}
    assert p["low effort"] == {"n_trials": 1, "acc_h": 0.0, "acc_m": 1.0, "acc_s": 0.0}
    assert p["none"]["n_trials"] == 1                         # the trial without a conversation does not vanish
    assert sum(v["n_trials"] for v in p.values()) == len(trials)


def test_conversation_kind():
    conv = pd.DataFrame({"conv_has_answer": [True, True, False, None], "conv_answer_letter": ["B", None, None, None]})
    assert list(rv.conversation_kind(conv)) == ["answer", "out_of_options", "no_answer", "no_answer"]


def test_describe():
    trials = pd.DataFrame({"question_id": ["q1", "q2", "q3", "q4"], "acc": [0.0, 1.0, 0.0, 1.0], "acc_ai": [1.0, 0.0, 0.0, 1.0],
                           "selected_answer": ["A", "B", "C", "D"], "selected_answer_ai": ["B", "C", "D", "D"]})
    d = rv.describe(trials, pd.Series({"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0}))
    assert d == {"n_trials": 4, "acc_h": 0.5, "acc_m": 0.75, "acc_s": 0.5, "changes": 0.75,
                 "wrong_to_right": 0.25, "right_to_wrong": 0.25}


def test_in_chat_widened():
    pairs = pd.DataFrame({"worker_id": ["w1", "w1", "w2", "w3"], "question_id": ["q1", "q2", "q1", "q2"],
                          "acc": [1.0, 0.0, 0.0, 1.0], "acc_ai": [1.0, 1.0, 0.0, 1.0]})
    conv = pd.DataFrame({"worker_id": ["w1", "w1", "w2", "w3"], "question_id": ["q1", "q2", "q1", "q2"],
                         "conv_has_answer": [True, True, True, False], "conv_answer_letter": ["A", "C", None, None]})
    r = rv.in_chat_widened(pairs, conv, key=pd.Series({"q1": "A", "q2": "B"}), machine=pd.Series({"q1": 1.0, "q2": 0.0}))
    assert r["strict"] == {"n_trials": 2, "acc_m_alone": 0.5, "acc_m_chat": 0.5}       # w1-q1 correct, w1-q2 wrong
    assert r["widened"] == {"n_trials": 3, "acc_m_alone": 2 / 3, "acc_m_chat": 1 / 3}  # w2-q1 out of options: wrong
