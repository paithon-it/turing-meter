"""Closeness net of difficulty (`angle/difficulty.py`), on hand-built tables and on stories simulated with a known answer."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from angle import difficulty as df  # noqa: E402
from angle import sdt  # noqa: E402
from angle.measure import compare_votes  # noqa: E402


# --- strata -------------------------------------------------------------------------------------------

def test_strata_merge_the_levels_from_the_hard_end():
    score = np.array([0, 0, 1, 2, 2, 2, 3, 3, 3, 3])
    assert df.strata_from_score(score, 3).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]


def test_a_short_tail_at_the_easy_end_joins_the_stratum_before():
    score = np.array([0, 0, 0, 1, 1, 1, 2, 5])
    assert df.strata_from_score(score, 3).tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert df.strata_from_score(np.array([4, 4, 7]), 5).tolist() == [0, 0, 0]          # fewer stories than one stratum


def test_quantile_strata_have_equal_sizes_when_there_are_no_ties():
    s = df.quantile_strata(np.arange(100.0), 4)
    assert np.bincount(s).tolist() == [25, 25, 25, 25] and s[0] == 0 and s[-1] == 3


# --- one correlation for all strata -------------------------------------------------------------------

@pytest.mark.parametrize("p1,p2,rho", [(0.84, 0.92, 0.68), (0.6, 0.7, -0.3), (0.9, 0.95, 0.0)])
def test_one_stratum_is_the_tetrachoric(p1, p2, rho):
    z1, z2 = norm.ppf(p1), norm.ppf(p2)
    p11 = float(sdt.bvn(z1, z2, rho))
    table = np.array([[p11, p1 - p11, p2 - p11, 1 - p1 - p2 + p11]]) * 1000
    r = df.common_rho(table)
    assert r["rho"] == pytest.approx(rho, abs=1e-3)
    assert r["rho"] == pytest.approx(sdt.tetrachoric(p1, p2, p11), abs=0.01)             # the article's estimator, on its grid
    assert r["strata_used"] == 1


def test_strata_with_no_errors_carry_no_information():
    informative = np.array([400.0, 100.0, 150.0, 350.0])
    alone = df.common_rho(informative[None, :])
    with_empty = df.common_rho(np.array([informative, [500.0, 0.0, 0.0, 0.0], [300.0, 0.0, 200.0, 0.0]]))
    assert with_empty["rho"] == pytest.approx(alone["rho"], abs=1e-6)
    assert with_empty["strata_used"] == 1 and with_empty["strata"] == 3
    assert np.isnan(df.common_rho(np.array([[500.0, 0.0, 0.0, 0.0]]))["rho"])


def test_mantel_haenszel_gives_back_a_common_odds_ratio():
    tables = np.array([[40.0, 10.0, 20.0, 20.0], [30.0, 5.0, 12.0, 8.0], [9.0, 3.0, 6.0, 8.0]])   # each with odds ratio 4
    assert df.mantel_haenszel(tables) == pytest.approx(4.0)
    assert df.mantel_haenszel(tables.sum(0, keepdims=True)) != pytest.approx(4.0, abs=0.1)       # pooled, it is another number


# --- stories with a known answer ----------------------------------------------------------------------

def _stories(n=4000, levels=8, load_person=0.6, load_machine=0.5, load_other=0.7, votes=100, seed=0):
    """Stories at several levels of difficulty; within a level a story has one more factor shared by the minds.

    A vote, a machine and a second machine have their own noise on top. Within a level the latent correlation
    between a vote and the machine is load_person · load_machine; between the two machines, load_machine ·
    load_other. Across levels the thresholds move together for everyone: the pooled correlation is larger.
    """
    g = np.random.default_rng(seed)
    level = g.integers(0, levels, n)
    shift = np.linspace(-1.4, 1.4, levels)[level]              # the higher the level, the easier the story for all
    z = g.standard_normal(n)
    x = load_person * z[:, None] + np.sqrt(1 - load_person ** 2) * g.standard_normal((n, votes))
    right = ((x + shift[:, None] + 0.6) > 0).sum(1)
    m1 = ((load_machine * z + np.sqrt(1 - load_machine ** 2) * g.standard_normal(n) + shift + 0.9) > 0).astype(float)
    m2 = ((load_other * z + np.sqrt(1 - load_other ** 2) * g.standard_normal(n) + shift + 1.1) > 0).astype(float)
    return level, right, np.full(n, votes), m1, m2


def test_person_and_machine_net_of_a_known_difficulty():
    level, right, total, m1, _ = _stories()
    r = df.net(right, total, m1, level, repetitions=200)
    assert r["raw"]["rho"] > 0.55                                             # difficulty and likeness together
    assert r["net"]["rho"] == pytest.approx(0.6 * 0.5, abs=0.04)              # the likeness alone
    assert r["net"]["rho_ci"][0] < 0.30 < r["net"]["rho_ci"][1]
    assert r["net"]["phi"] == pytest.approx(sdt.angle_degrees(r["net"]["rho"]))
    assert r["raw"]["rho_ci"][0] > r["net"]["rho_ci"][1]


def test_two_machines_net_of_a_known_difficulty():
    level, _, _, m1, m2 = _stories(seed=1)
    r = df.net(m1, np.ones(m1.size), m2, level, repetitions=200)
    assert r["raw"]["rho"] > 0.6
    assert r["net"]["rho"] == pytest.approx(0.5 * 0.7, abs=0.06)
    assert r["net"]["odds_ratio"] > 1 and r["net"]["odds_ratio"] < r["raw"]["odds_ratio"]


def test_minds_alike_only_through_difficulty_are_at_ninety_degrees():
    level, right, total, m1, _ = _stories(load_person=0.0, seed=2)
    r = df.net(right, total, m1, level, repetitions=200)
    assert r["raw"]["phi"] < 75
    assert r["net"]["phi"] == pytest.approx(90.0, abs=3.0)
    assert r["net"]["rho_ci"][0] < 0 < r["net"]["rho_ci"][1]


def test_coarse_strata_leave_difficulty_in():
    """Three bands of a difficulty that has eight levels: the correlation stays above the true one. Not a proof of likeness."""
    level, right, total, m1, _ = _stories(load_person=0.0, seed=3)
    coarse = df.net(right, total, m1, level // 3, repetitions=0)
    assert coarse["net"]["rho"] > 0.08


# --- the vote step, and the contested stories ---------------------------------------------------------

def test_votes_step_is_the_articles_estimator():
    g = np.random.default_rng(4)
    n = 600
    key = g.integers(0, 2, n)
    h = np.clip(g.beta(6, 1.5, n), 0.01, 0.99)
    ok = (g.random(n) < 0.3 + 0.6 * h).astype(float)
    a = compare_votes(h, np.full(n, 100.0), ok, m=2, repetitions=1, answer_class=key)
    b = df.votes_step(h, ok, key, np.ones(n))
    for name in ("dh", "dm", "rho", "phi", "useful_angle", "k"):
        assert b[name] == pytest.approx(a[name])
    assert b["margin"] == pytest.approx(a["ideal_phi"] / max(a["dh"], a["dm"]) - 1)


def test_contested_on_half_of_the_votes():
    g = np.random.default_rng(5)
    n = 800
    key = g.integers(0, 2, n)
    right = np.r_[g.binomial(100, 0.9, n - 100), g.binomial(100, 0.25, 100)]        # the last hundred stories are contested
    total = np.full(n, 100)
    ok = (g.random(n) < np.where(np.arange(n) < n - 100, 0.9, 0.3)).astype(float)
    r = df.contested_split(right, total, ok, key, splits=40, repetitions=100)
    assert r["same_votes"]["stories_removed"] == 100
    assert r["split"]["stories_removed"] == pytest.approx(100, abs=1)
    assert r["all_stories"]["phi"] < r["same_votes"]["phi"]                              # the contested stories close the angle
    assert r["split"]["phi"] == pytest.approx(r["same_votes"]["phi"], abs=6)
    assert r["split"]["phi_ci"][0] < r["split"]["phi"] < r["split"]["phi_ci"][1]


def test_no_bias_at_the_size_of_alphanli():
    """1,532 stories, eighteen strata, a strong machine with few errors in the easy strata: the mean over a dozen
    simulated corpora stays on the true correlation."""
    person, machines = [], []
    for seed in range(12):
        g = np.random.default_rng(100 + seed)
        n, levels = 1532, 18
        level = g.integers(0, levels, n)
        shift = np.linspace(-1.2, 2.0, levels)[level]
        z = g.standard_normal(n)
        right = ((0.6 * z[:, None] + 0.8 * g.standard_normal((n, 100)) + shift[:, None] + 0.4) > 0).sum(1)
        a = ((0.5 * z + np.sqrt(0.75) * g.standard_normal(n) + shift + 1.3) > 0).astype(float)
        b = ((0.8 * z + 0.6 * g.standard_normal(n) + shift + 1.3) > 0).astype(float)
        person.append(df.net(right, np.full(n, 100), a, level, repetitions=0)["net"]["rho"])
        machines.append(df.net(a, np.ones(n), b, level, repetitions=0)["net"]["rho"])
    assert np.mean(person) == pytest.approx(0.6 * 0.5, abs=0.03)
    assert np.mean(machines) == pytest.approx(0.5 * 0.8, abs=0.06)
