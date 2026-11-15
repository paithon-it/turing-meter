"""The ceiling of a pair that sees only a verdict (`angle/verdict.py`), where the answer is known."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from angle import sdt  # noqa: E402
from angle import verdict as vd  # noqa: E402

N = 20_000


# --- the error scale ----------------------------------------------------------------------------------

@pytest.mark.parametrize("a", [0.3, 0.58, 0.9, 0.99])
def test_error_agrees_with_the_articles_sensitivity(a):
    d = sdt.d_from_accuracy(a, 4)
    assert vd.error_mafc(d, 4) == pytest.approx(1 - a, rel=1e-6)
    assert vd.d_from_error(1 - a, 4) == pytest.approx(d, abs=1e-6)


@pytest.mark.parametrize("d", [1.0, 4.0, 10.0, 16.0])
def test_error_with_two_options_is_exact_also_in_the_tail(d):
    """Two options, the article's convention d′ = 2 z(a): the error is Φ(−d/2), down to 1e-15."""
    assert vd.error_mafc(d, 2) / norm.sf(d / 2) == pytest.approx(1.0, abs=1e-5)       # a ratio: at d = 16 the error is 6e-16
    assert vd.d_from_error(norm.sf(d / 2), 2) == pytest.approx(d, rel=1e-6)


def test_choice_probabilities_sum_to_one_and_match_the_forced_choice():
    mu = np.array([[1.3, 0.0, 0.0, 0.0], [0.2, -0.4, 1.1, 0.0]])
    p = vd.choice_probabilities(mu, 1.0)
    assert p.sum(1) == pytest.approx([1.0, 1.0], abs=1e-7)
    assert p[0, 0] == pytest.approx(sdt.pc_mafc(1.3, 4), abs=1e-7)
    assert p[0, 1] == pytest.approx((1 - sdt.pc_mafc(1.3, 4)) / 3, abs=1e-7)
    half = vd.choice_probabilities(mu, 0.5)                 # a smaller noise is a larger distance
    assert half[0, 0] == pytest.approx(sdt.pc_mafc(2.6, 4), abs=1e-7)


# --- the optimal observer where it has a closed form --------------------------------------------------

def _exact_independent(d_own, d_other, m=4):
    """ρ = 0: the verdict is worth a fixed bonus ln(p/q)/d on its option. Two one-dimensional integrals."""
    p = 1 - vd.error_mafc(d_other, m)
    bonus = math.log(p / ((1 - p) / (m - 1))) / d_own
    x, w = np.polynomial.legendre.leggauss(400)
    x, w = 10 * x, 10 * w * norm.pdf(10 * x)
    right = np.sum(w * ndtr(x + d_own + bonus) ** (m - 1))                       # the verdict is right
    wrong = np.sum(w * ndtr(x + d_own - bonus) * ndtr(x + d_own) ** (m - 2))     # the verdict names a wrong option
    return p * right + (1 - p) * wrong


@pytest.mark.parametrize("d_own,d_other", [(1.09, 1.31), (1.09, 2.18), (2.18, 1.09), (1.09, 3.27)])
def test_independent_errors_match_the_closed_form(d_own, d_other):
    r = vd.pair(d_own, d_other, 0.0, m=4, n=N)
    assert r["acc_pair"] == pytest.approx(_exact_independent(d_own, d_other), abs=4 * r["acc_pair_se"] + 1e-4)
    assert r["acc_pair_se"] < 0.003


def _exact_two_options(d_own, d_other, rho):
    """Two options, correlated noise: only the difference u between the observer's two cues matters."""
    so, st, s = d_own / math.sqrt(2), d_other / math.sqrt(2), math.sqrt(1 - rho * rho)
    u = np.linspace(so - 14, so + 14, 200_001)
    f0 = norm.pdf(u, so, math.sqrt(2))
    p0 = ndtr((st + rho * (u - so)) / (s * math.sqrt(2)))              # verdict 0 if option 0 is the true one
    p1 = ndtr((-st + rho * (u + so)) / (s * math.sqrt(2)))             # verdict 0 if option 1 is the true one
    eps = 1e-300
    says0_after0 = so * u + np.log(p0 + eps) - np.log(p1 + eps) > 0
    says0_after1 = so * u + np.log(1 - p0 + eps) - np.log(1 - p1 + eps) > 0
    return float(np.sum(f0 * (p0 * says0_after0 + (1 - p0) * says0_after1)) * (u[1] - u[0]))


@pytest.mark.parametrize("d_own,d_other,rho", [(1.0, 1.5, 0.3), (1.0, 2.5, 0.6), (2.0, 1.2, -0.4), (1.2, 1.2, 0.0)])
def test_two_options_with_correlated_errors_match_the_integral(d_own, d_other, rho):
    r = vd.pair(d_own, d_other, rho, m=2, n=N)
    assert r["acc_pair"] == pytest.approx(_exact_two_options(d_own, d_other, rho), abs=4 * r["acc_pair_se"] + 2e-4)


def test_a_verdict_that_knows_nothing_changes_nothing():
    r = vd.pair(1.09, 0.0, 0.0, m=4, n=N)
    assert r["acc_pair_sample"] == pytest.approx(r["acc_own_sample"], abs=1e-9)
    assert r["acc_own_sample"] == pytest.approx(sdt.pc_mafc(1.09, 4), abs=0.01)


def test_a_strong_observer_is_drawn_with_a_wider_noise_and_keeps_its_error_rate():
    """An observer wrong three times in a hundred thousand: its errors are met only with the wider noise, and
    weighed back they give its exact error rate. A verdict from someone who knows nothing leaves it there."""
    r = vd.pair(6.0, 0.0, 0.0, m=4, n=N)
    assert r["noise_widened"] == pytest.approx(2.0)
    exact = vd.error_mafc(6.0, 4)
    assert 2e-5 < exact < 5e-5
    assert 1 - r["acc_own_sample"] == pytest.approx(exact, rel=0.2)
    assert r["errors_to_stronger"] == pytest.approx(1.0, abs=1e-9)
    assert r["d_pair"] == pytest.approx(6.0, abs=1e-6)


def test_the_wider_noise_gives_the_closed_form_too():
    """Independent errors, the observer much the stronger: the pair's error rate against the two integrals."""
    r = vd.pair(4.5, 1.09, 0.0, m=4, n=N)
    assert r["noise_widened"] == pytest.approx(1.5) and r["stronger"] == "own"
    exact = 1 - _exact_independent(4.5, 1.09)
    assert 1e-3 < exact < vd.error_mafc(4.5, 4)                     # the verdict of the weaker removes some errors
    assert r["error_pair"] == pytest.approx(exact, rel=0.08)
    assert r["errors_removed"] == pytest.approx(1 - exact / vd.error_mafc(4.5, 4), abs=4 * r["errors_removed_se"] + 0.01)


@pytest.mark.parametrize("k,rho", [(1.2, 0.0), (2.0, 0.26), (3.0, 0.26), (0.6, 0.5), (2.0, -0.3)])
def test_the_optimal_observer_is_between_the_stronger_and_the_shared_cues(k, rho):
    d_own = 1.09
    r = vd.pair(d_own, k * d_own, rho, m=4, n=N)
    assert r["acc_pair_sample"] >= max(r["acc_own_sample"], r["acc_other_sample"]) - 1e-9   # it can always copy one of the two
    free = math.sqrt((d_own ** 2 + (k * d_own) ** 2 - 2 * rho * k * d_own ** 2) / (1 - rho * rho))  # no stop at the stronger
    assert r["d_pair"] <= free + 0.03
    assert r["margin"] >= -0.01


# --- the article's table ------------------------------------------------------------------------------

def test_shared_cues_give_back_the_articles_table():
    """«Quanto spazio resta alla coppia»: useful angle, margin and errors removed, as printed in the article.

    The article's person is the ChatBench group next to GPT-4o, right 57.7 times in a hundred (printed as 58).
    """
    printed = {1.2: (34, 16, 30, 17, 30), 2: (60, 3, 12, 8, 28), 3: (70.5, 0.3, 5, 2, 27),
               5: (78.5, 0, 2, 0, 26), 10: (84.3, 0, 0.5, 0, 26)}
    dh = sdt.d_from_accuracy(0.5774, 4)
    as_printed = lambda value, shown: round(value, 1) if shown != round(shown) else round(value)
    for k, (useful, m75, m90, e75, e90) in printed.items():
        a, b = vd.shared(dh, k * dh, vd.rho_of(75), 4), vd.shared(dh, k * dh, vd.rho_of(90), 4)
        assert as_printed(sdt.useful_angle(dh, k * dh), useful) == useful
        assert as_printed(100 * a["margin"], m75) == m75 and as_printed(100 * b["margin"], m90) == m90
        assert round(100 * a["errors_removed"]) == e75 and round(100 * b["errors_removed"]) == e90


def test_rows_of_the_table_have_what_the_article_needs():
    row = vd.table_row(1.09, 2.0, 90, m=4, n=8000)
    assert row["rho"] == 0.0 and row["k"] == 2.0
    for key in ("shared", "machine_verdict", "person_verdict"):
        assert {"margin", "errors_removed"} <= set(row[key])
    assert row["machine_verdict"]["margin"] <= row["shared"]["margin"] + 0.01
