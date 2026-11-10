"""The calibration: the generator does what the model says, and the true values are the right ones.

Fast tests: the real run is `python -m angle.calibration`.
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from angle import measure, sdt  # noqa: E402
from angle import calibration as T  # noqa: E402


def _expected(d_mean, sd, f):
    """E[f(d_i)] with d_i ~ Normal(d_mean, sd) clipped at D_MIN, by Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / weights.sum()
    return float(sum(p * f(x) for p, x in zip(weights, np.clip(d_mean + sd * nodes, T.D_MIN, None))))


@pytest.mark.parametrize("name", ["B", "C", "G"])
def test_generator_reproduces_accuracies(name):
    """Person, machine and pair: the accuracies of the model, within the sampling error."""
    sc = T.BY_NAME[name]
    people = 40_000
    items = int(sc.items * people / sc.people)
    _, _, correct, _, m_trial, pair = T._raw(sc, np.random.default_rng(11), people, items)
    w, r = sc.weight, sc.rho
    scale = math.sqrt(1 + w * w + 2 * w * r)            # the variance of the noise of a + w·b, per option
    expected_h = _expected(sc.dh, sc.sd, lambda d: sdt.pc_mafc(d, 4))
    expected_m = sdt.pc_mafc(sc.dm, 4)
    expected_s = _expected(sc.dh, sc.sd, lambda d: sdt.pc_mafc((d + w * sc.dm) / scale, 4))
    n = correct.size
    n_m = n if sc.per_trial else items                  # the machine per item: the sample is the items
    for obs, exp, nn in ((correct.mean(), expected_h, n), (m_trial.mean(), expected_m, n_m), (pair.mean(), expected_s, n)):
        se = math.sqrt(exp * (1 - exp) / (nn / 2))      # half of the sample: margin for the trials on the same item
        assert abs(obs - exp) < 4 * se, (name, obs, exp, se)


@pytest.mark.parametrize("per_trial", [False, True])
def test_with_zero_weight_the_pair_answers_like_the_person(per_trial):
    p = T.generate(replace(T.BY_NAME["A"], w=0.0, per_trial=per_trial), 3)
    assert np.array_equal(p.with_machine, p.correct)


def test_copy_the_true_turing_is_k():
    """In scenario D the pair gives the machine's answer: dS = dM and turing = aggregate dM/dH."""
    sc = T.BY_NAME["D"]
    p = T.generate(sc, 5)
    assert np.array_equal(p.with_machine, p.machine[p.item])
    v = T.true_values(sc, n=300_000)
    assert v["ds"] == v["dm"]
    assert v["turing"] == pytest.approx(v["k"], rel=1e-12)
    dh_aggregate = sdt.d_from_accuracy(_expected(sc.dh, sc.sd, lambda d: sdt.pc_mafc(d, 4)), 4)
    assert v["turing"] == pytest.approx(sc.dm / dh_aggregate, rel=0.02)


def test_true_values_deterministic():
    sc = T.BY_NAME["E"]
    a, b = T.true_values(sc, n=100_000), T.true_values(sc, n=100_000)
    assert a == b
    assert T.true_values(sc, n=100_000, seed=1)["turing"] != a["turing"]
    assert np.array_equal(T.generate(sc, 9).with_machine, T.generate(sc, 9).with_machine)


def test_optimal_weight_reaches_the_ideal():
    """With all people equal, w* brings the pair to the ideal: true η = 1."""
    v = T.true_values(replace(T.BY_NAME["E"], sd=0.0), n=400_000)
    assert v["eta_phi"] == pytest.approx(1.0, abs=0.015)


def test_scenarios_do_what_the_protocol_asks():
    v = {n: T.true_values(T.BY_NAME[n], n=200_000) for n in "ABCE"}
    assert v["A"]["eta_phi"] == pytest.approx(1.0, abs=0.02)                       # ideal pair
    assert 1.0 < v["B"]["turing"] < v["B"]["apparent_turings"] - 0.2                # halfway
    assert v["C"]["turing"] < 1.0                                                  # turing below one
    assert v["E"]["k"] == pytest.approx(1.0, abs=0.15)                             # machine equal to the person


def test_apparent_turings_do_not_change_the_instrument():
    """The wrapper adds |S|/dH in every resample and touches nothing else of compare."""
    p = T.generate(T.BY_NAME["A"], 7)
    a = measure.compare(p, m=4, repetitions=25, seed=3)
    b = T.compare_with_apparent(p, m=4, repetitions=25, seed=3)
    assert measure._one_pass.__name__ == "_one_pass"            # the original is back in its place
    assert {k: b[k] for k in a} == a
    assert set(b) - set(a) == {"apparent_turings", "apparent_turings_ci"}
    assert b["apparent_turings"] == pytest.approx(b["ideal_phi"] / b["dh"])
    lo, hi = b["apparent_turings_ci"]
    assert lo < hi


def test_statistics_and_criterion():
    rows = [{"turing": 1.0 + x, "turing_ci": [0.9 + x, 1.1 + x]} for x in (-0.2, -0.05, 0.0, 0.05, 0.3)]
    st = T.statistics(rows, 1.0, "turing")
    assert st["coverage_pct"] == pytest.approx(60.0)
    assert st["bias"] == pytest.approx(0.02)
    assert st["width_mean"] == pytest.approx(0.2)
    assert st["rmse"] == pytest.approx(math.sqrt((0.04 + 0.0025 + 0.0025 + 0.09) / 5))
    assert T.criterion("A", st)["passes"] is False
    ok = {"bias_pct": 2.0, "coverage_pct": 95.0}
    assert T.criterion("A", ok)["passes"] is True
    assert T.criterion("A", dict(ok, coverage_pct=99.0))["passes"] is False
    assert T.criterion("F", ok)["passes"] is None
