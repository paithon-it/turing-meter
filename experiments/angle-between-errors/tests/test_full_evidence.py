"""E, the law at full evidence: the combiner and the conversion on data simulated from the 16-option model.

Each simulated network sees, for each option, a Gaussian cue (d on the correct option) and gives
its posterior probability, softmax(d · x), rounded to five decimals like the real outputs. Between
the two networks the cues of each option have a known correlation ρ.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from angle import full_evidence as fe  # noqa: E402
from angle import sdt  # noqa: E402

M = 16


def _softmax(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def simulate(n, d_a, d_b, rho, seed=0, noise_b=False):
    """Two networks calibrated on the same items; with `noise_b` the second does not see the correct option."""
    g = np.random.default_rng(seed)
    y = g.integers(0, M, n)
    r = np.arange(n)
    common = g.standard_normal((n, M))
    xa = math.sqrt(rho) * common + math.sqrt(1 - rho) * g.standard_normal((n, M))
    xb = math.sqrt(rho) * common + math.sqrt(1 - rho) * g.standard_normal((n, M))
    xa[r, y] += d_a
    if not noise_b:
        xb[r, y] += d_b
    pa = np.round(_softmax(d_a * xa), 5)
    pb = np.round(_softmax((1.0 if noise_b else d_b) * xb), 5)
    return pa, pb, y


@pytest.fixture(scope="module")
def conv():
    return fe.Conversion()


@pytest.mark.parametrize("d_a,d_b,rho", [(1.2, 2.0, 0.0), (1.5, 2.0, 0.3), (1.0, 1.6, 0.7)])
def test_combiner_reaches_the_ideal(d_a, d_b, rho):
    """Out of sample, the pair reaches |S|(φ) with the true sensitivities (ρ = 0.7 > dA/dB: at the stronger)."""
    pa, pb, y = simulate(8000, d_a, d_b, rho, seed=1)
    acc = fe.cv_combiner(pa, pb, y)["acc"]
    d_s = sdt.d_from_accuracy(acc, M)
    expected = sdt.ideal(d_a, d_b, rho)
    assert abs(d_s - expected) < 0.1, (d_s, expected)


def test_instrument_recovers_rho_and_the_prediction(conv):
    """Tetrachoric + conversion on the correct answers of the two simulated networks: true ρ within 0.05."""
    d_a, d_b, rho = 1.4, 2.2, 0.4
    pa, pb, y = simulate(20000, d_a, d_b, rho, seed=2)
    ok_a = (pa.argmax(1) == y).astype(float)
    ok_b = (pb.argmax(1) == y).astype(float)
    da, db = sdt.d_from_accuracy(ok_a.mean(), M), sdt.d_from_accuracy(ok_b.mean(), M)
    estimate = conv(sdt.rho_machines(ok_a, ok_b), da, db)["rho"]
    assert abs(estimate - rho) < 0.05, estimate
    assert abs(sdt.ideal(da, db, estimate) - sdt.ideal(d_a, d_b, rho)) < 0.08


def test_a_pure_noise_machine_does_not_worsen_the_mind():
    pa, pb, y = simulate(4000, 1.2, 0.0, 0.0, seed=3, noise_b=True)
    acc_a = float((pa.argmax(1) == y).mean())
    c = fe.cv_combiner(pa, pb, y)
    se = math.sqrt(acc_a * (1 - acc_a) / y.size)
    assert c["acc"] >= acc_a - 2 * se, (c["acc"], acc_a)
    assert c["share_b"] < 0.15, c


def test_the_mind_is_always_chosen_the_same_way(conv):
    assert fe.choose_mind(("x", "y"), (0.6, 0.8), (0.5, 0.7)) == (0, 1)
    assert fe.choose_mind(("y", "x"), (0.8, 0.6), (0.7, 0.5)) == (1, 0)
    # equal accuracy: the average confidence on the correct answer decides, then the name
    assert fe.choose_mind(("x", "y"), (0.7, 0.7), (0.6, 0.5)) == (1, 0)
    assert fe.choose_mind(("y", "x"), (0.7, 0.7), (0.5, 0.6)) == (0, 1)
    assert fe.choose_mind(("b", "a"), (0.7, 0.7), (0.5, 0.5)) == (1, 0)
    pa, pb, y = simulate(400, 1.0, 1.8, 0.3, seed=4)
    u1 = fe.pair_unit("net-a", pa, "net-b", pb, y, conv, eps_probe=())
    u2 = fe.pair_unit("net-b", pb, "net-a", pa, y, conv, eps_probe=())
    assert (u1["mind"], u1["machine"]) == (u2["mind"], u2["machine"]) == ("net-a", "net-b")
    for k in ("acc_A", "acc_B", "acc_pair", "dS", "rho", "ideal_phi"):
        assert u1[k] == pytest.approx(u2[k], abs=1e-9), k


def test_fast_conversion_coincides_with_sdt(conv):
    assert conv(0.55, 1.1, 2.3)["rho"] == pytest.approx(sdt.convert_rho_mafc(0.55, 1.1, 2.3, M), abs=1e-12)


def test_tied_at_the_maximum():
    s = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert fe.correct_with_ties(s, np.array([0, 1, 2])).tolist() == pytest.approx([0.5, 0.0, 1 / 3])


def test_order_of_the_levels():
    high = sorted(["0.4", "inf", "1.0", "3.0", "0.7"], key=fe.FAMILIES["high-pass"])
    assert high == ["inf", "3.0", "1.0", "0.7", "0.4"]
    assert sorted(["c05", "c100", "c30", "c01"], key=fe.FAMILIES["contrast"]) == ["c100", "c30", "c05", "c01"]
    assert sorted(["16-0-10", "1-0-10", "128-0-10", "2-0-10"], key=fe.FAMILIES["eidolonIII"])[0] == "1-0-10"
