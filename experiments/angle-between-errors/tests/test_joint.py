"""A15 and the correction of A13: the weighted pieces give back the unweighted ones."""
import numpy as np

from angle import sdt
from angle.joint import _pair_rho


def test_pair_rho_with_integer_weights_is_the_repeated_sample():
    gen = np.random.default_rng(1)
    a, b = (gen.random(200) < 0.7).astype(float), (gen.random(200) < 0.6).astype(float)
    w = gen.poisson(1.0, 200).astype(float)
    q = {"task": "t", "a": a, "b": b, "ii": np.arange(200)}
    rep = np.repeat(np.arange(200), w.astype(int))
    assert abs(_pair_rho(q, {("t", "items"): w}) - sdt.rho_machines(a[rep], b[rep])) < 1e-12


def test_two_equal_observers():
    for rho in (0.0, 0.3, 0.81):
        assert abs(sdt.ideal(1.2, 1.2, rho) / 1.2 - 1 - (np.sqrt(2 / (1 + rho)) - 1)) < 1e-12
