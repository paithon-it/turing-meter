"""A17: the weighted correlation gives back the unweighted one."""
import numpy as np

from angle import sdt
from angle.position import _tet


def test_tet_with_unit_weights_is_rho_machines():
    gen = np.random.default_rng(3)
    a, b = (gen.random(500) < 0.8).astype(float), (gen.random(500) < 0.7).astype(float)
    assert abs(_tet(a, b, np.ones(500)) - sdt.rho_machines(a, b)) < 1e-12
