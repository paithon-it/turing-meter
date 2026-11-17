"""A16: the majority of three, ties by expectation."""
import numpy as np

from angle.third_run import majority, margin


def test_majority_of_three():
    given = np.array([[0, 0, 0],    # all right
                      [0, 0, 1],    # two right
                      [1, 1, 0],    # two agree on a wrong letter
                      [0, 1, 2],    # all differ, one right: 1/3
                      [1, 2, 3],    # all differ, none right
                      [2, 1, 2]])   # first and third agree, wrong
    right = np.zeros(6, dtype=int)
    assert np.allclose(majority(given, right), [1, 1, 0, 1 / 3, 0, 0])


def test_margin_is_zero_when_the_majority_equals_a_run():
    single = np.array([1, 1, 0, 1], dtype=float)
    assert abs(margin(single, single.copy())) < 1e-12
