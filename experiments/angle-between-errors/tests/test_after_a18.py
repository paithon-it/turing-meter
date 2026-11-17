"""Added after A18: the machine asked again, from the counts of right answers per question."""
import numpy as np
import pandas as pd

from angle import after_a18 as a


def test_two_answers_differ_without_replacement():
    q = pd.DataFrame({"n": [2.0, 50.0, 50.0], "k": [1.0, 50.0, 0.0]})
    grid = np.linspace(-0.3, 0.99, 5)
    s = a._stats(q, np.ones(3), grid, np.linspace(-0.2, 0.99, 5))
    assert abs(s["differ"] - 1 / 3) < 1e-12          # only the question with one right answer of two differs, always
    assert abs(s["accuracy"] - 0.5) < 1e-12
