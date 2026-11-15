"""The letter is read from the text of a closed model: what counts as an answer, and what does not."""
import pytest

from closed.machines_claude import read_letter


@pytest.mark.parametrize("text, k, expected", [
    ("B", 4, 1),
    ("**B**", 4, 1),
    ("B.", 4, 1),
    ("2/70 = 1/35\n\nD", 4, 3),                                   # the working, then the letter
    ("10 - (11 + 6) = 10 - 17 = -7\n\n**A**", 4, 0),
    ("P(X>3) = 1/8.\nP(at least one > 3) = 15/64.\n\n**B**", 4, 1),
    ("B. 15/64", 4, 1),
    ("B. 15/64\n\nBecause the two variables are independent.", 4, 1),
    ("The answer is C", 4, 2),
    ("Answer: A", 2, 0),
    ("C\n\nD", 4, 3),                                             # the last line is the verdict
])
def test_reads_the_letter(text, k, expected):
    assert read_letter(text, k) == expected


@pytest.mark.parametrize("text, k", [
    ("", 4),
    ("I cannot determine the answer from the information given.", 4),
    ("A probability of 1/35 seems right, but none of the options matches.", 4),   # the article, not the option
    ("E", 4),                                                       # not one of the four
    ("C", 2),
])
def test_does_not_guess(text, k):
    assert read_letter(text, k) is None


def test_margin_is_what_the_ideal_pair_promises_over_the_stronger():
    from angle import sdt
    from angle.frontier import summary

    dh, dm = 1.0, 2.0                                  # the machine is worth twice the person: useful angle 60 degrees
    def comparison(rho):
        return {"dh": dh, "dm": dm, "phi": sdt.angle_degrees(rho), "useful_angle": sdt.useful_angle(dh, dm),
                "ideal_phi": sdt.ideal(dh, dm, rho), "ideal_90": sdt.ideal(dh, dm, 0.0)}

    independent, below = summary(comparison(0.0)), summary(comparison(0.7))
    assert independent["margin"] == pytest.approx(5 ** 0.5 / 2 - 1)     # Pythagoras over the machine alone
    assert independent["above_useful_angle"] and independent["margin"] == pytest.approx(independent["margin_90"])
    assert below["margin"] == pytest.approx(0.0) and not below["above_useful_angle"]   # the ideal stops at the stronger


def test_where_the_points_go_is_a_partition():
    import pandas as pd
    from angle.conversations import where_points_go

    x = pd.DataFrame({"letter": [True, True, False, False], "chat_ok": [1.0, 0.0, 0.0, 0.0],
                      "m_alone": [1.0, 0.5, 0.8, 0.2], "acc_ai": [1.0, 0.0, 0.0, 1.0]})
    w = where_points_go(x)
    assert w["total"] == pytest.approx(x.acc_ai.mean() - x.m_alone.mean())
    assert w["conversation_with_answer"] + w["decision_with_answer"] + w["chats_without_answer"] == pytest.approx(w["total"])


def test_identical_people_have_no_spread_beyond_the_binomial():
    import numpy as np
    from angle.conversations import _spread

    n = np.full(400, 10.0)
    k = np.random.default_rng(1).binomial(10, 0.6, 400).astype(float)
    mu, var = _spread(k, n)
    assert mu == pytest.approx(0.6, abs=0.02)
    assert var < 0.003
