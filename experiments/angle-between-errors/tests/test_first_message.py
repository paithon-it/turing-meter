"""A19, the first message of the person (PROTOCOL.md, After the fifth review (v4): A19): written before any run."""
import numpy as np
import pandas as pd

from angle import first_message as fm


def test_the_first_message_is_the_persons_and_not_empty():
    h = [{"role": "AI", "content": "Hello, how can I help?"}, {"role": "User", "content": "  "},
         {"role": "User", "content": "What is 2+2?"}, {"role": "User", "content": "A) 3 B) 4"}]
    assert fm.first_person_message(h) == "What is 2+2?"
    assert fm.first_person_message([{"role": "AI", "content": "hi"}]) is None
    assert fm.first_person_message(None) is None


def test_options_the_question_contains_or_too_short_are_not_usable():
    q = "Which is larger, 15/64 or 1/4?"
    assert fm.usable_options(q, ["15/64", "1/4", "3/8", "1"]) == ["3/8"]
    assert fm.usable_options("Pick a colour", ["Red", "Blue", "x", "Green"]) == ["red", "blue", "green"]


def test_classify():
    q, o = "What is the main cause of the seasons?", ["Axial tilt", "Distance from the Sun", "Solar flares", "Clouds"]
    assert fm.classify(None, q, o) == ("no message", None)
    assert fm.classify("hi", q, o) == ("missing", True)
    assert fm.classify(q, q, o) == ("missing", False)                             # the question pasted, no options
    assert fm.classify(q + " A) Axial tilt B) Distance from the Sun", q, o) == ("present", False)
    assert fm.classify("tilt?", q, o)[0] == "missing"                             # not the whole option
    p = "Is 2 or 3 prime?"                                                        # one usable option only
    assert fm.classify(p, p, ["2", "3", "5", "both"]) == ("not classifiable", False)
    assert fm.classify(p, p, ["2", "3", "both", "neither"]) == ("missing", False)


def _synthetic() -> pd.DataFrame:
    rows = []
    # letter: 4 trials, 1 missing; silent: 4 trials, 2 missing, 1 no message; outside: 2 trials
    spec = [("letter", "missing", True, 1, 1), ("letter", "present", False, 1, 1), ("letter", "present", False, 0, 1),
            ("letter", "present", False, 1, 0),
            ("silent", "missing", True, 0, 1), ("silent", "missing", False, 0, 1), ("silent", "present", False, 1, 1),
            ("silent", "no message", None, 0, 1),
            ("outside", "present", False, 0, 1), ("outside", "not classifiable", True, 1, 1)]
    for i, (g, c, s, acc, alone) in enumerate(spec):
        rows.append({"worker_id": f"w{i % 3}", "question_id": f"q{i}", "group": g, "category": c, "short": s,
                     "letter": g == "letter", "acc_ai": float(acc), "m_alone": float(alone),
                     "chat_ok": float(alone)})
    return pd.DataFrame(rows)


def test_shares_and_differences():
    x = _synthetic()
    s = fm.stats(x, np.ones(len(x)))
    assert s["letter"]["missing"] == 0.25 and abs(s["silent"]["missing"] - 2 / 3) < 1e-12
    assert abs(s["diff_missing"] - (2 / 3 - 0.25)) < 1e-12
    assert abs(s["silent"]["short"] - 1 / 3) < 1e-12 and s["letter"]["short"] == 0.25
    assert s["silent"]["no_message"] == 0.25
    assert abs(s["silent_given_missing"] - 2 / 3) < 1e-12 and abs(s["silent_given_present"] - 1 / 5) < 1e-12


def test_the_parts_add_up_to_the_points_lost_without_a_letter():
    x = _synthetic()
    s = fm.stats(x, np.ones(len(x)))
    a = ~x.letter
    expected = a.mean() * (x.acc_ai[a].mean() - x.m_alone[a].mean())
    assert abs(s["chats_without_answer"] - expected) < 1e-12
    assert abs(s["parts"]["silent_missing"]["points"] - 2 / 10 * (0 - 1)) < 1e-12
    w = np.arange(1, len(x) + 1, dtype=float)
    sw = fm.stats(x, w)
    ew = w[a].sum() / w.sum() * (np.average(x.acc_ai[a], weights=w[a]) - np.average(x.m_alone[a], weights=w[a]))
    assert abs(sw["chats_without_answer"] - ew) < 1e-12
