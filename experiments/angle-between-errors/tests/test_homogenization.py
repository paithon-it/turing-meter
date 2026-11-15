"""Homogenization (PROTOCOL.md, «After the seventh review», A), on synthetic trials, before any run on ChatBench."""
import itertools

import numpy as np
import pandas as pd

from angle import homogenization as h


def synthetic(shared: float, copy_letter: float, n_items: int = 300, per_item: int = 8, seed: int = 1) -> pd.DataFrame:
    """People answer alone, then with a chat whose item-level component is shared by everyone on that item."""
    g = np.random.default_rng(seed)
    b = g.standard_normal(n_items)
    m = g.standard_normal(n_items)
    machine_wrong = g.integers(1, 4, n_items)            # the wrong letter the chat pushes, 1..3 (0 is the key)
    rows = []
    for i in range(n_items):
        for k in range(per_item):
            z = 0.3 + b[i] + g.standard_normal()
            z2 = 0.8 + np.sqrt(1 - shared) * b[i] + np.sqrt(shared) * 1.5 * m[i] + g.standard_normal()
            a, a2 = int(z > 0), int(z2 > 0)
            la = "A" if a else "BCD"[g.integers(0, 3)]
            if a2:
                l2 = "A"
            else:
                l2 = "ABCD"[machine_wrong[i]] if g.random() < copy_letter else "BCD"[g.integers(0, 3)]
            rows.append((f"p{i}-{k}", f"q{i}", a, a2, la, l2, "ABCD"[machine_wrong[i]], True))
    return pd.DataFrame(rows, columns=["worker_id", "question_id", "acc", "acc_ai", "selected_answer",
                                       "selected_answer_ai", "conv_answer_letter", "letter"])


def test_same_letter_counts_pairs_by_brute_force():
    items = np.array([0, 0, 0, 0, 1, 1, 2])
    wrong = np.array([True, True, True, False, True, True, True])
    letters = np.array(["B", "B", "C", "A", "D", "D", "B"])
    same, pairs = h.same_letter(items, wrong, letters, np.ones(3))
    brute_same = brute_pairs = 0
    for i, j in itertools.combinations(range(len(items)), 2):
        if items[i] == items[j] and wrong[i] and wrong[j]:
            brute_pairs += 1
            brute_same += letters[i] == letters[j]
    assert (same, pairs) == (brute_same, brute_pairs) == (2, 4)
    same2, pairs2 = h.same_letter(items, wrong, letters, np.array([2.0, 1.0, 1.0]))
    assert (same2, pairs2) == (3, 7)


def test_identical_conditions_give_zero_difference():
    x = synthetic(shared=0.0, copy_letter=0.0)
    x["acc_ai"], x["selected_answer_ai"] = x.acc, x.selected_answer
    r = h.measure(x, repetitions=40, subsets=False)
    assert abs(r["delta"]) < 1e-9 and r["delta_ci"] == [0.0, 0.0]
    assert abs(r["same_letter_diff"]) < 1e-12


def test_a_shared_chat_closes_the_angle_and_shares_the_wrong_letter():
    r = h.measure(synthetic(shared=0.6, copy_letter=0.8), repetitions=60, subsets=False)
    assert r["phi_ss"] < r["phi_hh"] and r["passes"]
    assert r["same_letter_chat"] > r["same_letter_alone"] and r["a3_passes"]
    assert r["shared_letter_from_chat"]["both_chats_gave_it"] > 0.5


def test_a_chat_without_a_shared_part_does_not_pass():
    r = h.measure(synthetic(shared=0.0, copy_letter=0.0, seed=3), repetitions=60, subsets=False)
    assert not r["a3_passes"]
    assert r["delta_ci"][0] < 0 < r["delta_ci"][1] or not r["passes"]
