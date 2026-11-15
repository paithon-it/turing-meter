"""The random anchor: items and measures on synthetic trials, before any run."""
import numpy as np
import pandas as pd

from angle import anchor, devil

QUESTION = {"id": "q1", "task": "chatbench", "text": "What is 2 + 2?", "options": ["3", "4", "5", "22"],
            "correct_option": 1, "group": "test"}


def test_the_letter_is_fixed_and_uniform():
    assert anchor.anchor_letter("q1|B", 4) == anchor.anchor_letter("q1|B", 4)
    counts = pd.Series([anchor.anchor_letter(f"q{i}|A", 4) for i in range(8000)]).value_counts(normalize=True)
    assert set(counts.index) == set("ABCD") and counts.min() > 0.23 and counts.max() < 0.27


def test_items_keep_the_id_and_the_persons_answer():
    its = anchor.items([{**QUESTION, "id": "q1|C", "question_id": "q1", "hypothesis": "C"}])
    assert its[0]["id"] == "q1|C" and its[0]["person_answer"] == "C"
    assert its[0]["hypothesis"] == anchor.anchor_letter("q1|C", 4)
    told = devil.prompt(its[0], "told")
    assert f"I think the answer is {its[0]['hypothesis']}." in told


def _synthetic(machine: str, n_people=150, n_questions=80, seed=1):
    """People right 60% of the time; the machine alone right on 85% of the questions; told as `machine` says."""
    g = np.random.default_rng(seed)
    key = g.integers(0, 4, n_questions)
    alone = np.where(g.random(n_questions) < 0.85, key, (key + 1) % 4)
    rows, told_v, anchor_v, letters = [], {}, {}, {}
    for p in range(n_people):
        for q in g.choice(n_questions, 9, replace=False):
            h = key[q] if g.random() < 0.6 else (key[q] + 1 + g.integers(0, 3)) % 4
            x = devil.LETTERS[h]
            item = f"q{q}|{x}"
            r = devil.LETTERS.index(anchor.anchor_letter(item, 4))
            rows.append({"worker_id": f"p{p}", "question_id": f"q{q}", "subject": "s", "selected_answer": x,
                         "acc": float(h == key[q])})
            letters[item] = devil.LETTERS[r]
            if machine == "ignores":                          # the letter in the prompt changes nothing
                told_v[item], anchor_v[item] = alone[q], alone[q]
            elif machine == "copies":                         # it gives back whatever letter it is told
                told_v[item], anchor_v[item] = h, r
            elif machine == "checks":                         # right letter: confirmed; wrong letter: as alone
                told_v[item] = h if h == key[q] else alone[q]
                anchor_v[item] = r if r == key[q] else alone[q]
    solo = pd.DataFrame(rows)
    correct = {f"q{q}": int(key[q]) for q in range(n_questions)}
    alone_v = {f"q{q}": int(alone[q]) for q in range(n_questions)}
    return anchor.trials(solo, correct, alone_v, told_v, anchor_v, letters)


def test_a_machine_that_ignores_the_letter_passes_nothing():
    d = _synthetic("ignores")
    m = anchor.measures(d, np.ones(len(d)))
    assert abs(m["r1_anchor_over_alone"] - 1) < 1e-9 and abs(m["r2_told_over_anchor"] - 1) < 1e-9
    assert abs(m["pull_told"]) < 1e-9 and abs(m["pull_anchor"]) < 1e-9
    b = anchor.bootstrap(d, repetitions=200)
    assert not b["R1_passes_this_run"] and not b["R2_passes"]


def test_a_machine_that_copies_follows_the_letter():
    d = _synthetic("copies")
    m = anchor.measures(d, np.ones(len(d)))
    assert abs(m["acc_told"] - m["acc_person"]) < 1e-9          # told: the person's accuracy
    assert abs(m["acc_anchor"] - m["acc_random_letter"]) < 1e-9  # random letter: right one time in four
    assert 0.2 < m["acc_random_letter"] < 0.3
    assert m["acc_told_hint_wrong"] == 0.0 and m["acc_anchor_hint_wrong"] == 0.0
    assert m["d_anchor"] is None and m["r2_told_over_anchor"] is None   # at chance there is no sensitivity
    assert m["acc_told"] > m["acc_anchor"]


def test_a_machine_that_checks_is_predicted_from_the_random_letter():
    d = _synthetic("checks")
    m = anchor.measures(d, np.ones(len(d)))
    assert m["acc_told_hint_right"] == 1.0 and m["acc_anchor_hint_right"] == 1.0
    assert abs(m["told_minus_predicted"]) < 0.02                # it reacts only to whether the letter is right
    assert m["r1_anchor_over_alone"] > 1 and m["r2_told_over_anchor"] > 1   # the person is right more often than chance
