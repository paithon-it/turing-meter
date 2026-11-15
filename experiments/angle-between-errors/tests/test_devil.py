"""The devil's advocate: items, prompts and measures on synthetic trials, before any run."""
import numpy as np
import pandas as pd

from angle import devil
from gpu.machines_llm import format_prompt

QUESTION = {"id": "q1", "task": "chatbench", "text": "What is 2 + 2?", "options": ["3", "4", "5", "22"],
            "correct_option": 1, "group": "test"}


def test_alone_is_the_letter_prompt_of_the_article():
    assert devil.prompt(QUESTION, "alone") == format_prompt(QUESTION, "letter")


def test_told_and_devil_carry_the_persons_answer():
    c = {**QUESTION, "id": "q1|C", "question_id": "q1", "hypothesis": "C"}
    told, dev = devil.prompt(c, "told"), devil.prompt(c, "devil")
    assert "I think the answer is C." in told and told.endswith("(A to D).")
    assert "I think the answer is C." in dev and "strongest arguments against my answer" in dev
    assert dev.endswith("Answer: Y, where Y is the letter of the correct option (A to D).")
    assert told.startswith(QUESTION["text"]) and dev.startswith(QUESTION["text"])


def test_one_item_per_question_and_letter():
    solo = pd.DataFrame({"worker_id": ["a", "b", "c", "d", "e"], "question_id": ["q1", "q1", "q1", "q2", "q1"],
                         "subject": "s", "selected_answer": ["B", "B", "C", "A", "E"], "acc": [1, 1, 0, 0, 0]})
    bank = [QUESTION, {**QUESTION, "id": "q2"}]
    its = devil.items(solo, bank)
    assert [c["id"] for c in its] == ["q1|B", "q1|C", "q2|A"]          # E is not one of the four options
    assert all(c["hypothesis"] == c["id"][-1] and c["question_id"] == c["id"][:2] for c in its)


def _synthetic(n_people=120, n_questions=60, seed=0, told="copy", dev="correct"):
    """People right 60% of the time; the machine alone right on 80% of the questions; told and devil as asked."""
    g = np.random.default_rng(seed)
    key = g.integers(0, 4, n_questions)
    alone_ok = g.random(n_questions) < 0.8
    alone = np.where(alone_ok, key, (key + 1) % 4)
    rows, told_v, dev_v = [], {}, {}
    for p in range(n_people):
        for q in g.choice(n_questions, 9, replace=False):
            h = key[q] if g.random() < 0.6 else (key[q] + 1 + g.integers(0, 3)) % 4
            x = devil.LETTERS[h]
            rows.append({"worker_id": f"p{p}", "question_id": f"q{q}", "subject": "s", "selected_answer": x,
                         "acc": float(h == key[q])})
            item = f"q{q}|{x}"
            told_v[item] = {"copy": h, "alone": alone[q]}[told]
            dev_v[item] = {"correct": key[q], "alone": alone[q]}[dev]
    solo = pd.DataFrame(rows)
    correct = {f"q{q}": int(key[q]) for q in range(n_questions)}
    alone_v = {f"q{q}": int(alone[q]) for q in range(n_questions)}
    return devil.trials(solo, correct, alone_v, told_v, dev_v)


def test_measures_see_a_machine_that_copies_the_person():
    d = _synthetic()
    m = devil.measures(d, np.ones(len(d)))
    assert m["acc_told_person_wrong"] == 0.0                              # it repeats every wrong answer
    assert m["trap_told"] == 1.0
    assert m["acc_told"] == m["acc_person"]
    assert m["acc_devil"] == 1.0 and m["acc_devil_person_wrong"] == 1.0
    assert m["h1_told_minus_alone_person_wrong"] < 0 < m["h2_devil_minus_told_person_wrong"]


def test_the_criteria_pass_on_a_clear_effect_and_not_on_a_null():
    effect = devil.bootstrap(_synthetic(), repetitions=200)
    assert effect["H1_passes"] and effect["H2_passes"]
    null = devil.bootstrap(_synthetic(told="alone", dev="alone"), repetitions=200)
    assert not null["H1_passes"] and not null["H2_passes"] and not null["H3_passes"]
    assert null["h3_yield_devil"] == 1.0


def test_a_missing_item_drops_the_trial():
    d = _synthetic(n_people=10, n_questions=20)
    solo = d[["worker_id", "question_id", "subject", "selected_answer", "acc"]]
    correct = dict(zip(d.question_id, d.key))
    alone = dict(zip(d.question_id, d.m0))
    told = dict(zip(d["item"], d.mA))
    dev = dict(zip(d["item"], d.mB))
    gone = d["item"].iloc[0]
    del dev[gone]
    assert len(devil.trials(solo, correct, alone, told, dev)) == len(d) - (d["item"] == gone).sum()


def test_claude_keeps_the_arguments_and_reads_the_letter(monkeypatch):
    import json
    from closed import devil_claude
    seen = {}

    def fake(text, model, effort, sch, timeout=300):
        seen["schema"], seen["text"] = sch, text
        return json.dumps({"structured_output": {"arguments": "4 is 2 + 2, not 5.", "answer": "B"},
                           "usage": {"output_tokens": 30}, "modelUsage": {"claude-haiku-5-5": {}}})
    monkeypatch.setattr(devil_claude, "launch", fake)
    c = {**QUESTION, "id": "q1|C", "question_id": "q1", "hypothesis": "C"}
    line, missing = devil_claude.ask(c, "devil", "claude-haiku-5-5", "low")
    assert missing is None and line["prob"] == [0.0, 1.0, 0.0, 0.0] and line["head"] == "4 is 2 + 2, not 5."
    assert list(seen["schema"]["properties"]) == ["arguments", "answer"] and "against my answer" in seen["text"]
    line, _ = devil_claude.ask(c, "told", "claude-haiku-5-5", "low")
    assert list(seen["schema"]["properties"]) == ["answer"] and "I think the answer is C." in seen["text"]


def test_the_reasoning_controls_match_the_article_and_the_devil():
    c = {**QUESTION, "id": "q1|C", "question_id": "q1", "hypothesis": "C"}
    assert devil.prompt(QUESTION, "alone_reason") == format_prompt(QUESTION, "reason")
    told = devil.prompt(c, "told_reason")
    assert "I think the answer is C." in told and "Think step by step, briefly." in told
    assert told.endswith("Answer: Y, where Y is the letter of the correct option (A to D).")
    assert devil.prompt(c, "devil").endswith(told.split("briefly. ", 1)[1])      # the same closing line as the devil
