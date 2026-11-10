"""The sources: from the authors' tables to the trials we measure with."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .measure import Trials

CB_SUBJECTS = {"math": "mathematics", "physics": "physics", "moral": "ethics"}


def _index(series: pd.Series) -> tuple[np.ndarray, list]:
    codes, unique = pd.factorize(series, sort=True)
    return codes, list(unique)


def chatbench_pairs(base: Path) -> pd.DataFrame:
    """The (person, question) pairs of the answer-first condition: alone and with the AI."""
    u = pd.read_csv(base / "chatbench/full_study/user_answers_filtered.csv")
    a = u[(u.condition == "answer-first") & (u.phase == 2) & u.question_id.str.contains("-redux-")]
    alone = a[a.answer_type == "user-alone"].set_index(["worker_id", "question_id"])
    with_ai = a[a.answer_type == "user-AI"].set_index(["worker_id", "question_id"])
    m = alone.join(with_ai[["acc", "selected_answer"]], rsuffix="_ai", how="inner").reset_index()
    return m


def chatbench_machine_alone(base: Path, model: str) -> pd.Series:
    a = pd.read_csv(base / "chatbench/ai_alone_answers.csv")
    x = a[a.model == model].set_index("question_id")
    return x["free_text_acc"]


def machine_from_prob(base: Path, folder: Path, task: str, name: str, mode: str) -> pd.Series:
    """Index = item id, value = 1 if the choice is correct."""
    rows = [json.loads(l) for l in open(folder / f"{task}__{name}__{mode}.jsonl", encoding="utf-8")]
    return pd.Series({r["id"]: float(int(np.argmax(r["prob"])) == r["correct_option"]) for r in rows})


def chatbench_trials(pairs: pd.DataFrame, machine: pd.Series, subject: str | None) -> Trials:
    d = pairs if subject is None else pairs[pairs.subject == subject]
    d = d[d.question_id.isin(machine.index)].dropna(subset=["acc", "acc_ai"])
    p, _ = _index(d.worker_id)
    c, items = _index(d.question_id)
    return Trials(person=p, item=c, correct=d.acc.to_numpy(float), machine=machine.reindex(items).to_numpy(float),
                  with_machine=d.acc_ai.to_numpy(float))


def haiid_trials(base: Path, task: str, machine: pd.Series | None) -> Trials:
    """HAIID: the answer before the advice, and the answer after. machine=None uses the advice itself."""
    d = pd.read_csv(base / "haiid/haiid_dataset.csv", low_memory=False)
    d = d[d.task_name == task].copy()
    d["id"] = task + "-" + d.task_instance_id.astype(str)
    p, _ = _index(d.participant_id)
    c, items = _index(d.id)
    cl, _ = _index(d.correct_label)
    y = (d.response_1 > 0).to_numpy(float)
    y2 = (d.response_2 > 0).to_numpy(float)
    if machine is None:
        return Trials(p, c, y, (d.advice > 0).to_numpy(float), with_machine=y2, machine_per_trial=True, answer_class=cl)
    x = machine.reindex(items)
    keep = x.notna().to_numpy()[c]
    pv, _ = pd.factorize(p[keep], sort=True)
    cv, items2 = pd.factorize(c[keep], sort=True)
    return Trials(pv, cv, y[keep], x.to_numpy(float)[items2], answer_class=cl[keep])


def load_alpha_votes(base: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(base / "items/alphanli.jsonl", encoding="utf-8")]
    return pd.DataFrame([{"id": r["id"], "h": r["votes_correct"] / r["votes"], "n": r["votes"]} for r in rows]).set_index("id")


def chatbench_machine_in_chat(base: Path) -> pd.DataFrame:
    """The answer the machine gave in that conversation, where it gave one with a letter."""
    q = pd.read_csv(base / "chatbench/questions.csv").set_index("question_id")["answer"]
    d = pd.read_json(base / "chatbench/full_study/conversations.json")
    d = d[(d.conv_has_answer == True) & d.conv_answer_letter.notna() & d.question_id.isin(q.index)].copy()  # noqa: E712
    d["m_ok"] = (d.conv_answer_letter == d.question_id.map(q)).astype(float)
    return d[["worker_id", "question_id", "model", "condition", "m_ok"]]


def chatbench_trials_in_chat(pairs: pd.DataFrame, in_chat: pd.DataFrame, model: str, subject: str | None) -> Trials:
    d = pairs[pairs.model == model].merge(in_chat[in_chat.model == model], on=["worker_id", "question_id"], suffixes=("", "_c"))
    if subject is not None:
        d = d[d.subject == subject]
    d = d.dropna(subset=["acc", "acc_ai", "m_ok"])
    p, _ = _index(d.worker_id)
    c, _ = _index(d.question_id)
    return Trials(person=p, item=c, correct=d.acc.to_numpy(float), machine=d.m_ok.to_numpy(float),
                  with_machine=d.acc_ai.to_numpy(float), machine_per_trial=True)


def chatbench_all_answers(base: Path) -> pd.DataFrame:
    return pd.read_csv(base / "chatbench/full_study/user_answers_filtered.csv")
