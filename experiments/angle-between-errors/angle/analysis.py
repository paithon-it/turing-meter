"""All person-machine comparisons, and the test of the law. Writes results/map.json.

Usage, from the experiment folder:  python -m angle.analysis GPU_RESULTS_DIR
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import combiner, sdt, sources
from .sources import _index
from .measure import Trials, compare, compare_votes, conversion_table

BASE = Path("data")
REPETITIONS = 2000
LETTERS = "ABCD"
LOGIT_MAX = 8.0


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.clip(np.log(p / (1 - p)), -LOGIT_MAX, LOGIT_MAX)


def load_machine(folder: Path, task: str, name: str, mode: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(folder / f"{task}__{name}__{mode}.jsonl", encoding="utf-8")]
    t = pd.DataFrame([{"id": r["id"], "top": int(np.argmax(r["prob"])), "ptop": float(max(r["prob"])),
                       "correct_option": r["correct_option"], "ok": float(int(np.argmax(r["prob"])) == r["correct_option"])} for r in rows])
    return t.drop_duplicates("id").set_index("id")


def _d(a, m):
    return sdt.d_from_accuracy(float(np.clip(a, 1e-4, 1 - 1e-4)), m) if a > 1 / m else float("nan")


def _d_combo_classes(outcome, answer_class, items, weights=None):
    """The d' of the combiner with the same rule as dm: class by class, free criterion, correction on distinct items.

    The rows are person x item: counting the rows in the correction of 1/2 would let z explode when the
    combiner learns to always say the same class on few items.
    """
    outcome, answer_class, items = np.asarray(outcome, float), np.asarray(answer_class), np.asarray(items)
    w = np.ones(outcome.size) if weights is None else np.asarray(weights, float)
    c1, c0 = answer_class == 1, answer_class == 0
    return sdt.d_from_classes(float(np.average(outcome[c1], weights=w[c1])), float(np.average(outcome[c0], weights=w[c0])),
                              n1=float(np.unique(items[c1]).size), n0=float(np.unique(items[c0]).size))


def law(dh, dm, rho, m, levels, combo_result, n=250_000, d_combo=None):
    """The predictions compared: with the measured rho and at 90 degrees, on the same information as the combiner."""
    if not (dh > 0 and dm > 0):
        return {}
    pred_phi = _d(combiner.predict(dh, dm, rho, m, levels, n=n), m)
    pred_90 = _d(combiner.predict(dh, dm, 0.0, m, levels, n=n), m)
    return {"d_combo": _d(combo_result, m) if d_combo is None else d_combo, "pred_phi": pred_phi, "pred_90": pred_90,
            "dmax": max(dh, dm), "acc_combo": float(combo_result)}


def _report(r, combo, y_h, y_m, items, weights=None, people=None):
    """The combiner without the rows, with the margin over the better of the two."""
    r["combo"] = {k: v for k, v in combo.items() if k != "outcome"}
    r["combo"].update(combiner.margin(combo, y_h, y_m, items, weights, repetitions=REPETITIONS, people=people))


# --- ChatBench -----------------------------------------------------------------

_CB = {}


def _cb_init():
    if not _CB:
        _CB["pairs"] = sources.chatbench_pairs(BASE)


def chatbench_one(args):
    folder, name, mode, subject = args
    _cb_init()
    c = _CB["pairs"]
    t = load_machine(folder, "chatbench", name, mode)
    d = c if subject is None else c[c.subject == subject]
    d = d[d.question_id.isin(t.index)].dropna(subset=["acc", "acc_ai"]).reset_index(drop=True)
    p = sources.chatbench_trials(d, t.ok, None)
    p.with_machine = None        # the answer after the chat is with GPT-4o or Llama, not with this machine
    r = compare(p, m=4, repetitions=REPETITIONS)
    r.update(source="chatbench", task=subject or "all", machine=name, mode=mode)
    if "excluded" in r:
        return r
    t_all = t.reindex(d.question_id)
    conf = d.confidence.map({"not-confident": 0.0, "somewhat-confident": 1.0, "very-confident": 2.0}).fillna(1.0).to_numpy()
    op_h = d.selected_answer.map(lambda x: LETTERS.index(x) if x in LETTERS else -1).to_numpy()
    op_m = t_all.top.to_numpy()
    y_h = d.acc.to_numpy(float)
    y_m = t_all.ok.to_numpy(float)
    items = pd.factorize(d.question_id)[0]
    lm = logit(t_all.ptop.to_numpy())
    X = np.column_stack([conf, lm, conf * lm])
    combo = combiner.combine(y_h, y_m, op_h, op_m, X, items)
    levels = list(np.bincount(conf.astype(int), minlength=3) / conf.size)
    r["law"] = law(r["dh"], r["dm"], r["rho"], 4, levels, combo["acc_pair"])
    _report(r, combo, y_h, y_m, items, people=pd.factorize(d.worker_id)[0])
    return r


# --- αNLI -----------------------------------------------------------------------

def alphanli_one(args):
    folder, name, mode = args
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    t = load_machine(folder, "alphanli", name, mode)
    items = items.join(t[["ok", "ptop", "top"]], how="inner")
    h = items.votes_correct / items.votes
    r = compare_votes(h.to_numpy(float), items.votes.to_numpy(float), items.ok.to_numpy(float), m=2, repetitions=REPETITIONS, answer_class=items.correct_option.to_numpy(int))
    r.update(source="alphanli", task="alphanli", machine=name, mode=mode)
    if "excluded" in r:
        return r
    # each vote is a row: the person who votes, the machine with its probability
    n_correct = items.votes_correct.to_numpy(int)
    n_total = items.votes.to_numpy(int)
    correct_option = items.correct_option.to_numpy(int)
    top = items.top.to_numpy(int)
    lm = logit(items.ptop.to_numpy())
    y_h = np.r_[np.ones(n_correct.size), np.zeros(n_correct.size)]
    y_m = np.r_[items.ok.to_numpy(float), items.ok.to_numpy(float)]
    op_h = np.r_[correct_option, 1 - correct_option]
    op_m = np.r_[top, top]
    weights = np.r_[n_correct, n_total - n_correct].astype(float)
    keep = weights > 0
    groups = np.r_[np.arange(n_correct.size), np.arange(n_correct.size)]
    sign = 2.0 * np.r_[top, top] - 1.0            # which option it chose: the machine may lean to one side
    X = np.column_stack([np.r_[lm, lm], sign, np.r_[lm, lm] * sign])
    combo = combiner.combine(y_h[keep], y_m[keep], op_h[keep], op_m[keep], X[keep], groups[keep], weights=weights[keep])
    classes = np.r_[correct_option, correct_option][keep]
    r["law"] = law(r["dh"], r["dm"], r["rho"], 2, None, combo["acc_pair"], d_combo=_d_combo_classes(combo["outcome"], classes, groups[keep], weights[keep]))
    _report(r, combo, y_h[keep], y_m[keep], groups[keep], weights[keep])
    return r


# --- HAIID -----------------------------------------------------------------------

def _haiid_table(task):
    d = pd.read_csv(BASE / "haiid/haiid_dataset.csv", low_memory=False)
    d = d[d.task_name == task].copy()
    d["id"] = task + "-" + d.task_instance_id.astype(str)
    return d


def haiid_one(args):
    folder, task, name, mode = args
    d = _haiid_table(task)
    if name == "advice":
        p = sources.haiid_trials(BASE, task, None)
        r = compare(p, m=2, repetitions=REPETITIONS)
        r.update(source="haiid", task=task, machine="advice", mode="given")
        return r
    t = load_machine(folder, f"haiid_{task}", name, mode)
    p = sources.haiid_trials(BASE, task, t.ok)
    r = compare(p, m=2, repetitions=REPETITIONS)
    r.update(source="haiid", task=task, machine=name, mode=mode)
    if "excluded" in r:
        return r
    d = d[d.id.isin(t.index)]
    t_r = t.reindex(d.id)
    y_h = (d.response_1 > 0).to_numpy(float)
    y_m = t_r.ok.to_numpy(float)
    items = pd.factorize(d.id)[0]
    # the person chooses the correct label if response_1 > 0, the other if < 0; on the machine's two options
    correct_option = t_r.correct_option.to_numpy(int)
    op_h = np.where(y_h > 0, correct_option, 1 - correct_option)
    op_m = t_r.top.to_numpy(int)
    conf = np.abs(d.response_1.to_numpy(float))
    lm = logit(t_r.ptop.to_numpy())
    sign = 2.0 * op_m - 1.0                       # which option it chose: the machine may lean to one side
    X = np.column_stack([conf, lm, conf * lm, sign, lm * sign])
    combo = combiner.combine(y_h, y_m, op_h, op_m, X, items)
    r["law"] = law(r["dh"], r["dm"], r["rho"], 2, None, combo["acc_pair"], d_combo=_d_combo_classes(combo["outcome"], correct_option, items))
    _report(r, combo, y_h, y_m, items, people=pd.factorize(d.participant_id)[0])
    return r


# --- people among themselves, machines among themselves -------------------------------

def machine_vs_machine(folder: Path, task: str, names: list[tuple[str, str]]) -> list:
    tab = {}
    for n, mo in names:
        try:
            tab[(n, mo)] = load_machine(folder, task, n, mo).ok
        except FileNotFoundError:
            continue
    out = []
    listing = list(tab)
    for i in range(len(listing)):
        for j in range(i + 1, len(listing)):
            a, b = tab[listing[i]], tab[listing[j]]
            idx = a.index.intersection(b.index)
            r = sdt.rho_machines(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())
            out.append({"task": task, "a": "/".join(listing[i]), "b": "/".join(listing[j]), "rho_correct": r, "items": len(idx)})
    return out


def _tasks_in_folder(folder: Path):
    names = {}
    for f in sorted(folder.glob("*__*__*.jsonl")):     # sorted: the order of the output must not depend on the file system
        if f.stat().st_size == 0 or f.name.startswith("vision__"):   # interrupted runs; vision has its own analysis
            continue
        task, name, mode = f.name[:-6].split("__")
        names.setdefault(task, set()).add((name, mode))
    return names


def run(folder: Path) -> None:
    names = _tasks_in_folder(folder)
    jobs: list = []
    for name, mode in sorted(names.get("chatbench", [])):
        for s in (None, "math", "physics", "moral"):
            jobs.append((chatbench_one, (folder, name, mode, s)))
    for name, mode in sorted(names.get("alphanli", [])):
        jobs.append((alphanli_one, (folder, name, mode)))
    for k, entries in names.items():
        if not k.startswith("haiid_"):
            continue
        for name, mode in sorted(entries):
            jobs.append((haiid_one, (folder, k[6:], name, mode)))
    for c in ("sarcasm", "census", "art", "cities", "dermatology"):
        jobs.append((haiid_one, (folder, c, "advice", "given")))
    print(len(jobs), "comparisons", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(fn, a) for fn, a in jobs]
        for i, fu in enumerate(futures):
            try:
                results.append(fu.result())
            except Exception as e:  # one that fails does not stop the others; the reason is written
                if isinstance(e, ValueError) and ("sensitivit" in str(e)):   # the machine stays below chance, at the point or in a resample
                    results.append({"excluded": "sensitivity not above chance", "args": str(jobs[i][1])})
                else:
                    results.append({"error": repr(e), "args": str(jobs[i][1])})
                print("ERROR", jobs[i][1], e, flush=True)
            if (i + 1) % 20 == 0:
                print(i + 1, flush=True)
    output = {"comparisons": results}
    output["machine_vs_machine"] = []
    for task, entries in names.items():
        output["machine_vs_machine"] += machine_vs_machine(folder, task, sorted(entries))
    # person with person on αNLI, from the votes
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True)
    h = (items.votes_correct / items.votes).to_numpy()
    n = items.votes.to_numpy(float)
    p = (h * (n * h - 1) / (n - 1)).mean()
    a = h.mean()
    output["person_person_alphanli"] = {"rho": sdt.tetrachoric(float(a), float(a), float(p)), "acc": float(a)}
    Path("results").mkdir(exist_ok=True)
    Path("results/map.json").write_text(json.dumps(output, indent=0))
    print("done", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
