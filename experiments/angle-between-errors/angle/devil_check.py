"""A9: the devil's advocate, checked. Pre-registered in PROTOCOL.md, "After the critical review (v4): the new measures".

(i) devil.bootstrap resamples people and questions; here the same with the weights on the items (question ×
answer given alone) instead of the questions, and how much the intervals change. (ii) The distinct calls per
condition, counted from the answer files. (iii) The two stacks: Llama 3.1 8B in FP8, letter only, against the
Llama of ChatBench on the same questions (its zero-shot letter-only and free-text accuracies).

Usage, from the experiment folder:  python -m angle.devil_check   → results/devil-check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import devil

KEYS = ("acc_alone", "acc_told", "acc_devil", "h1_told_minus_alone_person_wrong", "h2_devil_minus_told_person_wrong",
        "h3_yield_devil", "yield_told")


def _boot(d: pd.DataFrame, level: str) -> dict:
    p, _ = pd.factorize(d.worker_id, sort=True)
    q, _ = pd.factorize(d[level], sort=True)
    gen = np.random.default_rng(devil.SEED)
    runs = [devil.measures(d, gen.poisson(1.0, p.max() + 1)[p] * gen.poisson(1.0, q.max() + 1)[q].astype(float))
            for _ in range(devil.REPETITIONS)]
    out = {}
    for k in KEYS:
        s = np.array([r[k] for r in runs if r.get(k) is not None], float)
        out[k] = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    return out


def run() -> dict:
    solo = devil.solo_answers()
    bank = [json.loads(l) for l in open(devil.BASE / "items/chatbench.jsonl", encoding="utf-8")]
    correct = {c["id"]: c["correct_option"] for c in bank}
    out = {"bootstrap": {}, "calls": {}, "stacks": {}}
    for name, files in devil.MACHINES.items():
        if not all(f.exists() for f in files):
            continue
        d = devil.trials(solo, correct, *(devil.verdicts(f) for f in files))
        point = devil.measures(d, np.ones(len(d)))
        by_q, by_item = _boot(d, "question_id"), _boot(d, "item")
        out["bootstrap"][name] = {"trials": int(len(d)), "questions": int(d.question_id.nunique()), "items": int(d["item"].nunique()),
                                  **{k: {"point": point[k], "people_and_questions": by_q[k], "people_and_items": by_item[k],
                                         "width_ratio": (by_item[k][1] - by_item[k][0]) / (by_q[k][1] - by_q[k][0])} for k in KEYS}}
    for f in sorted(Path("devil-results").glob("*.jsonl")):
        ids = [json.loads(l)["id"] for l in open(f, encoding="utf-8")]
        out["calls"][f.name] = {"answers": len(ids), "distinct_items": len(set(ids))}
    # (iii) the two Llamas on the same questions
    llama = devil.verdicts(Path("devil-results/chatbench__llama-3.1-8b__alone.jsonl"))
    keys = {c["id"]: c["correct_option"] for c in bank}
    qs = sorted(set(llama) & set(keys))
    cb = pd.read_csv(devil.BASE / "chatbench/ai_alone_answers.csv")
    cb = cb[cb.model == "llama-3.1-8b"].set_index("question_id")
    common = [q for q in qs if q in cb.index]
    out["stacks"] = {
        "questions": len(common),
        "llama_fp8_letter_only": float(np.mean([llama[q] == keys[q] for q in common])),
        "chatbench_llama_letter_only_zero_shot": float(cb.loc[common, "letter_only_zero_shot_acc"].mean()),
        "chatbench_llama_free_text": float(cb.loc[common, "free_text_acc"].mean()),
        "on_the_2903_trials": {"llama_fp8_alone": out["bootstrap"].get("llama-3.1-8b", {}).get("acc_alone", {}).get("point")},
    }
    return out


def main() -> None:
    out = run()
    Path("results/devil-check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for name, v in out["bootstrap"].items():
        print(name, v["trials"], "trials,", v["questions"], "questions,", v["items"], "items;",
              {k: round(v[k]["width_ratio"], 2) for k in KEYS})
    print("calls", {k: v["distinct_items"] for k, v in out["calls"].items()})
    print("stacks", out["stacks"])


if __name__ == "__main__":
    main()
