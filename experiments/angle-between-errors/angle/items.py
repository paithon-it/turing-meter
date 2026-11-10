"""Builds the item banks: the same items the people saw, one per line in JSONL.

Usage:  python3 -I angle/items.py data

Each item: id, task, text, options, correct_option (index), image (relative path or null).
The order of the options does not depend on the correct answer (alphabetical or the source's).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def chatbench(base: Path):
    rows = list(csv.DictReader(open(base / "chatbench/questions.csv", encoding="utf-8")))
    for r in rows:
        if "-redux-" not in r["question_id"]:           # drops the attention check and the example
            continue
        yield {"id": r["question_id"], "task": "chatbench", "text": r["question"],
               "options": [r["option_A"], r["option_B"], r["option_C"], r["option_D"]],
               "correct_option": "ABCD".index(r["answer"]), "group": r["dataset"]}


def alphanli(base: Path):
    for r in map(json.loads, open(base / "chaos/alphanli.jsonl", encoding="utf-8")):
        e = r["example"]
        text = (f"Beginning: {e['obs1']}\nEnding: {e['obs2']}\n\n"
                "Which of the two hypotheses better explains what happened between the beginning and the ending?")
        yield {"id": r["uid"], "task": "alphanli", "text": text, "options": [e["hyp1"], e["hyp2"]],
               "correct_option": int(r["old_label"]) - 1, "group": "alphanli",
               "votes_correct": r["label_count"][int(r["old_label"]) - 1], "votes": sum(r["label_count"])}


QUESTIONS = {
    "sarcasm": "Is the following Reddit comment sarcastic?\n\nComment: {x}",
    "census": ("Based on this 1990s US census record, is the person's yearly income above or below 50,000 dollars?\n\n{x}"),
    "art": "This is a painting. Which art period does it belong to?",
    "cities": "This street-level photo was taken in one of two US cities. Which one?",
    "dermatology": "This is a dermatology photograph of a skin lesion. Is it benign or malignant?",
}
LABELS = {"no_sarcasm": "not sarcastic", "sarcasm": "sarcastic", "<50k": "below 50,000 dollars",
          ">=50k": "above 50,000 dollars", "benign": "benign", "malignant": "malignant"}


def haiid(base: Path):
    seen = set()
    for r in csv.DictReader(open(base / "haiid/haiid_dataset.csv", encoding="utf-8")):
        key = (r["task_name"], r["task_instance_id"], r["correct_label"], r["incorrect_label"])
        if key in seen:
            continue
        seen.add(key)
        t = r["task_name"]
        options = sorted([r["correct_label"], r["incorrect_label"]])
        correct_option = options.index(r["correct_label"])
        f = base / "haiid/tasks" / r["path_to_task"]
        if not f.exists():                       # the CSV writes SanFrancisco, the folder sanfrancisco
            f = base / "haiid/tasks" / Path(*[p.lower() if i == 1 else p for i, p in enumerate(Path(r["path_to_task"]).parts)])
        image = None
        if t == "sarcasm":
            text = QUESTIONS[t].format(x=f.read_text(encoding="utf-8").strip())
        elif t == "census":
            fields = json.loads(f.read_text(encoding="utf-8"))
            text = QUESTIONS[t].format(x="\n".join(f"{k}: {v}" for k, v in fields.items()))
        else:
            text, image = QUESTIONS[t], str(f.relative_to(base))
        yield {"id": f"{t}-{r['task_instance_id']}", "task": f"haiid_{t}", "text": text,
               "options": [LABELS.get(o, o) for o in options], "correct_option": correct_option, "group": t, "image": image}


if __name__ == "__main__":
    base = Path(sys.argv[1]).resolve()
    (base / "items").mkdir(exist_ok=True)
    per_task: dict[str, list] = {}
    for source in (chatbench, alphanli, haiid):
        for c in source(base):
            per_task.setdefault(c["task"], []).append(c)
    for name, items in per_task.items():
        seen = set()
        unique = [c for c in items if not (c["id"] in seen or seen.add(c["id"]))]
        with open(base / "items" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in unique:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(name, len(unique))
