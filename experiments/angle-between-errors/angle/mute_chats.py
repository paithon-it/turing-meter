"""The chats without an answer (PROTOCOL.md, «After the seventh review», B).

Usage, from the experiment folder:

  python3 -m angle.mute_chats sample     # results/mute-chats-sample.jsonl: 50 conversations to read, blind
  python3 -m angle.mute_chats analyse    # results/mute-chats.json, with the readings in results/mute-chats-labels.json

In 330 of the 1,493 GPT-4o trials ChatBench's automatic annotation says no answer came out of the chat. Of these,
the trials with no message from the machine are counted apart; the others are searched automatically for the text
of the options, and a sample of 50 is read with the question, the options and the conversation, without the key
and without the person's answer.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from angle import conversations, sources

BASE = Path("data")
RESULTS = Path("results")
SEED = 534
SAMPLE = 50
SECOND_READER = 10
CATEGORIES = {
    "a": "answer without the letter: a final answer that matches exactly one option, in words or as an equivalent value",
    "b": "answer outside the options: a final answer that matches none of the options",
    "c": "partial help: an explanation or an intermediate step, with no final answer to the question",
    "d": "missing question or clarification: the machine asks for the question, the options or what is meant, "
         "or answers a different question because the message was incomplete",
    "e": "other: off topic, a refusal, a greeting",
}


# ---- matching --------------------------------------------------------------------------------------------------

def normalize(s: str) -> str:
    s = str(s).lower()
    for t in ("\\(", "\\)", "\\[", "\\]", "$"):
        s = s.replace(t, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".").strip()


def named_options(text: str, options: list[str]) -> list[int]:
    """The options whose text appears in `text` as a whole token. Options shorter than two characters are skipped."""
    t = normalize(text)
    out = []
    for i, o in enumerate(options):
        n = normalize(o)
        if len(n) < 2:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", t):
            out.append(i)
    return out


def machine_text(history) -> str:
    return "\n\n".join(str(m.get("content", "")) for m in (history or []) if m.get("role") == "AI" and str(m.get("content", "")).strip())


def wilson(k: float, n: float, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


# ---- the data --------------------------------------------------------------------------------------------------

def mute(model: str = "gpt-4o") -> pd.DataFrame:
    """The trials whose chat, by ChatBench's annotation, gave no answer, with the conversation and the options."""
    x = conversations.trials(model)
    x = x[x.conv_has_answer == False].copy()  # noqa: E712
    conv = sources.chatbench_conversations(BASE)
    conv = conv[conv.model == model].drop_duplicates(["worker_id", "question_id"])
    x = x.merge(conv[["worker_id", "question_id", "chat_history"]], on=["worker_id", "question_id"], how="left")
    q = pd.read_csv(BASE / "chatbench/questions.csv").set_index("question_id")
    x["question"] = x.question_id.map(q.question)
    x["options"] = [[q.loc[i, f"option_{L}"] for L in "ABCD"] for i in x.question_id]
    x["key"] = x.question_id.map(q.answer)
    x["machine_text"] = x.chat_history.map(machine_text)
    x["no_machine_message"] = x.machine_text.str.len() == 0
    x["id"] = x.worker_id + "|" + x.question_id
    return x.sort_values(["worker_id", "question_id"]).reset_index(drop=True)


def drawn(x: pd.DataFrame) -> pd.DataFrame:
    """The sample to read: SAMPLE of the trials with a message from the machine, seed 534."""
    pool = x[~x.no_machine_message].reset_index(drop=True)
    idx = np.random.default_rng(SEED).choice(len(pool), size=min(SAMPLE, len(pool)), replace=False)
    return pool.iloc[idx].reset_index(drop=True)


def write_sample(x: pd.DataFrame, path: Path) -> None:
    """One line per conversation, with what the reader may see: no key, no answer of the person."""
    with path.open("w", encoding="utf-8") as f:
        for _, r in drawn(x).iterrows():
            conv = [{"role": m.get("role"), "content": m.get("content")} for m in (r.chat_history or [])]
            f.write(json.dumps({"id": r.id, "question": r.question, "options": dict(zip("ABCD", r.options)),
                                "conversation": conv}, ensure_ascii=False) + "\n")


# ---- the analysis ----------------------------------------------------------------------------------------------

def automatic(x: pd.DataFrame) -> dict:
    y = x[~x.no_machine_message].copy()
    y["named"] = [named_options(t, o) for t, o in zip(y.machine_text, y.options)]
    y["n_named"] = y.named.map(len)
    one = y[y.n_named == 1].copy()
    one["named_letter"] = one.named.map(lambda v: "ABCD"[v[0]])
    sel = one.selected_answer_ai.astype(str).str.strip().str.upper()
    n = len(y)
    return {
        "examined": n,
        "names_none": int((y.n_named == 0).sum()),
        "names_one": int(len(one)), "names_one_share": len(one) / n if n else None, "names_one_ci": wilson(len(one), n),
        "names_two_or_more": int((y.n_named >= 2).sum()),
        "names_one_is_the_key": int((one.named_letter == one.key).sum()),
        "names_one_person_chose_it": int((sel == one.named_letter).sum()),
        "names_one_acc_with_chat": float(one.acc_ai.mean()) if len(one) else None,
    }


def readings(x: pd.DataFrame, labels: dict) -> dict:
    s = drawn(x)
    first = labels["reader_1"]
    missing = [i for i in s.id if i not in first]
    if missing:
        raise SystemExit(f"{len(missing)} conversations of the sample have no reading")
    s["cat"] = s.id.map(first)
    n = len(s)
    out = {"read": n, "pool": int((~x.no_machine_message).sum()), "categories": {}}
    for c in CATEGORIES:
        k = int((s.cat == c).sum())
        lo, hi = wilson(k, n)
        g = s[s.cat == c]
        out["categories"][c] = {"n": k, "share": k / n, "ci": [lo, hi],
                                "estimated_in_pool": [lo * out["pool"], hi * out["pool"]],
                                "acc_with_chat": float(g.acc_ai.mean()) if k else None}
    second = labels.get("reader_2", {})
    both = [i for i in s.id[:SECOND_READER] if i in second]
    out["second_reader"] = {"read": len(both), "agree": int(sum(first[i] == second[i] for i in both))}
    out["rule"] = "(a) at least 10 of 50: the article qualifies the 330 as partly an annotation artefact"
    out["rule_met"] = out["categories"]["a"]["n"] >= 10
    return out


def run(labels_path: Path) -> dict:
    x = mute()
    res = {"protocol": "PROTOCOL.md, «After the seventh review», B", "trials_without_answer": int(len(x)),
           "no_machine_message": int(x.no_machine_message.sum()),
           "acc_with_chat_no_message": float(x[x.no_machine_message].acc_ai.mean()),
           "acc_with_chat_with_message": float(x[~x.no_machine_message].acc_ai.mean()),
           "automatic": automatic(x)}
    if labels_path.exists():
        res["readings"] = readings(x, json.loads(labels_path.read_text()))
    return res


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    if sys.argv[1:] == ["sample"]:
        x = mute()
        write_sample(x, RESULTS / "mute-chats-sample.jsonl")
        print("trials", len(x), "without a machine message", int(x.no_machine_message.sum()), "sample", SAMPLE, file=sys.stderr)
    elif sys.argv[1:] == ["analyse"]:
        out = run(RESULTS / "mute-chats-labels.json")
        (RESULTS / "mute-chats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        raise SystemExit(__doc__)
