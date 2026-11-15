"""Do two people become more alike with the chat beside them? (PROTOCOL.md, «After the seventh review», A)

Usage, from the experiment folder:

  python3 -m angle.homogenization        # results/homogenization.json

On ChatBench, answer-first condition, each person answers a question alone and then again after talking with the
machine. The angle between two different people on the same question is measured on the answers alone (φ_HH) and
on the answers with the chat (φ_SS): same people, same questions, same pairs, so the comparison is paired. The
bootstrap weighs the questions (Poisson), with the same weights for the two conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from angle import conversations, sdt

RESULTS = Path("results")
REPETITIONS = 2000
SEED = 534
M = 4


# ---- the measures ----------------------------------------------------------------------------------------------

def conversion_table(acc: float, m: int = M) -> tuple[np.ndarray, np.ndarray]:
    """From ρ of the correct answers to ρ of the cues, for two people with the same mean accuracy.

    The same conversion as the article's angle between two people (`run_chatbench.convert_person_person`). It is
    computed once per condition, at the observed mean accuracy, and every replica is read through it: so it was
    done for the interval of the 66 degrees.
    """
    dh = sdt.d_from_accuracy(float(acc), m)
    g = np.linspace(-0.3, 0.95, 26)
    t = np.array([sdt.rho_correct_from_rho_evidence(float(x), dh, dh, m, n=120_000) for x in g])
    return t, g


def to_cues(r_correct: float, table: tuple[np.ndarray, np.ndarray]) -> float:
    t, g = table
    return float(np.interp(r_correct, t, g))


def same_letter(items: np.ndarray, wrong: np.ndarray, letters: np.ndarray, item_weights: np.ndarray) -> tuple[float, float]:
    """Over the pairs of different people on the same question who are both wrong: (pairs with the same letter, pairs).

    Weighted by the question's weight, as the bootstrap does: a question with weight 2 counts twice.
    """
    d = pd.DataFrame({"item": items[wrong], "letter": letters[wrong]})
    if d.empty:
        return 0.0, 0.0
    by = d.groupby(["item", "letter"]).size()
    same = (by * (by - 1) / 2).groupby(level=0).sum()
    n = d.groupby("item").size()
    pairs = n * (n - 1) / 2
    w = item_weights[pairs.index.to_numpy()]
    return float((same.reindex(pairs.index).to_numpy() * w).sum()), float((pairs.to_numpy() * w).sum())


def shared_chat_letter(items, wrong, letters, chat_letters, item_weights) -> dict:
    """Among the pairs both wrong with the same letter: the share where both chats gave that letter, and where at
    least one did. Descriptive (measure 5)."""
    both = one = total = 0.0
    df = pd.DataFrame({"item": items, "wrong": wrong, "letter": letters, "chat": chat_letters})
    for it, g in df[df.wrong].groupby("item"):
        w = item_weights[it]
        rows = g[["letter", "chat"]].to_numpy()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][0] != rows[j][0]:
                    continue
                total += w
                hits = (rows[i][1] == rows[i][0]) + (rows[j][1] == rows[j][0])
                both += w * (hits == 2)
                one += w * (hits >= 1)
    return {"pairs": total, "both_chats_gave_it": both / total if total else None,
            "at_least_one_chat_gave_it": one / total if total else None}


def _angles(p, q, alone, with_chat, w, tables) -> dict:
    r_h = sdt.rho_person_pairs(p, q, alone, item_weights=w)
    r_s = sdt.rho_person_pairs(p, q, with_chat, item_weights=w)
    c_h, c_s = to_cues(r_h, tables[0]), to_cues(r_s, tables[1])
    return {"rho_hh": c_h, "rho_ss": c_s, "phi_hh": sdt.angle_degrees(c_h), "phi_ss": sdt.angle_degrees(c_s)}


def measure(x: pd.DataFrame, repetitions: int = REPETITIONS, seed: int = SEED, subsets: bool = True) -> dict:
    """x: one row per trial, with worker_id, question_id, acc, acc_ai, selected_answer, selected_answer_ai and,
    optionally, letter (the chat gave a letter) and conv_answer_letter."""
    p, _ = pd.factorize(x.worker_id, sort=True)
    q, _ = pd.factorize(x.question_id, sort=True)
    alone, chat = x.acc.to_numpy(float), x.acc_ai.to_numpy(float)
    l_alone = x.selected_answer.astype(str).str.strip().str.upper().to_numpy()
    l_chat = x.selected_answer_ai.astype(str).str.strip().str.upper().to_numpy()
    tables = (conversion_table(alone.mean()), conversion_table(chat.mean()))
    nq = int(q.max()) + 1
    ones = np.ones(nq)

    def letters(w):
        s_h, n_h = same_letter(q, alone == 0, l_alone, w)
        s_s, n_s = same_letter(q, chat == 0, l_chat, w)
        return {"same_alone": s_h / n_h, "same_chat": s_s / n_s, "pairs_alone": n_h, "pairs_chat": n_s}

    point = _angles(p, q, alone, chat, ones, tables)
    point.update(letters(ones))
    gen = np.random.default_rng(seed)
    reps = []
    for _ in range(repetitions):
        w = gen.poisson(1.0, nq).astype(float)
        r = _angles(p, q, alone, chat, w, tables)
        r.update(letters(w))
        reps.append(r)
    reps = pd.DataFrame(reps)
    ci = lambda s: [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]  # noqa: E731
    out = {
        "trials": int(len(x)), "people": int(p.max() + 1), "questions": nq,
        "acc_alone": float(alone.mean()), "acc_chat": float(chat.mean()),
        "phi_hh": point["phi_hh"], "phi_hh_ci": ci(reps.phi_hh),
        "phi_ss": point["phi_ss"], "phi_ss_ci": ci(reps.phi_ss),
        "delta": point["phi_hh"] - point["phi_ss"], "delta_ci": ci(reps.phi_hh - reps.phi_ss),
        "rho_hh": point["rho_hh"], "rho_ss": point["rho_ss"],
        "crowd_ceiling_alone": float(1 / np.sqrt(point["rho_hh"])) if point["rho_hh"] > 0 else None,
        "crowd_ceiling_chat": float(1 / np.sqrt(point["rho_ss"])) if point["rho_ss"] > 0 else None,
        "same_letter_alone": point["same_alone"], "same_letter_alone_ci": ci(reps.same_alone),
        "same_letter_chat": point["same_chat"], "same_letter_chat_ci": ci(reps.same_chat),
        "same_letter_diff": point["same_chat"] - point["same_alone"],
        "same_letter_diff_ci": ci(reps.same_chat - reps.same_alone),
        "pairs_both_wrong_alone": point["pairs_alone"], "pairs_both_wrong_chat": point["pairs_chat"],
    }
    if "conv_answer_letter" in x:
        chat_l = x.conv_answer_letter.where(x.conv_answer_letter.notna(), "").astype(str).str.strip().str.upper().to_numpy()
        out["shared_letter_from_chat"] = shared_chat_letter(q, chat == 0, l_chat, chat_l, ones)
    if subsets and "letter" in x:
        out["subsets"] = {}
        for name, mask in (("chat_gave_a_letter", x.letter.to_numpy(bool)), ("chat_gave_no_letter", ~x.letter.to_numpy(bool))):
            s = measure(x[mask].reset_index(drop=True), repetitions=repetitions // 4, seed=seed, subsets=False)
            out["subsets"][name] = {k: s[k] for k in ("trials", "acc_alone", "acc_chat", "phi_hh", "phi_ss", "delta", "delta_ci")}
    out["passes"] = out["delta_ci"][0] > 0
    out["a3_passes"] = out["same_letter_diff_ci"][0] > 0
    return out


def run() -> dict:
    res = {"protocol": "PROTOCOL.md, «After the seventh review», A", "repetitions": REPETITIONS, "seed": SEED}
    for model in conversations.MODELS:
        x = conversations.trials(model)
        res[model] = measure(x)
        r = res[model]
        print(model, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "subsets"}, flush=True)
    res["A1_gpt-4o"] = res["gpt-4o"]["passes"]
    res["A2_llama"] = res["llama-3.1-8b"]["passes"]
    res["A3"] = {m: res[m]["a3_passes"] for m in conversations.MODELS}
    return res


if __name__ == "__main__":
    out = run()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "homogenization.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("written", RESULTS / "homogenization.json", file=sys.stderr)
