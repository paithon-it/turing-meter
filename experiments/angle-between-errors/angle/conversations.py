"""What happens inside the ChatBench conversations. Exploratory: added on 15 November 2026, after two reviews.

1. Where the points go: the gap between the pair and the machine queried alone, in points of accuracy,
   split into the conversations that produced one of the options (the conversation, then the decision)
   and those that did not. A partition of the trials: it needs no convention for the mute chats.
2. Extraction and filter on the conversations with an answer: Tu = k · e · f with k = dM/dH,
   e = dC/dM, f = dS/dC (from results/chatbench-source.json).
3. Following the chat: on the trials where the chat's letter differs from the person's answer alone,
   how often the final answer takes the chat's letter, when it is right and when it is wrong; the
   difference with two intervals (resampling people; people and questions); and what a linear observer
   with no gating would do (a fixed bonus to the machine's option, calibrated on the observed accuracy).
4. Always and never: among the people with at least two disagreements, the share who always and who
   never followed the chat, against what identical people would give.
5. People split by skill: half of each person's trials, drawn at random, to classify; the other half to
   measure. Median over many draws, overall and within mathematics, because each person had one subject.
6. Zero errors: the bound on k and on the margin at 90 degrees for a model with no error on ChatBench,
   from the rule of three, since resampling cannot move zero errors.

Writes results/conversations.json. Usage, from the experiment folder:  python -m angle.conversations
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import sdt, sources

BASE = Path("data")
SEED = 534
MODELS = ("gpt-4o", "llama-3.1-8b")


def _d(a: float) -> float:
    return sdt.d_from_accuracy(float(min(max(a, 0.2501), 0.9999)), 4)


def trials(model: str) -> pd.DataFrame:
    """The answer-first trials of one arm, with the machine alone, the chat's letter and the key."""
    pairs = sources.chatbench_pairs(BASE)
    machine = sources.chatbench_machine_alone(BASE, model)
    x = pairs[(pairs.model == model) & pairs.question_id.isin(machine.index)].dropna(subset=["acc", "acc_ai"]).copy()
    conv = pd.read_json(BASE / "chatbench/full_study/conversations.json")
    conv = conv[conv.model == model].drop_duplicates(["worker_id", "question_id"])
    x = x.merge(conv[["worker_id", "question_id", "conv_has_answer", "conv_answer_letter"]], on=["worker_id", "question_id"], how="left")
    key = pd.read_csv(BASE / "chatbench/questions.csv").set_index("question_id")["answer"]
    x["m_alone"] = x.question_id.map(machine)
    x["letter"] = (x.conv_has_answer == True) & x.conv_answer_letter.notna()  # noqa: E712
    x["chat_ok"] = (x.conv_answer_letter == x.question_id.map(key)).astype(float)
    return x.reset_index(drop=True)


def where_points_go(x: pd.DataFrame) -> dict:
    a, b = x[x.letter], x[~x.letter]
    sa, sb = len(a) / len(x), len(b) / len(x)
    out = {"trials": int(len(x)), "with_answer": int(len(a)), "without_answer": int(len(b)),
           "machine_alone": float(x.m_alone.mean()), "pair": float(x.acc_ai.mean()),
           "conversation_with_answer": float(sa * (a.chat_ok.mean() - a.m_alone.mean())),
           "decision_with_answer": float(sa * (a.acc_ai.mean() - a.chat_ok.mean())),
           "chats_without_answer": float(sb * (b.acc_ai.mean() - b.m_alone.mean()))}
    out["total"] = out["pair"] - out["machine_alone"]
    parts = out["conversation_with_answer"] + out["decision_with_answer"] + out["chats_without_answer"]
    assert abs(parts - out["total"]) < 1e-9
    return out


def extraction_and_filter(model: str) -> dict:
    d = json.loads(Path("results/chatbench-source.json").read_text())
    alone = next(r for r in d["same_trials"] if r["machine"] == model)
    chat = next(r for r in d["comparisons"] if r["machine"] == model and r["n_trials"] == alone["n_trials"]
                and r.get("subject", "all") == "all" and abs(r["dm"] - alone["dm"]) > 1e-9)
    k, e, f = alone["dm"] / alone["dh"], chat["dm"] / alone["dm"], chat["ds"] / chat["dm"]
    return {"trials": alone["n_trials"], "k": k, "e": e, "f": f, "turing": alone["turing"], "product": k * e * f}


def _disagreements(x: pd.DataFrame) -> pd.DataFrame:
    d = x[x.letter & (x.conv_answer_letter != x.selected_answer)].copy()
    d["followed"] = (d.selected_answer_ai == d.conv_answer_letter).astype(float)
    return d


def _diff(d: pd.DataFrame, w=None) -> float:
    w = np.ones(len(d)) if w is None else w
    r, s = d.chat_ok.to_numpy() == 1, d.followed.to_numpy()
    return float(np.average(s[r], weights=w[r]) - np.average(s[~r], weights=w[~r]))


def linear_observer(dh: float, p_chat: float, acc_pair: float, n: int = 2_000_000, seed: int = 1) -> dict:
    """An observer that adds a fixed bonus to the chat's option: no gating. Bonus calibrated on the pair's accuracy."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4))
    x[:, 0] += dh
    first = x.argmax(1)
    ok = rng.random(n) < p_chat
    chat = np.where(ok, 0, rng.integers(1, 4, n))

    def final(b):
        y = x.copy()
        y[np.arange(n), chat] += b
        return y.argmax(1)

    lo, hi = 0.0, 8.0
    for _ in range(40):
        b = (lo + hi) / 2
        lo, hi = (b, hi) if (final(b) == 0).mean() < acc_pair else (lo, b)
    f = final(b)
    dis = chat != first
    return {"bonus": b, "follow_right": float((f[dis & ok] == 0).mean()), "follow_wrong": float((f[dis & ~ok] == chat[dis & ~ok]).mean())}


def following(x: pd.DataFrame, repetitions: int = 2000) -> dict:
    d = _disagreements(x)
    gen = np.random.default_rng(SEED)
    people, items = pd.factorize(d.worker_id)[0], pd.factorize(d.question_id)[0]
    by_people, by_both = [], []
    for _ in range(repetitions):
        wp = gen.poisson(1.0, people.max() + 1)[people].astype(float)
        wi = gen.poisson(1.0, items.max() + 1)[items].astype(float)
        for w, out in ((wp, by_people), (wp * wi, by_both)):
            r = d.chat_ok.to_numpy() == 1
            if w[r].sum() > 0 and w[~r].sum() > 0:
                out.append(_diff(d, w))
    a = x[x.letter]
    sim = linear_observer(sdt.d_from_accuracy(a.acc.mean(), 4), a.chat_ok.mean(), a.acc_ai.mean())
    return {"disagreements": int(len(d)), "chat_right": int((d.chat_ok == 1).sum()), "chat_wrong": int((d.chat_ok == 0).sum()),
            "follow_right": float(d[d.chat_ok == 1].followed.mean()), "follow_wrong": float(d[d.chat_ok == 0].followed.mean()),
            "difference": _diff(d), "ci_people": [float(np.percentile(by_people, 2.5)), float(np.percentile(by_people, 97.5))],
            "ci_people_and_questions": [float(np.percentile(by_both, 2.5)), float(np.percentile(by_both, 97.5))],
            "linear_observer_no_gating": sim}


def always_never(x: pd.DataFrame, repetitions: int = 3000) -> dict:
    d = _disagreements(x)
    count = d.groupby("worker_id").size()
    d = d[d.worker_id.isin(count[count >= 2].index)]
    obs = d.groupby("worker_id").followed.mean()
    rate = d.groupby("chat_ok").followed.mean()
    p, w = d.chat_ok.map(rate).to_numpy(), d.worker_id.to_numpy()
    gen = np.random.default_rng(SEED)
    sa, sn = [], []
    for _ in range(repetitions):
        g = pd.Series((gen.random(len(d)) < p).astype(float)).groupby(w).mean()
        sa.append((g == 1).mean())
        sn.append((g == 0).mean())
    return {"people": int(len(obs)), "always": float((obs == 1).mean()), "never": float((obs == 0).mean()),
            "identical_people_always": [float(np.mean(sa)), float(np.percentile(sa, 2.5)), float(np.percentile(sa, 97.5))],
            "identical_people_never": [float(np.mean(sn)), float(np.percentile(sn, 2.5)), float(np.percentile(sn, 97.5))]}


def by_skill(x: pd.DataFrame, draws: int = 300) -> dict:
    out = {}
    for scope, y in (("all", x), ("math", x[x.subject == "math"])):
        rows = []
        for seed in range(draws):
            gen = np.random.default_rng(seed)
            half = np.zeros(len(y), bool)
            for _, idx in y.groupby("worker_id").indices.items():
                idx = gen.permutation(idx)
                half[idx[: len(idx) // 2]] = True
            cl = y[half].groupby("worker_id").acc.mean()
            groups = {"less_skilled": cl[cl <= cl.quantile(1 / 3)].index, "more_skilled": cl[cl >= cl.quantile(2 / 3)].index}
            z = y[~half]
            for name, who in groups.items():
                g = z[z.worker_id.isin(who)]
                dh, dm, ds = _d(g.acc.mean()), _d(g.m_alone.mean()), _d(g.acc_ai.mean())
                rows.append({"group": name, "people": len(who), "acc_h": g.acc.mean(), "k": dm / dh, "turing": ds / dh, "yield": ds / dm,
                             "moral_share": float((g.drop_duplicates("worker_id").subject == "moral").mean())})
        r = pd.DataFrame(rows)
        out[scope] = {name: {c: [float(g[c].median()), float(g[c].quantile(0.05)), float(g[c].quantile(0.95))]
                             for c in ("people", "acc_h", "k", "turing", "yield", "moral_share")}
                      for name, g in r.groupby("group")}
        for name in out[scope]:
            out[scope][name]["useful_angle"] = math.degrees(math.acos(min(1.0, 1 / out[scope][name]["k"][0]))) if out[scope][name]["k"][0] > 1 else math.degrees(math.acos(out[scope][name]["k"][0]))
    return out


def zero_errors(items: int = 396) -> dict:
    """A machine with no error on ChatBench: the upper bound of its error rate is about 3/n (rule of three)."""
    f = json.loads(Path("results/frontier.json").read_text())
    r = next(c for c in f["comparisons"] if c["source"] == "chatbench" and c["task"] == "all" and c["machine"] == "fable-5.2")
    dm_low = _d(1 - 3 / items)
    dh_high = r["dm"] / r["k_ci"][0]                   # the interval of k there comes only from the people
    out = {"dm_at_rule_of_three": dm_low, "dh": r["dh"], "dh_high": dh_high}
    for name, dh in (("k_low_point_dh", r["dh"]), ("k_low_high_dh", dh_high)):
        k = dm_low / dh
        out[name] = k
        out[name.replace("k_low", "margin90")] = math.sqrt(1 + k * k) / k - 1
    return out


def run() -> dict:
    out = {"zero_errors_fable": zero_errors()}
    for model in MODELS:
        x = trials(model)
        out[model] = {"where_points_go": where_points_go(x), "extraction_and_filter": extraction_and_filter(model),
                      "following": following(x), "always_never": always_never(x), "by_skill": by_skill(x)}
    Path("results").mkdir(exist_ok=True)
    Path("results/conversations.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
