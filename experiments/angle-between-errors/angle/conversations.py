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
7. (added later the same day, after a third review) Group against person: the turing of the group
   against the ratio of the average individual sensitivities, from the spread of the people's accuracies
   net of binomial noise (Macmillan and Kaplan, 1985), alone and with the machine.
8. The chat when the person started wrong: on the conversations with an answer, how much the chat beats
   the machine queried alone when the person's answer alone was right and when it was wrong; and how often
   a wrong chat gives exactly the person's wrong answer, against the trap of the question (the wrong
   answers the other people chose on their own).
9. A hundred questions: how many the people get wrong, how many the machine, and how many both, against
   what independent errors would give.
10. A fourth factor in sensitivity over all trials, counting the mute chats as guesses: the contact factor
    and why it was not used (it depends on the convention for the mute chats).
11. (added later the same day, after a further review) Agreement and disagreement: how often the chat is wrong
    when it agrees with the person's answer alone and when it does not, who is right in a disagreement, by
    subject and by the confidence the person stated; and, in the model, the weights of the ideal pair, the
    "more confident one wins" rule (equal weights) and an observer that weighs the chat's verdict as it deserves.

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
    conv = sources.chatbench_conversations(BASE)
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


def _spread(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Mean and between-person variance of the true accuracies, net of the binomial noise (method of moments)."""
    a = k / n
    mu = float(k.sum() / n.sum())
    var = float(np.average((a - mu) ** 2, weights=n) - np.average(mu * (1 - mu) / n, weights=n))
    return mu, max(var, 0.0)


_GRID = np.linspace(0.0005, 0.9995, 2001)
_D_GRID: np.ndarray | None = None


def _mean_d(mu: float, var: float) -> float:
    """E[d'(a)] for accuracies with a Beta distribution of that mean and variance; d' = 0 at or below chance."""
    global _D_GRID
    from scipy.stats import beta as beta_dist
    if var <= 0:
        return _d(mu)
    common = mu * (1 - mu) / var - 1
    if common <= 0:
        return float("nan")
    if _D_GRID is None:                       # the inversion of the m-AFC formula is slow: done once
        _D_GRID = np.array([_d(x) if x > 0.2501 else 0.0 for x in _GRID])
    w = beta_dist.pdf(_GRID, mu * common, (1 - mu) * common)
    return float(np.sum(w * _D_GRID) / np.sum(w))


def group_against_person(x: pd.DataFrame, repetitions: int = 500) -> dict:
    g = x.groupby("worker_id").agg(n=("acc", "size"), h=("acc", "sum"), s=("acc_ai", "sum"))
    def one(g):
        mh, vh = _spread(g.h.to_numpy(float), g.n.to_numpy(float))
        ms, vs = _spread(g.s.to_numpy(float), g.n.to_numpy(float))
        pooled = _d(ms) / _d(mh)
        individual = _mean_d(ms, vs) / _mean_d(mh, vh)
        return {"acc_alone": mh, "sd_alone": math.sqrt(vh), "acc_with": ms, "sd_with": math.sqrt(vs),
                "turing_group": pooled, "turing_mean_of_people": individual, "ratio": individual / pooled}
    out = one(g)
    gen = np.random.default_rng(SEED)
    runs = [one(g.iloc[gen.integers(0, len(g), len(g))]) for _ in range(repetitions)]
    for key in ("turing_group", "turing_mean_of_people", "ratio", "sd_alone", "sd_with"):
        v = np.array([r[key] for r in runs if not math.isnan(r[key])])
        out[key + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def started_wrong(x: pd.DataFrame, repetitions: int = 2000) -> dict:
    a = x[x.letter].copy()
    a["lift"] = a.chat_ok - a.m_alone
    def lifts(z):
        r, w = z[z.acc == 1], z[z.acc == 0]
        return float(r.lift.mean()), float(w.lift.mean())
    lr, lw = lifts(a)
    people = a.worker_id.unique()
    idx = {p: np.flatnonzero(a.worker_id.to_numpy() == p) for p in people}
    gen = np.random.default_rng(SEED)
    diffs = []
    for _ in range(repetitions):
        rows = np.concatenate([idx[p] for p in gen.choice(people, people.size)])
        r, w = lifts(a.iloc[rows])
        diffs.append(r - w)
    wrong = a[(a.acc == 0) & (a.chat_ok == 0)]
    same = float((wrong.conv_answer_letter == wrong.selected_answer).mean())
    trap = _the_trap_of_the_question(wrong, repetitions)
    return {"trials_person_right": int((a.acc == 1).sum()), "trials_person_wrong": int((a.acc == 0).sum()),
            "chat_person_right": float(a[a.acc == 1].chat_ok.mean()), "alone_person_right": float(a[a.acc == 1].m_alone.mean()),
            "chat_person_wrong": float(a[a.acc == 0].chat_ok.mean()), "alone_person_wrong": float(a[a.acc == 0].m_alone.mean()),
            "lift_person_right": lr, "lift_person_wrong": lw, "difference": lr - lw,
            "difference_ci_people": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "wrong_chats_after_wrong_person": int(len(wrong)), "same_wrong_answer_as_person": same,
            "same_wrong_answer_by_chance": 1 / 3, "trap": trap}


def _the_trap_of_the_question(wrong: pd.DataFrame, repetitions: int) -> dict:
    """The reference for «the chat repeats the person's wrong answer» is not one in three: some wrong options
    attract everybody. For each wrong chat after a wrong person, the share of the other people who got that
    question wrong on their own and chose the chat's letter; the excess over it is what depends on the person.
    Intervals with Poisson weights over people, and over people and questions (deviation of 15 November)."""
    pairs = sources.chatbench_pairs(BASE)
    others = pairs[pairs.acc == 0][["worker_id", "question_id", "selected_answer"]]
    rows = []
    for r in wrong[wrong.conv_answer_letter.isin(list("ABCD"))].itertuples():
        o = others[(others.question_id == r.question_id) & (others.worker_id != r.worker_id)]
        if len(o):
            rows.append((r.worker_id, r.question_id, float(r.conv_answer_letter == r.selected_answer),
                         float((o.selected_answer == r.conv_answer_letter).mean())))
    d = pd.DataFrame(rows, columns=["w", "q", "observed", "reference"])
    gap = (d.observed - d.reference).to_numpy()
    gen = np.random.default_rng(SEED)
    people, questions = d.w.unique(), d.q.unique()
    by_people, by_both = [], []
    for _ in range(repetitions):
        wp = d.w.map(pd.Series(gen.poisson(1.0, people.size), index=people)).to_numpy(float)
        wq = d.q.map(pd.Series(gen.poisson(1.0, questions.size), index=questions)).to_numpy(float)
        if wp.sum() > 0:
            by_people.append(np.average(gap, weights=wp))
        if (wp * wq).sum() > 0:
            by_both.append(np.average(gap, weights=wp * wq))
    return {"cases": int(len(d)), "observed": float(d.observed.mean()), "reference": float(d.reference.mean()),
            "excess": float(gap.mean()), "excess_ci_people": [float(np.percentile(by_people, 2.5)), float(np.percentile(by_people, 97.5))],
            "excess_ci_people_and_questions": [float(np.percentile(by_both, 2.5)), float(np.percentile(by_both, 97.5))]}


def hundred_questions(x: pd.DataFrame) -> dict:
    wh, wm = 1 - x.acc.mean(), 1 - x.m_alone.mean()
    both = float(((1 - x.acc) * (1 - x.m_alone)).mean())
    return {"person_wrong": float(wh), "machine_wrong": float(wm), "both_wrong": both, "both_if_independent": float(wh * wm)}


def contact_factor(x: pd.DataFrame) -> dict:
    a = x[x.letter]
    acc_itt = float(np.where(x.letter, x.chat_ok, 0.25).mean())
    acc_own = float(np.where(x.letter, x.chat_ok, x.acc).mean())
    dc_ans, dm = _d(a.chat_ok.mean()), _d(x.m_alone.mean())
    out = {}
    for name, acc in (("mute_as_guess", acc_itt), ("mute_as_own_answer", acc_own)):
        dc = _d(acc)
        out[name] = {"contact": dc / dc_ans, "extraction": dc_ans / dm, "filter": _d(x.acc_ai.mean()) / dc}
    return out


def weight_ratio(k: float, phi_degrees: float) -> float:
    """How many times the machine's cue weighs the person's in the ideal pair (w ∝ Σ⁻¹d), each cue in units of
    its own noise: (k − cos φ)/(1 − k cos φ). k at 90 degrees; it diverges at the useful angle, below which the
    free optimum would give the person a negative weight and the article stops at the stronger."""
    c = math.cos(math.radians(phi_degrees))
    return (k - c) / (1 - k * c)


def equal_weights(dh: float, dm: float, rho: float) -> dict:
    """«The more confident one wins» amounts to equal weights on the two cues: (dH + dM)/sqrt(2(1 + ρ)). It beats
    the stronger only if k < 1/(sqrt(2(1 + ρ)) − 1), 2.41 at 90 degrees (Bahrami et al., 2010)."""
    d = (dh + dm) / math.sqrt(2 * (1 + rho))
    return {"d": d, "over_machine": d / max(dh, dm), "k_threshold": 1 / (math.sqrt(2 * (1 + rho)) - 1)}


def weighing_the_verdict(dh: float, p_chat: float, n: int = 2_000_000, seed: int = 1) -> dict:
    """An observer with the person's graded cue and only the chat's verdict, which adds to the chat's option the
    bonus the likelihood ratio gives, ln(p/((1 − p)/3))/dH: how often it follows the chat in a disagreement.
    Errors of person and chat independent; it is a model, not a measurement."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4))
    x[:, 0] += dh
    first = x.argmax(1)
    ok = rng.random(n) < p_chat
    chat = np.where(ok, 0, rng.integers(1, 4, n))
    b = math.log(p_chat / ((1 - p_chat) / 3)) / dh
    y = x.copy()
    y[np.arange(n), chat] += b
    f = y.argmax(1)
    dis = chat != first
    return {"bonus": b, "follows_in_disagreements": float((f[dis] == chat[dis]).mean()), "accuracy": float((f == 0).mean())}


def _outcomes(z: pd.DataFrame) -> dict:
    followed = (z.selected_answer_ai == z.conv_answer_letter)
    return {"disagreements": int(len(z)), "chat_right": float(z.chat_ok.mean()), "person_right": float(z.acc.mean()),
            "both_wrong": float(((z.chat_ok == 0) & (z.acc == 0)).mean()), "person_follows_chat": float(followed.mean()),
            "kept_own": float((z.selected_answer_ai == z.selected_answer).mean()), "pair_right": float(z.acc_ai.mean())}


def agreement_and_disagreement(x: pd.DataFrame, model: str, repetitions: int = 2000) -> dict:
    """On the chats with a letter: a disagreement is a chat letter different from the person's answer alone."""
    a = x[x.letter].reset_index(drop=True)
    dis = (a.conv_answer_letter != a.selected_answer).to_numpy()
    chat_ok, pair = a.chat_ok.to_numpy(float), a.acc_ai.to_numpy(float)

    def gaps(w):
        err_agree = 1 - np.average(chat_ok[~dis], weights=w[~dis])
        err_dis = 1 - np.average(chat_ok[dis], weights=w[dis])
        return err_dis - err_agree, np.average(pair[dis], weights=w[dis]) - np.average(chat_ok[dis], weights=w[dis])

    gen = np.random.default_rng(SEED)
    people, items = pd.factorize(a.worker_id)[0], pd.factorize(a.question_id)[0]
    runs = {"people": [], "people_and_questions": []}
    for _ in range(repetitions):
        wp = gen.poisson(1.0, people.max() + 1)[people].astype(float)
        wi = gen.poisson(1.0, items.max() + 1)[items].astype(float)
        for key, w in (("people", wp), ("people_and_questions", wp * wi)):
            if w[dis].sum() > 0 and w[~dis].sum() > 0:
                runs[key].append(gaps(w))
    ci = lambda key, i: [float(np.percentile([r[i] for r in runs[key]], 2.5)), float(np.percentile([r[i] for r in runs[key]], 97.5))]
    d, agree = a[dis], a[~dis]
    wrong_agree = agree[agree.acc == 0]
    out = {"chats_with_letter": int(len(a)), "agreements": int((~dis).sum()), "disagreements": int(dis.sum()),
           "chat_error_agreement": float(1 - chat_ok[~dis].mean()), "chat_error_disagreement": float(1 - chat_ok[dis].mean()),
           "chat_error_gap": gaps(np.ones(len(a)))[0], "chat_error_gap_ci_people": ci("people", 0),
           "chat_error_gap_ci_people_and_questions": ci("people_and_questions", 0),
           "chat_errors": int((chat_ok == 0).sum()), "chat_errors_in_disagreement": int(((chat_ok == 0) & dis).sum()),
           "pair_errors": int((pair == 0).sum()), "pair_errors_in_disagreement": int(((pair == 0) & dis).sum()),
           "agreements_on_a_wrong_answer": int(len(wrong_agree)), "pair_right_there": int(wrong_agree.acc_ai.sum()),
           "always_follow_minus_pair": -gaps(np.ones(len(a)))[1],
           "pair_minus_always_follow_ci_people": ci("people", 1), "pair_minus_always_follow_ci_people_and_questions": ci("people_and_questions", 1),
           "outcomes": {"all": _outcomes(d), **{subj: _outcomes(d[d.subject == subj]) for subj in ("math", "physics", "moral")}}}
    conf = {}
    for subj, z in (("all", d), ("math", d[d.subject == "math"])):
        conf[subj] = {lvl: {"n": int((z.confidence == lvl).sum()), "person_right": int(z[z.confidence == lvl].acc.sum()),
                            "chat_right": int(z[z.confidence == lvl].chat_ok.sum()), "moral": int(((z.confidence == lvl) & (z.subject == "moral")).sum())}
                      for lvl in ("not-confident", "somewhat-confident", "very-confident")}
    zm = d[d.subject == "math"].reset_index(drop=True)
    pm = pd.factorize(zm.worker_id)[0]
    h, lv = zm.acc.to_numpy(float), zm.confidence.to_numpy()
    def vc_nc(w):
        v, nc = lv == "very-confident", lv == "not-confident"
        return np.average(h[v], weights=w[v]) - np.average(h[nc], weights=w[nc]) if w[v].sum() > 0 and w[nc].sum() > 0 else np.nan
    g2 = np.random.default_rng(SEED)
    vv = np.array([vc_nc(g2.poisson(1.0, pm.max() + 1)[pm].astype(float)) for _ in range(repetitions)])
    vv = vv[~np.isnan(vv)]
    conf["math_very_minus_not_confident_person_right"] = vc_nc(np.ones(len(zm)))
    conf["math_very_minus_not_confident_ci_people"] = [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))]
    out["confidence_in_disagreements"] = conf
    src = json.loads(Path("results/chatbench-source.json").read_text())
    r = next(c for c in src["comparisons"] if c["machine"] == model and c["subject"] == "all" and c["kind"] == "alone")
    out["model"] = {"k": r["k"], "phi": r["phi"], "phi_ci": r["phi_ci"],
                    "weight_ratio_at_phi": weight_ratio(r["k"], r["phi"]),
                    "weight_ratio_at_ci": [weight_ratio(r["k"], p) if math.cos(math.radians(p)) < 1 / r["k"] else None for p in r["phi_ci"]],
                    "useful_angle": math.degrees(math.acos(min(1.0, 1 / r["k"]))) if r["k"] > 1 else None,
                    "equal_weights": equal_weights(r["dh"], r["dm"], r["rho"]),
                    "weighing_the_verdict": weighing_the_verdict(sdt.d_from_accuracy(a.acc.mean(), 4), a.chat_ok.mean())}
    return out


def branches(x: pd.DataFrame) -> dict:
    """The tree of the chat figure: a hundred trials split by what comes out of the chat, with the correct answers of
    the machine alone and of the pair in each branch, per hundred trials (the article's figure reads them from here)."""
    n = len(x)
    per_hundred = lambda s: float(100.0 * s / n)
    none = x[~x.letter]
    same = x[x.letter & (x.conv_answer_letter == x.selected_answer)]
    other = x[x.letter & (x.conv_answer_letter != x.selected_answer)]
    out = {"trials": int(n)}
    for name, b in (("no_letter", none), ("same_letter", same), ("other_letter", other)):
        out[name] = {"trials": int(len(b)), "share": per_hundred(len(b)), "machine": per_hundred(b.m_alone.sum()), "pair": per_hundred(b.acc_ai.sum())}
    out["other_letter"].update(chat_right=per_hundred(other.chat_ok.sum()), chat_wrong=per_hundred((1 - other.chat_ok).sum()))
    f = following(x, repetitions=1)
    out.update(follow_right=f["follow_right"], follow_wrong=f["follow_wrong"],
               machine=per_hundred(x.m_alone.sum()), pair=per_hundred(x.acc_ai.sum()))
    parts = sum(out[b]["share"] for b in ("no_letter", "same_letter", "other_letter"))
    assert abs(parts - 100.0) < 1e-9 and abs(sum(out[b]["machine"] for b in ("no_letter", "same_letter", "other_letter")) - out["machine"]) < 1e-9
    return out


def run() -> dict:
    out = {"zero_errors_fable": zero_errors()}
    for model in MODELS:
        x = trials(model)
        out[model] = {"where_points_go": where_points_go(x), "extraction_and_filter": extraction_and_filter(model),
                      "following": following(x), "always_never": always_never(x), "by_skill": by_skill(x),
                      "group_against_person": group_against_person(x), "started_wrong": started_wrong(x),
                      "hundred_questions": hundred_questions(x), "contact_factor": contact_factor(x),
                      "agreement_and_disagreement": agreement_and_disagreement(x, model)}
    Path("results").mkdir(exist_ok=True)
    Path("results/conversations.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["branches"]:          # only the tree of the chat figure → results/chat-branches.json
        Path("results/chat-branches.json").write_text(json.dumps({"gpt-4o": branches(trials("gpt-4o"))}, indent=1))
        print(Path("results/chat-branches.json").read_text())
    else:
        print(json.dumps(run(), indent=1))
