"""Four robustness checks requested by a review (2026-11-10). Exploratory: decided after seeing the data.

1. αNLI without the contested keys: φ of each machine and its rank correlation with dM, (a) dropping the items
   where the majority of the 100 votes contradicts the key, (b) using the majority as the key.
2. The useful angle with free weights: the full-evidence combiner of `full_evidence`, on the same data and with the
   same split, without the constraint a ≥ 0, b ≥ 0.
3. The ChatBench turing on the trials where the chat did not record an answer.
4. The uncertainty on the trends: exact permutation p and bootstrap interval over the machines of the rank
   correlations between dM and φ.

It changes no existing function and rewrites no result: it reads `results/*.json` and the data, and writes
`results/review.json`.

Usage, from the experiment folder:  python -m angle.review GPU_RESULTS_DIR [--checks 1,2,3,4]
With --checks only those checks are redone; the others stay as they are in the file.
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest, rankdata, spearmanr

from . import full_evidence as fe
from . import sdt, sources
from .measure import _d, compare, compare_votes

R = Path("results")
BASE = Path("data")
OUTPUT = R / "review.json"
REPETITIONS = 2000              # as analysis.py and run_chatbench.py
MACHINE_REPETITIONS = 10_000    # bootstrap over the machines (ten points: cheap)
SEED = 534
MODES = ("letter", "reason")


# --- 4. rank correlations: exact p, bootstrap over the machines --------------------------------------

@lru_cache(maxsize=4)
def _all_permutations(n: int) -> np.ndarray:
    """The n! permutations of range(n), one per row (n ≤ 10: 3.6 million rows of one byte)."""
    return np.fromiter((i for p in permutations(range(n)) for i in p), dtype=np.int8,
                       count=math.factorial(n) * n).reshape(-1, n)


def spearman_exact(x, y, n_max_exact: int = 10, monte_carlo: int = 200_000, seed: int = SEED) -> dict:
    """Spearman ρ with the permutation p: exact up to `n_max_exact` points, otherwise Monte Carlo.

    The ranks of y are permuted against those of x (average ranks if there are ties, as scipy). The two-sided p
    counts the permutations with |ρ| ≥ |observed ρ|; the two one-sided ones, those with ρ ≤ and ≥ the observed.
    """
    rx = rankdata(np.asarray(x, float))
    ry = rankdata(np.asarray(y, float))
    n = rx.size
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if den == 0:
        return {"rho": float("nan"), "n": int(n)}
    obs = float(rx @ ry) / den
    tol = 1e-12
    if n <= n_max_exact:
        perm = _all_permutations(n)
        method = "exact"
    else:
        gen = np.random.default_rng(seed)
        perm = np.argsort(gen.random((monte_carlo, n)), axis=1)
        method = "monte carlo"
    le = ge = ab = 0
    for i in range(0, perm.shape[0], 400_000):
        s = ry[perm[i:i + 400_000]] @ rx / den
        le += int(np.sum(s <= obs + tol))
        ge += int(np.sum(s >= obs - tol))
        ab += int(np.sum(np.abs(s) >= abs(obs) - tol))
    tot = perm.shape[0]
    return {"rho": obs, "n": int(n), "p_two_sided": ab / tot, "p_negative": le / tot, "p_positive": ge / tot,
            "permutations": int(tot), "method": method}


def spearman_bootstrap(x, y, repetitions: int = MACHINE_REPETITIONS, seed: int = SEED) -> dict:
    """Percentile interval of Spearman ρ resampling the machines (the pairs x, y) with replacement."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gen = np.random.default_rng(seed)
    out = []
    for _ in range(repetitions):
        i = gen.integers(0, x.size, x.size)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:      # a constant column: ρ is not defined
            continue
        out.append(spearmanr(x[i], y[i]).statistic)
    out = np.asarray(out, float)
    return {"ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "share_below_zero": float(np.mean(out < 0)), "valid_repetitions": int(out.size)}


def leave_one_out(names, x, y) -> dict:
    """Spearman ρ removing one machine at a time."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rows = {}
    for i, name in enumerate(names):
        t = np.ones(x.size, bool)
        t[i] = False
        rows[name] = float(spearmanr(x[t], y[t]).statistic)
    v = np.array(list(rows.values()))
    return {"per_machine_removed": rows, "min": float(v.min()), "max": float(v.max())}


def trend(names, dm, phi, bootstrap: bool = True) -> dict:
    out = spearman_exact(dm, phi)
    if bootstrap:
        out["bootstrap_machines"] = spearman_bootstrap(dm, phi)
    out["without_one_machine"] = leave_one_out(names, dm, phi)
    return out


# --- 1. αNLI: the contested key ------------------------------------------------------------------------

def alphanli_keys(items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Three versions of the αNLI items, with the columns `key` (option index) and `key_votes`.

    - original: the key of the story, as in the article;
    - without_contested: without the items where the majority of the votes goes against the key (ties stay: there
      the majority does not contradict);
    - majority: the key is the most voted option; tied items have no majority and are dropped.
    """
    against = 2 * items.votes_correct < items.votes
    ties = 2 * items.votes_correct == items.votes
    orig = items.assign(key=items.correct_option.astype(int), key_votes=items.votes_correct)
    major = orig[~ties].copy()
    flip = against[~ties].to_numpy()
    major["key"] = np.where(flip, 1 - major.correct_option, major.correct_option).astype(int)
    major["key_votes"] = np.where(flip, major.votes - major.votes_correct, major.votes_correct)
    return {"original": orig, "without_contested": orig[~against.to_numpy()], "majority": major}


def _votes_table(items_v: pd.DataFrame, machine: pd.DataFrame):
    d = items_v.join(machine[["top"]], how="inner")
    h = (d.key_votes / d.votes).to_numpy(float)
    ok = (d.top.to_numpy(int) == d.key.to_numpy(int)).astype(float)
    return h, d.votes.to_numpy(float), ok, d.key.to_numpy(int), d.index.to_numpy()


def dm_phi_votes(h, ok, answer_class, w, m: int = 2):
    """The step of `measure.compare_votes` reduced to (dM, φ), with the same formulas, for weights w on the items."""
    ah = float(np.average(h, weights=w))
    pm = float(np.average(ok, weights=w))
    p11 = float(np.average(h * ok, weights=w))
    r_g = sdt.tetrachoric(np.clip(ah, 1e-4, 1 - 1e-4), np.clip(pm, 1e-4, 1 - 1e-4), p11)
    dh, dm = _d(ah, m), _d(pm, m)
    if math.isnan(dh) or math.isnan(dm):
        return None
    if m == 2 and answer_class is not None:
        c1, c0 = answer_class == 1, answer_class == 0
        if w[c1].sum() > 0 and w[c0].sum() > 0:
            dm = sdt.d_from_classes(float(np.average(ok[c1], weights=w[c1])), float(np.average(ok[c0], weights=w[c0])),
                                    n1=float(w[c1].sum()), n0=float(w[c0].sum()))
    return dm, sdt.angle_degrees(r_g)


def spearman_paired_items(tables: list, repetitions: int = REPETITIONS, seed: int = SEED) -> dict:
    """Spearman ρ between dM and φ resampling the items with the same weights for all machines.

    It is the measurement uncertainty of the trend, with the machines fixed. With seed 534 and the same items the
    weights are those that `compare_votes` used for the intervals of each machine.
    """
    n = tables[0][0].size
    if any(t[0].size != n for t in tables):
        raise ValueError("the machines do not have the same items")
    gen = np.random.default_rng(seed)
    out = []
    for _ in range(repetitions):
        w = gen.poisson(1.0, n).astype(float)
        points = [dm_phi_votes(h, ok, cl, w) for h, ok, cl in tables]
        if any(p is None for p in points):
            continue
        dm, phi = np.array(points).T
        out.append(spearmanr(dm, phi).statistic)
    out = np.asarray(out, float)
    return {"ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "share_below_zero": float(np.mean(out < 0)), "valid_repetitions": int(out.size)}


def check_1(folder: Path, repetitions: int = REPETITIONS) -> dict:
    from .analysis import load_machine           # imported here: analysis reads data/ only when needed
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    variants = alphanli_keys(items)
    map_data = json.loads((R / "map.json").read_text())
    orig = {(c["machine"], c["mode"]): c for c in map_data["comparisons"] if c.get("source") == "alphanli" and "phi" in c}
    names = sorted({n for n, _ in orig})
    out = {"items": {"total": int(len(items)),
                     "majority_against_key": int(np.sum(2 * items.votes_correct < items.votes)),
                     "tied_50_50": int(np.sum(2 * items.votes_correct == items.votes))},
           "variants": {}}
    for mode in MODES:
        machines = {n: load_machine(folder, "alphanli", n, mode) for n in names}
        for name_v, cv in variants.items():
            rows, tables = [], []
            for n in names:
                h, votes, ok, key, idx = _votes_table(cv, machines[n])
                r = compare_votes(h, votes, ok, m=2, repetitions=repetitions, answer_class=key)
                o = orig[(n, mode)]
                rows.append({"machine": n, "items": int(h.size), "acc_h": r["acc_h"], "acc_m": r["acc_m"],
                             "dh": r["dh"], "dm": r["dm"], "phi": r["phi"], "phi_ci": r["phi_ci"],
                             "useful_angle": r["useful_angle"],
                             "original": {"dm": o["dm"], "phi": o["phi"], "phi_ci": o["phi_ci"], "items": o["items"]}})
                tables.append((h, ok, key))
            rows.sort(key=lambda r: r["dm"])
            dm = [r["dm"] for r in rows]
            phi = [r["phi"] for r in rows]
            t = trend([r["machine"] for r in rows], dm, phi)
            t["bootstrap_paired_items"] = spearman_paired_items(tables, repetitions)
            v = out["variants"].setdefault(name_v, {})
            v[mode] = {"rows": rows, "spearman_dm_phi": t["rho"], "trend": t,
                       "items": sorted({r["items"] for r in rows}),
                       "phi_weakest_to_strongest": [rows[0]["phi"], rows[-1]["phi"]]}
            if name_v == "original":                    # the same run must give map.json back
                v[mode]["reproduces_map_max_diff_phi"] = float(max(abs(r["phi"] - r["original"]["phi"]) for r in rows))
                v[mode]["reproduces_map_max_diff_ci"] = float(max(abs(a - b) for r in rows
                                                                  for a, b in zip(r["phi_ci"], r["original"]["phi_ci"])))
            print("check 1", name_v, mode, "ρs", round(t["rho"], 3), "p", t["p_two_sided"],
                  [(r["machine"][:8], round(r["phi"], 1)) for r in rows], flush=True)
    return out


# --- 2. the useful angle with free weights -------------------------------------------------------------

def fit_free_weights(la: np.ndarray, lb: np.ndarray, y: np.ndarray, ridge: float = fe.RIDGE) -> np.ndarray:
    """As `full_evidence.fit_weights`, with a and b also negative (stopped at ±WEIGHT_MAX only to stay finite)."""
    res = minimize(fe._nll, x0=np.ones(2), args=(la, lb, y, ridge), jac=True, method="L-BFGS-B",
                   bounds=[(-fe.WEIGHT_MAX, fe.WEIGHT_MAX)] * 2)
    return res.x


def cv_combiner_free(pa: np.ndarray, pb: np.ndarray, y: np.ndarray, eps: float = fe.EPS,
                     folds: int = fe.FOLDS, seed: int = fe.SEED) -> dict:
    """`full_evidence.cv_combiner` with free weights: same score, same split, same tie handling."""
    la, lb = np.log(pa + eps), np.log(pb + eps)
    correct = np.empty(y.size)
    weights = []
    for tr, te in fe.cv_split(y, folds, seed):
        a, b = fit_free_weights(la[tr], lb[tr], y[tr])
        weights.append((a, b))
        correct[te] = fe.correct_with_ties(a * la[te] + b * lb[te], y[te])
    weights = np.array(weights)
    return {"acc": float(correct.mean()), "a": float(weights[:, 0].mean()), "b": float(weights[:, 1].mean()),
            "a_negative": int(np.sum(weights[:, 0] < 0)), "b_negative": int(np.sum(weights[:, 1] < 0)),
            "weights_at_limit": int(np.sum(np.abs(weights).max(1) >= 0.999 * fe.WEIGHT_MAX))}


def ideal_unclipped(dh: float, dm: float, rho: float) -> float:
    """|S| with free weights: the same formula as `sdt.ideal` without stopping at the stronger below the useful angle."""
    return math.sqrt((dh * dh + dm * dm - 2.0 * rho * dh * dm) / (1.0 - rho * rho))


def _condition_free(args):
    ds, cond, y, P, pairs, verify = args
    out = []
    for ia, ib, key in pairs:
        c = cv_combiner_free(P[ia], P[ib], y)
        r = {"key": key, "acc_free": c["acc"], "a_free": c["a"], "b_free": c["b"],
             "a_negative": c["a_negative"], "b_negative": c["b_negative"], "weights_at_limit_free": c["weights_at_limit"]}
        if verify:                                      # the constrained combiner, redone: it must give the file back
            r["acc_constrained_redone"] = fe.cv_combiner(P[ia], P[ib], y)["acc"]
        out.append(r)
    return out


def _region(df: pd.DataFrame, column: str) -> dict:
    """How many units beat the strongest network (dS > dB) and by how much, in units of the strongest."""
    x = df[df[column].notna()]
    r = (x[column] / x.dB).to_numpy()
    beats = (x[column] > x.dB + 1e-12).to_numpy()          # the same threshold as full_evidence.summarize
    out = {"units": int(len(x)), "beats_B": int(beats.sum()), "equal_to_B": int(np.sum(np.abs(x[column] - x.dB) < 1e-12)),
           "below_B": int(np.sum(x[column] < x.dB - 1e-12)), "beyond_B_by_5_percent": int(np.sum(r > 1.05)),
           "dS_over_dB_median": float(np.median(r)) if r.size else None,
           "dS_over_dB_q1_q3": [float(np.percentile(r, 25)), float(np.percentile(r, 75))] if r.size else None,
           "dS_over_dB_median_among_those_beating_B": float(np.median(r[beats])) if beats.any() else None}
    # if beating B were only noise, above and below B would be equally likely: sign test, ties excluded
    out["sign_against_B_p"] = _sign_test(out["beats_B"], out["below_B"])
    return out


def _sign_test(up: int, down: int) -> float | None:
    return float(binomtest(up, up + down, 0.5).pvalue) if up + down > 0 else None


def summarize_free(df: pd.DataFrame) -> dict:
    below = (df.phi < df.useful_angle).to_numpy()
    out = {}
    for name, s in (("all", np.ones(len(df), bool)), ("below_useful_angle", below), ("above_useful_angle", ~below)):
        x = df[s]
        free = x[x.free_state == "valid"]
        o = {"constrained": _region(x, "dS"), "free": _region(free, "dS_free"),
             "free_at_ceiling": int(np.sum(x.free_state == "at ceiling")),
             "free_at_chance": int(np.sum(x.free_state == "at chance level")),
             "units_with_negative_a_in_every_fold": int(np.sum(x.a_negative == fe.FOLDS)),
             "units_with_negative_a_in_at_least_one_fold": int(np.sum(x.a_negative > 0)),
             "units_with_negative_b_in_at_least_one_fold": int(np.sum(x.b_negative > 0)),
             "folds_with_weights_at_limit": int(x.weights_at_limit_free.sum())}
        if len(free):
            g_meas = (free.dS_free / free.dB).to_numpy()
            g_pred = (free.ideal_unclipped / free.dB).to_numpy()
            dv = (free.dS_free - free.dS).to_numpy()
            o.update(free_minus_constrained_median=float(np.median(dv)),
                     free_above_constrained=int(np.sum(dv > 1e-12)), free_below_constrained=int(np.sum(dv < -1e-12)),
                     sign_free_vs_constrained_p=_sign_test(int(np.sum(dv > 1e-12)), int(np.sum(dv < -1e-12))),
                     predicted_unclipped_over_dB_median=float(np.median(g_pred)),
                     eta_unclipped_median=float(np.median(free.dS_free / free.ideal_unclipped)),
                     spearman_gain_over_B_measured_predicted_unclipped=fe._spearman(g_meas, g_pred))
        out[name] = o
    return out


def check_2(folder: Path, processes: int = 2, check_every: int = 1) -> dict:
    """The valid units of full-evidence.json, redone with free weights. `check_every` = 1 also redoes the
    constrained combiner on all of them (a check that data and pairs are the same)."""
    t0 = time.time()
    fe_json = json.loads((R / "full-evidence.json").read_text())
    units = [u for u in fe_json["units"] if u["state"] == "valid"]
    data = fe.load(folder)
    networks = data["networks"]
    if networks != fe_json["parameters"]["networks"]:
        raise ValueError("the networks are not those of the full_evidence run")
    per_cond: dict = {}
    for i, u in enumerate(units):
        per_cond.setdefault((u["dataset"], u["condition"]), []).append(
            (networks.index(u["mind"]), networks.index(u["machine"]), i))
    tasks = [(ds, cond, data["conditions"][(ds, cond)]["y"], data["conditions"][(ds, cond)]["P"], pairs,
              k % check_every == 0) for k, ((ds, cond), pairs) in enumerate(sorted(per_cond.items()))]
    print("check 2:", len(units), "valid units in", len(tasks), "conditions", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=processes) as ex:
        for k, out in enumerate(ex.map(_condition_free, tasks)):
            results += out
            if (k + 1) % 10 == 0:
                print("  ", k + 1, "conditions,", round(time.time() - t0), "s", flush=True)
    results.sort(key=lambda r: r["key"])
    df = pd.DataFrame(units)
    free = pd.DataFrame(results).set_index("key").sort_index()
    df = df.join(free)
    acc = df.acc_free.to_numpy()
    df["free_state"] = np.where(acc > fe.CEILING, "at ceiling", np.where(acc <= 1.0 / fe.M, "at chance level", "valid"))
    df["dS_free"] = [fe._d(a) if s == "valid" else np.nan for a, s in zip(acc, df.free_state)]
    df["ideal_unclipped"] = [ideal_unclipped(a, b, r) for a, b, r in zip(df.dA, df.dB, df.rho)]
    redone = df[df.acc_constrained_redone.notna()] if "acc_constrained_redone" in df else df.iloc[:0]
    out = {
        "valid_units": int(len(df)), "conditions": int(len(tasks)),
        "split": "the same as full_evidence: 5 folds stratified by category, seed 534; ε 1e-5, ridge 1e-6",
        "weight_limits": [-fe.WEIGHT_MAX, fe.WEIGHT_MAX],
        "constrained_check": {"units_redone": int(len(redone)),
                              "max_abs_diff_acc": float((redone.acc_constrained_redone - redone.acc_pair).abs().max()) if len(redone) else None},
        "summary": summarize_free(df),
        "original_summary_for_comparison": {k: fe_json["summary"][k] for k in
                                            ("dS_above_dB", "dS_equal_dB", "dS_below_dB", "below_useful_angle",
                                             "below_ua_turing_above_k", "above_ua_gain_over_B_measured_median",
                                             "pair_beyond_B_by_5_percent")},
        "seconds": round(time.time() - t0, 1),
        "units": df[["dataset", "condition", "mind", "machine", "phi", "useful_angle", "dA", "dB", "rho", "dS",
                     "acc_pair", "acc_free", "dS_free", "free_state", "a_free", "b_free", "a_negative",
                     "b_negative", "weights_at_limit_free", "ideal_unclipped"]].to_dict("records"),
    }
    s = out["summary"]
    for reg in ("below_useful_angle", "above_useful_angle"):
        print("check 2", reg, "constrained", s[reg]["constrained"], "free", s[reg]["free"], flush=True)
    return out


# --- 3. ChatBench: the trials without an answer in the chat ----------------------------------------------

def complement(pairs: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """The rows of `pairs` whose pair (worker_id, question_id) is not in `keys`."""
    k = keys[["worker_id", "question_id"]].drop_duplicates()
    x = pairs.merge(k, on=["worker_id", "question_id"], how="left", indicator=True)
    return x[x["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)


def by_effort(trials: pd.DataFrame, conversations: pd.DataFrame, machine: pd.Series) -> dict:
    """The trials split by the ChatBench annotation of how the person used the chat (`user_effort`:
    "tried", "low effort", "known"); "none" where the conversation is missing. Per group: trials, accuracy
    of the person alone, of the machine alone, of the person with the machine. No intervals: it describes."""
    k = conversations[["worker_id", "question_id", "user_effort"]].drop_duplicates(["worker_id", "question_id"])
    y = trials.merge(k, on=["worker_id", "question_id"], how="left")
    y["m_ok"] = y.question_id.map(machine)
    return {name: {"n_trials": int(len(z)), "acc_h": float(z.acc.mean()), "acc_m": float(z.m_ok.mean()),
                   "acc_s": float(z.acc_ai.mean())}
            for name, z in y.groupby(y.user_effort.fillna("none"))}


def conversation_kind(conv: pd.DataFrame) -> pd.Series:
    """What the chat recorded: "answer" (a letter among the options), "out_of_options" (an annotated answer that
    is none of the options, for example "sqrt(74)"), "no_answer" (no answer)."""
    has = (conv.conv_has_answer == True).to_numpy()  # noqa: E712
    letter = conv.conv_answer_letter.notna().to_numpy()
    return pd.Series(np.where(has & letter, "answer", np.where(has, "out_of_options", "no_answer")), index=conv.index)


def describe(trials: pd.DataFrame, machine: pd.Series) -> dict:
    """Trials; accuracy of the person alone, of the machine alone, of the person with the machine; share of the
    trials where the person changes answer after the chat, and in which direction. No intervals: it describes."""
    m_ok = trials.question_id.map(machine)
    return {"n_trials": int(len(trials)), "acc_h": float(trials.acc.mean()), "acc_m": float(m_ok.mean()),
            "acc_s": float(trials.acc_ai.mean()), "changes": float((trials.selected_answer != trials.selected_answer_ai).mean()),
            "wrong_to_right": float(((trials.acc == 0) & (trials.acc_ai == 1)).mean()),
            "right_to_wrong": float(((trials.acc == 1) & (trials.acc_ai == 0)).mean())}


def in_chat_widened(pairs: pd.DataFrame, conv: pd.DataFrame, key: pd.Series, machine: pd.Series) -> dict:
    """The machine of the conversation where the chat recorded an answer. "strict": only the answers that are
    one of the options, as in the article; "widened": also those outside the options, counted as wrong
    (a lower bound: some are correct written in another form). Next to it, the machine alone on the same trials."""
    k = conv[conv.conv_has_answer == True][["worker_id", "question_id", "conv_answer_letter"]]  # noqa: E712
    d = pairs.merge(k.drop_duplicates(["worker_id", "question_id"]), on=["worker_id", "question_id"])
    d = d[d.question_id.isin(machine.index)].dropna(subset=["acc", "acc_ai"])
    ok = (d.conv_answer_letter == d.question_id.map(key)).astype(float)
    alone = d.question_id.map(machine)
    strict = d.conv_answer_letter.notna()
    return {"strict": {"n_trials": int(strict.sum()), "acc_m_alone": float(alone[strict].mean()), "acc_m_chat": float(ok[strict].mean())},
            "widened": {"n_trials": int(len(d)), "acc_m_alone": float(alone.mean()), "acc_m_chat": float(ok.mean())}}


FIELDS_3 = ("n_trials", "people", "items", "acc_h", "acc_m", "acc_s", "dh", "dm", "ds", "turing", "turing_ci",
            "yield_on_machine", "yield_on_machine_ci", "phi", "phi_ci", "repetitions")


def check_3(repetitions: int = REPETITIONS) -> dict:
    c = sources.chatbench_pairs(BASE)
    in_chat = sources.chatbench_machine_in_chat(BASE)
    source = json.loads((R / "chatbench-source.json").read_text())
    conv = pd.read_json(BASE / "chatbench/full_study/conversations.json")
    key = pd.read_csv(BASE / "chatbench/questions.csv").set_index("question_id")["answer"]
    out = {}
    for model in ("gpt-4o", "llama-3.1-8b"):
        m = sources.chatbench_machine_alone(BASE, model)
        keys = in_chat[in_chat.model == model].dropna(subset=["m_ok"])
        x = complement(c[c.model == model], keys)
        r = compare(sources.chatbench_trials(x, m, None), m=4, repetitions=repetitions)
        row = {k: r.get(k) for k in FIELDS_3}
        alone = next(z for z in source["comparisons"] if z["machine"] == model and z["subject"] == "all" and z["kind"] == "alone")
        same = next(z for z in source["same_trials"] if z["machine"] == model)
        row["per_comparison"] = {name: {k: z.get(k) for k in ("n_trials", "acc_h", "acc_m", "acc_s", "turing", "turing_ci")}
                                 for name, z in (("all_trials", alone), ("with_chat_answer", same))}
        row["trials_add_up"] = bool(row["n_trials"] + same["n_trials"] == alone["n_trials"])
        inside = x[x.question_id.isin(m.index)].dropna(subset=["acc", "acc_ai"])
        row["by_effort"] = by_effort(inside, conv[conv.model == model], m)
        assert sum(v["n_trials"] for v in row["by_effort"].values()) == row["n_trials"]
        # what is inside the trials without a letter in the chat; and, for comparison, those with a letter
        cm = conv[conv.model == model].drop_duplicates(["worker_id", "question_id"])
        cm = cm.assign(kind=conversation_kind(cm))
        every = c[(c.model == model) & c.question_id.isin(m.index)].dropna(subset=["acc", "acc_ai"])
        y = every.merge(cm[["worker_id", "question_id", "kind", "user_effort"]], on=["worker_id", "question_id"], how="left")
        y["kind"] = y.kind.fillna("no_conversation")
        row["by_kind"] = {name: describe(z, m) for name, z in y.groupby("kind")}
        row["no_answer_tried"] = describe(y[(y.kind == "no_answer") & (y.user_effort == "tried")], m)
        row["low_effort_share"] = {"all": float((y.user_effort == "low effort").mean()),
                                   **{name: float((z.user_effort == "low effort").mean()) for name, z in y.groupby("kind")}}
        assert row["by_kind"]["answer"]["n_trials"] == same["n_trials"]
        assert sum(v["n_trials"] for k_, v in row["by_kind"].items() if k_ != "answer") == row["n_trials"]
        row["in_chat_widened"] = in_chat_widened(c[c.model == model], conv[conv.model == model], key, m)
        assert row["in_chat_widened"]["strict"]["n_trials"] == same["n_trials"]
        out[model] = row
        print("check 3", model, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                 if k in ("n_trials", "acc_h", "acc_m", "acc_s", "turing")}, row["turing_ci"], flush=True)
    return out


# --- 4. the trends -----------------------------------------------------------------------------------------

def check_4(check1: dict | None = None) -> dict:
    summ = json.loads((R / "summary.json").read_text())
    out = {}
    vis = summ["vision"]["networks"]
    out["vision"] = trend([r["machine"] for r in vis], [r["dm_median"] for r in vis], [r["phi_median"] for r in vis])
    out["vision"]["spearman_in_summary"] = summ["vision"]["spearman_dm_phi"]
    for key, v in summ["h7"].items():
        rows = v["rows"]
        names = [r["machine"] for r in rows]
        dm = [r["dm"] for r in rows]
        phi = [r["phi"] for r in rows]
        t = trend(names, dm, phi)
        t["spearman_in_summary"] = v["spearman_dm_phi"]
        if key.startswith("chatbench"):
            strongest = names[int(np.argmax(dm))]
            keep = [i for i, n in enumerate(names) if n != strongest]
            t["without_the_strongest"] = {"removed": strongest, **spearman_exact([dm[i] for i in keep], [phi[i] for i in keep])}
            g = "gemma-4-26b"
            if g in names and g != strongest:
                keep = [i for i, n in enumerate(names) if n != g]
                t["without_gemma_4_26b"] = spearman_exact([dm[i] for i in keep], [phi[i] for i in keep])
        out[key] = t
        print("check 4", key, round(t["rho"], 3), "p", t["p_two_sided"], "boot", t["bootstrap_machines"]["ci95"], flush=True)
    if check1:
        for name_v, v in check1["variants"].items():
            for mode, x in v.items():
                out[f"alphanli-{name_v}/{mode}"] = {k: x["trend"][k] for k in ("rho", "p_two_sided", "p_negative", "permutations")}
    return out


# --- run ------------------------------------------------------------------------------------------------------

def main(folder: Path, checks: set[int]) -> dict:
    old = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    res = dict(old)
    res["note"] = ("Exploratory analyses, decided on 2026-11-10 after seeing the data, at the request of a review: "
                   "PROTOCOL.md, \"Deviations\". They do not replace any result of the article.")
    res.setdefault("times", {})
    if 1 in checks:
        t = time.time()
        res["check_1_alphanli_keys"] = check_1(folder)
        res["times"]["check_1_s"] = round(time.time() - t, 1)
    if 3 in checks:
        t = time.time()
        res["check_3_chatbench_without_chat_answer"] = check_3()
        res["times"]["check_3_s"] = round(time.time() - t, 1)
    if 4 in checks:
        t = time.time()
        res["check_4_trends"] = check_4(res.get("check_1_alphanli_keys"))
        res["times"]["check_4_s"] = round(time.time() - t, 1)
    if 2 in checks:
        t = time.time()
        res["check_2_free_weights"] = check_2(folder)
        res["times"]["check_2_s"] = round(time.time() - t, 1)
    R.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(fe._clean(res), indent=1, ensure_ascii=False))
    print("written", OUTPUT, flush=True)
    return res


if __name__ == "__main__":
    checks = {1, 2, 3, 4}
    if "--checks" in sys.argv:
        checks = {int(x) for x in sys.argv[sys.argv.index("--checks") + 1].split(",")}
    main(Path(sys.argv[1]), checks)
