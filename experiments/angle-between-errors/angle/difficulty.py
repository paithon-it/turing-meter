"""Closeness net of difficulty, on αNLI. Added after the sixth review (15 November 2026): exploratory.

In the angle between the errors two things end up together: how alike two minds are, and how hard the stories are
for everyone. For the pair the two are the same thing (PROTOCOL.md, «Model and definitions»), and the margin is
computed with the whole correlation. To say how much of the closeness is likeness, the difficulty has to be
taken out, and it has to be estimated on minds that are not the two being compared.

On αNLI the «person» is one vote out of a hundred anonymous votes on a story: the correlation between person and
machine is, by construction, how much the machine fails where many people fail. Taking out a difficulty
estimated on the people themselves would take out everything (`human_strata`, below, shows it: the finer the
strata, the closer to 90 degrees). So:

- person and machine, net of the difficulty for a third population: the open machines (ten, in both modes: twenty
  answers per story; for an open machine, the other nine: eighteen answers). The difficulty of a story is how
  many of those answers are right.
- machine and machine, net of the difficulty for the people: the votes for the key, out of a hundred.

Strata. The consecutive levels of the difficulty score, merged from the hard end until each stratum holds at
least MIN_STORIES stories (60): fine enough that little difficulty is left inside a stratum, large enough that a
stratum has errors to count. The same rule for the two scores; 30 and 120 are reported as a check.

Estimator. One latent correlation for all strata, each stratum with its own two thresholds, taken from its
margins: the ρ that maximizes the sum over strata of the log-likelihood of the 2×2 tables (both right, first
only, second only, both wrong). With one stratum it is the tetrachoric of the article. A stratum where one of
the two makes no error, or only errors, carries no information and drops out: with the strongest machines the
easy strata do. Next to it the Mantel-Haenszel common odds ratio, which needs no thresholds, as a check.
Intervals: the stories are resampled (Poisson weights), the strata stay fixed.

What it cannot say. The difficulty score is a count of twenty answers, or of a hundred votes: what it leaves
inside a stratum stays in the correlation, which is therefore an upper bound on the likeness. And the open
machines are not neutral judges of difficulty: what they share with a Claude model goes out with the difficulty.
There is no single «pure» angle: it depends on who measures the difficulty.

Also here, because it uses the same votes: the contested stories defined on half of the votes, with the angle
measured on the other half (`contested_split`).

Writes results/difficulty.json. Usage, from the experiment folder:
    python -m angle.difficulty CLOSED_RESULTS_DIR GPU_RESULTS_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from . import sdt

SEED = 534
REPETITIONS = 2000
MIN_STORIES = 60
CHECK_MIN_STORIES = (30, 120)
BANDS = (3, 10, 25)                    # for the strata built on the people themselves
_GRID = np.linspace(-0.98, 0.99, 40)
_EDGE = 1e-9


# --- strata -------------------------------------------------------------------------------------------

def strata_from_score(score, min_stories: int = MIN_STORIES) -> np.ndarray:
    """Strata from an integer score (low is hard): consecutive levels merged until each holds min_stories stories.

    A short tail at the easy end joins the stratum before it.
    """
    score = np.asarray(score)
    levels, counts = np.unique(score, return_counts=True)
    label, current, held = np.empty(levels.size, int), 0, 0
    for i, c in enumerate(counts):
        label[i] = current
        held += c
        if held >= min_stories:
            current, held = current + 1, 0
    if held and current > 0:
        label[label == current] = current - 1
    return label[np.searchsorted(levels, score)]


def quantile_strata(x, bands: int) -> np.ndarray:
    """Strata of about equal size on a continuous score."""
    x = np.asarray(x, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bands + 1))[1:-1])
    return np.digitize(x, edges, right=True)


# --- tables and estimators ----------------------------------------------------------------------------

def tables(right, total, second, strata, w=None) -> np.ndarray:
    """One 2×2 table per stratum: both right, first only, second only, both wrong.

    The first is given as right answers out of a total for each story (a person: votes for the key out of a
    hundred; a machine: 0 or 1 out of one); the second as 0 or 1.
    """
    right, total, second = (np.asarray(x, float) for x in (right, total, second))
    w = np.ones(right.size) if w is None else np.asarray(w, float)
    k = int(np.max(strata)) + 1
    r, t = right * w, total * w
    return np.stack([np.bincount(strata, r * second, k), np.bincount(strata, r * (1 - second), k),
                     np.bincount(strata, (t - r) * second, k), np.bincount(strata, (t - r) * (1 - second), k)], axis=1)


def common_rho(t) -> dict:
    """One latent correlation for all strata, each with its own two thresholds (from its margins)."""
    t = np.asarray(t, float)
    n = t.sum(1)
    alive = n > 0
    p1, p2 = np.zeros(n.size), np.zeros(n.size)
    p1[alive], p2[alive] = (t[alive, 0] + t[alive, 1]) / n[alive], (t[alive, 0] + t[alive, 2]) / n[alive]
    keep = alive & (p1 > _EDGE) & (p1 < 1 - _EDGE) & (p2 > _EDGE) & (p2 < 1 - _EDGE)
    out = {"rho": float("nan"), "strata": int(n.size), "strata_used": int(keep.sum()),
           "weight_used": float(n[keep].sum() / n.sum()) if n.sum() > 0 else 0.0, "used": keep}
    if not keep.any():
        return out
    a, b, c = p1[keep], p2[keep], t[keep]
    z1, z2 = norm.ppf(a), norm.ppf(b)

    def minus_loglik(r):
        p11 = sdt.bvn(z1, z2, np.asarray(r, float)[..., None])
        cells = np.stack([p11, a - p11, b - p11, 1 - a - b + p11], axis=-1)
        return -(c * np.log(np.clip(cells, 1e-12, None))).sum(axis=(-2, -1))

    i = int(np.argmin(minus_loglik(_GRID)))
    best = minimize_scalar(lambda r: float(minus_loglik(r)), bounds=(_GRID[max(i - 1, 0)], _GRID[min(i + 1, _GRID.size - 1)]),
                           method="bounded", options={"xatol": 1e-5})
    out["rho"] = float(best.x)
    return out


def mantel_haenszel(t) -> float:
    """The common odds ratio of being right together, across strata. It needs no thresholds."""
    t = np.asarray(t, float)
    n = t.sum(1)
    alive = n > 0
    den = float((t[alive, 1] * t[alive, 2] / n[alive]).sum())
    return float((t[alive, 0] * t[alive, 3] / n[alive]).sum() / den) if den > 0 else float("nan")


def _pooled_rho(t: np.ndarray) -> float:
    """The article's estimator on all the stories together: the tetrachoric, on its grid."""
    s = t.sum(0)
    n = s.sum()
    p1, p2 = (s[0] + s[1]) / n, (s[0] + s[2]) / n
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return float("nan")
    return sdt.tetrachoric(float(np.clip(p1, 1e-4, 1 - 1e-4)), float(np.clip(p2, 1e-4, 1 - 1e-4)), float(s[0] / n))


def _estimates(right, total, second, strata, w=None) -> dict:
    t = tables(right, total, second, strata, w)
    fit = common_rho(t)
    return {"raw": _pooled_rho(t), "net": fit["rho"], "raw_or": mantel_haenszel(t.sum(0, keepdims=True)),
            "net_or": mantel_haenszel(t), "fit": fit}


def _interval(values) -> list | None:
    v = np.asarray([x for x in values if x == x], float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None


def _angles(ci) -> list | None:
    return None if ci is None else sorted(sdt.angle_degrees(r) for r in ci)


def net(right, total, second, strata, repetitions: int = REPETITIONS, seed: int = SEED) -> dict:
    """The correlation between a first (right answers out of a total per story) and a second (0 or 1), on all
    the stories together and net of the strata, with intervals resampling the stories."""
    right, total, second = (np.asarray(x, float) for x in (right, total, second))
    strata = np.asarray(strata, int)
    point = _estimates(right, total, second, strata)
    fit = point["fit"]
    in_used = fit["used"][strata]
    out = {"stories": int(right.size),
           "raw": {"rho": point["raw"], "phi": sdt.angle_degrees(point["raw"]), "odds_ratio": point["raw_or"]},
           "net": {"rho": point["net"], "phi": sdt.angle_degrees(point["net"]) if point["net"] == point["net"] else None,
                   "odds_ratio": point["net_or"], "share_of_raw": point["net"] / point["raw"] if point["raw"] else None,
                   "strata": fit["strata"], "strata_used": fit["strata_used"], "stories_used": int(in_used.sum()),
                   "errors_of_second": int((second == 0).sum()), "errors_of_second_used": int(((second == 0) & in_used).sum())}}
    if repetitions:
        gen = np.random.default_rng(seed)
        runs = [_estimates(right, total, second, strata, gen.poisson(1.0, right.size).astype(float)) for _ in range(repetitions)]
        out["raw"]["rho_ci"] = _interval(r["raw"] for r in runs)
        out["raw"]["phi_ci"] = _angles(out["raw"]["rho_ci"])
        out["raw"]["odds_ratio_ci"] = _interval(r["raw_or"] for r in runs)
        out["net"]["rho_ci"] = _interval(r["net"] for r in runs)
        out["net"]["phi_ci"] = _angles(out["net"]["rho_ci"])
        out["net"]["odds_ratio_ci"] = _interval(r["net_or"] for r in runs)
        out["net"]["share_of_raw_ci"] = _interval(r["net"] / r["raw"] for r in runs if r["raw"])
        out["repetitions"] = len(runs)
    return out


def medians(pairs: list, strata, repetitions: int, seed: int = SEED) -> dict:
    """Median over many pairs of machines of the correlation, on all the stories and net of the strata.
    The same resampled stories for all the pairs of a repetition."""
    strata = np.asarray(strata, int)
    ones = np.ones(strata.size)

    def one(w):
        e = [_estimates(a, ones, b, strata, w) for a, b in pairs]
        return float(np.nanmedian([x["raw"] for x in e])), float(np.nanmedian([x["net"] for x in e]))

    raw, cleaned = one(None)
    gen = np.random.default_rng(seed)
    runs = [one(gen.poisson(1.0, strata.size).astype(float)) for _ in range(repetitions)]
    return {"pairs": len(pairs), "raw_median": raw, "raw_median_ci": _interval(r[0] for r in runs),
            "net_median": cleaned, "net_median_ci": _interval(r[1] for r in runs), "repetitions": len(runs)}


# --- the people as their own measure of difficulty: by construction, 90 degrees ------------------------------

def human_strata(right, total, second, bands=BANDS, splits: int = 50, repetitions: int = 300, seed: int = SEED) -> dict:
    """Person and machine within strata of the people's own difficulty: half of the votes make the strata, the
    other half is the person. Shown to say why it is not the analysis: with few bands what is left is difficulty
    inside the band, with many nothing is left."""
    right, total, second = np.asarray(right, int), np.asarray(total, int), np.asarray(second, float)
    gen = np.random.default_rng(seed)
    half = total // 2

    def one(q, w):
        a = gen.hypergeometric(right, total - right, half)
        return common_rho(tables(right - a, total - half, second, quantile_strata(a / half, q), w))["rho"]

    out = {}
    for q in bands:
        rho = float(np.mean([one(q, None) for _ in range(splits)]))
        ci = _interval(one(q, gen.poisson(1.0, right.size).astype(float)) for _ in range(repetitions))
        out[str(q)] = {"rho": rho, "phi": sdt.angle_degrees(rho), "rho_ci": ci, "phi_ci": _angles(ci)}
    return out


# --- the contested stories, on half of the votes ------------------------------------------------------

def votes_step(h, ok, key, w) -> dict:
    """The step of `measure.compare_votes`, for weights w on the stories, with the margin of the ideal pair."""
    ah, pm = float(np.average(h, weights=w)), float(np.average(ok, weights=w))
    rho = sdt.tetrachoric(np.clip(ah, 1e-4, 1 - 1e-4), np.clip(pm, 1e-4, 1 - 1e-4), float(np.average(h * ok, weights=w)))
    c1, c0 = key == 1, key == 0
    n1, n0 = float(w[c1].sum()), float(w[c0].sum())
    dh = sdt.d_from_classes(float(np.average(h[c1], weights=w[c1])), float(np.average(h[c0], weights=w[c0])), n1=n1, n0=n0)
    dm = sdt.d_from_classes(float(np.average(ok[c1], weights=w[c1])), float(np.average(ok[c0], weights=w[c0])), n1=n1, n0=n0)
    return {"acc_h": ah, "acc_m": pm, "dh": dh, "dm": dm, "k": dm / dh, "rho": rho, "phi": sdt.angle_degrees(rho),
            "useful_angle": sdt.useful_angle(dh, dm), "margin": sdt.ideal(dh, dm, rho) / max(dh, dm) - 1.0}


def contested_split(right, total, ok, key, splits: int = 200, repetitions: int = REPETITIONS, seed: int = SEED) -> dict:
    """A story is contested when most voters go against its key. The article takes those stories out looking at
    the same votes that then measure the angle. Here half of the votes of each story, drawn at random, say
    whether it is contested, and the other half is the person. Point: mean over the splits; interval: a new
    split and resampled stories at each repetition."""
    right, total, ok, key = np.asarray(right, int), np.asarray(total, int), np.asarray(ok, float), np.asarray(key, int)
    gen = np.random.default_rng(seed)
    half = total // 2
    ones = np.ones(right.size)
    same = ~(2 * right < total)
    out = {"all_stories": votes_step(right / total, ok, key, ones),
           "same_votes": {**votes_step(right[same] / total[same], ok[same], key[same], ones[same]), "stories_removed": int((~same).sum())}}

    def one(w):
        a = gen.hypergeometric(right, total - right, half)
        keep = ~(2 * a < half)
        r = votes_step((right - a)[keep] / (total - half)[keep], ok[keep], key[keep], w[keep])
        return {**r, "stories_removed": float((~keep).sum())}

    names = ("phi", "rho", "k", "useful_angle", "margin", "stories_removed")
    points = [one(ones) for _ in range(splits)]
    runs = [one(gen.poisson(1.0, right.size).astype(float)) for _ in range(repetitions)]
    out["split"] = {name: float(np.mean([p[name] for p in points])) for name in names}
    for name in names[:-1]:
        out["split"][name + "_ci"] = _interval(r[name] for r in runs)
    out["split"]["splits"], out["split"]["repetitions"] = splits, len(runs)
    return out


# --- the run ------------------------------------------------------------------------------------------

def run(closed: Path, gpu: Path, repetitions: int = REPETITIONS) -> dict:
    from .analysis import BASE, _tasks_in_folder, load_machine          # here: they read the data folders

    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    right, total, key = items.votes_correct.to_numpy(int), items.votes.to_numpy(int), items.correct_option.to_numpy(int)

    def answers(folder: Path) -> dict:
        out = {}
        for name, mode in sorted(_tasks_in_folder(folder).get("alphanli", [])):
            ok = load_machine(folder, "alphanli", name, mode).ok.reindex(items.index)
            if ok.isna().any():
                raise ValueError(f"{name}/{mode} did not answer every story")
            out[(name, mode)] = ok.to_numpy(float)
        return out

    open_m, closed_m = answers(gpu), answers(closed)
    kept = {"all": np.ones(right.size, bool), "without_contested": ~(2 * right < total)}
    out = {"note": "Exploratory, added after the sixth review, 15 November 2026 (PROTOCOL.md). On αNLI the person is one vote "
                   "of the hundred on a story. Intervals: 95 percent, resampling the stories.",
           "min_stories": MIN_STORIES, "repetitions": repetitions,
           "stories": {k: int(v.sum()) for k, v in kept.items()},
           "open_machines": sorted({n for n, _ in open_m}), "closed_machines": sorted({n for n, _ in closed_m})}

    def pool(target: str, modes=("letter", "reason")) -> np.ndarray:
        return np.sum([ok for (n, mo), ok in open_m.items() if n != target and mo in modes], axis=0).astype(int)

    # 1. person and machine, net of the difficulty for the open machines
    rows = []
    for (name, mode), ok in [*closed_m.items(), *open_m.items()]:
        score = pool(name)
        row = {"machine": name, "mode": mode, "closed": (name, mode) in closed_m,
               "pool_answers": int(sum(1 for (n, _) in open_m if n != name))}
        for variant, keep in kept.items():
            r = net(right[keep], total[keep], ok[keep], strata_from_score(score[keep]), repetitions)
            r["check"] = {f"min_stories_{m}": _estimates(right[keep], total[keep], ok[keep], strata_from_score(score[keep], m))["net"]
                          for m in CHECK_MIN_STORIES}
            r["check"]["letter_mode_pool"] = _estimates(right[keep], total[keep], ok[keep],
                                                        strata_from_score(pool(name, ("letter",))[keep]))["net"]
            row[variant] = r
        rows.append(row)
        print("person-machine", name, mode, {v: (round(row[v]["raw"]["phi"], 1), round(row[v]["net"]["phi"], 1), [round(x, 1) for x in row[v]["net"]["phi_ci"]])
                                             for v in kept}, flush=True)
    out["person_machine"] = rows

    # 2. machine and machine, net of the difficulty for the people
    everyone = {**closed_m, **open_m}
    strongest = [("gemma-4-26b", "letter"), ("qwen3.8-27b", "letter")]
    named = [(a, b) for i, a in enumerate(closed_m) for b in list(closed_m)[i + 1:]]
    named += [(a, b) for a in strongest if a in open_m for b in closed_m]
    pairs = []
    for a, b in named:
        row = {"a": "/".join(a), "b": "/".join(b)}
        for variant, keep in kept.items():
            x, y, ones = everyone[a][keep], everyone[b][keep], np.ones(int(keep.sum()))
            r = net(x, ones, y, strata_from_score(right[keep]), repetitions)
            r["check"] = {f"min_stories_{m}": _estimates(x, ones, y, strata_from_score(right[keep], m))["net"] for m in CHECK_MIN_STORIES}
            row[variant] = r
        pairs.append(row)
        print("machine-machine", row["a"], row["b"], {v: (round(row[v]["raw"]["rho"], 2), round(row[v]["net"]["rho"], 2),
                                                           [round(x, 2) for x in row[v]["net"]["rho_ci"]]) for v in kept}, flush=True)
    keys = list(open_m)
    between = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys)) if keys[i][0] != keys[j][0]]
    out["machine_machine"] = {"pairs": pairs,
                              "open_pairs": {variant: medians([(open_m[a][keep], open_m[b][keep]) for a, b in between],
                                                              strata_from_score(right[keep]), max(repetitions // 8, 1))
                                             for variant, keep in kept.items()}}
    print("open pairs", {v: (round(r["raw_median"], 2), round(r["net_median"], 2), [round(x, 2) for x in r["net_median_ci"]])
                         for v, r in out["machine_machine"]["open_pairs"].items()}, flush=True)

    # 3. why not the people's own difficulty: by construction it goes to 90 degrees
    out["human_strata"] = [{"machine": name, "mode": mode, "raw_phi": sdt.angle_degrees(_estimates(right, total, ok, np.zeros(right.size, int))["raw"]),
                            "bands": human_strata(right, total, ok, repetitions=max(repetitions // 8, 1))}
                           for (name, mode), ok in closed_m.items()]

    # 4. the contested stories defined on half of the votes
    out["contested_split"] = [{"machine": name, "mode": mode, **contested_split(right, total, ok, key, repetitions=repetitions)}
                              for (name, mode), ok in [*closed_m.items(), *[(s, open_m[s]) for s in strongest if s in open_m]]]
    for r in out["contested_split"]:
        print("contested", r["machine"], round(r["all_stories"]["phi"], 1), round(r["same_votes"]["phi"], 1),
              round(r["split"]["phi"], 1), [round(x, 1) for x in r["split"]["phi_ci"]], flush=True)

    Path("results").mkdir(exist_ok=True)
    Path("results/difficulty.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist()))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
