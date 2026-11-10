"""E, the law at full evidence: ten networks in pairs, each with the probabilities of the 16 categories.

Protocol, "The instrument on trial", section E. For each pair of networks and each condition of
model-vs-human (without the cue-conflict) the weaker acts as mind A and the stronger as machine B:
d′ from the accuracies with 16 options, ρ with the article's instrument (tetrachoric of the
correct answers, converted for m = 16), the prediction |S| with the measured angle and at 90 degrees,
and the sensitivity of a full-evidence combiner that adds the logarithms of the two probabilities
with two learned weights, evaluated on unseen images. Writes results/full-evidence.json.

Usage, from the experiment folder:  python -m angle.full_evidence GPU_RESULTS_DIR [--summarize]
With --summarize it redoes only the summaries, on the units already written in the file (the whole
run takes about 20 minutes with two processes).
"""
from __future__ import annotations

import ast
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest, spearmanr
from sklearn.model_selection import KFold, StratifiedKFold

from . import sdt
from .analysis_vision import BASE, load_network, load_humans, image_key

M = 16
EPS = 1e-5                      # the step of the saved probabilities (5 decimals): a saved 0 is worth less than 5e-6
EPS_PROBE = (1e-6, 1e-4)        # sensitivity to the choice of ε
FOLDS = 5
SEED = 534
CEILING = 0.995
MIN_IMAGES = 100
RIDGE = 1e-6                    # keeps the optimum finite when the training images are fully separated
WEIGHT_MAX = 1e3
RESAMPLES = 4000
CONV_GRID = np.linspace(-0.3, 0.95, 26)                         # the one of sdt.convert_rho_mafc
EXTENDED_GRID = np.r_[CONV_GRID, 0.97, 0.98, 0.99, 0.995]       # only for the sensitivity analysis
N_CONV = 120_000                                                 # as sdt.convert_rho_mafc
MIND_2014 = "vgg16"
CATEGORIES = sorted(["knife", "keyboard", "elephant", "bicycle", "airplane", "clock", "oven", "chair", "bear", "boat",
                     "cat", "bottle", "truck", "car", "bird", "dog"])

# degradation families: the level is read from the name of the condition; the key grows with the difficulty
FAMILIES = {
    "contrast": lambda c: -float(c.lstrip("c")),              # c100 ... c01: less contrast, harder
    "uniform-noise": float,                                    # amplitude of the noise
    "low-pass": float,                                         # standard deviation of the blur
    "high-pass": lambda c: -float(c),                          # inf = whole image; smaller σ, less is left
    "phase-scrambling": float,                                 # degrees of scrambled phase
    "eidolonI": lambda c: float(c.split("-")[0]),              # reach 1 ... 128
    "eidolonII": lambda c: float(c.split("-")[0]),
    "eidolonIII": lambda c: float(c.split("-")[0]),
}
UNORDERED = {
    "rotation": "0, 90, 180, 270 degrees: 90 and 270 are mirror images, the difficulty order cannot be derived from the name",
    "colour": "two levels", "false-colour": "two levels", "power-equalisation": "two levels",
    "edge": "one level", "silhouette": "one level", "sketch": "one level", "stylized": "one level",
}


# --- data -----------------------------------------------------------------------

def years() -> dict[str, int]:
    """Name and year of the networks, read from gpu/machines_vision.py without running it (importing it would need torch)."""
    text = (Path(__file__).resolve().parent.parent / "gpu" / "machines_vision.py").read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "MODELS":
            return {k: int(v[1]) for k, v in ast.literal_eval(node.value).items()}
    raise ValueError("MODELS not found in gpu/machines_vision.py")


def categories(folder: Path) -> list[str]:
    """The order of the 16 probabilities, with the same rule as analysis_vision."""
    f = Path(folder).parent / "categories.json"
    return json.load(open(f))["categories"] if f.exists() else list(CATEGORIES)


def load(folder: Path) -> dict:
    """Each condition with its images: correct category and probabilities of each network, in the same order.

    Condition and category of an image come from the people's files, through the same `image_key` of
    analysis_vision: the people saw the same images as the networks, with another prefix.
    """
    index = {c: i for i, c in enumerate(categories(folder))}
    networks = [n for n in years() if (folder / f"vision__{n}.jsonl").exists()]
    datasets = sorted(p.name for p in BASE.iterdir() if p.is_dir() and "cue" not in p.name)
    mapping = {}
    for ds in datasets:
        d = load_humans(ds)
        d["key"] = d.imagename.map(lambda n, ds=ds: image_key(n, ds))
        g = d.groupby("key").agg(n_cond=("condition", "nunique"), n_cat=("category", "nunique"),
                                 cond=("condition", "first"), categ=("category", "first"))
        if (g.n_cond > 1).any() or (g.n_cat > 1).any():
            raise ValueError(f"{ds}: the same image with more than one condition or category")
        for k, r in g.iterrows():
            mapping[(ds, k)] = (str(r["cond"]), index[r["categ"]])
    raw: dict = {}
    for name in networks:
        for r in load_network(folder, name):
            ds = r["rel"].split("/")[0]
            if ds not in datasets:                     # cue-conflict
                continue
            cond, y = mapping[(ds, image_key(r["img"], ds))]
            c = raw.setdefault((ds, cond), {})
            c.setdefault(r["rel"], {"y": y, "p": {}})["p"][name] = r["prob"]
    conditions = {}
    for (ds, cond), c in sorted(raw.items()):
        rel = sorted(c)
        if any(len(c[x]["p"]) != len(networks) for x in rel):
            raise ValueError(f"{ds}/{cond}: images that not all networks saw")
        y = np.array([c[x]["y"] for x in rel], int)
        P = np.stack([np.array([c[x]["p"][n] for x in rel], float) for n in networks])
        conditions[(ds, cond)] = {"y": y, "P": P}
    return {"networks": networks, "conditions": conditions}


# --- ρ: tetrachoric of the correct answers, converted for m = 16 ---------------------

class Conversion:
    """`sdt.convert_rho_mafc` with the same random draws, computed only once.

    `sdt.rho_correct_from_rho_evidence` draws with seed 534 three n × m matrices of normals and counts the
    answer as correct when option 0, shifted by d, is the highest. The part that does not depend on d (option 0
    before the shift and the maximum of the others) is computed here once for each ρ of the grid;
    the correct answer is then `A0 + d >= max of the others`, the same floating-point operation as
    `a[:, 0] += d; a.argmax(1) == 0`. The result coincides with `sdt.convert_rho_mafc`, without
    rounding the sensitivities: the run and the tests check it.
    """

    def __init__(self, grid=EXTENDED_GRID, m: int = M, n: int = N_CONV, seed: int = 534):
        gen = np.random.default_rng(seed)
        common = gen.standard_normal((n, m))
        z1 = gen.standard_normal((n, m))
        z2 = gen.standard_normal((n, m))
        self.grid = np.asarray(grid, float)
        self.n_base = int(np.sum(np.isin(self.grid, CONV_GRID)))
        self._tab = []
        for x in self.grid:
            rho = float(x)
            if abs(rho) < 1e-9:
                a, b = z1, z2
            else:
                w = math.sqrt(abs(rho))
                sign = 1.0 if rho >= 0 else -1.0
                a = w * common + math.sqrt(1 - abs(rho)) * z1
                b = sign * w * common + math.sqrt(1 - abs(rho)) * z2
            self._tab.append((a[:, 0].copy(), a[:, 1:].max(1), b[:, 0].copy(), b[:, 1:].max(1)))

    def table(self, d_a: float, d_b: float, extended: bool = False) -> np.ndarray:
        """The tetrachoric of the correct answers for each ρ of the evidence on the grid."""
        count = len(self._tab) if extended else self.n_base
        t = np.empty(count)
        for i, (a0, ma, b0, mb) in enumerate(self._tab[:count]):
            oa = (a0 + d_a) >= ma
            ob = (b0 + d_b) >= mb
            t[i] = sdt.tetrachoric(float(oa.mean()), float(ob.mean()), float((oa & ob).mean()))
        return t

    def __call__(self, r_correct: float, d_a: float, d_b: float, extended: bool = False) -> dict:
        t = self.table(d_a, d_b, extended)
        g = self.grid[: t.size]
        limit = "high" if r_correct >= t[-1] else ("low" if r_correct <= t[0] else None)
        return {"rho": float(np.interp(r_correct, t, g)), "limit": limit, "monotonic": bool(np.all(np.diff(t) > 0))}


# --- the full-evidence combiner -------------------------------------------------------

def _nll(theta, la, lb, y, ridge):
    s = theta[0] * la + theta[1] * lb
    mx = s.max(1, keepdims=True)
    e = np.exp(s - mx)
    z = e.sum(1, keepdims=True)
    p = e / z
    r = np.arange(y.size)
    lse = mx[:, 0] + np.log(z[:, 0])
    f = -np.mean(s[r, y] - lse) + ridge * float(theta @ theta)
    ga = -np.mean(la[r, y] - (p * la).sum(1)) + 2 * ridge * theta[0]
    gb = -np.mean(lb[r, y] - (p * lb).sum(1)) + 2 * ridge * theta[1]
    return f, np.array([ga, gb])


def fit_weights(la: np.ndarray, lb: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """a, b ≥ 0 that maximize the multinomial likelihood of softmax(a·la + b·lb)."""
    res = minimize(_nll, x0=np.ones(2), args=(la, lb, y, ridge), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, WEIGHT_MAX)] * 2)
    return res.x


def correct_with_ties(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correct answer per image; if several categories tie at the maximum, the correct one counts 1/how many they are."""
    ties = s == s.max(1, keepdims=True)
    return ties[np.arange(y.size), y] / ties.sum(1)


def cv_split(y: np.ndarray, folds: int = FOLDS, seed: int = SEED):
    """The folds of the cross-validation, stratified by category when every category has enough images."""
    count = np.bincount(y)
    if count[count > 0].min() >= folds:
        return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(y.size), y))
    return list(KFold(folds, shuffle=True, random_state=seed).split(np.zeros(y.size)))


def cv_combiner(pa: np.ndarray, pb: np.ndarray, y: np.ndarray, eps: float = EPS,
                folds: int = FOLDS, seed: int = SEED) -> dict:
    """Out-of-sample accuracy of the score a·log(pA + ε) + b·log(pB + ε), weights learned on the other folds."""
    la, lb = np.log(pa + eps), np.log(pb + eps)
    correct = np.empty(y.size)
    weights = []
    for tr, te in cv_split(y, folds, seed):
        a, b = fit_weights(la[tr], lb[tr], y[tr])
        weights.append((a, b))
        correct[te] = correct_with_ties(a * la[te] + b * lb[te], y[te])
    weights = np.array(weights)
    tot = weights.sum(1)
    return {"acc": float(correct.mean()), "a": float(weights[:, 0].mean()), "b": float(weights[:, 1].mean()),
            "share_b": float(np.mean(np.where(tot > 0, weights[:, 1] / np.where(tot > 0, tot, 1), np.nan))),
            "weights_at_limit": int(np.sum(weights.max(1) >= 0.999 * WEIGHT_MAX)),
            "b_zero": int(np.sum(weights[:, 1] <= 1e-8)), "a_zero": int(np.sum(weights[:, 0] <= 1e-8))}


# --- a unit: a pair of networks in a condition ----------------------------------------

def choose_mind(names, acc, p_correct) -> tuple[int, int]:
    """(mind, machine): the mind is the less accurate; at equal accuracy the one less sure of the
    correct answer on average, then the name. It does not depend on the order in which the two networks arrive."""
    k0 = (acc[0], p_correct[0], names[0])
    k1 = (acc[1], p_correct[1], names[1])
    return (0, 1) if k0 <= k1 else (1, 0)


def _d(a: float) -> float:
    return sdt.d_from_accuracy(float(a), M)


def pair_unit(name1: str, p1: np.ndarray, name2: str, p2: np.ndarray, y: np.ndarray, conv: Conversion,
              eps_probe=EPS_PROBE) -> dict:
    rows = np.arange(y.size)
    acc = [float((p.argmax(1) == y).mean()) for p in (p1, p2)]
    pg = [float(p[rows, y].mean()) for p in (p1, p2)]
    ia, ib = choose_mind((name1, name2), acc, pg)
    pa, pb = (p1, p2)[ia], (p1, p2)[ib]
    ok_a = (pa.argmax(1) == y).astype(float)
    ok_b = (pb.argmax(1) == y).astype(float)
    combo = cv_combiner(pa, pb, y)
    u = {"mind": (name1, name2)[ia], "machine": (name1, name2)[ib], "images": int(y.size),
         "acc_A": acc[ia], "acc_B": acc[ib], "acc_pair": combo["acc"],
         "weight_a": combo["a"], "weight_b": combo["b"], "share_b": combo["share_b"],
         "weights_at_limit": combo["weights_at_limit"], "b_zero": combo["b_zero"], "a_zero": combo["a_zero"],
         "pA_correct_zero": float(np.mean(pa[rows, y] == 0.0)), "pB_correct_zero": float(np.mean(pb[rows, y] == 0.0)),
         "p_A_above_chance": float(binomtest(int(ok_a.sum()), int(y.size), 1.0 / M, alternative="greater").pvalue)}
    for e in eps_probe:
        u[f"acc_pair_eps{e:g}"] = cv_combiner(pa, pb, y, eps=e)["acc"]
    if y.size < MIN_IMAGES:
        u["state"] = "few images"
    elif min(acc) <= 1.0 / M:
        u["state"] = "at chance level"
    elif max(u["acc_A"], u["acc_B"], u["acc_pair"]) > CEILING:
        u["state"] = "at ceiling"
    elif u["acc_pair"] <= 1.0 / M:
        # A and B just above chance, the pair out of sample not: dS is not estimated. It is counted
        # apart and, in the sensitivity analysis, it comes back with dS = 0
        u["state"] = "pair at chance level"
    else:
        u["state"] = "valid"
    if u["state"] not in ("valid", "pair at chance level"):
        return u
    d_a, d_b = _d(u["acc_A"]), _d(u["acc_B"])
    d_s = _d(u["acc_pair"]) if u["state"] == "valid" else 0.0
    r_g = sdt.rho_machines(ok_a, ok_b)
    c = conv(r_g, d_a, d_b)
    rho = c["rho"]
    i_phi, i_90 = sdt.ideal(d_a, d_b, rho), sdt.ideal(d_a, d_b, 0.0)
    u.update(dA=d_a, dB=d_b, k=d_b / d_a, rho_correct=r_g, rho=rho, rho_limit=c["limit"], table_monotonic=c["monotonic"],
             phi=sdt.angle_degrees(rho), useful_angle=sdt.useful_angle(d_a, d_b), ideal_phi=i_phi, ideal_90=i_90,
             apparent_turings_phi=sdt.apparent_turings(d_a, d_b, rho), apparent_turings_90=sdt.apparent_turings(d_a, d_b, 0.0),
             dS=d_s, turing=d_s / d_a, eta=d_s / i_phi, eta_90=d_s / i_90,
             err_phi=abs(d_s - i_phi), err_90=abs(d_s - i_90))
    u["below_useful_angle"] = bool(u["phi"] < u["useful_angle"])
    ce = conv(r_g, d_a, d_b, extended=True)
    u.update(rho_extended=ce["rho"], rho_extended_limit=ce["limit"])
    u["rho_rounded"] = conv(r_g, round(d_a, 1), round(d_b, 1))["rho"]
    for e in eps_probe:
        a = u[f"acc_pair_eps{e:g}"]
        u[f"dS_eps{e:g}"] = _d(a) if 1.0 / M < a <= CEILING else None
    return u


_CONV: Conversion | None = None


def _init():
    global _CONV
    _CONV = Conversion()


def analyze_condition(args) -> list[dict]:
    ds, cond, networks, year, y, P = args
    out = []
    for i, j in combinations(range(len(networks)), 2):
        u = pair_unit(networks[i], P[i], networks[j], P[j], y, _CONV)
        u.update(dataset=ds, condition=cond, year_mind=year[u["mind"]], year_machine=year[u["machine"]])
        out.append(u)
    return out


# --- summaries -----------------------------------------------------------------------

def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.median(np.repeat(x, w.astype(int))))


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:      # a constant column: not defined
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _clusters(groups, repetitions: int, seed: int):
    """Weights of the units for each resample of the conditions (cluster bootstrap)."""
    cid = pd.factorize(pd.Series(groups))[0]
    c = int(cid.max()) + 1
    gen = np.random.default_rng(seed)
    for _ in range(repetitions):
        yield np.bincount(gen.integers(0, c, c), minlength=c)[cid].astype(float)


def summarize(df: pd.DataFrame, repetitions: int = RESAMPLES, seed: int = SEED) -> dict:
    """The criteria of E and the counts, on the valid units of `df` (columns dS, ideal_phi, ideal_90, ...)."""
    if df.empty:
        return {"units": 0}
    e_phi = np.abs(df.dS - df.ideal_phi).to_numpy()
    e_90 = np.abs(df.dS - df.ideal_90).to_numpy()
    eta = (df.dS / df.ideal_phi).to_numpy()
    groups = (df.dataset + "|" + df.condition).to_numpy()
    diff, med = [], []
    for w in _clusters(groups, repetitions, seed):
        diff.append(float(np.sum(w * (e_phi - e_90)) / w.sum()))
        med.append(_weighted_median(eta, w))
    ci_diff = [float(x) for x in np.percentile(diff, [2.5, 97.5])]
    ci_med = [float(x) for x in np.percentile(med, [2.5, 97.5])]
    lo = np.minimum(df.ideal_phi, df.ideal_90).to_numpy()
    hi = np.maximum(df.ideal_phi, df.ideal_90).to_numpy()
    ds = df.dS.to_numpy()
    between = (ds > lo) & (ds < hi)
    equal = e_phi < 1e-9
    turing = (df.dS / df.dA).to_numpy()
    below = (df.phi < df.useful_angle).to_numpy()
    k = df.k.to_numpy()
    g_meas = (df.dS / df.dB).to_numpy()                 # the pair in the unit of the machine: gain over B
    g_phi = (df.ideal_phi / df.dB).to_numpy()
    g_90 = (df.ideal_90 / df.dB).to_numpy()
    out = {
        "units": int(len(df)), "conditions": int(len(set(groups))),
        "mae_phi": float(e_phi.mean()), "mae_90": float(e_90.mean()),
        "difference": float(e_phi.mean() - e_90.mean()), "difference_ci95": ci_diff,
        "criterion_1_passes": bool(ci_diff[1] < 0),
        "closer_phi": int(np.sum(e_phi < e_90)), "closer_90": int(np.sum(e_90 < e_phi)),
        "ties": int(np.sum(e_phi == e_90)),
        "between_the_two": int(between.sum()), "between_the_two_closer_phi": int(np.sum(between & (e_phi < e_90))),
        "equal_to_prediction_phi": int(equal.sum()),
        "below_both": int(np.sum((ds < lo) & ~equal)),
        "above_both": int(np.sum((ds > hi) & ~equal)),
        "equal_predictions": int(np.sum(lo == hi)),
        "rho_negative": int(np.sum(df.rho < 0)),
        "eta_median": float(np.median(eta)), "eta_q1": float(np.percentile(eta, 25)), "eta_q3": float(np.percentile(eta, 75)),
        "eta_median_ci95": ci_med,
        "criterion_2_passes": bool(0.9 <= ci_med[0] and ci_med[1] <= 1.1),
        "criterion_2_point_estimate_inside": bool(0.9 <= float(np.median(eta)) <= 1.1),
        "eta_above_1_1": int(np.sum(eta > 1.1)), "eta_below_0_9": int(np.sum(eta < 0.9)),
        "eta_90_median": float(np.median(df.dS / df.ideal_90)),
        "dS_equal_dB": int(np.sum(np.abs(ds - df.dB.to_numpy()) < 1e-12)),
        "dS_above_dB": int(np.sum(ds > df.dB.to_numpy() + 1e-12)), "dS_below_dB": int(np.sum(ds < df.dB.to_numpy() - 1e-12)),
        "spearman_turing_apparent_turings_phi": _spearman(turing, df.apparent_turings_phi),
        "spearman_turing_apparent_turings_90": _spearman(turing, df.apparent_turings_90),
        "spearman_turing_k": _spearman(turing, k),
        # net of k: does the pair beat the machine where the law says it should?
        "gain_over_B_measured_median": float(np.median(g_meas)),
        "gain_over_B_predicted_phi_median": float(np.median(g_phi)),
        "gain_over_B_predicted_90_median": float(np.median(g_90)),
        "spearman_gain_over_B_phi": _spearman(g_meas, g_phi),
        "spearman_gain_over_B_90": _spearman(g_meas, g_90),
        "pair_beyond_B_by_5_percent": int(np.sum(g_meas > 1.05)),
        "below_useful_angle": int(below.sum()),
        # the combiner that gives weight zero to A in every fold is B alone: below the useful angle it coincides with |S|(φ) = dB
        "combiner_reduced_to_B": int(np.sum(df.a_zero == FOLDS)) if "a_zero" in df else None,
        "phi_error_zero": int(np.sum(e_phi < 1e-9)),
    }
    if below.any():
        out.update(below_ua_turing_median=float(np.median(turing[below])), below_ua_k_median=float(np.median(k[below])),
                   below_ua_turing_over_k_median=float(np.median(turing[below] / k[below])),
                   below_ua_turing_above_k=int(np.sum(turing[below] > k[below])))
    if (~below).any():
        out.update(above_ua_turing_median=float(np.median(turing[~below])),
                   above_ua_turing_over_k_median=float(np.median(turing[~below] / k[~below])),
                   above_ua_eta_median=float(np.median(eta[~below])),
                   above_ua_gain_over_B_measured_median=float(np.median(g_meas[~below])),
                   above_ua_gain_over_B_predicted_phi_median=float(np.median(g_phi[~below])),
                   above_ua_gain_over_B_predicted_90_median=float(np.median(g_90[~below])),
                   above_ua_spearman_gain_over_B_phi=_spearman(g_meas[~below], g_phi[~below]),
                   above_ua_spearman_gain_over_B_90=_spearman(g_meas[~below], g_90[~below]))
    return out


def _valid(units_: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([u for u in units_ if u["state"] == "valid"])


def sensitivity_analysis(units_: list[dict], vision_conditions: set | None) -> dict:
    """Sensitivity analyses, exploratory: added here, not in the protocol."""
    df = _valid(units_)
    out = {}
    if vision_conditions:
        keep = [(a, b) in vision_conditions for a, b in zip(df.dataset, df.condition)]
        out["only_vision_json_conditions"] = summarize(df[keep])
    out["only_A_above_chance_p05"] = summarize(df[df.p_A_above_chance < 0.05])
    out["without_rho_at_limit"] = summarize(df[df.rho_limit.isna()])
    out["only_above_useful_angle"] = summarize(df[df.phi >= df.useful_angle])
    out["without_combiner_reduced_to_B"] = summarize(df[df.a_zero < FOLDS])
    # whoever copies the machine (dS = dB): do the two criteria tell it apart from the combiner? (deviation of 9 November on the turing
    # read from the machine's side)
    copied = df.copy()
    copied["dS"] = copied.dB
    out["machine_copy"] = summarize(copied)
    out["machine_copy_above_useful_angle"] = summarize(copied[copied.phi >= copied.useful_angle])
    out["with_chance_pairs_at_dS_0"] = summarize(pd.DataFrame(
        [u for u in units_ if u["state"] in ("valid", "pair at chance level")]))
    ext = df.copy()
    ext["rho"] = ext.rho_extended
    ext["ideal_phi"] = [sdt.ideal(a, b, r) for a, b, r in zip(ext.dA, ext.dB, ext.rho)]
    ext["phi"] = [sdt.angle_degrees(r) for r in ext.rho]
    ext["apparent_turings_phi"] = ext.ideal_phi / ext.dA
    out["conversion_grid_up_to_0_995"] = summarize(ext)
    for e in EPS_PROBE:
        c = f"dS_eps{e:g}"
        x = df[df[c].notna()].copy()
        x["dS"] = x[c].astype(float)
        out[f"epsilon_{e:g}"] = summarize(x)
        out[f"epsilon_{e:g}"]["extra_units_at_ceiling"] = int(df[c].isna().sum())
    return out


def mind_2014(units_: list[dict], year: dict) -> dict:
    """VGG16 always the mind, the other nine machines: medians over the valid conditions."""
    rows, strong = [], 0
    for u in units_:
        if MIND_2014 not in (u["mind"], u["machine"]):
            continue
        other = u["machine"] if u["mind"] == MIND_2014 else u["mind"]
        vgg_mind = u["mind"] == MIND_2014
        r = {"machine": other, "dataset": u["dataset"], "condition": u["condition"], "state": u["state"],
             "vgg_stronger": not vgg_mind}
        if u["state"] == "valid":
            d_v = u["dA"] if vgg_mind else u["dB"]
            d_n = u["dB"] if vgg_mind else u["dA"]
            acc_n = u["acc_B"] if vgg_mind else u["acc_A"]
            r.update(k=d_n / d_v, phi=u["phi"], apparent_turings=u["ideal_phi"] / d_v, turing=u["dS"] / d_v, eta=u["eta"],
                     over_machine=u["dS"] / d_n, acc_over_machine=u["acc_pair"] - acc_n)
        rows.append(r)
    t = pd.DataFrame(rows)
    v = t[t.state == "valid"]
    tab = []
    for name in [n for n in year if n != MIND_2014]:
        x = v[v.machine == name]
        every = t[t.machine == name]
        tab.append({"machine": name, "year": year[name], "conditions": int(len(x)),
                    "k": float(x.k.median()), "phi": float(x.phi.median()), "apparent_turings": float(x.apparent_turings.median()),
                    "turing": float(x.turing.median()), "eta": float(x.eta.median()),
                    "dS_over_d_machine": float(x.over_machine.median()),
                    "pair_better_than_machine": int((x.acc_over_machine > 0).sum()),
                    "pair_worse_than_machine": int((x.acc_over_machine < 0).sum()),
                    "vgg_stronger_valid": int(x.vgg_stronger.sum()), "vgg_stronger_all": int(every.vgg_stronger.sum()),
                    "at_ceiling": int((every.state == "at ceiling").sum()), "at_chance": int((every.state == "at chance level").sum())})
        strong += int(x.vgg_stronger.sum())
    return {"table": tab, "vgg_stronger_in_valid_conditions": strong,
            "vgg_stronger_in_all": int(t.vgg_stronger.sum())}


def families(units_: list[dict], data: dict, strongest: str) -> dict:
    """For the pair VGG16 + the strongest network, level by level inside each orderable family."""
    networks = data["networks"]
    out = {"strongest_network": strongest, "unordered": UNORDERED, "families": {}}
    pair_map = {(u["dataset"], u["condition"]): u for u in units_ if {u["mind"], u["machine"]} == {MIND_2014, strongest}}
    for fam, key_fn in FAMILIES.items():
        levels = sorted([c for (d, c) in data["conditions"] if d == fam], key=key_fn)
        if len(levels) < 4:
            continue
        rows = []
        means = []
        for c in levels:
            u = pair_map[(fam, c)]
            acc_v = u["acc_A"] if u["mind"] == MIND_2014 else u["acc_B"]
            acc_f = u["acc_B"] if u["mind"] == MIND_2014 else u["acc_A"]
            r = {"level": c, "state": u["state"], "acc_A": acc_v, "acc_B": acc_f, "acc_pair": u["acc_pair"],
                 "gain": u["acc_pair"] - acc_v, "gain_over_B": u["acc_pair"] - acc_f, "vgg_mind": u["mind"] == MIND_2014}
            if u["state"] == "valid":
                d_v = u["dA"] if u["mind"] == MIND_2014 else u["dB"]
                d_f = u["dB"] if u["mind"] == MIND_2014 else u["dA"]
                r.update(k=d_f / d_v, phi=u["phi"], turing=u["dS"] / d_v, apparent_turings=u["ideal_phi"] / d_v, eta=u["eta"])
            rows.append(r)
            y, P = data["conditions"][(fam, c)]["y"], data["conditions"][(fam, c)]["P"]
            means.append(float(np.mean([(P[i].argmax(1) == y).mean() for i in range(len(networks))])))
        rho_order = float(spearmanr(np.arange(len(levels)), means).statistic)
        out["families"][fam] = {"levels": rows, "mean_accuracy_of_networks": means,
                                "spearman_order_accuracy": rho_order}
    return out


def checks(units_: list[dict], data: dict, conv: Conversion) -> dict:
    """That the fast conversion coincides with sdt.convert_rho_mafc; how much rounding d to 0.1 would move ρ;
    how saturated the probabilities are."""
    df = _valid(units_)
    picked = df.iloc[np.linspace(0, len(df) - 1, 6).astype(int)]
    comparison = []
    for _, u in picked.iterrows():
        slow = sdt.convert_rho_mafc(u.rho_correct, u.dA, u.dB, M)
        comparison.append({"dataset": u.dataset, "condition": u.condition, "mind": u.mind, "machine": u.machine,
                           "rho_fast": float(u.rho), "rho_sdt": float(slow), "difference": float(u.rho - slow)})
    arr = (df.rho_rounded - df.rho).abs()
    # sdt.convert_rho_mafc interpolates without making the table monotonic (measure.conversion_table does):
    # how much ρ and the prediction would change if it were made monotonic
    mono, example = [], None
    for _, u in df[~df.table_monotonic.astype(bool)].iterrows():
        t = conv.table(u.dA, u.dB)
        rm = float(np.interp(u.rho_correct, np.maximum.accumulate(t), CONV_GRID))
        mono.append((rm - u.rho, sdt.ideal(u.dA, u.dB, rm) - u.ideal_phi, int(np.sum(np.diff(t) < 0))))
        if example is None and abs(rm - u.rho) > 0.02:
            example = {"dataset": u.dataset, "condition": u.condition, "mind": u.mind, "machine": u.machine,
                       "r_correct": float(u.rho_correct), "dA": float(u.dA), "dB": float(u.dB),
                       "rho_convert_rho_mafc": float(u.rho), "rho_monotonic_table": rm,
                       "last_three_table_values": [float(x) for x in t[-3:]]}
    mono = np.array(mono, float).reshape(-1, 3)
    sat = {}
    for i, name in enumerate(data["networks"]):
        one = sum(int(np.sum(c["P"][i].max(1) >= 0.99999)) for c in data["conditions"].values())
        zero_c = sum(int(np.sum(c["P"][i][np.arange(c["y"].size), c["y"]] == 0.0)) for c in data["conditions"].values())
        tied = sum(int(np.sum((c["P"][i] == c["P"][i].max(1, keepdims=True)).sum(1) > 1)) for c in data["conditions"].values())
        tot = sum(c["y"].size for c in data["conditions"].values())
        sat[name] = {"images": tot, "max_at_least_0_99999": one, "correct_at_zero": zero_c, "tied_at_max": tied}
    return {
        "fast_conversion_vs_sdt": comparison,
        "rounding_d_to_0_1": {"units": int(len(arr)), "max_abs_delta_rho": float(arr.max()),
                              "p95_abs_delta_rho": float(np.percentile(arr, 95)), "beyond_0_02": int(np.sum(arr > 0.02))},
        "rho_at_limit": {"high_0_95": int(np.sum(df.rho_limit == "high")), "low_-0_3": int(np.sum(df.rho_limit == "low")),
                         "high_with_extended_grid_0_995": int(np.sum(df.rho_extended_limit == "high"))},
        "rho_correct_at_grid_edge": int(np.sum(df.rho_correct >= sdt.RHO_GRID[-1] - 1e-9)),
        "non_monotonic_tables": {"units": int(len(mono)), "with_decreasing_steps": int(np.sum(mono[:, 2] > 0)),
                                 "rho_shifted_beyond_0_02": int(np.sum(np.abs(mono[:, 0]) > 0.02)),
                                 "max_abs_delta_rho": float(np.abs(mono[:, 0]).max()) if len(mono) else 0.0,
                                 "max_abs_delta_prediction_phi": float(np.abs(mono[:, 1]).max()) if len(mono) else 0.0,
                                 "example": example},
        "combiner": {"folds_with_weights_at_limit": int(df.weights_at_limit.sum()), "folds_with_b_zero": int(df.b_zero.sum()),
                     "folds_with_a_zero": int(df.a_zero.sum()),
                     "units_with_a_zero_in_every_fold": int(np.sum(df.a_zero == FOLDS))},
        "saturation": sat,
    }


def per_dataset(units_: list[dict]) -> list[dict]:
    df = _valid(units_)
    out = []
    for ds, x in df.groupby("dataset"):
        out.append({"dataset": ds, "units": int(len(x)), "mae_phi": float(x.err_phi.mean()), "mae_90": float(x.err_90.mean()),
                    "eta_median": float(x.eta.median()), "phi_median": float(x.phi.median())})
    return out


# --- run ------------------------------------------------------------------------------

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _f(x, n=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def print_report(r: dict) -> None:
    s = r["summary"]
    c = r["counts"]
    print("\nE, the law at full evidence")
    print(f"  units: {c['total']} = {c['valid']} valid + {c['at ceiling']} at ceiling + {c['at chance level']} at chance"
          f" + {c['pair at chance level']} with the pair at chance + {c['few images']} with fewer than {MIN_IMAGES} images; conditions with valid units: {s['conditions']}")
    print(f"  mean absolute error: with φ {_f(s['mae_phi'])}, at 90° {_f(s['mae_90'])}; difference {_f(s['difference'])}"
          f" [{_f(s['difference_ci95'][0])}, {_f(s['difference_ci95'][1])}] -> criterion 1 {'PASSES' if s['criterion_1_passes'] else 'FAILS'}")
    print(f"  closer with φ in {s['closer_phi']} units, at 90° in {s['closer_90']}, ties {s['ties']}")
    print(f"  dS between the two predictions: {s['between_the_two']} units, closer to the one with φ in {s['between_the_two_closer_phi']};"
          f" equal to the one with φ {s['equal_to_prediction_phi']}, below both {s['below_both']}, above both {s['above_both']}")
    print(f"  η = dS/|S|(φ): median {_f(s['eta_median'])} [{_f(s['eta_median_ci95'][0])}, {_f(s['eta_median_ci95'][1])}],"
          f" quartiles {_f(s['eta_q1'])}-{_f(s['eta_q3'])}; > 1.1: {s['eta_above_1_1']}, < 0.9: {s['eta_below_0_9']}"
          f" -> criterion 2 {'PASSES' if s['criterion_2_passes'] else 'FAILS'}")
    print(f"  dS versus dB: equal in {s['dS_equal_dB']}, above in {s['dS_above_dB']}, below in {s['dS_below_dB']}")
    print(f"  combiner reduced to B (a = 0 in every fold): {s['combiner_reduced_to_B']} units; zero error with φ: {s['phi_error_zero']}")
    print(f"  Spearman turing / apparent turings: with φ {_f(s['spearman_turing_apparent_turings_phi'])}, at 90° {_f(s['spearman_turing_apparent_turings_90'])};"
          f" turing / k {_f(s['spearman_turing_k'])}")
    print(f"  gain over B (dS/dB): measured {_f(s['gain_over_B_measured_median'])}, predicted with φ {_f(s['gain_over_B_predicted_phi_median'])},"
          f" at 90° {_f(s['gain_over_B_predicted_90_median'])}; Spearman measured-predicted with φ {_f(s['spearman_gain_over_B_phi'])},"
          f" at 90° {_f(s['spearman_gain_over_B_90'])}; beyond +5%: {s['pair_beyond_B_by_5_percent']}")
    print(f"  above the useful angle: η median {_f(s['above_ua_eta_median'])}, gain over B measured {_f(s['above_ua_gain_over_B_measured_median'])}"
          f" predicted with φ {_f(s['above_ua_gain_over_B_predicted_phi_median'])}, at 90° {_f(s['above_ua_gain_over_B_predicted_90_median'])};"
          f" Spearman with φ {_f(s['above_ua_spearman_gain_over_B_phi'])}, at 90° {_f(s['above_ua_spearman_gain_over_B_90'])}")
    if s.get("below_useful_angle"):
        print(f"  below the useful angle: {s['below_useful_angle']} units; turing median {_f(s['below_ua_turing_median'])},"
              f" k median {_f(s['below_ua_k_median'])}, turing/k median {_f(s['below_ua_turing_over_k_median'])},"
              f" turing > k in {s['below_ua_turing_above_k']}")
    print("\n  A mind of 2014 (VGG16)")
    print(f"  {'machine':16s} year  cond     k      φ  a.tur turing     η dS/dM  better/worse  VGG stronger")
    for t in r["mind_2014"]["table"]:
        print(f"  {t['machine']:16s} {t['year']} {t['conditions']:5d} {_f(t['k'], 2):>6s} {_f(t['phi'], 1):>6s}"
              f" {_f(t['apparent_turings'], 2):>6s} {_f(t['turing'], 2):>6s} {_f(t['eta'], 3):>5s} {_f(t['dS_over_d_machine'], 3)}"
              f"  {t['pair_better_than_machine']:3d}/{t['pair_worse_than_machine']:<3d}  {t['vgg_stronger_valid']}")
    print(f"  VGG16 stronger than the machine in {r['mind_2014']['vgg_stronger_in_valid_conditions']} valid conditions")
    print(f"\n  Families, VGG16 + {r['families']['strongest_network']}")
    for fam, x in r["families"]["families"].items():
        print(f"  {fam} (Spearman order-mean accuracy {_f(x['spearman_order_accuracy'], 2)})")
        for v in x["levels"]:
            print(f"    {v['level']:>10s} A {_f(v['acc_A'], 3)} B {_f(v['acc_B'], 3)} pair {_f(v['acc_pair'], 3)}"
                  f" gain {_f(v['gain'], 3):>6s} turing {_f(v.get('turing'), 2):>5s} {v['state']}")
    print("\n  Sensitivity (exploratory)")
    for name, x in r["sensitivity_analysis"].items():
        if x.get("units", 0) == 0:
            print(f"    {name}: no units")
            continue
        print(f"    {name}: n {x['units']}, difference {_f(x['difference'])} [{_f(x['difference_ci95'][0])}, {_f(x['difference_ci95'][1])}],"
              f" η median {_f(x['eta_median'])} [{_f(x['eta_median_ci95'][0])}, {_f(x['eta_median_ci95'][1])}]")


def main(folder: Path, summary_only: bool = False) -> dict:
    year = years()
    data = load(folder)
    networks = data["networks"]
    output = Path("results/full-evidence.json")
    if summary_only:                   # the units of the previous run, without recomputing them (they are deterministic)
        units_ = json.loads(output.read_text())["units"]
        print(len(units_), "units read from", output, flush=True)
    else:
        tasks = [(ds, cond, networks, year, c["y"], c["P"]) for (ds, cond), c in data["conditions"].items()]
        print(len(networks), "networks,", len(tasks), "conditions,", len(tasks) * len(networks) * (len(networks) - 1) // 2, "units", flush=True)
        units_ = []
        with ProcessPoolExecutor(max_workers=2, initializer=_init) as ex:
            for i, out in enumerate(ex.map(analyze_condition, tasks)):
                units_ += out
                if (i + 1) % 10 == 0:
                    print(i + 1, "conditions", flush=True)
    conv = Conversion()
    mean_acc = {n: float(np.mean([(c["P"][i].argmax(1) == c["y"]).mean() for c in data["conditions"].values()]))
                for i, n in enumerate(networks)}
    strongest = max(mean_acc, key=mean_acc.get)
    vision_conditions = None
    f = Path("results/vision.json")
    if f.exists():
        vision_conditions = {(r["dataset"], r["condition"]) for r in json.loads(f.read_text()) if "condition" in r}
    states = pd.Series([u["state"] for u in units_]).value_counts().to_dict()
    counts = {"total": len(units_)}
    for s in ("valid", "at ceiling", "at chance level", "pair at chance level", "few images"):
        counts[s] = int(states.get(s, 0))
    df = _valid(units_)
    res = {
        "parameters": {"m": M, "epsilon": EPS, "folds": FOLDS, "seed": SEED, "ceiling": CEILING, "min_images": MIN_IMAGES,
                       "ridge": RIDGE, "weight_max": WEIGHT_MAX, "condition_resamples": RESAMPLES,
                       "conversion": "sdt.convert_rho_mafc reproduced without rounding dA and dB (grid -0.3..0.95, 26 points, n=120000, seed 534)",
                       "networks": networks, "years": year, "mean_accuracy": mean_acc},
        "counts": counts,
        "summary": summarize(df),
        "sensitivity_analysis": sensitivity_analysis(units_, vision_conditions),
        "mind_2014": mind_2014(units_, year),
        "families": families(units_, data, strongest),
        "per_dataset": per_dataset(units_),
        "checks": checks(units_, data, conv),
        "conditions_missing_from_vision_json": sorted(f"{d}/{c}" for d, c in data["conditions"] if vision_conditions and (d, c) not in vision_conditions),
        "units": units_,
    }
    Path("results").mkdir(exist_ok=True)
    output.write_text(json.dumps(_clean(res), indent=0, ensure_ascii=False))
    print_report(res)
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]), summary_only="--summarize" in sys.argv[2:])
