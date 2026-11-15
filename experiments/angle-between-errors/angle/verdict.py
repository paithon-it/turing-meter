"""The ceiling of a pair when one of the two gives only a verdict. Added after the sixth review (15 November 2026).

The article's ideal pair, |S|, pools the graded cues of person and machine before they become verdicts. Someone
who works next to a machine usually gets its verdict, not its cue. Here one member of the pair keeps its graded
cues and of the other sees only the option chosen, and decides as the optimal observer would (maximum a
posteriori), under the same model as the rest of the experiment: m options, one Gaussian cue per option with unit
noise, the true option shifted by the sensitivity, and the noise of the two correlated option by option with
correlation rho (the model of `sdt._simulate_correct`).

How it is computed. The observer's cues `a` are drawn (the true option is the first). Given `a` and a hypothesis
j on the true option, the other's cues are independent normals with means rho·a_i + (d_other − rho·d_own)·[i = j]
and standard deviation sqrt(1 − rho²): the probability of each verdict c is a one-dimensional integral. The
posterior of j given (a, c) is proportional to P(c | a, j)·exp(d_own·a_j), and the observer answers its maximum.
The verdict is not drawn: every trial is averaged over the m verdicts with their probabilities, so the only Monte
Carlo noise left is over `a`. Errors are reported as a ratio to the errors of the stronger member on the same
trials, and anchored to its exact error rate.

The observer is exact, so it is not held to non-negative weights as the article's ideal is: below the useful
angle whatever it gains over the stronger is reported as it comes. With the machine's verdict it is nothing.
With the person's verdict and a machine five or ten times as sensitive it is a fraction of a percent: the
machine can use the person's verdict as a reference for the noise the two share. When the observer is the
stronger and almost never wrong, its cues are drawn with a wider noise and each trial is weighed back.

It is a model, not a measurement: no person is in it. A machine that also gives its reasons, or a graded
confidence, gives more than a verdict, and this ceiling does not bind it.

Writes results/verdict-ceiling.json:
  table      the rows of the article's margin table (a person right 58 times in a hundred, four options, a
             machine k times as sensitive, at 75 and 90 degrees), with shared cues, with only the machine's
             verdict (the person keeps the cues) and with only the person's verdict (the machine keeps them);
  chatbench  the two ChatBench pairs at their measured sensitivities and angle (results/chatbench-source.json).

Usage, from the experiment folder:  python -m angle.verdict
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from . import sdt

SEED = 534
TRIALS = 200_000
WIDE_NOISE_FROM = 3.0                         # a stronger observer beyond this sensitivity is drawn with a wider noise
PERSON = "gpt-4o"                             # the person of the article's table: the ChatBench group that worked with GPT-4o
K_VALUES = (1.2, 2.0, 3.0, 5.0, 10.0)
ANGLES = (75.0, 90.0)
MODELS = ("gpt-4o", "llama-3.1-8b")



def _normal_nodes(count: int, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes for the integral of a function against the standard normal density."""
    x, w = np.polynomial.legendre.leggauss(count)
    z = half_width * x
    return z, half_width * w * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


_Z_FINE, _W_FINE = _normal_nodes(96, 9.0)       # for an error rate, down to 1e-15
_Z, _W = _normal_nodes(48, 8.0)                 # for the probability of a verdict, trial by trial


def rho_of(angle_degrees: float) -> float:
    return 0.0 if abs(angle_degrees - 90.0) < 1e-9 else math.cos(math.radians(angle_degrees))


def _shift(d: float, m: int) -> float:
    """The distance between the true option's cue and the others. With two options the article's d′ is 2 z(a)."""
    return d / math.sqrt(2.0) if m == 2 else d


def error_mafc(d: float, m: int) -> float:
    """Probability of a wrong answer among m options. Accurate in the tail, where 1 − sdt.pc_mafc is not."""
    return float(np.sum(_W_FINE * -np.expm1((m - 1) * log_ndtr(_Z_FINE + _shift(d, m)))))


def d_from_error(error: float, m: int) -> float:
    """The sensitivity that gives that error rate: the inverse of `error_mafc`."""
    if not 0.0 < error < 1.0:
        raise ValueError(f"error rate outside (0, 1): {error!r}")
    if error >= 1.0 - 1.0 / m:
        raise ValueError("error rate at chance level or above")
    target = math.log(error)
    return float(brentq(lambda d: math.log(error_mafc(d, m)) - target, 1e-9, 40.0, xtol=1e-10))


def choice_probabilities(mu: np.ndarray, s: float) -> np.ndarray:
    """P(option c has the largest cue), for cues that are independent normals with means mu (n, m) and sd s."""
    mu = np.asarray(mu, float)
    m = mu.shape[1]
    t = ndtr((mu[:, :, None, None] - mu[:, None, :, None]) / s + _Z)      # [trial, c, i, node]: cue c beats cue i
    idx = np.arange(m)
    t[:, idx, idx, :] = 1.0
    return t.prod(axis=2) @ _W


def pair(d_own: float, d_other: float, rho: float, m: int = 4, n: int = TRIALS, seed: int = SEED, chunk: int = 2500) -> dict:
    """The optimal observer with its own graded cues (sensitivity d_own) and the other's verdict (d_other)."""
    if abs(rho) > 0.99:
        raise ValueError("correlation too close to one: the other's cues would be the observer's own")
    so, st, s = _shift(d_own, m), _shift(d_other, m), math.sqrt(1.0 - rho * rho)
    stronger = "other" if d_other >= d_own else "own"
    # When the observer is the stronger and almost never wrong, its errors are too rare to meet by chance: the
    # cues are drawn with a wider noise (tau times) and each trial weighs the ratio of the two densities.
    tau = max(1.0, so / WIDE_NOISE_FROM) if stronger == "own" else 1.0
    gen = np.random.default_rng(seed)
    e_pair, e_other, e_own, weights = [], [], [], []
    for start in range(0, n, chunk):
        noise = tau * gen.standard_normal((min(chunk, n - start), m))
        weights.append(np.exp(m * math.log(tau) - 0.5 * (noise * noise).sum(1) * (1.0 - 1.0 / (tau * tau))))
        a = noise
        a[:, 0] += so
        logpost = np.empty((a.shape[0], m, m))                            # [trial, verdict, hypothesis]
        for j in range(m):
            mu = rho * a
            mu[:, j] += st - rho * so
            p = choice_probabilities(mu, s)
            if j == 0:
                p_true = p                                                # the first option is the true one
            logpost[:, :, j] = np.log(np.maximum(p, 1e-300)) + so * a[:, [j]]
        wrong = logpost.argmax(2) != 0
        e_pair.append((p_true * wrong).sum(1))
        e_other.append(p_true[:, 1:].sum(1))
        e_own.append((a.argmax(1) != 0).astype(float))
    w = np.concatenate(weights)
    x, y, o = w * np.concatenate(e_pair), w * np.concatenate(e_other), w * np.concatenate(e_own)
    exact = {"own": error_mafc(d_own, m), "other": error_mafc(d_other, m)}
    ref = y if stronger == "other" else o
    ratio = float(x.sum() / ref.sum())
    ratio_se = float(np.sqrt(np.sum((x - ratio * ref) ** 2)) / ref.sum())
    d_stronger = max(d_own, d_other)
    chance = 1.0 - 1.0 / m

    def d_at(r: float) -> float:
        return d_from_error(min(max(exact[stronger] * r, 1e-300), chance - 1e-9), m)

    d_pair = d_at(ratio)
    return {"d_own": d_own, "d_other": d_other, "rho": rho, "m": m, "trials": int(x.size), "stronger": stronger, "noise_widened": tau,
            "error_own": exact["own"], "error_other": exact["other"], "error_pair": exact[stronger] * ratio,
            "acc_pair": 1.0 - exact[stronger] * ratio, "acc_pair_se": exact[stronger] * ratio_se,
            "errors_to_stronger": ratio, "errors_removed": 1.0 - ratio, "errors_removed_se": ratio_se,
            "d_pair": d_pair, "margin": d_pair / d_stronger - 1.0,
            "margin_se": abs(d_at(ratio + ratio_se) - d_at(max(ratio - ratio_se, 1e-12))) / (2.0 * d_stronger),
            "acc_pair_sample": float(1.0 - x.mean()), "acc_other_sample": float(1.0 - y.mean()), "acc_own_sample": float(1.0 - o.mean())}


def shared(dh: float, dm: float, rho: float, m: int = 4) -> dict:
    """The article's ideal pair, which pools the graded cues: its margin over the stronger and the errors it removes."""
    ideal, stronger = sdt.ideal(dh, dm, rho), max(dh, dm)
    return {"d_pair": ideal, "margin": ideal / stronger - 1.0,
            "error_pair": error_mafc(ideal, m), "errors_removed": 1.0 - error_mafc(ideal, m) / error_mafc(stronger, m)}


_KEEP = ("d_pair", "margin", "margin_se", "error_pair", "errors_removed", "errors_removed_se")


def table_row(dh: float, k: float, angle: float, m: int = 4, n: int = TRIALS) -> dict:
    dm, rho = k * dh, rho_of(angle)
    mv, pv = pair(dh, dm, rho, m, n), pair(dm, dh, rho, m, n)
    return {"k": k, "angle": angle, "rho": rho, "dh": dh, "dm": dm, "useful_angle": sdt.useful_angle(dh, dm),
            "error_person": error_mafc(dh, m), "error_machine": error_mafc(dm, m),
            "shared": shared(dh, dm, rho, m),
            "machine_verdict": {key: mv[key] for key in _KEEP},          # the person keeps the cues, the machine gives a verdict
            "person_verdict": {key: pv[key] for key in _KEEP}}           # the machine keeps the cues, the person gives a verdict


def _source(model: str) -> dict:
    """The ChatBench pair with the machine queried alone, all trials (results/chatbench-source.json)."""
    src = json.loads(Path("results/chatbench-source.json").read_text())
    return next(c for c in src["comparisons"] if c["machine"] == model and c["subject"] == "all" and c["kind"] == "alone")


def _say(*what) -> None:
    print(*what, flush=True)


def margin_table(n: int = TRIALS) -> dict:
    """The article's table «Quanto spazio resta alla coppia». Its person is the ChatBench group next to GPT-4o,
    right 57.7 times in a hundred on its own (the article prints 58): with exactly 0.58 the errors removed would
    read one point higher than printed."""
    person = _source(PERSON)
    rows = []
    for k in K_VALUES:
        for angle in ANGLES:
            rows.append(table_row(person["dh"], k, angle, 4, n))
            _say("table", k, angle, {key: round(100 * rows[-1][key]["margin"], 2) for key in ("shared", "machine_verdict", "person_verdict")})
    return {"accuracy_alone": person["acc_h"], "dh": person["dh"], "rows": rows}


def chatbench(n: int = TRIALS) -> list[dict]:
    """The two ChatBench pairs, machine queried alone, all trials: what each ceiling promises over the machine."""
    out = []
    for model in MODELS:
        r = _source(model)
        dh, dm = r["dh"], r["dm"]
        row = {"machine": model, "acc_h": r["acc_h"], "acc_m": r["acc_m"], "acc_s": r["acc_s"], "dh": dh, "dm": dm, "ds": r["ds"],
               "k": r["k"], "phi": r["phi"], "phi_ci": r["phi_ci"], "useful_angle": r["useful_angle"],
               "measured_over_machine": r["yield_on_machine"], "measured_over_machine_ci": r["yield_on_machine_ci"], "at": {}}
        for name, angle in (("measured_angle", r["phi"]), ("angle_ci_low", r["phi_ci"][0]), ("angle_ci_high", r["phi_ci"][1]), ("90", 90.0)):
            rho = rho_of(angle)
            mv, pv = pair(dh, dm, rho, 4, n), pair(dm, dh, rho, 4, n)
            sh = shared(dh, dm, rho, 4)
            row["at"][name] = {"angle": angle, "rho": rho,
                               "shared_over_machine": sh["d_pair"] / dm, "shared_accuracy": 1.0 - sh["error_pair"],
                               "machine_verdict_over_machine": mv["d_pair"] / dm, "machine_verdict_se": mv["margin_se"],
                               "machine_verdict_accuracy": mv["acc_pair"], "machine_verdict_errors_removed": mv["errors_removed"],
                               "person_verdict_over_machine": pv["d_pair"] / dm, "person_verdict_se": pv["margin_se"],
                               "person_verdict_accuracy": pv["acc_pair"]}
            _say("chatbench", model, name, round(row["at"][name]["machine_verdict_over_machine"], 4))
        out.append(row)
    return out


def run(n: int = TRIALS) -> dict:
    out = {"note": "Model, not measurement: the optimal observer under the article's Gaussian model. Added after the sixth "
                   "review, 15 November 2026 (PROTOCOL.md). The standard errors are Monte Carlo errors, not sampling intervals.",
           "trials": n, "options": 4, "table": margin_table(n), "chatbench": chatbench(n)}
    Path("results").mkdir(exist_ok=True)
    Path("results/verdict-ceiling.json").write_text(json.dumps(out, indent=1))
    return out


def report(out: dict) -> None:
    pct = lambda v: f"{100 * v:5.1f}%"
    print("k, angle | margin: shared, machine's verdict (se), person's verdict | errors removed: shared, machine's verdict (se), person's verdict")
    for r in out["table"]["rows"]:
        s, mv, pv = r["shared"], r["machine_verdict"], r["person_verdict"]
        print(f"  {r['k']:4.1f} {r['angle']:3.0f} | {pct(s['margin'])} {pct(mv['margin'])} ({100 * mv['margin_se']:.2f}) {pct(pv['margin'])}"
              f" | {pct(s['errors_removed'])} {pct(mv['errors_removed'])} ({100 * mv['errors_removed_se']:.2f}) {pct(pv['errors_removed'])}")
    for r in out["chatbench"]:
        print(f"{r['machine']}: k {r['k']:.2f}, measured dS/dM {r['measured_over_machine']:.3f} {[round(v, 2) for v in r['measured_over_machine_ci']]}")
        for name, a in r["at"].items():
            print(f"  {name:15s} {a['angle']:5.1f} deg | shared {a['shared_over_machine']:.3f} | machine's verdict {a['machine_verdict_over_machine']:.3f}"
                  f" ({a['machine_verdict_se']:.3f}) | person's verdict {a['person_verdict_over_machine']:.3f}")


if __name__ == "__main__":
    report(run())
