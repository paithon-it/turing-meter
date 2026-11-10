"""T, the calibration: the article's instrument on data where the truth is known.

Specification: PROTOCOL.md, "The instrument on trial", subsection "T, the calibration".

The process: for each item a vector of cues common to the m options; person i has
sensitivity d_i ~ Normal(dH, sd), clipped at D_MIN, and cues with correlation ρ with those of
the machine (common component with weight sqrt(ρ), as in `sdt._simulate_correct`); the machine has
cues per item (the same answer for whoever sees that item) or, in scenario G, per
trial. The pair picks the option that maximizes person_cues + w · machine_cues;
in the "copy" scenario it gives the machine's answer. The correct answer is option 0.

The true value of each quantity comes from the same process on a huge sample, with the
same inversions as the instrument (aggregate accuracy over the trials -> `sdt.d_from_accuracy`)
and with the generative ρ for the angle and for the ideal.

Usage, from the experiment folder:
    python -m angle.calibration                 # all replicas, writes results/calibration.json
    python -m angle.calibration --replicas 20   # a short run
    python -m angle.calibration --stopwatch     # how long one call of compare takes

The run of the saved results (9 November 2026), in two parts because of the machine load:
    python -m angle.calibration --replicas 300 --scenarios AB
    python -m angle.calibration --replicas 150 --scenarios CDEFG --merge
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import measure, sdt
from .measure import Trials

SEED = 20261109            # declared base seed: data, bootstrap and true values derive from it
D_MIN = 0.02               # the minimum of a person's sensitivity (the same as sdt.estimate_from_evidence)
REPETITIONS = 300          # the resamples of compare, as the protocol asks
N_TRUE = 2_000_000         # trials of the huge sample
N_INSTRUMENT = 1_000_000   # trials of the huge sample on which the instrument itself runs
QUANTITIES = ("turing", "apparent_turings", "eta_phi", "phi")
CRITERION = {"error_pct_max": 5.0, "coverage_min": 90.0, "coverage_max": 98.0,
             "scenarios": ("A", "B", "C", "D", "E", "G")}


# --- scenarios -------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    dh: float = 1.2
    sd: float = 0.4
    dm: float = 2.6
    rho: float = 0.2
    w: float | None = None          # None: the optimal weight w*
    copy: bool = False
    per_trial: bool = False
    people: int = 160
    per_person: int = 7
    items: int = 380
    m: int = 4

    @property
    def weight(self) -> float:
        """w* = (dM − ρ dH) / (dH − ρ dM): the weight that maximizes the sensitivity of the sum."""
        if self.w is not None:
            return self.w
        return (self.dm - self.rho * self.dh) / (self.dh - self.rho * self.dm)

    def parameters(self) -> dict:
        d = asdict(self)
        d["w"] = None if self.copy else self.weight
        d["w_optimal"] = self.w is None and not self.copy
        return d


SCENARIOS = (
    Scenario("A", "ideal pair"),
    Scenario("B", "pair halfway", w=0.5),
    Scenario("C", "weak machine followed too much, turing below one", dm=0.6, w=1.5),
    Scenario("D", "the person copies the machine, turing equal to k", copy=True),
    Scenario("E", "machine equal to the person", dm=1.3, rho=0.3),
    Scenario("F", "weak person and thirty lone people, as in physics",
             dh=0.7, sd=0.3, dm=2.4, rho=0.45, w=1.0, people=30, per_person=8, items=130),
    Scenario("G", "like A, with the machine answering trial by trial", per_trial=True),
)
BY_NAME = {s.name: s for s in SCENARIOS}


# --- the process -------------------------------------------------------------------

def _choose_items(gen: np.random.Generator, people: int, per_person: int, items: int) -> np.ndarray:
    """per_person different items for each person, at random among `items` (rejection sampling of the rows with duplicates)."""
    if per_person > items:
        raise ValueError("more questions per person than items")
    chosen = gen.integers(0, items, (people, per_person))
    while True:
        dup = (np.diff(np.sort(chosen, axis=1), axis=1) == 0).any(axis=1)
        if not dup.any():
            return chosen
        chosen[dup] = gen.integers(0, items, (int(dup.sum()), per_person))


def _raw(sc: Scenario, gen: np.random.Generator, people: int, items: int):
    """One round of the process: (person, item, correct, machine per item or per trial, machine on the trial, pair)."""
    m, r = sc.m, sc.rho
    if not 0.0 <= r < 1.0:
        raise ValueError("need 0 ≤ ρ < 1")
    sr, sn = math.sqrt(r), math.sqrt(1.0 - r)
    common = gen.standard_normal((items, m))
    di = np.clip(sc.dh + sc.sd * gen.standard_normal(people), D_MIN, None)
    chosen = _choose_items(gen, people, sc.per_person, items)
    person = np.repeat(np.arange(people), sc.per_person)
    item = chosen.ravel()
    a = sr * common[item] + sn * gen.standard_normal((item.size, m))
    a[:, 0] += di[person]
    if sc.per_trial:
        b = sr * common[item] + sn * gen.standard_normal((item.size, m))
        b[:, 0] += sc.dm
        machine = (b.argmax(1) == 0).astype(float)
        m_trial = machine
    else:
        b_item = sr * common + sn * gen.standard_normal((items, m))
        b_item[:, 0] += sc.dm
        machine = (b_item.argmax(1) == 0).astype(float)
        b = b_item[item]
        m_trial = machine[item]
    correct = (a.argmax(1) == 0).astype(float)
    if sc.copy:
        pair = m_trial.copy()
    else:
        pair = ((a + sc.weight * b).argmax(1) == 0).astype(float)
    return person, item, correct, machine, m_trial, pair


def generate(sc: Scenario, seed) -> Trials:
    """A simulated experiment with the design of the scenario, in the format that `measure.compare` reads."""
    gen = np.random.default_rng(seed)
    person, item, correct, machine, _, pair = _raw(sc, gen, sc.people, sc.items)
    return Trials(person=person, item=item, correct=correct, machine=machine,
                  with_machine=pair, machine_per_trial=sc.per_trial)


def _derived(acc_h: float, acc_m: float, acc_s: float, rho: float, m: int) -> dict:
    """The quantities from the aggregate accuracies, with the instrument's inversions and the given ρ."""
    dh, dm, ds = (sdt.d_from_accuracy(a, m) for a in (acc_h, acc_m, acc_s))
    ideal = sdt.ideal(dh, dm, rho)
    return {"acc_h": acc_h, "acc_m": acc_m, "acc_s": acc_s, "dh": dh, "dm": dm, "ds": ds,
            "k": dm / dh, "rho": rho, "ideal_phi": ideal,
            "turing": ds / dh, "apparent_turings": sdt.apparent_turings(dh, dm, rho),
            "eta_phi": ds / ideal, "phi": sdt.angle_degrees(rho)}


def true_values(sc: Scenario, n: int = N_TRUE, seed=None, chunks: int = 10) -> dict:
    """The true value of each quantity: the same process on `n` trials, in chunks for memory.

    Each chunk is an experiment with the same design scaled up (same questions per person, same
    ratio between trials and items). The accuracies are summed over all chunks; the dispersion
    between chunks gives the Monte Carlo error of the true value (`_se`).
    """
    gen = np.random.default_rng(np.random.SeedSequence((SEED, ord(sc.name), 999_999)) if seed is None else seed)
    people = max(int(round(n / chunks / sc.per_person)), 1)
    items = max(int(round(sc.items * people / sc.people)), sc.per_person)
    sums = np.zeros(3)
    trials = 0
    per_chunk = []
    for _ in range(chunks):
        _, _, correct, _, m_trial, pair = _raw(sc, gen, people, items)
        s = np.array([correct.sum(), m_trial.sum(), pair.sum()])
        sums += s
        trials += correct.size
        per_chunk.append(_derived(*(s / correct.size), sc.rho, sc.m))
    out = _derived(*(sums / trials), sc.rho, sc.m)
    out["trials"] = trials
    for g in QUANTITIES + ("dh", "dm", "ds", "acc_h", "acc_m", "acc_s"):
        v = np.array([x[g] for x in per_chunk])
        out[g + "_se"] = float(v.std(ddof=1) / math.sqrt(chunks))
    return out


# --- the instrument ------------------------------------------------------------------

@contextmanager
def _with_apparent_turings():
    """`measure.compare` as it is, with the apparent turings also computed in every resample.

    `compare` does not keep the single bootstrap replicas: to get the interval of |S|/dH we
    wrap `_one_pass` (which `compare` looks up in the module at every call) adding
    `apparent_turings = sdt.apparent_turings(dh, dm, rho)`. The rest is identical: no key
    changes and no extra random number is drawn. The file measure.py is not touched.
    """
    original = measure._one_pass

    def one_pass(*args, **kwargs):
        out = original(*args, **kwargs)
        if out is not None:
            out["apparent_turings"] = sdt.apparent_turings(out["dh"], out["dm"], out["rho"])
        return out

    measure._one_pass = one_pass
    try:
        yield
    finally:
        measure._one_pass = original


def compare_with_apparent(p: Trials, m: int = 4, repetitions: int = REPETITIONS, seed: int = 534) -> dict:
    with _with_apparent_turings():
        return measure.compare(p, m=m, repetitions=repetitions, seed=seed)


def instrument_true_value(sc: Scenario, n: int = N_INSTRUMENT) -> dict:
    """The protocol read to the letter: what the instrument returns on a huge sample.

    A single experiment with the same design scaled up (same questions per person, same ratio between
    trials and items), measured by `compare` without bootstrap. For dH, dM and dS it coincides with
    `true_values` up to sampling; for ρ it also contains the error the estimator has with infinite data.
    """
    f = n / (sc.people * sc.per_person)
    big = replace(sc, people=int(round(sc.people * f)), items=int(round(sc.items * f)))
    out = compare_with_apparent(generate(big, np.random.SeedSequence((SEED, ord(sc.name), 888_888))),
                                m=sc.m, repetitions=0)
    return {k: out.get(k) for k in QUANTITIES + ("rho", "rho_correct", "dh", "dm", "ds", "acc_h", "acc_m", "acc_s", "n_trials")}


def _seeds(i_sc: int, r: int) -> tuple[np.random.SeedSequence, int]:
    data = np.random.SeedSequence((SEED, i_sc, r, 0))
    boot = int(np.random.SeedSequence((SEED, i_sc, r, 1)).generate_state(1)[0])
    return data, boot


def one_replica(arg) -> dict:
    """One replica: a simulated experiment, then the instrument. Returns only what the statistics need."""
    i_sc, r, repetitions = arg
    sc = SCENARIOS[i_sc]
    data_seed, boot_seed = _seeds(i_sc, r)
    p = generate(sc, data_seed)
    t0 = time.perf_counter()
    out = compare_with_apparent(p, m=sc.m, repetitions=repetitions, seed=boot_seed)
    row = {"replica": r, "seconds": round(time.perf_counter() - t0, 3)}
    if "excluded" in out:
        row["excluded"] = out["excluded"]
        return row
    for key in QUANTITIES + ("dh", "dm", "ds", "rho", "acc_h", "acc_m", "acc_s"):
        row[key] = out.get(key)
    for g in QUANTITIES:
        row[g + "_ci"] = out.get(g + "_ci")
    row["valid_repetitions"] = out.get("repetitions")
    return row


# --- statistics ---------------------------------------------------------------------

def statistics(rows: list[dict], true: float, g: str) -> dict:
    """Bias, root mean squared error, coverage and width of the 95 percent interval."""
    estimates = np.array([x[g] for x in rows if x.get(g) is not None], float)
    ci = np.array([x[g + "_ci"] for x in rows if x.get(g) is not None and x.get(g + "_ci") is not None], float)
    if estimates.size == 0:
        return {"n": 0}
    out = {"true": true, "n": int(estimates.size), "mean": float(estimates.mean()), "median": float(np.median(estimates)),
           "sd": float(estimates.std(ddof=1)) if estimates.size > 1 else None,
           "bias": float(estimates.mean() - true),
           "bias_se": float(estimates.std(ddof=1) / math.sqrt(estimates.size)) if estimates.size > 1 else None,
           "median_error": float(np.median(estimates) - true),
           "rmse": float(math.sqrt(np.mean((estimates - true) ** 2)))}
    out["bias_pct"] = 100.0 * out["bias"] / true
    out["bias_pct_se"] = (100.0 * out["bias_se"] / abs(true)
                          if out["bias_se"] is not None else None)
    if ci.size:
        inside = (ci[:, 0] <= true) & (true <= ci[:, 1])
        c = float(inside.mean())
        out.update({"n_ci": int(ci.shape[0]), "coverage_pct": 100.0 * c,
                    "coverage_pct_se": 100.0 * math.sqrt(c * (1 - c) / ci.shape[0]),
                    "true_below_ci_pct": 100.0 * float((true < ci[:, 0]).mean()),
                    "true_above_ci_pct": 100.0 * float((true > ci[:, 1]).mean()),
                    "width_mean": float((ci[:, 1] - ci[:, 0]).mean()),
                    "width_median": float(np.median(ci[:, 1] - ci[:, 0]))})
    return out


def criterion(name: str, st: dict) -> dict:
    """The protocol's criterion on the turing: bias within 5 % and coverage between 90 and 98 %."""
    e, c = st.get("bias_pct"), st.get("coverage_pct")
    out = {"bias_pct": e, "coverage_pct": c, "coverage_pct_se": st.get("coverage_pct_se")}
    if e is None or c is None:
        out.update(passes=False, reason="no estimate")
        return out
    ok_e = abs(e) <= CRITERION["error_pct_max"]
    ok_c = CRITERION["coverage_min"] <= c <= CRITERION["coverage_max"]
    if name not in CRITERION["scenarios"]:
        out.update(passes=None, subject_to_criterion=False,
                   note="Scenario F: worse is expected, we report by how much",
                   error_within_5pct=ok_e, coverage_within_90_98=ok_c)
        return out
    out.update(passes=bool(ok_e and ok_c), subject_to_criterion=True,
               error_within_5pct=ok_e, coverage_within_90_98=ok_c)
    return out


def summarize(sc: Scenario, true: dict, rows: list[dict], instrument_true: dict | None = None) -> dict:
    excluded = [x for x in rows if "excluded" in x]
    good = [x for x in rows if "excluded" not in x]
    reasons: dict[str, int] = {}
    for x in excluded:
        reasons[x["excluded"]] = reasons.get(x["excluded"], 0) + 1
    entry = {"parameters": sc.parameters(),
             "replicas": len(rows), "excluded_count": len(excluded), "exclusion_reasons": reasons,
             "without_turing": sum(1 for x in good if x.get("turing") is None),
             "seconds_per_replica": float(np.mean([x["seconds"] for x in rows])) if rows else None,
             "true_values": true,
             "quantities": {g: statistics(good, true[g], g) for g in QUANTITIES},
             "rho_estimated": statistics(good, true["rho"], "rho") if good else {}}
    if instrument_true is not None:
        entry["instrument_values_on_huge_sample"] = instrument_true
        entry["versus_instrument_on_huge_sample"] = {
            g: {k: v for k, v in statistics(good, instrument_true[g], g).items()
                if k in ("true", "bias", "bias_pct", "coverage_pct", "coverage_pct_se")}
            for g in QUANTITIES if instrument_true.get(g) is not None}
    return entry


# --- run -----------------------------------------------------------------------------

def stopwatch(times: int = 3, repetitions: int = REPETITIONS) -> list[float]:
    elapsed = []
    for r in range(times):
        p = generate(SCENARIOS[0], np.random.SeedSequence((SEED, 0, 10_000 + r, 0)))
        t0 = time.perf_counter()
        compare_with_apparent(p, m=4, repetitions=repetitions)
        elapsed.append(time.perf_counter() - t0)
    return elapsed


def _fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def table(res: dict) -> str:
    rows = [f"{'sc':<3}{'quantity':<18}{'true':>8}{'mean':>8}{'err %':>8}{'rmse':>8}{'cov %':>7}{'width':>8}  R  excl"]
    for name, v in res["scenarios"].items():
        for g in QUANTITIES:
            s = v["quantities"][g]
            rows.append(f"{name:<3}{g:<18}{_fmt(s.get('true')):>8}{_fmt(s.get('mean')):>8}"
                        f"{_fmt(s.get('bias_pct'), 1):>8}{_fmt(s.get('rmse')):>8}"
                        f"{_fmt(s.get('coverage_pct'), 1):>7}{_fmt(s.get('width_mean')):>8}"
                        f"  {v['replicas']} {v['excluded_count']}")
    rows.append("")
    for name, c in res["criterion"].items():
        outcome = {True: "passes", False: "FAILS", None: "not subject"}[c["passes"]]
        rows.append(f"criterion {name}: {outcome}  (error {_fmt(c['bias_pct'], 2)} %, "
                    f"coverage {_fmt(c['coverage_pct'], 1)} ± {_fmt(c['coverage_pct_se'], 1)} %)")
    return "\n".join(rows)


def run(replicas: int, repetitions: int = REPETITIONS, processes: int = 2, scenarios=None,
        output: Path = Path("results/calibration.json"), merge: bool = False) -> dict:
    """Runs the chosen scenarios. With `merge` it keeps the scenarios already in the file and replaces only those run now.

    The seeds depend only on scenario and replica: the first R replicas of a scenario are the same
    whatever the total number, and the scenarios can be run in several goes.
    """
    chosen = [s for s in SCENARIOS if scenarios is None or s.name in scenarios]
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_file = output.with_name(output.stem + "-replicas.json")
    res = {"protocol": "PROTOCOL.md, \"The instrument on trial\", T",
           "instrument": "measure.compare(Trials(...), m=4, repetitions=%d), bootstrap seed different for each replica" % repetitions,
           "seed": SEED, "replicas_per_scenario": {}, "bootstrap_repetitions": repetitions,
           "n_true": N_TRUE, "d_min": D_MIN,
           "notes": ["True value: aggregate accuracies over %d trials of the same process, inverted with "
                     "sdt.d_from_accuracy(a, m); ρ, φ, ideal and apparent turings with the generative ρ." % N_TRUE,
                     "In addition, to the letter of the protocol: the instrument itself on %d trials (no bootstrap), "
                     "and the statistics against that value." % N_INSTRUMENT,
                     "Coverage: share of the replicas whose 95 %% percentile interval contains the true value.",
                     "Excluded replicas: compare returns the key 'excluded'; they do not enter the statistics."],
           "protocol_criterion": dict(CRITERION, scenarios=list(CRITERION["scenarios"])),
           "scenarios": {}, "criterion": {}}
    raw = {}
    if merge and output.exists() and raw_file.exists():
        old = json.loads(output.read_text())
        raw = json.loads(raw_file.read_text())
        if old.get("bootstrap_repetitions") != repetitions:
            raise ValueError("only runs with the same number of resamples can be merged")
        for key in ("scenarios", "criterion"):
            res[key] = old.get(key, {})
        res["replicas_per_scenario"] = {n: v["replicas"] for n, v in res["scenarios"].items()}
        res["previous_run_seconds"] = old.get("total_seconds", 0.0) + old.get("previous_run_seconds", 0.0)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=processes) as ex:
        for sc in chosen:
            i_sc = SCENARIOS.index(sc)
            t_sc = time.time()
            true = true_values(sc)
            true_i = instrument_true_value(sc)
            rows = list(ex.map(one_replica, [(i_sc, r, repetitions) for r in range(replicas)], chunksize=2))
            entry = summarize(sc, true, rows, true_i)
            entry["wall_seconds"] = round(time.time() - t_sc, 1)
            res["scenarios"][sc.name] = entry
            res["criterion"][sc.name] = criterion(sc.name, entry["quantities"]["turing"])
            res["replicas_per_scenario"][sc.name] = replicas
            for key in ("scenarios", "criterion", "replicas_per_scenario"):        # in the order A-G
                res[key] = {n: res[key][n] for n in BY_NAME if n in res[key]}
            raw[sc.name] = rows
            res["total_seconds"] = round(time.time() - t0, 1)
            output.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            raw_file.write_text(json.dumps(raw, ensure_ascii=False))
            print(f"scenario {sc.name}: {entry['wall_seconds']} s, excluded {entry['excluded_count']}", flush=True)
    return res


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--replicas", type=int, default=200)
    a.add_argument("--repetitions", type=int, default=REPETITIONS)
    a.add_argument("--processes", type=int, default=2)
    a.add_argument("--scenarios", default=None, help="for example ABC; all if absent")
    a.add_argument("--output", default="results/calibration.json")
    a.add_argument("--stopwatch", action="store_true")
    a.add_argument("--merge", action="store_true", help="keeps the scenarios already in the output file")
    x = a.parse_args(argv)
    if x.stopwatch:
        t = stopwatch(repetitions=x.repetitions)
        print("seconds per call of compare:", [round(v, 2) for v in t])
        return
    res = run(x.replicas, x.repetitions, x.processes, x.scenarios, Path(x.output), x.merge)
    print(table(res))


if __name__ == "__main__":
    main()
