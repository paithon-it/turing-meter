"""A7: the calibration with heterogeneous questions (scenario H). Pre-registered in PROTOCOL.md, "After the critical
review (v4): the new measures".

Scenario H is A plus two departures from the article's model: the difficulty changes from question to question (the
correct option's mean shifted by δ_q ~ N(0, 0.5²), the same for person and machine), and on each question one wrong
option attracts (its mean raised by 0.6 for everyone). With the model misspecified, ρ has no generative true value:
the turing is judged against its true value from a huge sample, with the criteria of the other scenarios (error at
most 5 percent, coverage between 90 and 98); the angle against the instrument's own value on a huge sample
(coverage only). Everything else as in calibration.py: same seeds scheme, same number of replicas as A (300), same
resamples (300).

Usage, from the experiment folder:  python -m angle.calibration_h [REPLICAS]   → results/calibration-h.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import calibration as C
from .measure import Trials

H = C.Scenario("H", "like A, with heterogeneous questions and an attractive wrong option")
SPREAD, ATTRACTION = 0.5, 0.6
I_SC = len(C.SCENARIOS)          # its index in the seed scheme: after the six of version 3


def _raw_h(sc, gen, people, items):
    """C._raw with the question effects: the same draws in the same order, then the shifts."""
    m, r = sc.m, sc.rho
    sr, sn = math.sqrt(r), math.sqrt(1.0 - r)
    common = gen.standard_normal((items, m))
    di = np.clip(sc.dh + sc.sd * gen.standard_normal(people), C.D_MIN, None)
    chosen = C._choose_items(gen, people, sc.per_person, items)
    person = np.repeat(np.arange(people), sc.per_person)
    item = chosen.ravel()
    shift = np.zeros((items, m))
    shift[:, 0] = SPREAD * gen.standard_normal(items)      # difficulty of the question, shared
    shift[:, 1] = ATTRACTION                                 # an attractive wrong option, for everyone
    a = sr * common[item] + sn * gen.standard_normal((item.size, m)) + shift[item]
    a[:, 0] += di[person]
    b_item = sr * common + sn * gen.standard_normal((items, m)) + shift
    b_item[:, 0] += sc.dm
    machine = (b_item.argmax(1) == 0).astype(float)
    b = b_item[item]
    m_trial = machine[item]
    correct = (a.argmax(1) == 0).astype(float)
    pair = ((a + sc.weight * b).argmax(1) == 0).astype(float)
    return person, item, correct, machine, m_trial, pair


def generate(sc, seed) -> Trials:
    gen = np.random.default_rng(seed)
    person, item, correct, machine, _, pair = _raw_h(sc, gen, sc.people, sc.items)
    return Trials(person=person, item=item, correct=correct, machine=machine, with_machine=pair)


def true_turing(n: int = C.N_TRUE, chunks: int = 10) -> dict:
    gen = np.random.default_rng(np.random.SeedSequence((C.SEED, ord("H"), 999_999)))
    people = max(int(round(n / chunks / H.per_person)), 1)
    items = max(int(round(H.items * people / H.people)), H.per_person)
    sums, trials, per = np.zeros(3), 0, []
    for _ in range(chunks):
        _, _, correct, _, m_trial, pair = _raw_h(H, gen, people, items)
        s = np.array([correct.sum(), m_trial.sum(), pair.sum()])
        sums += s
        trials += correct.size
        a = s / correct.size
        per.append(C.sdt.d_from_accuracy(a[2], H.m) / C.sdt.d_from_accuracy(a[0], H.m))
    a = sums / trials
    return {"acc_h": a[0], "acc_m": a[1], "acc_s": a[2], "turing": C.sdt.d_from_accuracy(a[2], H.m) / C.sdt.d_from_accuracy(a[0], H.m),
            "turing_se": float(np.std(per, ddof=1) / math.sqrt(chunks)), "trials": trials}


def instrument_on_huge_sample(n: int = C.N_INSTRUMENT) -> dict:
    f = n / (H.people * H.per_person)
    big = replace(H, people=int(round(H.people * f)), items=int(round(H.items * f)))
    out = C.compare_with_apparent(generate(big, np.random.SeedSequence((C.SEED, ord("H"), 888_888))), m=H.m, repetitions=0)
    return {k: out.get(k) for k in ("turing", "phi", "rho", "rho_correct", "dh", "dm", "ds", "acc_h", "acc_m", "acc_s")}


def one_replica(arg) -> dict:
    r, repetitions = arg
    data_seed, boot_seed = C._seeds(I_SC, r)
    out = C.compare_with_apparent(generate(H, data_seed), m=H.m, repetitions=repetitions, seed=boot_seed)
    if "excluded" in out:
        return {"replica": r, "excluded": out["excluded"]}
    return {"replica": r, **{k: out.get(k) for k in ("turing", "turing_ci", "phi", "phi_ci", "rho", "acc_h", "acc_m", "acc_s")}}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    replicas = int(argv[0]) if argv else 300
    t0 = time.time()
    true = true_turing()
    huge = instrument_on_huge_sample()
    with ProcessPoolExecutor(max_workers=2) as ex:
        rows = list(ex.map(one_replica, [(r, C.REPETITIONS) for r in range(replicas)]))
    good = [x for x in rows if "excluded" not in x]
    st_turing = C.statistics(good, true["turing"], "turing")
    st_phi = C.statistics(good, huge["phi"], "phi")
    crit = C.criterion("H", st_turing)
    crit.pop("note", None)                       # C.criterion writes F's note for any scenario outside its list
    crit["subject_to_criterion"] = True
    ok_e = abs(st_turing["bias_pct"]) <= C.CRITERION["error_pct_max"]
    ok_c = C.CRITERION["coverage_min"] <= st_turing["coverage_pct"] <= C.CRITERION["coverage_max"]
    crit.update(passes=bool(ok_e and ok_c), error_within_5pct=ok_e, coverage_within_90_98=ok_c)
    out = {"scenario": H.parameters(), "spread": SPREAD, "attraction": ATTRACTION, "replicas": replicas,
           "excluded": len(rows) - len(good), "true_turing": true, "instrument_on_huge_sample": huge,
           "turing": st_turing, "turing_criterion": crit, "phi_against_huge_sample": st_phi, "seconds": round(time.time() - t0)}
    Path("results/calibration-h.json").write_text(json.dumps(out, indent=1))
    print("turing: true %.3f, bias %.2f%%, coverage %.1f%% -> passes %s" % (true["turing"], st_turing["bias_pct"], st_turing["coverage_pct"], crit["passes"]))
    print("phi: huge-sample instrument %.1f, mean estimate %.1f, coverage %.1f%%" % (huge["phi"], st_phi["mean"], st_phi.get("coverage_pct", float("nan"))))


if __name__ == "__main__":
    main()
