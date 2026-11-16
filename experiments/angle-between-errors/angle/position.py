"""A17, the position of the answer and the similarity among machines. Pre-registered in PROTOCOL.md, "After the sixth
review (v4): A17" (cc2dc24, on the server at 20:28:45Z), before computing it.

On αNLI the ten open models with the letter only answered the stories in both orders (A11). The averaged machine answers
the option with the larger mean of the two probabilities, story by story (on an exact tie, the original order). The same
quantities as H2 are computed on three versions of the machines (original, swapped, averaged): the tetrachoric
correlation of the correct answers for the 45 pairs of machines, and between the people's votes and each machine as in
`measure.compare_votes`. Intervals: 2,000 replicas, seed 534, a Poisson(1) weight on every story shared by every pair.

Usage, from the experiment folder:  python -m angle.position   → results/position.json
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from . import sdt
from .analysis import load_machine

SEED = 534
REPLICAS = 2000
BASE = Path("data")
ORIGINAL = Path("gpu-results")
SWAPPED = Path("contamination-results")
VERSIONS = ("original", "swapped", "averaged")


def _rows(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def answers(name: str, ids: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """The correct answers (0/1) of one machine on the stories `ids`, in the three versions."""
    o, s = _rows(ORIGINAL / f"alphanli__{name}__letter.jsonl"), _rows(SWAPPED / f"alphanli__{name}__letter.jsonl")
    po = np.array([o[i]["prob"] for i in ids], float)
    ps = np.array([s[i]["prob"] for i in ids], float)
    assert all(o[i]["correct_option"] == s[i]["correct_option"] == c for i, c in zip(ids, correct)), name
    avg = (po + ps) / 2
    pick_avg = np.where(avg[:, 0] == avg[:, 1], po.argmax(axis=1), avg.argmax(axis=1))
    return {"original": (po.argmax(axis=1) == correct).astype(float), "swapped": (ps.argmax(axis=1) == correct).astype(float),
            "averaged": (pick_avg == correct).astype(float)}


def _tet(a: np.ndarray, b: np.ndarray, w: np.ndarray, clip: bool = False) -> float:
    p1, p2, p11 = np.average(a, weights=w), np.average(b, weights=w), np.average(a * b, weights=w)
    if clip:      # as measure.compare_votes does for person and machine
        p1, p2 = np.clip(p1, 1e-4, 1 - 1e-4), np.clip(p2, 1e-4, 1 - 1e-4)
    elif not (0 < p1 < 1 and 0 < p2 < 1):
        return float("nan")
    return sdt.tetrachoric(float(p1), float(p2), float(p11))


def build() -> dict:
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    names = sorted(p.name.split("__")[1] for p in SWAPPED.glob("alphanli__*__letter.jsonl"))
    common = set(items.index)
    for n in names:
        common &= set(_rows(ORIGINAL / f"alphanli__{n}__letter.jsonl")) & set(_rows(SWAPPED / f"alphanli__{n}__letter.jsonl"))
    ids = np.array(sorted(common))
    correct = items.loc[ids, "correct_option"].to_numpy(int)
    h = (items.loc[ids, "votes_correct"] / items.loc[ids, "votes"]).to_numpy(float)
    machines = {n: answers(n, ids, correct) for n in names}
    for n in names:   # the original version is the machine of the map
        ok = load_machine(ORIGINAL, "alphanli", n, "letter").ok.reindex(ids).to_numpy(float)
        assert np.array_equal(ok, machines[n]["original"]), n
    return {"ids": ids, "h": h, "names": names, "machines": machines}


DATA: dict = {}


def _stats(w: np.ndarray) -> dict:
    d = DATA
    out = {}
    for v in VERSIONS:
        mm = [_tet(d["machines"][a][v], d["machines"][b][v], w) for a, b in combinations(d["names"], 2)]
        pm = [_tet(d["h"], d["machines"][n][v], w, clip=True) for n in d["names"]]
        out[v] = {"mm": mm, "pm": pm, "mm_median": float(np.nanmedian(mm)), "pm_median": float(np.nanmedian(pm))}
    out["delta_mm"] = out["original"]["mm_median"] - out["averaged"]["mm_median"]
    out["d_averaged"] = out["averaged"]["mm_median"] - out["averaged"]["pm_median"]
    out["d_original"] = out["original"]["mm_median"] - out["original"]["pm_median"]
    return out


def _replica(r: int) -> dict:
    gen = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(r,)))
    s = _stats(gen.poisson(1.0, DATA["ids"].size).astype(float))
    return {k: s[k] for k in ("delta_mm", "d_averaged", "d_original")}


def _ci(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> None:
    DATA.update(build())
    point = _stats(np.ones(DATA["ids"].size))
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("fork")) as ex:
        reps = list(ex.map(_replica, range(REPLICAS), chunksize=50))
    out = {"protocol": "PROTOCOL.md, After the sixth review (cc2dc24)", "seed": SEED, "replicas": REPLICAS,
           "stories": int(DATA["ids"].size), "machines": DATA["names"],
           "pairs": [f"{a}/{b}" for a, b in combinations(DATA["names"], 2)],
           "accuracy": {n: {v: float(DATA["machines"][n][v].mean()) for v in VERSIONS} for n in DATA["names"]},
           **{v: point[v] for v in VERSIONS}}
    for k in ("delta_mm", "d_averaged", "d_original"):
        out[k] = point[k]
        out[k + "_ci"] = _ci([r[k] for r in reps])
    out["a17a_passes"] = bool(out["delta_mm_ci"][0] > 0)
    out["a17b_passes"] = bool(out["d_averaged_ci"][0] > 0)
    Path("results/position.json").write_text(json.dumps(out, indent=1))
    for v in VERSIONS:
        print("%-9s machines %.3f  person-machine %.3f" % (v, out[v]["mm_median"], out[v]["pm_median"]))
    print("A17a delta_mm %.3f (%.3f %.3f) passes %s" % (out["delta_mm"], *out["delta_mm_ci"], out["a17a_passes"]))
    print("A17b D averaged %.3f (%.3f %.3f) passes %s; D original %.3f (%.3f %.3f)"
          % (out["d_averaged"], *out["d_averaged_ci"], out["a17b_passes"], out["d_original"], *out["d_original_ci"]))


if __name__ == "__main__":
    main()
