"""The closed models next to the people: is there still room for a pair above a stronger machine?

Reads the verdicts written by closed/machines_claude.py and, with the same estimators as the other
machines, gives for each model: how many times it is worth the people (k), the angle between its errors
and theirs, the useful angle, and the margin, what the ideal pair promises over the stronger of the two.
The people never worked with these models: there is no turing here, only what a pair could do.

On αNLI the same numbers are given three times, with the three readings of the key of angle/review.py
(the key of the story, without the stories where most voters go against the key, the majority as the key),
for the closed models and, when their folder is given, for the open ones on the same footing.

Exploratory: added after the article, on models the protocol did not have. Writes results/frontier.json.

Usage, from the experiment folder:  python -m angle.frontier CLOSED_RESULTS_DIR [GPU_RESULTS_DIR]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import review
from .analysis import BASE, REPETITIONS, _tasks_in_folder, chatbench_one, load_machine, machine_vs_machine
from .measure import compare_votes

SUBJECTS = (None, "math", "physics", "moral")
KEEP = ("n_trials", "people", "items", "acc_h", "acc_m", "dh", "dm", "k", "k_ci", "rho", "rho_ci", "phi", "phi_ci",
        "useful_angle", "useful_angle_ci", "ideal_phi", "ideal_90", "excluded")


def summary(r: dict) -> dict:
    """What is kept of a comparison, with the margin of the ideal pair over the stronger of the two."""
    out = {k: r[k] for k in KEEP if k in r}
    if "ideal_phi" in r:
        stronger = max(r["dh"], r["dm"])
        out["apparent_turings"] = r["ideal_phi"] / r["dh"]
        out["margin"] = r["ideal_phi"] / stronger - 1.0          # zero at the useful angle and below: the ideal stops at the stronger
        out["margin_90"] = r["ideal_90"] / stronger - 1.0        # the same, had the errors been independent
        out["above_useful_angle"] = bool(r["phi"] > r["useful_angle"])
    return out


def _wrong(folder: Path, task: str, name: str, mode: str) -> pd.Index:
    t = load_machine(folder, task, name, mode)
    return t.index[t.ok == 0]


def _chatbench(args) -> dict:
    folder, name, mode, subject = args
    base = {"source": "chatbench", "task": subject or "all", "machine": name, "mode": mode}
    try:
        return {**base, **summary(chatbench_one((folder, name, mode, subject)))}
    except ValueError as e:                               # a machine without errors has no sensitivity to estimate
        return {**base, "excluded": str(e)}


def _alphanli(args) -> dict:
    variant, folder, name, mode, closed = args
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    h, votes, ok, key, _ = review._votes_table(review.alphanli_keys(items)[variant], load_machine(folder, "alphanli", name, mode))
    r = compare_votes(h, votes, ok, m=2, repetitions=REPETITIONS, answer_class=key)
    return {"source": "alphanli", "key": variant, "machine": name, "mode": mode, "closed": closed, "items": int(h.size),
            "errors": int((ok == 0).sum()), **summary(r)}


def where_errors_fall(folder: Path, names: list[tuple[str, str]]) -> list[dict]:
    """On αNLI: how many of a machine's errors are on stories where most voters go against the key."""
    items = pd.read_json(BASE / "items/alphanli.jsonl", lines=True).set_index("id")
    share = items.votes_correct / items.votes
    against = set(items.index[2 * items.votes_correct < items.votes])
    out = []
    for name, mode in names:
        w = _wrong(folder, "alphanli", name, mode)
        out.append({"machine": name, "mode": mode, "errors": int(len(w)), "on_contested_keys": int(len(set(w) & against)),
                    "votes_for_key_there": float(share.loc[w].mean()), "votes_for_key_elsewhere": float(share.drop(w).mean())})
    return out


def run(closed: Path, gpu: Path | None) -> None:
    names = _tasks_in_folder(closed)
    jobs = [(_chatbench, (closed, n, mo, s)) for n, mo in sorted(names.get("chatbench", [])) for s in SUBJECTS]
    machines = [(closed, n, mo, True) for n, mo in sorted(names.get("alphanli", []))]
    if gpu is not None:
        machines += [(gpu, n, mo, False) for n, mo in sorted(_tasks_in_folder(gpu).get("alphanli", []))]
    jobs += [(_alphanli, (v, f, n, mo, c)) for v in ("original", "without_contested", "majority") for f, n, mo, c in machines]
    print(len(jobs), "comparisons", flush=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        results = [fu.result() for fu in [ex.submit(fn, a) for fn, a in jobs]]
    output = {"comparisons": results, "machine_vs_machine": [], "where_errors_fall": []}
    for task, entries in names.items():
        output["machine_vs_machine"] += machine_vs_machine(closed, task, sorted(entries))
    if "alphanli" in names:
        output["where_errors_fall"] = where_errors_fall(closed, sorted(names["alphanli"]))
    Path("results").mkdir(exist_ok=True)
    Path("results/frontier.json").write_text(json.dumps(output, indent=1))
    report(output)


def report(output: dict) -> None:
    def line(r: dict) -> str:
        if "phi" not in r:
            return f"  {r['machine']:14s} {r['mode']:6s} excluded: {r.get('excluded')}"
        return (f"  {r['machine']:14s} {r['mode']:6s} acc {r['acc_m']:.3f} k {r['k']:.2f} [{r['k_ci'][0]:.2f}-{r['k_ci'][1]:.2f}] "
                f"angle {r['phi']:.1f} [{r['phi_ci'][0]:.0f}-{r['phi_ci'][1]:.0f}] useful {r['useful_angle']:.0f} "
                f"margin {100 * r['margin']:.1f}% (at 90: {100 * r['margin_90']:.0f}%)")

    rows = output["comparisons"]
    for subject in ("all", "math", "physics", "moral"):
        print(f"chatbench, {subject}", flush=True)
        for r in rows:
            if r["source"] == "chatbench" and r["task"] == subject:
                print(line(r), flush=True)
    for variant in ("original", "without_contested", "majority"):
        print(f"alphanli, key: {variant}", flush=True)
        for r in sorted((r for r in rows if r["source"] == "alphanli" and r["key"] == variant), key=lambda r: r.get("dm", 0.0)):
            print(line(r) + ("   <- closed" if r["closed"] else ""), flush=True)
    for r in output["where_errors_fall"]:
        print(f"  {r['machine']:14s} errors {r['errors']}, on contested keys {r['on_contested_keys']}, "
              f"votes for the key there {r['votes_for_key_there']:.2f}, elsewhere {r['votes_for_key_elsewhere']:.2f}", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
