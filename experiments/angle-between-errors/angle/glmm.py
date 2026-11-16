"""A18, the angle against a probit mixed model. Pre-registered in PROTOCOL.md, "After the sixth review (v4): A18"
(cd8d98a, on the server at 20:56:19Z) and corrected before computing (02df70e, 20:57:29Z).

For each ChatBench comparison the people's answers alone and the machine's answers are stacked and fitted in R with
lme4::glmer (binomial, probit, Laplace): y ~ 0 + type + (0 + type | question) + (0 + is_person | worker). The quantity
compared is the correlation of the correct answers of a person and the machine on the same question,
ρ_c = c / sqrt((1 + σ_p²)(1 + σ_m²)); it goes to the evidence through the model's conversion among four options
(sdt.convert_rho_mafc at the comparison's d_H and d_M), then to the angle.

Usage, from the experiment folder:
  python -m angle.glmm export DIR    → DIR/NN.csv, one per comparison, and DIR/comparisons.json
  Rscript angle/glmm.R DIR           → DIR/fits.json  (on the L4)
  python -m angle.glmm read DIR      → results/glmm.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import sdt, sources
from .analysis import load_machine

BASE = Path("data")
M = 4


def _people_alone_for(t_index) -> pd.DataFrame:
    """The people of the map's ChatBench comparisons: all subjects, the answer alone (as analysis.chatbench_one)."""
    c = sources.chatbench_pairs(BASE)
    return c[c.question_id.isin(t_index)].dropna(subset=["acc", "acc_ai"]).reset_index(drop=True)


def _rows(people: pd.DataFrame, machine: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame({"type": "person", "worker": people.worker_id.astype(str), "question": people.question_id,
                      "succ": people.acc.astype(int), "n": 1})
    m = pd.DataFrame({"type": "machine", "worker": "machine", "question": machine.question,
                      "succ": machine.succ.astype(int), "n": machine.n.astype(int)})
    return pd.concat([p, m], ignore_index=True)


def comparisons() -> list[dict]:
    """The 25 comparisons, each with its in-house values and its data."""
    out = []
    mp = json.loads(Path("results/map.json").read_text())["comparisons"]
    for c in mp:
        if c.get("source") == "chatbench" and c.get("task") == "all":
            out.append({"set": "map", "folder": "gpu-results", "machine": c["machine"], "mode": c["mode"], "in_house": c})
    fr = json.loads(Path("results/frontier.json").read_text())["comparisons"]
    for name in ("haiku-5.5", "sonnet-5.5", "opus-5.5"):
        c = next(x for x in fr if x.get("source") == "chatbench" and x.get("task") == "all" and x["machine"] == name)
        out.append({"set": "claude", "folder": "closed-results", "machine": name, "mode": "low", "in_house": c})
    cs = json.loads(Path("results/chatbench-source.json").read_text())["comparisons"]
    for name in ("gpt-4o", "llama-3.1-8b"):
        c = next(x for x in cs if x["machine"] == name and x["subject"] == "all" and x["kind"] == "alone")
        out.append({"set": "table5", "machine": name, "mode": "alone", "in_house": c})
    return out


def data_for(c: dict) -> pd.DataFrame:
    if c["set"] in ("map", "claude"):
        t = load_machine(Path(c["folder"]), "chatbench", c["machine"], c["mode"])
        people = _people_alone_for(t.index)
        q = people.question_id.unique()
        machine = pd.DataFrame({"question": q, "succ": t.ok.reindex(q).to_numpy(), "n": 1})
    else:
        x = sources.chatbench_pairs(BASE)
        x = x[x.model == c["machine"]]
        a = pd.read_csv(BASE / "chatbench/ai_alone_answers.csv")
        a = a[a.model == c["machine"]].set_index("question_id")
        people = x[x.question_id.isin(a.index)].dropna(subset=["acc", "acc_ai"]).reset_index(drop=True)
        q = people.question_id.unique()
        n = a.free_text_count.reindex(q).to_numpy()
        machine = pd.DataFrame({"question": q, "succ": np.rint(a.free_text_acc.reindex(q).to_numpy() * n), "n": n})
    d = _rows(people, machine)
    ih = c["in_house"]       # the exported data are those of the in-house comparison
    pp = d[d.type == "person"]
    assert len(pp) == ih["n_trials"], (c["machine"], len(pp), ih["n_trials"])
    assert abs(pp.succ.mean() - ih["acc_h"]) < 1e-9, c["machine"]
    return d


def export(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, c in enumerate(comparisons()):
        d = data_for(c)
        d.to_csv(folder / f"{i:02d}.csv", index=False)
        ih = c["in_house"]
        meta.append({"i": i, "set": c["set"], "machine": c["machine"], "mode": c["mode"],
                     **{k: ih.get(k) for k in ("acc_h", "acc_m", "dh", "dm", "rho_correct", "rho", "phi", "phi_ci", "n_trials", "items")}})
    (folder / "comparisons.json").write_text(json.dumps(meta, indent=1))
    print(len(meta), "comparisons exported to", folder)


def read(folder: Path) -> None:
    meta = json.loads((folder / "comparisons.json").read_text())
    fits = {f["i"]: f for f in json.loads((folder / "fits.json").read_text())}
    rows = []
    for c in meta:
        f = fits[c["i"]]
        rc = f["rho_c"]
        rho = sdt.convert_rho_mafc(rc, c["dh"], c["dm"], M)
        phi = sdt.angle_degrees(rho)
        rows.append({**c, "glmm": {k: f[k] for k in f if k != "i"}, "glmm_rho_c": rc, "glmm_rho": rho, "glmm_phi": phi,
                     "diff_phi": phi - c["phi"], "within": bool(c["phi_ci"][0] <= phi <= c["phi_ci"][1])})
    pre = [r for r in rows if r["set"] != "table5-majority"]     # the 25 of the protocol; the majority rows came after
    d = np.array([abs(r["diff_phi"]) for r in pre])
    # frontier.json does not keep the in-house ρ_c of the Claude comparisons: the descriptive difference is on the others
    dr = np.array([abs(r["glmm_rho_c"] - r["rho_correct"]) for r in pre if r["rho_correct"] is not None])
    out = {"protocol": "PROTOCOL.md, A18 (cd8d98a, corrected 02df70e)", "comparisons": pre, "n": len(pre),
           "added_after_majority": [r for r in rows if r["set"] == "table5-majority"],
           "within": sum(r["within"] for r in pre), "passes": all(r["within"] for r in pre),
           "median_abs_diff_phi": float(np.median(d)), "max_abs_diff_phi": float(d.max()),
           "median_abs_diff_rho_c": float(np.median(dr)), "max_abs_diff_rho_c": float(dr.max()),
           "warnings": sum(bool(r["glmm"]["warnings"]) for r in pre), "singular": sum(bool(r["glmm"]["singular"]) for r in pre)}
    Path("results/glmm.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print("%-8s %-15s %-7s in-house %6.1f (%5.1f %5.1f) rho_c %s | glmm %6.1f rho_c %.3f %s%s" % (
            r["set"], r["machine"], r["mode"], r["phi"], *r["phi_ci"],
            "  -  " if r["rho_correct"] is None else "%.3f" % r["rho_correct"], r["glmm_phi"], r["glmm_rho_c"],
            "" if r["within"] else "OUT", " (warn)" if r["glmm"]["warnings"] else ""))
    print("within %d of %d; |Δφ| median %.1f max %.1f; |Δρ_c| median %.3f max %.3f; warnings %d singular %d"
          % (out["within"], out["n"], out["median_abs_diff_phi"], out["max_abs_diff_phi"], out["median_abs_diff_rho_c"],
             out["max_abs_diff_rho_c"], out["warnings"], out["singular"]))


if __name__ == "__main__" and sys.argv[1] in ("export", "read"):
    {"export": export, "read": read}[sys.argv[1]](Path(sys.argv[2]))


def majority(folder: Path) -> None:
    """Added after the outcome, descriptive: GPT-4o and Llama of table 5 with one answer per question, the majority of
    ChatBench's samples (right if at least half are right), through both estimators. Files 25 and 26."""
    from . import measure
    meta = json.loads((folder / "comparisons.json").read_text())
    extra = []
    for j, name in ((25, "gpt-4o"), (26, "llama-3.1-8b")):
        c = next(x for x in comparisons() if x["set"] == "table5" and x["machine"] == name)
        d = data_for(c)
        mrows = d.type == "machine"
        d.loc[mrows, "succ"] = (d.loc[mrows, "succ"] / d.loc[mrows, "n"] >= 0.5).astype(int)
        d.loc[mrows, "n"] = 1
        d.to_csv(folder / f"{j:02d}.csv", index=False)
        x = sources.chatbench_pairs(BASE)
        x = x[x.model == name]
        mach = d[mrows].set_index("question").succ.astype(float)
        t = sources.chatbench_trials(x, mach, None)
        r = measure.compare(t, m=M, repetitions=2000)
        extra.append({"i": j, "set": "table5-majority", "machine": name, "mode": "majority",
                      **{k: r.get(k) for k in ("acc_h", "acc_m", "dh", "dm", "rho_correct", "rho", "phi", "phi_ci", "n_trials", "items")}})
        print(name, "in-house with the majority: phi %.1f (%.1f %.1f), rho_c %.3f" % (r["phi"], *r["phi_ci"], r["rho_correct"]))
    (folder / "comparisons.json").write_text(json.dumps(meta + extra, indent=1))


if __name__ == "__main__" and sys.argv[1] == "majority":
    majority(Path(sys.argv[2]))
