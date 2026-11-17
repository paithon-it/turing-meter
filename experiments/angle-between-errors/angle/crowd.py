"""A5: the crowd. Pre-registered in PROTOCOL.md, "After the critical review (v4): the new measures".

The ceiling of a large crowd whose members have sensitivity d′ and pairwise correlation ρ between their cues is
d′/√ρ (classic, for pooled graded evidence and equal sensitivities). With ρ_HH (two people alone) and ρ_SS (two
people with the chat), as in "Two people with the chat beside them" (homogenization.py):
    Tu_crowd = (d′_S/√ρ_SS) / (d′_H/√ρ_HH) = Tu · √(ρ_HH/ρ_SS).
One bootstrap over people and questions (Poisson weights, 2,000 replicas, seed 534) carries d′_H, d′_S and the two
ρ with the same weights. Hypothesis: Tu_crowd/Tu < 1; passes if the upper end of its interval is below 1.

Usage, from the experiment folder:  python -m angle.crowd   → results/crowd.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import conversations, sdt
from .homogenization import conversion_table, to_cues

REPLICAS, SEED, M = 2000, 534, 4


def _one(p, q, alone, chat, wp, wq, tables) -> dict | None:
    w = wp[p] * wq[q]
    if w.sum() == 0:
        return None
    r_h = to_cues(sdt.rho_person_pairs(p, q, alone, item_weights=wq, person_weights=wp), tables[0])
    r_s = to_cues(sdt.rho_person_pairs(p, q, chat, item_weights=wq, person_weights=wp), tables[1])
    a_h, a_s = float(np.sum(w * alone) / w.sum()), float(np.sum(w * chat) / w.sum())
    if min(a_h, a_s) <= 1 / M:
        return None
    dh, ds = sdt.d_from_accuracy(a_h, M), sdt.d_from_accuracy(a_s, M)
    out = {"rho_hh": r_h, "rho_ss": r_s, "dh": dh, "ds": ds, "turing": ds / dh}
    if r_h > 0 and r_s > 0:
        out["ratio"] = float(np.sqrt(r_h / r_s))
        out["turing_crowd"] = out["turing"] * out["ratio"]
        out["crowd_alone"] = dh / np.sqrt(r_h)
        out["crowd_chat"] = ds / np.sqrt(r_s)
    return out


def measure(x: pd.DataFrame) -> dict:
    p, _ = pd.factorize(x.worker_id, sort=True)
    q, _ = pd.factorize(x.question_id, sort=True)
    alone, chat = x.acc.to_numpy(float), x.acc_ai.to_numpy(float)
    tables = (conversion_table(alone.mean()), conversion_table(chat.mean()))
    point = _one(p, q, alone, chat, np.ones(p.max() + 1), np.ones(q.max() + 1), tables)
    gen = np.random.default_rng(SEED)
    reps = [_one(p, q, alone, chat, gen.poisson(1.0, p.max() + 1).astype(float), gen.poisson(1.0, q.max() + 1).astype(float), tables)
            for _ in range(REPLICAS)]
    reps = [r for r in reps if r is not None]
    out = {"trials": int(len(x)), "people": int(p.max() + 1), "questions": int(q.max() + 1), "replicas": len(reps),
           "replicas_with_rho_not_positive": int(sum("ratio" not in r for r in reps)), **point}
    for k in ("rho_hh", "rho_ss", "turing", "ratio", "turing_crowd", "crowd_alone", "crowd_chat"):
        s = np.array([r[k] for r in reps if k in r], float)
        out[k + "_ci"] = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    out["phi_hh"], out["phi_ss"] = sdt.angle_degrees(point["rho_hh"]), sdt.angle_degrees(point["rho_ss"])
    out["passes"] = bool(out["ratio_ci"][1] < 1)
    return out


def main() -> None:
    out = {"replicas": REPLICAS, "seed": SEED}
    for model in conversations.MODELS:
        out[model] = measure(conversations.trials(model))
        r = out[model]
        print(model, "Tu %.2f (%.2f-%.2f)" % (r["turing"], *r["turing_ci"]), "| crowd %.2f (%.2f-%.2f)" % (r["turing_crowd"], *r["turing_crowd_ci"]),
              "| ratio %.3f (%.3f-%.3f)" % (r["ratio"], *r["ratio_ci"]), "| angles %.0f -> %.0f" % (r["phi_hh"], r["phi_ss"]),
              "| passes", r["passes"], flush=True)
    Path("results/crowd.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
