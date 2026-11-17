"""A3: the share of the machine's errors that the ideal pair removes at 90 degrees, and its limit for large k.

Pre-registered in PROTOCOL.md, "After the critical review (v4): the new measures". A property of the model,
not a measurement: from the data only the accuracy of the article's person. Writes results/errors-removed.json.

At 90 degrees the ideal pair with shared cues has |S| = sqrt(d_H² + d_M²), d_M = k·d_H. The ratio of the
pair's errors to the machine's, e(|S|)/e(d_M), for large k:
- choice among m options, e(d) ~ (m − 1)·Φ(−d/√2): ratio ~ (d_M/|S|)·exp(−d_H²/4) → exp(−d_H²/4);
- yes/no with d′ = 2z(a), e(d) = Φ(−d/2): ratio ~ (d_M/|S|)·exp(−d_H²/8) → exp(−d_H²/8).
The prefactor d_M/|S| = k/sqrt(1 + k²) goes to one slowly, so the limit holds only from some k on.

Usage, from the experiment folder:  python -m angle.errors_removed
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from . import sdt

KS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0)
DHS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
GAP = 0.05                       # the criterion: "does not depend on k" only where the gap from the limit is below 5 percent


D_MAX = 12.0                     # sensitivities beyond this have errors below 1e-17: not a case the article meets


def error(d: float, m: int | None) -> float:
    """Error of an observer with sensitivity d: m options (forced choice), or yes/no with d = 2z(a) when m is None.

    Computed directly, not as 1 − accuracy, which loses all precision for a near-perfect machine:
    e = ∫ φ(x − d)·[1 − Φ(x)^(m−1)] dx, with 1 − Φ^(m−1) = −expm1((m − 1)·log1p(−Φ(−x))).
    """
    if m is None:
        return float(norm.sf(d / 2.0))
    def f(x):
        with np.errstate(divide="ignore"):
            return float(norm.pdf(x - d) * -np.expm1((m - 1) * np.log1p(-norm.sf(x))))
    lo, hi = min(-12.0, d / 2 - 12), d + 12
    return float(quad(f, lo, hi, points=[d / 2, d], epsabs=0.0, epsrel=1e-10, limit=400)[0])


def limit(dh: float, m: int | None) -> float:
    return math.exp(-dh * dh / (8.0 if m is None else 4.0))


def run() -> dict:
    out = {"criterion_relative_gap": GAP, "conventions": {}}
    for label, m in (("m=2", 2), ("m=4", 4), ("yes/no", None)):
        rows = []
        for dh in DHS:
            lim = limit(dh, m)
            for k in KS:
                dm = k * dh
                if dm > D_MAX:
                    continue
                s = math.hypot(dh, dm)
                ratio = error(s, m) / error(dm, m)
                rows.append({"dh": dh, "k": k, "ratio": ratio, "removed": 1.0 - ratio, "limit": lim,
                             "relative_gap": abs(ratio - lim) / lim, "prefactor": dm / s})
        # from which k on the gap stays below the criterion, for each d_H
        from_k = {}
        for dh in DHS:
            r = [x for x in rows if x["dh"] == dh]
            ok = [x["k"] for x in r if all(y["relative_gap"] < GAP for y in r if y["k"] >= x["k"])]
            from_k[str(dh)] = min(ok) if ok else None
        out["conventions"][label] = {"rows": rows, "gap_below_criterion_from_k": from_k}
    # the article's person: the ChatBench group next to GPT-4o, right 57.7 times in a hundred among four options, the
    # same person as the margin table (verdict.margin_table). Up to the fourth review this used exactly 0.58 (d_H 1.089
    # against 1.080), and the two places of the article gave two limits for one person.
    person = json.loads(Path("results/chatbench-source.json").read_text())
    person = next(c for c in person["comparisons"] if c["machine"] == "gpt-4o" and c["subject"] == "all" and c["kind"] == "alone")
    dh58 = person["dh"]
    out["person_58_of_100"] = {"accuracy_alone": person["acc_h"], "dh": dh58, "limit_ratio": limit(dh58, 4), "limit_removed": 1.0 - limit(dh58, 4),
                               "by_k": [{"k": k, "removed": 1.0 - error(math.hypot(dh58, k * dh58), 4) / error(k * dh58, 4)} for k in KS if k * dh58 <= D_MAX]}
    return out


def main() -> None:
    out = run()
    Path("results/errors-removed.json").write_text(json.dumps(out, indent=1))
    for label, v in out["conventions"].items():
        print(label, "gap below 5% from k =", v["gap_below_criterion_from_k"])
    p = out["person_58_of_100"]
    print("person 58/100: d_H %.3f, limit removed %.3f;" % (p["dh"], p["limit_removed"]),
          " ".join("k=%g %.3f" % (x["k"], x["removed"]) for x in p["by_k"]))


if __name__ == "__main__":
    main()
