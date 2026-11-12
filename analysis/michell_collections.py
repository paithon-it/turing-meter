"""Which collections the buckets mix, and the subsets to rerun the test on.

METR's suite joins three collections: SWAA (five families of short tasks), RE-Bench
(the ai_rd_ families) and HCAST (all the others).
  1. how many tasks of each collection fall in each duration bucket, and which families
     fill the two middle buckets;
  2. writes two subsets of METR's file, without SWAA and HCAST only, to rerun
     michell_test.py and the follow-ups on each;
  3. lists the violations of double cancellation in the observed table, with their
     margins, to choose an example.

Usage: python michell_collections.py task_results_1_1.yaml OUTPUT_DIR [--bins 6]
"""
import argparse
import itertools
import os

import numpy as np
import yaml

import michell_models as mm
import michell_test as m

SWAA = {"file_selection", "alert_triage", "request_routing", "code_completion", "arithmetic"}
COLLECTIONS = ("SWAA", "HCAST", "RE-Bench")


def collection(task_id):
    family = task_id.split("/")[0]
    if family in SWAA:
        return "SWAA"
    if family.startswith("ai_rd_"):
        return "RE-Bench"
    return "HCAST"


def write_subset(path, keep, out):
    d = yaml.safe_load(open(path))
    for v in d["agents"].values():
        v["tasks"] = [t for t in v["tasks"] if keep(collection(t["task_id"]))]
    yaml.safe_dump(d, open(out, "w"), sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_results")
    ap.add_argument("output_dir")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--show", type=int, default=25, help="violations to list, largest margin first")
    args = ap.parse_args()
    D = m.Data(args.task_results)
    col = np.array([collection(t) for t in D.tasks])
    print("tasks per collection:", {c: int((col == c).sum()) for c in COLLECTIONS})
    nb = args.bins
    bucket, qs = mm.buckets(D, nb)
    for k in range(nb):
        sel = bucket == k
        print(f"  bucket {k+1} ({2**qs[k]:.2f}-{2**qs[k+1]:.2f} min): "
              + ", ".join(f"{c} {int((col[sel] == c).sum())}" for c in COLLECTIONS))
    for k in (2, 3):
        families = sorted({D.tasks[j].split("/")[0] for j in np.where(bucket == k)[0]})
        print(f"  families in bucket {k+1} ({len(families)}):", ", ".join(families))

    write_subset(args.task_results, lambda c: c != "SWAA", os.path.join(args.output_dir, "without_swaa.yaml"))
    write_subset(args.task_results, lambda c: c == "HCAST", os.path.join(args.output_dir, "hcast_only.yaml"))
    print(f"written: {os.path.join(args.output_dir, 'without_swaa.yaml')}, {os.path.join(args.output_dir, 'hcast_only.yaml')}")

    # If A in y >= B in x and B in z >= C in y, then A in z >= C in x. Margin: the smallest of the three gaps.
    order = mm.agent_order(D)
    P, W = m.bucket_matrix(D, D.S, bucket, order, nb)
    names = [D.agents[i] for i in order]
    found = []
    for a, b, c in itertools.permutations(range(D.nA), 3):
        for x, y, z in itertools.permutations(range(nb), 3):
            ay, bx, bz, cy, az, cx = P[a, y], P[b, x], P[b, z], P[c, y], P[a, z], P[c, x]
            if ay >= bx and bz >= cy and az < cx:
                margin = min(ay - bx, bz - cy, cx - az)
                found.append((margin, names[a], names[b], names[c], x + 1, y + 1, z + 1,
                              ay, bx, bz, cy, az, cx, int(min(W[a, y], W[b, x], W[b, z], W[c, y], W[a, z], W[c, x]))))
    found.sort(reverse=True)
    print(f"violations (all cells, no minimum of runs): {len(found)}; with margin >= 0.05: {sum(t[0] >= 0.05 for t in found)}; "
          f">= 0.10: {sum(t[0] >= 0.10 for t in found)}")
    for t in found[:args.show]:
        print(f"  margin {t[0]:.3f} | A={t[1]} B={t[2]} C={t[3]} | x={t[4]} y={t[5]} z={t[6]} | "
              f"Ay {t[7]:.2f} >= Bx {t[8]:.2f}; Bz {t[9]:.2f} >= Cy {t[10]:.2f}; but Az {t[11]:.2f} < Cx {t[12]:.2f} | fewest runs {t[13]}")


if __name__ == "__main__":
    main()
