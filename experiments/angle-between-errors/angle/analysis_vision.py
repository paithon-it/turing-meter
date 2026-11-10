"""model-vs-human: people and networks on the same images, 16 categories. Writes results/vision.json.

Usage:  python -m angle.analysis_vision GPU_RESULTS_DIR
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import combiner, sdt
from .measure import Trials, compare, conversion_table

BASE = Path("data/geirhos/raw-data")
m = 16


_NAME_PATTERN = re.compile(r"^\d+_[a-z]+_(?:s\d+|dnn)_(.+)$")


def image_key(name: str, dataset: str) -> str:
    """People and networks saw the same images under different names: the session prefix is removed."""
    found = _NAME_PATTERN.match(name)
    rest = found.group(1) if found else name
    if dataset in ("edge", "silhouette"):        # in the networks' files they are just called cat1.png
        return rest.rsplit("_", 1)[-1]
    return rest


def load_humans(dataset):
    rows = []
    for f in sorted((BASE / dataset).glob("*.csv")):
        rows.append(pd.read_csv(f))
    d = pd.concat(rows, ignore_index=True)
    d["correct"] = (d.object_response == d.category).astype(float)
    d["dataset"] = dataset
    return d


def load_network(folder, name):
    rows = [json.loads(l) for l in open(folder / f"vision__{name}.jsonl", encoding="utf-8")]
    return rows


def one(args):
    folder, name, dataset = args
    d = load_humans(dataset)
    if "cue" in dataset:
        return []                      # cue-conflict has no single correct answer
    cat = json.load(open(Path(folder).parent / "categories.json"))["categories"] if (Path(folder).parent / "categories.json").exists() else None
    if cat is None:
        cat = sorted(["knife", "keyboard", "elephant", "bicycle", "airplane", "clock", "oven", "chair", "bear", "boat", "cat",
                      "bottle", "truck", "car", "bird", "dog"])
    rows = load_network(folder, name)
    tab = {}
    for r in rows:
        if r["rel"].split("/")[0] != dataset:
            continue
        p = np.array(r["prob"])
        tab[image_key(r["img"], dataset)] = (int(p.argmax()), float(p.max()))
    d["imagename"] = d.imagename.map(lambda n: image_key(n, dataset))
    d = d[d.imagename.isin(tab)].copy()
    if d.empty:
        return [{"error": "no image matches", "dataset": dataset, "machine": name}]
    d["top"] = d.imagename.map(lambda x: tab[x][0])
    d["ptop"] = d.imagename.map(lambda x: tab[x][1])
    d["opt_h"] = d.object_response.map({c: i for i, c in enumerate(cat)}).fillna(-1).astype(int)
    d["correct_option"] = d.category.map({c: i for i, c in enumerate(cat)})
    d["m_ok"] = (d.top == d.correct_option).astype(float)
    out = []
    for cond, x in d.groupby("condition"):
        if x.imagename.nunique() < 20 or x.correct.mean() <= 1 / m or x.m_ok.mean() <= 1 / m:
            continue
        p, _ = pd.factorize(x.subj, sort=True)
        c, items = pd.factorize(x.imagename, sort=True)
        mi = x.drop_duplicates("imagename").set_index("imagename").m_ok.reindex(items).to_numpy(float)
        r = compare(Trials(p, c, x.correct.to_numpy(float), mi), m=m, repetitions=600)
        r.update(source="geirhos", dataset=dataset, condition=str(cond), machine=name, image_items=int(x.imagename.nunique()))
        if "excluded" not in r:
            lm = np.log(np.clip(x.ptop.to_numpy(), 1e-5, 1 - 1e-5) / (1 - np.clip(x.ptop.to_numpy(), 1e-5, 1 - 1e-5)))
            combo = combiner.combine(x.correct.to_numpy(float), x.m_ok.to_numpy(float), x.opt_h.to_numpy(), x.top.to_numpy(),
                                     np.column_stack([np.clip(lm, -8, 8)]), c)
            pred_phi = sdt.d_from_accuracy(float(np.clip(combiner.predict(r["dh"], r["dm"], r["rho"], m, None, n=200_000), 1e-4, 1 - 1e-4)), m)
            pred_90 = sdt.d_from_accuracy(float(np.clip(combiner.predict(r["dh"], r["dm"], 0.0, m, None, n=200_000), 1e-4, 1 - 1e-4)), m)
            acc = combo["acc_pair"]
            r["law"] = {"d_combo": sdt.d_from_accuracy(float(np.clip(acc, 1e-4, 1 - 1e-4)), m) if acc > 1 / m else float("nan"),
                        "pred_phi": pred_phi, "pred_90": pred_90, "dmax": max(r["dh"], r["dm"]), "acc_combo": acc}
            r["combo"] = {k: v for k, v in combo.items() if k != "outcome"}
        out.append(r)
    return out


def main(folder: Path):
    networks = sorted(p.name[len("vision__"):-len(".jsonl")] for p in folder.glob("vision__*.jsonl"))
    sets = sorted(p.name for p in BASE.iterdir() if p.is_dir())
    jobs = [(folder, r, d) for r in networks for d in sets]
    everything = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for out in ex.map(one, jobs):
            everything += out
    Path("results").mkdir(exist_ok=True)
    Path("results/vision.json").write_text(json.dumps(everything, indent=0))
    print(len(everything), "rows", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
