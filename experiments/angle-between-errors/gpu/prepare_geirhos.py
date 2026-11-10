"""Downloads the model-vs-human images and the mapping to 16 categories. Data only: no code from the authors is run.

Usage:  python3 -I gpu/prepare_geirhos.py DESTINATION
"""
import ast
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{}.tar.gz"
H = "https://raw.githubusercontent.com/bethgelab/model-vs-human/master/modelvshuman/helper/"
NAMES = ["colour", "contrast", "edge", "eidolonI", "eidolonII", "eidolonIII", "false-colour", "high-pass", "low-pass",
         "phase-scrambling", "power-equalisation", "rotation", "silhouette", "sketch", "stylized", "uniform-noise", "cue-conflict"]
UA = {"User-Agent": "angle-between-errors/1.0 (download of public data)"}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size:
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        dest.write_bytes(r.read())


def categories(dest: Path):
    """human_categories.py is read as text: only the WNID frozensets are taken, without running it."""
    source = urllib.request.urlopen(urllib.request.Request(H + "human_categories.py", headers=UA), timeout=60).read().decode()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HumanCategories")
    groups = {}
    for n in cls.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "frozenset":
            groups[n.targets[0].id] = set(ast.literal_eval(n.value.args[0]))
    cat = urllib.request.urlopen(urllib.request.Request(H + "categories.txt", headers=UA), timeout=60).read().decode().splitlines()
    wnid = [r.split()[0] for r in cat]
    names = sorted(["knife", "keyboard", "elephant", "bicycle", "airplane", "clock", "oven", "chair", "bear", "boat", "cat",
                    "bottle", "truck", "car", "bird", "dog"])
    (dest / "categories.json").write_text(json.dumps({"categories": names, "indices": [[i for i, w in enumerate(wnid) if w in groups[n]] for n in names]}))


def main():
    dest = Path(sys.argv[1]).resolve(); (dest / "img").mkdir(parents=True, exist_ok=True); (dest / "tar").mkdir(exist_ok=True)
    categories(dest)
    for n in NAMES:
        t = dest / "tar" / f"{n}.tar.gz"
        fetch(BASE.format(n), t)
        with tarfile.open(t) as ar:
            members = [m for m in ar.getmembers() if m.isfile() and m.name.lower().endswith(".png")]
            ar.extractall(dest / "img" / n, members=members, filter="data")
        print(n, len(members), flush=True)
        t.unlink()
    print("END", flush=True)


if __name__ == "__main__":
    main()
