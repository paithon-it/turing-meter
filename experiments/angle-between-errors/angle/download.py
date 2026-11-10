"""Downloads the source data: data files only, never the authors' code.

Usage, from the experiment folder:  python3 -I angle/download.py DESTINATION chaos haiid chatbench geirhos

Each source ends up in its own folder. Only the listed extensions are downloaded;
from archives, only the file that is needed is extracted.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "angle-between-errors/1.0 (download of public data)"}
GITHUB = "https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
EXTENSIONS = {".csv", ".json", ".jpg", ".jpeg", ".png", ".txt", ".jsonl"}


def fetch(url: str, dest: Path, attempts: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for n in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                dest.write_bytes(r.read())
            return
        except Exception:
            if n == attempts - 1:
                raise
            time.sleep(2 + 3 * n)


def tree(repo: str, ref: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(GITHUB.format(repo=repo, ref=ref), headers=UA), timeout=60) as r:
        return json.load(r)["tree"]


def from_github(repo: str, ref: str, prefixes: tuple[str, ...], dest: Path, extensions=EXTENSIONS) -> int:
    from urllib.parse import quote

    n = 0
    for entry in tree(repo, ref):
        p = entry["path"]
        if entry["type"] != "blob" or not p.startswith(prefixes) or Path(p).suffix.lower() not in extensions:
            continue
        fetch(RAW.format(repo=repo, ref=ref, path=quote(p)), dest / p)
        n += 1
    return n


def chaos(base: Path) -> None:
    """From the mirror on Hugging Face: the authors' Dropbox link no longer responds (CC BY-NC 4.0: not redistributed)."""
    dest = base / "chaos" / "alphanli.jsonl"
    fetch("https://huggingface.co/datasets/earino/chaosnli/resolve/main/raw/chaosNLI_alphanli.jsonl", dest)
    print("chaos", dest.stat().st_size)


def haiid(base: Path) -> None:
    n = from_github("kailas-v/human-ai-interactions", "main", ("haiid_dataset.csv", "tasks/"), base / "haiid")
    print("haiid", n)


def chatbench(base: Path) -> None:
    root = "https://huggingface.co/datasets/microsoft/ChatBench/resolve/main/"
    for f in ("questions.csv", "ai_alone_answers.csv", "full_study/user_answers_filtered.csv", "full_study/conversations.json"):
        fetch(root + f, base / "chatbench" / f)
    print("chatbench ok")


def geirhos(base: Path) -> None:
    n = from_github("bethgelab/model-vs-human", "master", ("raw-data/",), base / "geirhos", {".csv"})
    print("geirhos csv", n)


if __name__ == "__main__":
    base = Path(sys.argv[1]).resolve()
    for source in sys.argv[2:]:
        {"chaos": chaos, "haiid": haiid, "chatbench": chatbench, "geirhos": geirhos}[source](base)
