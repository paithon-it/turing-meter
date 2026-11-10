"""Task and people loading and validation (docs/SPEC.md, sections 4.1-4.2).

Task files may come from untrusted contributors: every path must stay inside
the task directory, and symlinks are resolved before the check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

CHECK_TYPES = {"script"}
KINDS = {"decision", "production"}
CONDITIONS = {"alone", "with_machine"}
FINAL_DECISIONS = {"human", "machine", "shared"}
ORDERS = {"alone_first", "machine_first", "counterbalanced"}


class TaskError(ValueError):
    """Invalid task or people file; the message names file, field, expectation."""


@dataclass(frozen=True)
class Person:
    id: str
    role: str
    experience: str
    year: int


@dataclass(frozen=True)
class Protocol:
    machine: str
    harness: str
    final_decision: str
    shows_confidence: bool
    order: str


@dataclass(frozen=True)
class People:
    people: dict[str, Person]
    protocols: dict[str, Protocol]

    def __contains__(self, pid: object) -> bool:
        return pid in self.people


@dataclass(frozen=True)
class HumanAttempt:
    person: str
    condition: str
    minutes: float
    passed: bool
    protocol: str | None = None


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    kind: str
    language: str
    prompt: str
    image: str
    files: tuple[str, ...]
    check_type: str
    check_command: str
    protected: tuple[str, ...]
    chance: float | None
    human: tuple[HumanAttempt, ...]
    path: Path


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise TaskError(f"{path}: cannot read file ({e.strerror})") from None
    except yaml.YAMLError as e:
        raise TaskError(f"{path}: invalid YAML ({e})") from None


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class _Ctx:
    def __init__(self, file: Path):
        self.file = file

    def err(self, field: str, expected: str, got: Any = None) -> TaskError:
        extra = "" if got is None else f", got {got!r}"
        return TaskError(f"{self.file}: field '{field}': expected {expected}{extra}")

    def get(self, d: dict, key: str, typ: type | tuple, label: str, prefix: str = "") -> Any:
        name = prefix + key
        if key not in d:
            raise self.err(name, f"{label} (field missing)")
        v = d[key]
        if isinstance(v, bool) and typ is not bool or not isinstance(v, typ):
            raise self.err(name, label, v)
        return v

    def str_list(self, v: Any, name: str) -> list[str]:
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            raise self.err(name, "a list of strings", v)
        return v


def _inside(root: Path, rel: str, ctx: _Ctx, field: str, must_exist: bool) -> None:
    if (not rel or "\x00" in rel or PurePosixPath(rel).is_absolute()
            or PureWindowsPath(rel).is_absolute() or PureWindowsPath(rel).drive
            or ".." in PurePosixPath(rel.replace("\\", "/")).parts):
        raise ctx.err(field, "a relative path inside the task directory", rel)
    target = root / rel
    resolved = target.resolve()  # follows symlinks
    if not resolved.is_relative_to(root.resolve()):
        raise ctx.err(field, "a path inside the task directory (symlink leaves it)", rel)
    if must_exist and not resolved.exists():
        raise ctx.err(field, "an existing file inside the task directory", rel)


def _walk_symlinks(root: Path, rel: str, ctx: _Ctx, field: str) -> None:
    """Reject symlinks inside a listed directory that point outside the task dir."""
    base = (root / rel)
    if not base.is_dir():
        return
    real_root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        for name in [*dirnames, *filenames]:
            p = Path(dirpath) / name
            if p.is_symlink() and not p.resolve().is_relative_to(real_root):
                raise ctx.err(field, "no symlink leaving the task directory",
                              str(p.relative_to(root)))


def load_people(path: Path) -> People:
    path = Path(path)
    data = _load_yaml(path)
    ctx = _Ctx(path)
    if not isinstance(data, dict):
        raise ctx.err("<root>", "a mapping with 'people' and 'protocols'", data)
    raw = ctx.get(data, "people", dict, "a mapping of person id to description")
    if not raw:
        raise ctx.err("people", "at least one person")
    people: dict[str, Person] = {}
    for pid, d in raw.items():
        p = f"people.{pid}."
        if not isinstance(d, dict):
            raise ctx.err(f"people.{pid}", "a mapping", d)
        people[str(pid)] = Person(
            id=str(pid),
            role=ctx.get(d, "role", str, "a string", p),
            experience=ctx.get(d, "experience", str, "a string", p),
            year=ctx.get(d, "year", int, "an integer year", p),
        )
    raw_pr = ctx.get(data, "protocols", dict, "a mapping of protocol id to description")
    if not raw_pr:
        raise ctx.err("protocols", "at least one protocol")
    protocols: dict[str, Protocol] = {}
    for qid, pr in raw_pr.items():
        pre = f"protocols.{qid}."
        if not isinstance(pr, dict):
            raise ctx.err(f"protocols.{qid}", "a mapping", pr)
        fd = ctx.get(pr, "final_decision", str, "a string", pre)
        if fd not in FINAL_DECISIONS:
            raise ctx.err(pre + "final_decision", f"one of {sorted(FINAL_DECISIONS)}", fd)
        order = ctx.get(pr, "order", str, "a string", pre)
        if order not in ORDERS:
            raise ctx.err(pre + "order", f"one of {sorted(ORDERS)}", order)
        protocols[str(qid)] = Protocol(
            machine=ctx.get(pr, "machine", str, "a string", pre),
            harness=ctx.get(pr, "harness", str, "a string", pre),
            final_decision=fd,
            shows_confidence=ctx.get(pr, "shows_confidence", bool, "a boolean", pre),
            order=order,
        )
    return People(people, protocols)


def load_task(task_dir: Path, people: People) -> Task:
    task_dir = Path(task_dir)
    file = task_dir / "task.yaml"
    if file.is_symlink() and not file.resolve().is_relative_to(task_dir.resolve()):
        raise TaskError(f"{file}: task.yaml is a symlink that leaves the task directory")
    data = _load_yaml(file)
    ctx = _Ctx(file)
    if not isinstance(data, dict):
        raise ctx.err("<root>", "a mapping", data)

    tid = ctx.get(data, "id", str, "a string")
    strs = {k: ctx.get(data, k, str, "a string") for k in ("family", "language", "prompt")}
    kind = ctx.get(data, "kind", str, "'decision' or 'production'")
    if kind not in KINDS:
        raise ctx.err("kind", "'decision' or 'production'", kind)

    if "chance" not in data:
        raise ctx.err("chance", "null, or a number in (0, 1) for decision tasks (field missing)")
    chance = data["chance"]
    if kind == "decision":
        if not _is_num(chance) or not 0 < chance < 1:
            raise ctx.err("chance", "a number in (0, 1) for a decision task", chance)
        chance = float(chance)
    elif chance is not None:
        raise ctx.err("chance", "null for a production task", chance)

    env = ctx.get(data, "environment", dict, "a mapping")
    image = ctx.get(env, "image", str, "a string", "environment.")
    files = ctx.str_list(ctx.get(env, "files", list, "a list of paths", "environment."),
                         "environment.files")
    for f in files:
        _inside(task_dir, f, ctx, "environment.files", must_exist=True)
        _walk_symlinks(task_dir, f, ctx, "environment.files")

    chk = ctx.get(data, "check", dict, "a mapping")
    ctype = ctx.get(chk, "type", str, "a string", "check.")
    if ctype not in CHECK_TYPES:
        raise ctx.err("check.type", f"one of {sorted(CHECK_TYPES)}", ctype)
    command = ctx.get(chk, "command", str, "a string", "check.")
    protected = ctx.str_list(ctx.get(chk, "protected", list, "a list of paths", "check."),
                             "check.protected")
    for f in protected:
        _inside(task_dir, f, ctx, "check.protected", must_exist=True)
        _walk_symlinks(task_dir, f, ctx, "check.protected")

    human: list[HumanAttempt] = []
    for i, e in enumerate(ctx.get(data, "human", list, "a list of attempts")):
        pre = f"human[{i}]."
        if not isinstance(e, dict):
            raise ctx.err(f"human[{i}]", "a mapping", e)
        person = ctx.get(e, "person", str, "a person id", pre)
        if person not in people:
            raise ctx.err(pre + "person",
                          f"an id declared in people.yaml ({sorted(people.people)})", person)
        cond = ctx.get(e, "condition", str, "'alone' or 'with_machine'", pre)
        if cond not in CONDITIONS:
            raise ctx.err(pre + "condition", "'alone' or 'with_machine'", cond)
        proto = e.get("protocol")
        if cond == "with_machine":
            if not isinstance(proto, str) or proto not in people.protocols:
                raise ctx.err(pre + "protocol",
                              f"a protocol id declared in people.yaml ({sorted(people.protocols)})",
                              proto)
        elif "protocol" in e:
            raise ctx.err(pre + "protocol", "absent for an 'alone' attempt", proto)
        minutes = e.get("minutes")
        if not _is_num(minutes) or not minutes > 0:
            raise ctx.err(pre + "minutes", "a positive number", minutes)
        passed = e.get("passed")
        if not isinstance(passed, bool):
            raise ctx.err(pre + "passed", "a boolean", passed)
        human.append(HumanAttempt(person, cond, float(minutes), passed, proto))

    return Task(id=tid, image=image, files=tuple(files), check_type=ctype,
                check_command=command, protected=tuple(protected), chance=chance,
                human=tuple(human), path=task_dir, kind=kind, **strs)


def validate_tree(root: Path) -> list[tuple[str, list[str]]]:
    """Validate every task under root. Returns (task dir, errors) pairs.

    Looks for people.yaml in the task's directory or any parent up to root.
    """
    root = Path(root)
    results: list[tuple[str, list[str]]] = []
    for tf in sorted(root.rglob("task.yaml")):
        d = tf.parent
        name = str(d)
        pf = next((p / "people.yaml" for p in [d, *d.parents]
                   if (p / "people.yaml").is_file()
                   and (p == root or root in p.parents)), None)
        try:
            if pf is None:
                raise TaskError(f"{tf}: no people.yaml found in {root} or its subdirectories")
            load_task(d, load_people(pf))
            results.append((name, []))
        except TaskError as e:
            results.append((name, [str(e)]))
    return results
