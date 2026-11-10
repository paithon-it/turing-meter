import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from turing_meter.tasks import TaskError, load_people, load_task, validate_tree

EX = Path(__file__).parent.parent / "tasks" / "examples"


@pytest.fixture
def tree(tmp_path):
    shutil.copytree(EX, tmp_path / "t")
    t = tmp_path / "t"
    f = t / "invoice-total-001" / "task.yaml"
    d = yaml.safe_load(f.read_text())
    d["human"] = [
        {"person": "p01", "condition": "alone", "minutes": 4.2, "passed": True},
        {"person": "p01", "condition": "with_machine", "protocol": "series",
         "minutes": 2.9, "passed": True},
    ]
    f.write_text(yaml.safe_dump(d))
    return t


def edit(tree, fn):
    p = tree / "invoice-total-001" / "task.yaml"
    d = yaml.safe_load(p.read_text())
    fn(d)
    p.write_text(yaml.safe_dump(d))


def errors(tree):
    people = load_people(tree / "people.yaml")
    with pytest.raises(TaskError) as e:
        load_task(tree / "invoice-total-001", people)
    return str(e.value)


def test_example_valid():
    people = load_people(EX / "people.yaml")
    t = load_task(EX / "invoice-total-001", people)
    assert t.id == "invoice-total-001"


def test_validate_tree_ok():
    res = validate_tree(EX)
    assert res and all(not errs for _, errs in res)


def test_missing_check(tree):
    edit(tree, lambda d: d.pop("check"))
    msg = errors(tree)
    assert "task.yaml" in msg and "check" in msg


def test_bad_check_type(tree):
    edit(tree, lambda d: d["check"].update(type="llm"))
    assert "check.type" in errors(tree)


def test_unknown_person(tree):
    edit(tree, lambda d: d["human"][0].update(person="ghost"))
    msg = errors(tree)
    assert "ghost" in msg and "person" in msg


def test_bad_condition(tree):
    edit(tree, lambda d: d["human"][0].update(condition="together"))
    assert "condition" in errors(tree)


def test_bad_kind(tree):
    edit(tree, lambda d: d.update(kind="other"))
    assert "kind" in errors(tree)


def test_decision_needs_chance(tree):
    edit(tree, lambda d: d.update(kind="decision", chance=None))
    assert "chance" in errors(tree)


@pytest.mark.parametrize("c", [0, 1, 1.5, -0.2, "half", True])
def test_chance_out_of_range(tree, c):
    edit(tree, lambda d: d.update(kind="decision", chance=c))
    assert "chance" in errors(tree)


def test_decision_with_chance_ok(tree):
    edit(tree, lambda d: d.update(kind="decision", chance=0.5))
    people = load_people(tree / "people.yaml")
    assert load_task(tree / "invoice-total-001", people).chance == 0.5


def test_production_chance_must_be_null(tree):
    edit(tree, lambda d: d.update(chance=0.5))
    assert "chance" in errors(tree)


def test_with_machine_needs_protocol(tree):
    edit(tree, lambda d: d["human"][1].pop("protocol"))
    assert "protocol" in errors(tree)


def test_unknown_protocol(tree):
    edit(tree, lambda d: d["human"][1].update(protocol="ghost"))
    msg = errors(tree)
    assert "ghost" in msg and "protocol" in msg


def test_alone_must_not_carry_protocol(tree):
    edit(tree, lambda d: d["human"][0].update(protocol="series"))
    assert "protocol" in errors(tree)


def test_protocols_must_be_nonempty(tree):
    p = tree / "people.yaml"
    d = yaml.safe_load(p.read_text())
    d["protocols"] = {}
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(TaskError) as e:
        load_people(p)
    assert "protocols" in str(e.value)


def test_example_has_no_invented_measurements():
    d = yaml.safe_load((EX / "invoice-total-001" / "task.yaml").read_text())
    assert d["human"] == []


def test_people_protocol_validated(tree):
    p = tree / "people.yaml"
    d = yaml.safe_load(p.read_text())
    d["protocols"]["series"]["final_decision"] = "nobody"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(TaskError) as e:
        load_people(p)
    assert "final_decision" in str(e.value)


def test_people_order_and_confidence_validated(tree):
    p = tree / "people.yaml"
    for key, val in (("order", "random"), ("shows_confidence", "no")):
        d = yaml.safe_load((EX / "people.yaml").read_text())
        d["protocols"]["series"][key] = val
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(TaskError) as e:
            load_people(p)
        assert key in str(e.value)


def test_path_traversal(tree):
    edit(tree, lambda d: d["check"].update(protected=["../x"]))
    assert "protected" in errors(tree)


def test_absolute_path(tree):
    edit(tree, lambda d: d["check"].update(protected=["/etc/passwd"]))
    assert "protected" in errors(tree)


def test_missing_protected_file(tree):
    edit(tree, lambda d: d["check"].update(protected=["nope.py"]))
    assert "nope.py" in errors(tree)


def test_environment_files_traversal(tree):
    edit(tree, lambda d: d["environment"].update(files=["../"]))
    assert "environment.files" in errors(tree)


def test_negative_minutes(tree):
    edit(tree, lambda d: d["human"][0].update(minutes=-1))
    assert "minutes" in errors(tree)


def test_passed_must_be_bool(tree):
    edit(tree, lambda d: d["human"][0].update(passed="yes"))
    assert "passed" in errors(tree)


def test_symlink_escape(tree, tmp_path):
    outside = tmp_path / "secret.py"
    outside.write_text("x")
    link = tree / "invoice-total-001" / "evil.py"
    os.symlink(outside, link)
    edit(tree, lambda d: d["check"].update(protected=["evil.py"]))
    assert "protected" in errors(tree)


def test_path_variants_rejected(tree):
    for bad in ["C:/x", "C:\\x", "\\\\srv\\share", "a/../../x", "a\\..\\x"]:
        edit(tree, lambda d: d["check"].update(protected=[bad]))
        assert "protected" in errors(tree), bad


def test_symlink_inside_files_dir_escapes(tree, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    os.symlink(outside, tree / "invoice-total-001" / "inputs" / "leak.txt")
    msg = errors(tree)
    assert "environment.files" in msg and "leak.txt" in msg


def test_symlink_inside_files_dir_ok(tree):
    d = tree / "invoice-total-001" / "inputs"
    os.symlink(d / "fattura-001.txt", d / "alias.txt")
    people = load_people(tree / "people.yaml")
    load_task(tree / "invoice-total-001", people)


def test_symlinked_subdir_escape(tree, tmp_path):
    outside = tmp_path / "outdir"
    outside.mkdir()
    (outside / "f").write_text("x")
    os.symlink(outside, tree / "invoice-total-001" / "inputs" / "sub")
    assert "environment.files" in errors(tree)


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "turing_meter", *args],
                          capture_output=True, text=True)


def test_cli_ok():
    r = run_cli("validate", str(EX))
    assert r.returncode == 0
    assert "invoice-total-001" in r.stdout


def test_cli_fail(tree):
    edit(tree, lambda d: d.pop("check"))
    r = run_cli("validate", str(tree))
    assert r.returncode == 1
    assert "invoice-total-001" in r.stdout
