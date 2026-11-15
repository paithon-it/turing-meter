"""`turing-meter pair`: the program tested on sheets whose truth is known (docs/PAIR-PROTOCOL.md)."""

import json
import math

import numpy as np
import pytest

from turing_meter import mchoice, pair
from turing_meter.cli import main

LETTERS = np.array(list("ABCD"))
DH, DM, RHO = 1.080, 2.338, 0.267   # the 58% person and the 88% machine: k 2.16, angle 74.5
TRUTH = {"k": DM / DH, "phi": math.degrees(math.acos(RHO)), "margin": mchoice.ideal(DH, DM, RHO) / DM - 1}


def choices(n, m, rho, dh, dm, gen):
    """The answers (0 is the right option) of two minds whose evidence has correlation rho."""
    common = gen.standard_normal((n, m))
    w = math.sqrt(abs(rho))
    a = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
    b = w * common + math.sqrt(1 - abs(rho)) * gen.standard_normal((n, m))
    a[:, 0] += dh
    b[:, 0] += dm
    return a.argmax(1), b.argmax(1)


def sheet(ca, cb, extra=None):
    d = {"key": ["A"] * ca.size, "person": list(LETTERS[ca]), "machine": list(LETTERS[cb])}
    if extra:
        d.update(extra)
    return d


def test_large_known_sheet_recovers_k_angle_margin():
    ca, cb = choices(6000, 4, RHO, DH, DM, np.random.default_rng(7))
    ref = pair.analyze(sheet(ca, cb), 4, B=300)
    assert ref["k"]["estimate"] == pytest.approx(TRUTH["k"], abs=0.1)
    assert ref["phi"]["estimate"] == pytest.approx(TRUTH["phi"], abs=3.0)
    assert ref["margin"]["estimate"] == pytest.approx(TRUTH["margin"], abs=0.02)
    assert ref["phi"]["status"] == "estimated"
    assert ref["zone"] == "A"
    lo, hi = ref["k"]["ci95"]
    assert lo < TRUTH["k"] < hi


def test_zero_machine_errors_gives_a_lower_bound_and_no_angle():
    gen = np.random.default_rng(7)
    ca, _ = choices(100, 4, 0.0, DH, 6.0, gen)
    ref = pair.analyze(sheet(ca, np.zeros(100, int)), 4, B=200)
    assert ref["errors_seen"]["machine"] == 0
    assert ref["phi"]["status"] == "not estimable"
    assert "lower bound" in ref["k"]["note"]
    assert ref["k"]["interval_used"] == "exact"
    lo, hi = ref["k"]["ci95_exact"]
    assert lo > 1.0 and ref["zone"] in ("A", "B")
    assert "Angle between the errors: not estimable" in pair.report(ref)


def test_yes_no_with_a_shared_bias_shows_no_systematic_error():
    """Both minds lean toward 'no': the per-class sensitivities and the angle do not move."""
    ks, phis = [], []
    for seed in range(4):
        gen = np.random.default_rng(100 + seed)
        n = 20000
        klass = np.repeat([1, 0], n // 2)
        e = gen.standard_normal((n, 2)) @ np.linalg.cholesky(np.array([[1.0, RHO], [RHO, 1.0]])).T
        sign = np.where(klass == 1, 1.0, -1.0)
        yes_h, yes_m = sign * DH / 2 + e[:, 0] > 0.4, sign * DM / 2 + e[:, 1] > 0.4
        data = {"key": list(np.where(klass == 1, "YES", "NO")), "person": list(np.where(yes_h, "YES", "NO")),
                "machine": list(np.where(yes_m, "YES", "NO"))}
        ref = pair.analyze(data, 2, B=60)
        ks.append(ref["k"]["estimate"])
        phis.append(ref["phi"]["estimate"])
    assert np.mean(ks) == pytest.approx(TRUTH["k"], abs=0.08)
    assert np.mean(phis) == pytest.approx(TRUTH["phi"], abs=2.0)


def test_two_phase_weights_narrow_the_angle_interval():
    gen = np.random.default_rng(7)
    ca, cb = choices(2000, 4, RHO, DH, DM, gen)
    err, ok = np.flatnonzero(cb != 0), np.flatnonzero(cb == 0)
    ne = min(100, err.size)
    idx = np.r_[gen.choice(err, ne, replace=False), gen.choice(ok, 200 - ne, replace=False)]
    w = np.r_[np.full(ne, err.size / ne), np.full(200 - ne, ok.size / (200 - ne))]
    weighted = pair.analyze(sheet(ca[idx], cb[idx], {"weight": [str(x) for x in w]}), 4, B=400)
    plain = pair.analyze(sheet(ca[gen.choice(2000, 200, replace=False)], cb[gen.choice(2000, 200, replace=False)]), 4, B=400)
    assert weighted["weighted"] and not plain["weighted"]
    width = lambda r: r["phi"]["ci95"][1] - r["phi"]["ci95"][0]
    assert width(weighted) < width(plain)
    assert weighted["phi"]["estimate"] == pytest.approx(TRUTH["phi"], abs=12)


def test_few_errors_make_the_angle_indicative():
    gen = np.random.default_rng(3)
    ca, cb = choices(200, 4, RHO, 1.6, 3.0, gen)
    ref = pair.analyze(sheet(ca, cb), 4, B=200)
    assert min(ref["errors_seen"]["machine"], ref["errors_seen"]["person"]) < pair.FEW_ERRORS
    assert ref["phi"]["status"] in ("indicative", "not estimable")


# --- the sheet ----------------------------------------------------------------------

ENGLISH = "case,stratum,key,person,confidence,machine,chat,pair,seconds_alone,seconds_pair,weight\n"
ITALIAN = "caso,strato,chiave,persona,sicurezza,macchina,chat,coppia,secondi_solo,secondi_coppia,peso\n"
ROWS = "c1,x,A,A,3,A,A,A,10,20,1\nc2,x,B,A,1,B,B,B,12,25,1\nc3,y,C,C,2,B,B,C,9,15,1\n,,,,,,,,,,\n"


def rows_of(text):
    import csv
    import io
    return list(csv.DictReader(io.StringIO(text)))


def test_both_column_languages_read_to_the_same_columns():
    en = pair.parse_sheet(rows_of(ENGLISH + ROWS))
    it = pair.parse_sheet(rows_of(ITALIAN + ROWS))
    assert en == it
    assert en["key"] == ["A", "B", "C"] and en["dropped"] == 1
    assert en["seconds_pair"] == ["20", "25", "15"] and en["stratum"] == ["x", "x", "y"]


def test_column_names_are_case_insensitive_and_order_free():
    d = pair.parse_sheet(rows_of("Machine,KEY,Person\nA,A,B\n"))
    assert d["key"] == ["A"] and d["person"] == ["B"] and d["machine"] == ["A"]


def test_missing_required_column_is_an_error():
    with pytest.raises(pair.SheetError, match="missing column 'machine'"):
        pair.parse_sheet(rows_of("key,person\nA,A\n"))
    with pytest.raises(pair.SheetError):
        pair.parse_sheet([])


def test_decimal_comma_in_weights_and_times():
    gen = np.random.default_rng(1)
    ca, cb = choices(300, 4, RHO, 1.2, 2.2, gen)
    plain = sheet(ca, cb)
    commas = dict(plain, weight=["1,0"] * 300)
    assert pair.analyze(commas, 4, B=50)["weighted"] is False


def test_cli_writes_the_report_and_the_json(tmp_path, capsys):
    gen = np.random.default_rng(5)
    ca, cb = choices(300, 4, RHO, 1.2, 2.2, gen)
    p = tmp_path / "s.csv"
    p.write_text("case,key,person,machine\n" + "\n".join(f"c{i},A,{LETTERS[a]},{LETTERS[b]}" for i, (a, b) in enumerate(zip(ca, cb))))
    out = tmp_path / "r.json"
    assert main(["pair", str(p), "--options", "4", "--resamples", "100", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("CALIBRATION: 300 cases, 4 options") and "ZONE " in text
    assert json.loads(out.read_text())["cases"] == 300


def test_cli_reports_a_bad_sheet_without_a_traceback(tmp_path, capsys):
    p = tmp_path / "s.csv"
    p.write_text("key,person\nA,A\n")
    assert main(["pair", str(p), "--options", "4"]) == 2
    assert "missing column" in capsys.readouterr().err


def test_zone_boundaries():
    assert pair.zone_from_k(1.6, 3.0)[0] == "A"
    assert pair.zone_from_k(1.2, 3.0)[0] == "B"
    assert pair.zone_from_k(0.8, 1.4)[0] == "C"
    assert pair.zone_from_k(0.5, 0.9)[0] == "D"
    assert pair.zone_from_k(0.3, 0.6)[0] == "E"
    assert pair.zone_from_k(0.8, 1.4)[1] == pytest.approx(math.sqrt(2) - 1)


def test_mchoice_matches_two_alternative_case():
    assert mchoice.d_from_accuracy(0.8, 2) == pytest.approx(2 * 0.8416212335729143)
    assert mchoice.pc_mafc(0.0, 4) == pytest.approx(0.25, abs=1e-6)
    assert mchoice.useful_angle(1.0, 2.0) == pytest.approx(60.0)
    assert mchoice.ideal(1.0, 1.0, 0.0) == pytest.approx(math.sqrt(2))
