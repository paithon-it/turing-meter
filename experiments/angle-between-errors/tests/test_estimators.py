"""The estimators on data simulated with each source's real design.

Protocol criterion: systematic error of ρ below 0.03 in absolute value.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from angle import sdt  # noqa: E402


def simulate_people(rho, items=396, people=330, per_person=12, mean=0.6, sd_ability=0.4, acc_m=0.8, seed=0):
    """Each person sees few items; difficulty and machine correlated as declared."""
    g = np.random.default_rng(seed)
    zm = norm.ppf(acc_m)
    z = g.standard_normal(items)                                    # the machine's evidence per item
    m = (z < zm).astype(float)
    th = norm.ppf(mean) + sd_ability * g.standard_normal(people)
    P, C, Y = [], [], []
    for j in range(people):
        sc = g.choice(items, per_person, replace=False)
        e = rho * z[sc] + math.sqrt(1 - rho * rho) * g.standard_normal(per_person)
        scale = math.sqrt(1 + sd_ability ** 2)
        P += [j] * per_person
        C += sc.tolist()
        Y += ((e * scale) < th[j] * scale / scale).astype(float).tolist()
    return np.array(P), np.array(C), np.array(Y), m


@pytest.mark.parametrize("true_rho", [0.0, 0.3, 0.6])
def test_rho_personal_thresholds_chatbench_design(true_rho):
    estimates = []
    for s in range(30):
        P, C, Y, m = simulate_people(true_rho, seed=s)
        estimates.append(sdt.rho_personal_thresholds(P, C, Y, m))
    assert abs(np.mean(estimates) - true_rho) < 0.03, (true_rho, np.mean(estimates), np.std(estimates))


def test_d_mafc_reconstructs():
    for d in (0.5, 1.0, 2.0, 3.0):
        assert sdt.d_from_accuracy(sdt.pc_mafc(d, 4), 4) == pytest.approx(d, abs=1e-4)


@pytest.mark.parametrize("true_rho", [0.0, 0.3, 0.6])
def test_mafc_conversion_returns_the_true_rho(true_rho):
    g = sdt.rho_correct_from_rho_evidence(true_rho, 1.2, 2.2, 4, n=300_000, seed=9)
    assert sdt.convert_rho_mafc(g, 1.2, 2.2, 4) == pytest.approx(true_rho, abs=0.03)


def test_ideal_matches_the_old_one():
    assert sdt.ideal(1, 2, 0.0) == pytest.approx(math.sqrt(5))
    assert sdt.ideal(1, 2, 0.5) == pytest.approx(2.0)      # ρ = min/max: the stronger alone
    assert sdt.ideal(1, 2, 0.8) == pytest.approx(2.0)      # beyond: no negative weights
    assert sdt.useful_angle(1, 2) == pytest.approx(60.0)


def test_person_pairs():
    """People with the same correlation ρ between them, different thresholds."""
    g = np.random.default_rng(3)
    true_rho = 0.5
    P, C, Y = [], [], []
    common = g.standard_normal(400)
    th = norm.ppf(0.65) + 0.3 * g.standard_normal(300)
    for j in range(300):
        sc = g.choice(400, 30, replace=False)
        e = math.sqrt(true_rho) * common[sc] + math.sqrt(1 - true_rho) * g.standard_normal(30)
        P += [j] * 30
        C += sc.tolist()
        Y += (e < th[j]).astype(float).tolist()
    r = sdt.rho_person_pairs(np.array(P), np.array(C), np.array(Y))
    assert abs(r - true_rho) < 0.05


def test_combiner_recovers_the_prediction():
    """On the model's data, the combiner reaches what the prediction function says."""
    from angle import combiner
    for m in (2, 4):
        dh, dm, rho = 1.0, 1.8, 0.3
        accuracy = combiner.predict(dh, dm, rho, m, None, n=200_000)
        assert accuracy > 0.5
        a = sdt.d_from_accuracy(accuracy, m)
        assert max(dh, dm) - 0.05 <= a <= sdt.ideal(dh, dm, rho) + 0.05, (m, a)


def test_predict_grows_if_the_errors_are_independent():
    from angle import combiner
    a = combiner.predict(1.2, 1.2, 0.0, 2, [0.3, 0.3, 0.4], n=200_000)
    b = combiner.predict(1.2, 1.2, 0.7, 2, [0.3, 0.3, 0.4], n=200_000)
    assert a > b + 0.01, (a, b)


def _simulated_pairs(true_rho=0.4, items=300, people=240, per_person=12, seed=0):
    g = np.random.default_rng(seed)
    common = g.standard_normal(items)
    th = norm.ppf(0.65) + 0.3 * g.standard_normal(people)
    P, C, Y = [], [], []
    for j in range(people):
        sc = g.choice(items, per_person, replace=False)
        e = math.sqrt(true_rho) * common[sc] + math.sqrt(1 - true_rho) * g.standard_normal(per_person)
        P += [j] * per_person
        C += sc.tolist()
        Y += (e < th[j]).astype(float).tolist()
    return np.array(P), np.array(C), np.array(Y)


def test_weighted_pairs_bootstrap_does_not_inflate():
    """Resampling the items with weights must not create pairs of a trial with itself."""
    p, c, y = _simulated_pairs()
    g = np.random.default_rng(1)
    v = [sdt.rho_person_pairs(p, c, y, item_weights=g.poisson(1.0, c.max() + 1)) for _ in range(60)]
    assert abs(np.mean(v) - 0.4) < 0.06, np.mean(v)


def test_d_with_shifted_criterion():
    """A machine that almost always says yes: z(hit) − z(fa), not 2 z(a)."""
    assert sdt.d_from_classes(0.97, 0.09) == pytest.approx(norm.ppf(0.97) + norm.ppf(0.09), abs=1e-9)
    # with a neutral criterion it coincides with 2 z(a)
    assert sdt.d_from_classes(0.8, 0.8) == pytest.approx(2 * norm.ppf(0.8), abs=1e-9)
    # the correction of ½ keeps the case 0 or 1 finite
    assert np.isfinite(sdt.d_from_classes(1.0, 0.75, n1=16, n0=16))


def test_measure_uses_the_machines_criterion():
    from angle.measure import Trials, compare
    P = np.repeat(np.arange(200), 32)
    C = np.tile(np.arange(32), 200)
    answer_class = (C < 16).astype(int)
    g = np.random.default_rng(0)
    Y = (g.random(C.size) < 0.7).astype(float)
    mac = np.r_[np.ones(16), np.full(16, 0.0)]        # class 1 always right, class 0 always wrong
    mac[16:28] = 1.0                                  # 75% right on class 0
    r = compare(Trials(P, C, Y, mac, answer_class=answer_class), m=2, repetitions=0)
    expected = sdt.d_from_classes(1.0, 0.75, n1=16, n0=16)
    assert r["dm"] == pytest.approx(expected, abs=0.01), (r["dm"], expected)


def _trials_m4(rho, m=4, items=396, people=330, per_person=12, dh=0.9, sd=0.5, dm=1.6, seed=0):
    from angle.measure import Trials
    g = np.random.default_rng(seed)
    common = g.standard_normal((items, m))
    b = (np.sqrt(rho) * common + np.sqrt(1 - rho) * g.standard_normal((items, m))) if rho > 0 else g.standard_normal((items, m))
    b[:, 0] += dm
    mok = (b.argmax(1) == 0).astype(float)
    di = np.clip(dh + sd * g.standard_normal(people), 0.05, None)
    P, C, Y = [], [], []
    for j in range(people):
        sc = g.choice(items, per_person, replace=False)
        a = (np.sqrt(rho) * common[sc] + np.sqrt(1 - rho) * g.standard_normal((per_person, m))) if rho > 0 else g.standard_normal((per_person, m))
        a[:, 0] += di[j]
        P += [j] * per_person
        C += sc.tolist()
        Y += (a.argmax(1) == 0).astype(float).tolist()
    return Trials(np.array(P), np.array(C), np.array(Y), mok)


@pytest.mark.parametrize("true_rho", [0.3, 0.6])
def test_m4_conversion_with_different_people(true_rho):
    """End to end, with the ChatBench design and people of different sensitivity."""
    from angle.measure import compare
    st = [compare(_trials_m4(true_rho, seed=s), m=4, repetitions=0)["rho"] for s in range(6)]
    assert abs(np.mean(st) - true_rho) < 0.03, (true_rho, np.mean(st))


def test_grid_does_not_act_as_ceiling():
    assert sdt.RHO_GRID.max() > 0.99 and sdt.RHO_GRID.min() < -0.8


def test_d_combo_corrects_on_the_items():
    """Many rows on the same few items must not push the combiner's d′ beyond what the items allow."""
    from angle.analysis import _d_combo_classes
    items = np.repeat(np.arange(24), 100)
    answer_class = (items < 12).astype(int)
    outcome = np.where(answer_class == 1, 1.0, (items % 2 == 0) * 1.0)         # always right on one class, half on the other
    d = _d_combo_classes(outcome, answer_class, items)
    expected = sdt.d_from_classes(1.0, 0.5, n1=12, n0=12)
    assert abs(d - expected) < 1e-9
    assert d < 2.0


def _pair_matrix(rho=0.5, P=60, I=400, seed=7):
    gen = np.random.default_rng(seed)
    c = gen.standard_normal(I)
    e = np.sqrt(rho) * c[None, :] + np.sqrt(1 - rho) * gen.standard_normal((P, I))
    return (e < 0.3).astype(float)


def test_person_weights_consistent():
    y = _pair_matrix()
    P, I = y.shape
    p, c = np.repeat(np.arange(P), I), np.tile(np.arange(I), P)
    base = sdt.rho_person_pairs(p, c, y.ravel())
    assert abs(sdt.rho_person_pairs(p, c, y.ravel(), person_weights=np.ones(P)) - base) < 1e-9
    assert abs(sdt.rho_person_pairs(p, c, y.ravel(), person_weights=2 * np.ones(P)) - base) < 1e-9


def test_weighted_people_bootstrap_does_not_inflate():
    """A network resampled twice must not pair with its own copy."""
    y = _pair_matrix()
    P, I = y.shape
    p, c = np.repeat(np.arange(P), I), np.tile(np.arange(I), P)
    base = sdt.rho_person_pairs(p, c, y.ravel())
    gen = np.random.default_rng(3)
    copies, weights = [], []
    for _ in range(40):
        source_idx = gen.integers(0, P, P)
        g = y[source_idx]
        copies.append(sdt.rho_person_pairs(p, c, g.ravel()))
        weights.append(sdt.rho_person_pairs(p, c, y.ravel(), person_weights=np.bincount(source_idx, minlength=P).astype(float)))
    assert np.mean(copies) > base + 0.005
    assert abs(np.mean(weights) - base) < abs(np.mean(copies) - base)


def test_apparent_turings_depends_on_k_and_on_the_angle():
    """|S|/dH: at 90 degrees it is sqrt(1 + k²), below the useful angle it stops at the stronger, and the scale does not matter."""
    assert sdt.apparent_turings(1.0, 2.0, 0.0) == pytest.approx(math.sqrt(5.0))
    assert sdt.apparent_turings(1.0, 2.0, 0.5) == pytest.approx(2.0)      # cos φ = 1/k: the useful angle
    assert sdt.apparent_turings(1.0, 2.0, 0.8) == pytest.approx(2.0)      # below: the machine alone
    assert sdt.apparent_turings(2.0, 1.0, 0.8) == pytest.approx(1.0)      # the person is the stronger
    assert sdt.apparent_turings(3.0, 6.0, 0.2) == pytest.approx(sdt.apparent_turings(1.0, 2.0, 0.2))
    k, rho = 2.0, 0.2
    assert sdt.apparent_turings(1.0, k, rho) == pytest.approx(math.sqrt(1 + k * k - 2 * rho * k) / math.sqrt(1 - rho * rho))


def test_turing_of_comparisons_summarizes():
    """The summary of the turing over a meta-analysis: median, how many below one, interval by study."""
    from angle import vaccaro
    rows = [{"paper": "a", "dh": 1.0, "dm": 2.0, "ds": 1.5, "year": 2020, "within": True},
            {"paper": "a", "dh": 1.0, "dm": 2.0, "ds": 0.9, "year": 2020, "within": False},
            {"paper": "b", "dh": 2.0, "dm": 1.0, "ds": 2.4, "year": 2023, "within": False}]
    t = vaccaro.turing_of_comparisons(rows, repetitions=200)
    assert t["comparisons"] == 3 and t["studies"] == 2
    assert t["median"] == pytest.approx(1.2)
    assert t["below_one"] == 1 and t["studies_below_one"] == 1
    assert t["above_the_best"] == 1                      # only the third beats the stronger of the two
    assert t["ci"][0] <= t["median"] <= t["ci"][1]
    assert t["years"] == [2020, 2023]
    assert t["same_people"] == {"comparisons": 1, "below_one": 0, "median": pytest.approx(1.5)}


def test_pair_ds_uses_the_convention_of_dh():
    """With two classes person and machine have the free-criterion d′: the pair must sit on the same scale."""
    from angle.measure import Trials, _one_pass
    g = np.random.default_rng(0)
    n = 4000
    answer_class = np.repeat([0, 1], n // 2)
    person = np.arange(n) % 40
    item = np.arange(n)
    correct = np.where(answer_class == 1, g.random(n) < 0.80, g.random(n) < 0.60).astype(float)
    pair = np.where(answer_class == 1, g.random(n) < 0.95, g.random(n) < 0.55).astype(float)   # very shifted criterion
    machine = np.where(answer_class == 1, g.random(n) < 0.90, g.random(n) < 0.85).astype(float)
    p = Trials(person, item, correct, machine, with_machine=pair, machine_per_trial=True, answer_class=answer_class)
    r = _one_pass(p, 2, np.ones(40), np.ones(n), lambda x, a=None, b=None: x)
    c1, c0 = answer_class == 1, answer_class == 0
    expected = sdt.d_from_classes(pair[c1].mean(), pair[c0].mean(), n1=c1.sum(), n0=c0.sum())
    assert r["ds"] == pytest.approx(expected)
    assert r["ds"] != pytest.approx(2 * norm.ppf(pair.mean()), abs=0.02)        # the old convention gave a different number
    assert r["turing"] == pytest.approx(expected / r["dh"])
    assert r["yield_on_machine"] == pytest.approx(expected / r["dm"])
