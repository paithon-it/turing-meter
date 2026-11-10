import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from turing_meter.apparent import (
    apparent,
    apparent_turings,
    apparent_turings_from_ratio,
    class_alignment,
    confidence_sharing,
    d_prime_from_accuracy,
    d_prime_from_auc,
    d_prime_from_rates,
    error_alignment,
    evidence_ratio,
    expected_weight_of_evidence,
    pair,
    people_alignment,
    pooled_alignment,
    practice_adjusted,
    tetrachoric,
    turing,
    useful_angle,
    verdict_ideal,
)


# --- the scale -------------------------------------------------------------

def test_chance_is_zero():
    assert d_prime_from_accuracy(0.5) == pytest.approx(0)
    assert d_prime_from_rates(0.3, 0.3) == pytest.approx(0)
    assert d_prime_from_auc(0.5) == pytest.approx(0)


def test_accuracy_matches_rates_for_an_unbiased_observer():
    a = 0.8
    assert d_prime_from_accuracy(a) == pytest.approx(d_prime_from_rates(a, 1 - a))


def test_auc_inverts_the_equal_variance_model():
    d = 1.3
    assert d_prime_from_auc(norm.cdf(d / math.sqrt(2))) == pytest.approx(d)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2])
def test_rates_outside_the_open_interval_are_refused(bad):
    with pytest.raises(ValueError):
        d_prime_from_accuracy(bad)


# --- the ideal of the pair ---------------------------------------------------

def test_independent_errors_give_the_quadrature_law():
    assert apparent(3, 4) == pytest.approx(5)


def test_equal_partners_with_correlated_errors():
    d, rho = 1.5, 0.4
    phi = math.degrees(math.acos(rho))
    assert apparent(d, d, phi) == pytest.approx(math.sqrt(2 * d * d / (1 + rho)))


def test_the_ideal_falls_to_the_stronger_when_errors_are_aligned_just_enough():
    dh, dm = 1.0, 2.0
    phi = math.degrees(math.acos(dh / dm))
    assert apparent(dh, dm, phi) == pytest.approx(dm)


@pytest.mark.parametrize("phi", [60, 45, 30, 10])
def test_aligned_errors_never_raise_the_ideal_above_the_stronger(phi):
    """No negative weight on the weaker partner: past rho = 1/2 the ideal is dM."""
    assert apparent(1.0, 2.0, phi) == pytest.approx(2.0)


def test_the_ideal_never_rises_as_errors_align():
    phis = range(170, 0, -10)
    vals = [apparent(1.0, 2.5, p) for p in phis]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("k,angle", [(1, 0.0), (2, 60.0), (5, math.degrees(math.acos(0.2))), (10, math.degrees(math.acos(0.1)))])
def test_useful_angle_is_arccos_of_one_over_k(k, angle):
    assert useful_angle(1.0, float(k)) == pytest.approx(angle)
    assert useful_angle(float(k), 1.0) == pytest.approx(angle)


@pytest.mark.parametrize("dh,dm", [(1.0, 2.0), (0.6, 3.0), (1.5, 2.5)])
def test_at_the_useful_angle_the_ideal_is_the_stronger_and_above_it_more(dh, dm):
    phi = useful_angle(dh, dm)
    assert apparent(dh, dm, phi) == pytest.approx(max(dh, dm))
    assert apparent(dh, dm, phi + 5) > max(dh, dm) + 1e-6
    assert apparent(dh, dm, phi - 5) == pytest.approx(max(dh, dm))


def test_useful_angle_needs_both_minds_above_chance():
    with pytest.raises(ValueError):
        useful_angle(0.0, 2.0)


def test_confidence_sharing_is_below_the_ideal_and_below_the_best_for_unequal_partners():
    assert confidence_sharing(1, 1) == pytest.approx(apparent(1, 1))
    assert confidence_sharing(1, 3) < apparent(1, 3)
    assert confidence_sharing(0.3, 1) < 1


def test_opposed_errors_raise_the_ideal_above_the_law():
    assert apparent(1, 1, 120) > apparent(1, 1, 90)


@pytest.mark.parametrize("phi", [0, 180])
def test_degenerate_angles_are_refused(phi):
    with pytest.raises(ValueError):
        apparent(1, 2, phi)


def test_the_law_is_the_optimum_of_two_independent_observers():
    """Simulate two observers and the best linear rule: accuracy matches |S|."""
    rng = np.random.default_rng(0)
    dh, dm, n = 0.8, 1.6, 400_000
    y = rng.choice([-1.0, 1.0], size=n)
    xh = y * dh / 2 + rng.standard_normal(n)
    xm = y * dm / 2 + rng.standard_normal(n)
    acc = np.mean(np.sign(dh * xh + dm * xm) == y)
    assert d_prime_from_accuracy(acc) == pytest.approx(apparent(dh, dm), abs=0.02)


# --- the numbers the SPEC reports -------------------------------------------

def test_turing_is_eta_over_cos_theta():
    p = pair(1.0, 1.5, 1.4)
    assert p.turing == pytest.approx(p.eta / p.cos_theta)


def test_anchors():
    dh, dm = 1.0, 2.0
    ignore, follow = pair(dh, dm, dh), pair(dh, dm, dm)
    ideal = pair(dh, dm, apparent(dh, dm))
    assert ignore.eta == pytest.approx(ignore.cos_theta)
    assert ignore.gain_share == pytest.approx(0)
    assert ignore.adoption == pytest.approx(0)
    assert ignore.turing == pytest.approx(1)
    assert follow.eta == pytest.approx(follow.eta_follow)
    assert follow.adoption == pytest.approx(1)
    assert ideal.eta == pytest.approx(1)
    assert ideal.gain_share == pytest.approx(1)


def test_eta_is_the_floor_plus_the_realized_share_of_the_gain():
    p = pair(0.7, 1.9, 1.2)
    assert p.eta == pytest.approx(p.cos_theta + p.gain_share * (1 - p.cos_theta))


def test_adoption_only_when_the_machine_is_stronger():
    assert pair(2.0, 1.0, 2.1).adoption is None
    assert pair(1.2, 1.2, 1.5).adoption is None


def test_theta_is_45_for_equal_partners():
    assert pair(1.2, 1.2, 1.5).theta_deg == pytest.approx(45)


def test_theta_follows_the_ideal_at_any_angle():
    p = pair(1.0, 1.5, 1.2, phi_deg=70)
    assert math.cos(math.radians(p.theta_deg)) == pytest.approx(p.cos_theta)


def test_turing_is_eta_times_the_apparent_turings():
    p = pair(1.0, 1.5, 1.4, phi_deg=70)
    assert p.turing == pytest.approx(p.eta * p.apparent_turings)
    assert p.apparent_turings == pytest.approx(1 / p.cos_theta)
    assert p.apparent_turings == pytest.approx(apparent_turings(1.0, 1.5, 70))


def test_a_person_who_copies_the_machine_is_worth_k_turings():
    """The turing is read beside k: read from the machine's side, a copier is worth 1."""
    dh, k = 0.8, 2.5
    copy = pair(dh, k * dh, k * dh)
    assert copy.turing == pytest.approx(k)
    assert copy.k == pytest.approx(k)
    assert copy.yield_on_machine == pytest.approx(1)


def test_near_the_useful_angle_eta_and_g_cannot_tell_copying_either():
    dh, k = 0.8, 2.5
    copy = pair(dh, k * dh, k * dh, phi_deg=useful_angle(1.0, k) + 2)
    assert copy.eta == pytest.approx(k / copy.apparent_turings)
    assert copy.gain_share == pytest.approx((k - 1) / (copy.apparent_turings - 1))
    assert copy.eta > 0.99 and copy.gain_share > 0.99


def test_a_turing_above_one_can_leave_the_pair_below_the_machine():
    p = pair(1.0, 2.0, 1.6)
    assert p.turing == pytest.approx(1.6)
    assert p.yield_on_machine == pytest.approx(0.8)


def test_pair_needs_both_minds_above_chance():
    with pytest.raises(ValueError):
        pair(0.0, 1.0, 0.5)


def test_turing_needs_the_person_above_chance():
    with pytest.raises(ValueError):
        turing(0.0, 1.0)


# --- the apparent turings ----------------------------------------------------

def test_apparent_turings_are_the_ideal_in_the_persons_unit():
    dh, dm, phi = 0.8, 1.9, 75
    assert apparent_turings(dh, dm, phi) == pytest.approx(apparent(dh, dm, phi) / dh)


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 5.0])
def test_apparent_turings_at_90_degrees_are_the_hypotenuse(k):
    assert apparent_turings(1.3, k * 1.3) == pytest.approx(math.sqrt(1 + k * k))
    assert apparent_turings_from_ratio(k) == pytest.approx(math.sqrt(1 + k * k))


@pytest.mark.parametrize("dh", [0.4, 1.0, 2.5])
@pytest.mark.parametrize("k", [0.3, 1.0, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("phi", [15, 45, 60, 75, 84, 90, 110, 150])
def test_the_closed_form_matches_the_ideal(dh, k, phi):
    """sqrt(1 + k^2 - 2 k cos phi) / sin phi, stopped at the stronger below the useful angle."""
    assert apparent_turings_from_ratio(k, phi) == pytest.approx(apparent_turings(dh, k * dh, phi))


@pytest.mark.parametrize("k", [0.25, 2.0, 5.0])
def test_below_the_useful_angle_the_apparent_turings_are_the_stronger_alone(k):
    phi = useful_angle(1.0, k)
    assert apparent_turings_from_ratio(k, phi - 5) == pytest.approx(max(1.0, k))
    assert apparent_turings_from_ratio(k, phi + 5) > max(1.0, k)


def test_a_stronger_machine_raises_the_apparent_turings():
    vals = [apparent_turings_from_ratio(k, 80) for k in (1.0, 1.5, 2.0, 3.0, 5.0)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_apparent_turings_need_the_person_above_chance():
    with pytest.raises(ValueError):
        apparent_turings(0.0, 1.0)


@pytest.mark.parametrize("k,phi", [(-1.0, 90), (2.0, 0), (2.0, 180)])
def test_apparent_turings_from_ratio_refuse_what_has_no_ideal(k, phi):
    with pytest.raises(ValueError):
        apparent_turings_from_ratio(k, phi)


# --- the square of the turing --------------------------------------------------

def test_the_weight_of_evidence_has_variance_twice_its_mean():
    """Turing's theorem, as Good (1985) reports it: here the mean is d'^2 / 2 nats."""
    rng = np.random.default_rng(3)
    d, n = 1.3, 400_000
    x = d / 2 + rng.standard_normal(n)       # cases whose right answer is yes
    llr = d * x                              # log-likelihood ratio, natural units
    assert llr.mean() == pytest.approx(expected_weight_of_evidence(d), rel=0.01)
    assert llr.var() == pytest.approx(2 * expected_weight_of_evidence(d), rel=0.01)


def test_the_evidence_ratio_is_the_square_of_the_turing_and_the_ratio_of_weights():
    dh, ds = 0.8, 1.4
    assert evidence_ratio(dh, ds) == pytest.approx(turing(dh, ds) ** 2)
    assert evidence_ratio(dh, ds) == pytest.approx(
        expected_weight_of_evidence(ds) / expected_weight_of_evidence(dh))


@pytest.mark.parametrize("n", [1, 2, 4])
def test_n_hypothetical_copies_with_independent_errors_give_an_evidence_ratio_of_n(n):
    """n copies of one person, independent noise, evidence pooled ideally."""
    rng = np.random.default_rng(n)
    dh, cases = 0.9, 400_000
    y = rng.choice([-1.0, 1.0], size=cases)
    evidence = (y * dh / 2)[:, None] + rng.standard_normal((cases, n))
    ds = d_prime_from_accuracy(np.mean(np.sign(evidence.sum(axis=1)) == y))
    assert evidence_ratio(dh, ds) == pytest.approx(n, rel=0.03)


@pytest.mark.parametrize("rho", [0.2, 0.5])
def test_two_copies_whose_errors_correlate_are_worth_less_than_two(rho):
    d = 1.1
    two = apparent(d, d, math.degrees(math.acos(rho)))
    assert evidence_ratio(d, two) == pytest.approx(2 / (1 + rho))


def test_weight_of_evidence_needs_a_sensitivity():
    with pytest.raises(ValueError):
        expected_weight_of_evidence(-0.5)


# --- the verdict-only ideal --------------------------------------------------

@pytest.mark.parametrize("dh,dm", [(0.5, 0.5), (0.5, 2.0), (2.0, 0.5), (1.0, 3.0), (3.0, 3.0)])
def test_verdict_ideal_sits_between_the_best_and_the_law(dh, dm):
    v = verdict_ideal(dh, dm)
    assert max(dh, dm) - 1e-9 <= v <= apparent(dh, dm) + 1e-9


def test_verdict_ideal_matches_simulation():
    rng = np.random.default_rng(1)
    dh, dm, n = 1.0, 1.8, 400_000
    y = rng.choice([-1.0, 1.0], size=n)
    xh = y * dh / 2 + rng.standard_normal(n)
    vote = np.sign(y * dm / 2 + rng.standard_normal(n))
    am = norm.cdf(dm / 2)
    llr = dh * xh + vote * math.log(am / (1 - am))
    acc = np.mean(np.sign(llr) == y)
    assert d_prime_from_accuracy(acc) == pytest.approx(verdict_ideal(dh, dm), abs=0.02)


# --- the angle between the errors --------------------------------------------

def _simulate(rho, dh, dm, n, seed):
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return noise[:, 0] > -dh / 2, noise[:, 1] > -dm / 2


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, -0.3])
def test_alignment_recovers_the_correlation_of_the_errors(rho):
    h, m = _simulate(rho, 1.2, 1.8, 40_000, seed=7)
    a = error_alignment(h, m)
    assert a.rho == pytest.approx(rho, abs=0.05)
    assert a.phi_deg == pytest.approx(math.degrees(math.acos(rho)), abs=4)
    assert a.items == 40_000


def test_alignment_refuses_minds_that_never_err():
    with pytest.raises(ValueError):
        error_alignment([True, True, True], [True, False, True])


def test_alignment_needs_the_same_items():
    with pytest.raises(ValueError):
        error_alignment([True, False], [True, False, True])


# --- the correlation behind a two-by-two table -------------------------------

@pytest.mark.parametrize("r", [-0.6, 0.0, 0.25, 0.7])
def test_tetrachoric_inverts_the_bivariate_normal(r):
    p1, p2 = 0.7, 0.4
    p11 = multivariate_normal.cdf([norm.ppf(p1), norm.ppf(p2)], mean=[0, 0], cov=[[1, r], [r, 1]])
    assert tetrachoric(p1, p2, p11) == pytest.approx(r, abs=1e-5)


def test_tetrachoric_at_a_median_split_and_on_arrays():
    r = np.array([-0.5, 0.1, 0.5])
    p11 = 0.25 + np.arcsin(r) / (2 * math.pi)
    assert tetrachoric(0.5, 0.5, p11) == pytest.approx(r, abs=1e-5)


# --- errors on the same cases, or a lean toward the same answer ---------------

def _classes(rho, n, c_person, c_machine, dh=1.2, dm=1.8, seed=11):
    """Balanced classes; evidence toward 'yes' is truth * d'/2 + noise; 'yes' above the criterion."""
    rng = np.random.default_rng(seed)
    truth = np.where(np.arange(n) % 2 == 0, 1, -1)
    noise = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    person_yes = truth * dh / 2 + noise[:, 0] > c_person
    machine_yes = truth * dm / 2 + noise[:, 1] > c_machine
    return person_yes, machine_yes, truth


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5])
def test_class_alignment_recovers_the_correlation_of_the_evidence(rho):
    h, m, y = _classes(rho, 60_000, 0.0, 0.0)
    a = class_alignment(h, m, y)
    assert a.rho == pytest.approx(rho, abs=0.03)
    assert len(a.by_class) == 2
    assert a.items == 60_000


def test_a_shared_criterion_inflates_the_errors_but_not_the_class_alignment():
    """Review of 2 November 2026: leaning toward the machine's answer is not sharing its errors."""
    h, m, y = _classes(0.25, 60_000, c_person=0.6, c_machine=0.6)
    correct_h, correct_m = h == (y > 0), m == (y > 0)
    assert class_alignment(h, m, y).rho == pytest.approx(0.25, abs=0.03)
    assert error_alignment(correct_h, correct_m).rho > 0.25 + 0.08


def test_class_alignment_needs_both_classes():
    h, m, y = _classes(0.2, 1_000, 0.0, 0.0)
    with pytest.raises(ValueError):
        class_alignment(h, m, np.ones_like(y))


def test_people_alignment_recovers_what_people_share():
    rng = np.random.default_rng(5)
    people, n, r = 40, 3_000, 0.3
    truth = np.where(np.arange(n) % 2 == 0, 1, -1)
    shared = rng.standard_normal(n)
    noise = math.sqrt(r) * shared + math.sqrt(1 - r) * rng.standard_normal((people, n))
    yes = truth * 0.6 + noise > rng.normal(0, 0.3, (people, 1))  # each with a criterion of their own
    assert people_alignment(yes, truth) == pytest.approx(r, abs=0.03)


# --- what practice alone does to the alignment --------------------------------

def _latent(tau2, sm2, sh2, load=1.0, a=0.0):
    """rho(person, machine) and rho(person, person) when the person weighs the shared difficulty by `load`."""
    shared_h = load * load * tau2
    hm = (load * tau2 + a * math.sqrt(sh2 * sm2)) / math.sqrt((shared_h + sh2) * (tau2 + sm2))
    hh = (shared_h + a * a * sh2) / (shared_h + sh2)
    return hm, hh


@pytest.mark.parametrize("sh2,load", [(1.95, 1.0), (1.2, 1.0), (0.6, 1.0), (1.2, 1.4), (0.8, 0.7)])
def test_practice_alone_leaves_the_adjusted_alignment_at_zero(sh2, load):
    tau2, sm2 = 0.3, 0.7
    q = tau2 / (tau2 + sm2)
    hm, hh = _latent(tau2, sm2, sh2, load)
    assert practice_adjusted(hm, hh, q) == pytest.approx(0, abs=1e-12)


def test_assimilation_shows_in_the_adjusted_alignment():
    tau2, sm2, sh2 = 0.3, 0.7, 1.2
    hm, hh = _latent(tau2, sm2, sh2, a=0.134)
    assert practice_adjusted(hm, hh, tau2 / (tau2 + sm2)) > 0.08


def _crowd(r, people=40, n=3_000, seed=5):
    """People and a machine sharing a component of their evidence: every pair correlates at r."""
    rng = np.random.default_rng(seed)
    truth = np.where(np.arange(n) % 2 == 0, 1, -1)
    shared = rng.standard_normal(n)
    machine = truth * 1.0 + math.sqrt(r) * shared + math.sqrt(1 - r) * rng.standard_normal(n) > 0.3
    noise = math.sqrt(r) * shared + math.sqrt(1 - r) * rng.standard_normal((people, n))
    sens = rng.normal(0.6, 0.15, (people, 1))
    yes = truth * sens + noise > rng.normal(0, 0.3, (people, 1))
    return yes, machine, truth


def test_pooled_alignment_keeps_each_persons_criterion():
    yes, machine, truth = _crowd(0.3)
    assert pooled_alignment(yes, machine, truth) == pytest.approx(0.3, abs=0.03)


def test_alignment_weights_give_one_estimate_per_draw():
    yes, machine, truth = _crowd(0.3, people=20, n=600)
    ones = np.ones((3, 20))
    assert pooled_alignment(yes, machine, truth, ones) == pytest.approx(
        [pooled_alignment(yes, machine, truth)] * 3, abs=1e-9)
    assert people_alignment(yes, truth, ones) == pytest.approx([people_alignment(yes, truth)] * 3, abs=1e-9)

