"""Tests for the core guarantees, one idea per test.

The style throughout: never check an example against a remembered answer
when the algorithm can be checked against exhaustive enumeration instead.

Run:  pytest test_ratrecon.py
"""
import math
import random
from fractions import Fraction

import pytest

from ratrecon import (
    SearchIncomplete,
    decimal_digits,
    farey_in_interval,
    identify,
    identify_shared,
    lattice_refine,
    observation_interval,
    parse_decimal,
    reconstruct,
    reconstruct_shared,
    simplest_in_interval,
    sufficient_digits,
    sufficient_digits_for_denominators,
)


def _member(f, lo, hi, inc_lo, inc_hi):
    if f < lo or f > hi:
        return False
    if f == lo and not inc_lo:
        return False
    return not (f == hi and not inc_hi)


def test_observation_cell_is_the_exact_preimage():
    """The central invariant: f lies in the cell iff decimal_digits(f)
    regenerates the observed string -- for both digit rules, both signs, and
    the half-open/signed-zero edge cases.  Every certificate rests on it."""
    rng = random.Random(13)
    for _ in range(250):
        x = Fraction(rng.randrange(-120, 121), rng.randrange(1, 60))
        k = rng.randrange(1, 5)
        for mode in ("round", "truncate"):
            s = decimal_digits(x, k, mode)
            cell = observation_interval(parse_decimal(s), mode)
            assert x in cell, (x, k, mode, s)
            for f in farey_in_interval(cell.lo, cell.hi, 60, cell.include_lo, cell.include_hi):
                assert decimal_digits(f, k, mode) == s, (x, f, k, mode)


def test_signed_zero_keeps_its_asymmetry_even_under_noise():
    """"0.00" and "-0.00" are different observations ([0, h) vs (-h, 0)),
    and noise widening must widen the correct signed base cell -- otherwise
    -1/200 counts as consistent with a *positive* zero."""
    pos = observation_interval(parse_decimal("0.00"), "round", "0.1")
    neg = observation_interval(parse_decimal("-0.00"), "round", "0.1")
    assert (pos.lo, pos.hi) == (Fraction(-1, 1000), Fraction(3, 500))
    assert (neg.lo, neg.hi) == (Fraction(-3, 500), Fraction(1, 1000))
    assert Fraction(-1, 200) not in pos and Fraction(1, 200) not in neg


def test_farey_enumeration_matches_brute_force():
    """The Farey neighbour walk claims to produce EVERY q <= Q fraction in an
    interval at O(1) each; the judge is a full double loop."""
    rng = random.Random(2)
    for _ in range(60):
        lo = Fraction(rng.randrange(-900, 900), rng.randrange(1, 300))
        hi = lo + Fraction(rng.randrange(1, 200), rng.randrange(1, 300))
        qmax = rng.randrange(1, 30)
        brute = sorted(
            {
                Fraction(p, q)
                for q in range(1, qmax + 1)
                for p in range(math.ceil(lo * q), math.floor(hi * q) + 1)
            }
        )
        assert farey_in_interval(lo, hi, qmax) == brute


def test_simplest_fraction_has_minimal_denominator():
    rng = random.Random(4)
    for _ in range(100):
        lo = Fraction(rng.randrange(-600, 600), rng.randrange(1, 200))
        hi = lo + Fraction(rng.randrange(1, 150), rng.randrange(1, 200))
        s = simplest_in_interval(lo, hi)
        assert lo <= s <= hi
        # nothing with a smaller denominator fits
        for q in range(1, s.denominator):
            assert not any(
                lo <= Fraction(p, q) <= hi
                for p in range(math.ceil(lo * q), math.floor(hi * q) + 1)
            )


def test_threshold_equality_suffices_and_is_verified_exhaustively():
    """10^k >= Q^2 determines q <= Q (equality included: the extremal Farey
    gap 1/Q^2 would need q = q' = Q, but same-denominator fractions are 1/Q
    apart, so the true minimum gap is >= 1/(Q(Q-1))).  Checked here for every
    proper fraction with q <= 100 at exactly k = 4 = sufficient_digits(100)."""
    assert sufficient_digits(100) == 4
    for q in range(2, 101):
        for p in range(1, q):
            if math.gcd(p, q) != 1:
                continue
            x = Fraction(p, q)
            for mode in ("round", "truncate"):  # exercise both supported digit rules
                rep = identify(decimal_digits(x, 4, mode), qmax=100, mode=mode)
                assert rep.identifiable and rep.witness == x, (x, mode)


def test_constrained_grid_threshold_is_linear_not_quadratic():
    """On a denominator family the spacing is 1/max_pairwise_lcm -- for
    {6, 10, 14} that is 1/70 (NOT 1/210: no pair attains the triple lcm), so
    two digits suffice where the unconstrained bound would demand more."""
    fams = [6, 10, 14]
    assert sufficient_digits_for_denominators(fams) == 2
    for q in fams:
        for p in range(1, q):
            x = Fraction(p, q)
            rep = identify(decimal_digits(x, 2), qmax=14, denominators=fams)
            assert rep.identifiable and rep.witness == x, (x,)


def test_ranking_is_prior_dominated_below_threshold():
    """The prior exponent is a real modelling choice: s=2 (Occam) and s=0
    (pure residual) rank the same ambiguous observation differently, and
    s=0's winner is exactly the zero-residual fraction."""
    occam = reconstruct("0.3", qmax=1000, max_candidates=1)[0].fraction
    ml = reconstruct("0.3", qmax=1000, prior_exponent=0.0, max_candidates=1)[0].fraction
    assert occam == Fraction(1, 3)  # simplest consistent
    assert ml == Fraction(3, 10)  # closest consistent (residual exactly 0)


def test_periodicity_is_verified_and_survives_rounded_last_digit():
    # 67/74 = 0.9(054); 2/3 -> 0.66667 breaks literal periodicity at the end
    top = reconstruct("0.9054054", qmax=100)[0]
    assert top.fraction == Fraction(67, 74) and top.periodic is not None
    vals = [c.fraction for c in reconstruct("0.66667", qmax=10)]
    assert Fraction(2, 3) in vals


def test_lattice_refinement_survives_noise_beyond_rounding():
    rng = random.Random(9)
    for _ in range(40):
        q = rng.randrange(2, 500)
        x = Fraction(rng.randrange(1, q), q)
        obs = x + Fraction(rng.randrange(-3000, 3000), 1000) * Fraction(1, 10 ** 9)
        assert x in lattice_refine(obs, 9, 500, noise_ulp=3)


def test_shared_reconstruction_grades_its_claims():
    """Shared-denominator guarantee levels, in one place:
    (a) pooling: four 3-digit decimals identify q = 61 though each one's top
        individual candidate differs from the generating fraction;
    (b) an LLL miss falls back to the exact scan (0.750/0.293 -> 27/92);
    (c) with the scan disabled the same case is undecided and raises rather
        than reporting 'no';
    (d) uniqueness counts all numerators per q, so '1.0' at qmax=20 has two
        tuples (1 and 19/20)."""
    decs = ["0.377", "0.656", "0.180", "0.852"]
    truths = [Fraction(p, 61) for p in (23, 40, 11, 52)]
    assert [reconstruct(d, qmax=200, max_candidates=1)[0].fraction for d in decs] != truths
    joint = reconstruct_shared(decs, qmax=200)
    assert list(joint.fractions) == truths and joint.method == "lll"
    assert identify_shared(decs, qmax=200).identifiable

    rescue = reconstruct_shared(["0.750", "0.293"], qmax=100)
    assert rescue.denominator == 92 and rescue.method == "exhaustive"
    with pytest.raises(SearchIncomplete):
        reconstruct_shared(["0.750", "0.293"], qmax=100, exhaustive_limit=0)

    rep = identify_shared(["1.0"], qmax=20)
    assert rep.consistent_count == 2 and not rep.identifiable


def test_max_candidates_never_changes_top_rank():
    """The candidate pool is built to a score bound, not to the output size:
    asking for one answer or twenty must never change which fraction ranks
    first, including at small prior exponents where the spare-doubling
    margin ceil(0.5 / (s ln 2)) is largest."""
    kwargs = dict(
        qmax=352, mode="truncate", noise_ulp="0.1", prior_exponent=0.1,
        include_periodic=False, include_lattice=False,
    )
    tops = {reconstruct("-0.0", max_candidates=n, **kwargs)[0].fraction
            for n in (1, 2, 5, 20)}
    assert tops == {Fraction(-1, 18)}  # the exhaustive-ranking winner


def test_constraint_semantics():
    # a huge grid generator means its admissible DIVISORS, not a 10^9 sweep
    assert reconstruct("0.5", qmax=10, denominators=[10 ** 9])[0].fraction == Fraction(1, 2)
    # an empty hard family is an empty hypothesis class: reject, don't guess
    with pytest.raises(ValueError):
        identify("0.5", qmax=10, denominators=[])
    with pytest.raises(ValueError):
        reconstruct("0.5", qmax=10, denominators=[0])
