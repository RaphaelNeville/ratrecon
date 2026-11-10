"""Tests for ratrecon.

Where possible the results are checked against brute-force enumeration.

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
    lll_reduce,
    minimum_spacing_denominator,
    observation_interval,
    parse_decimal,
    reconstruct,
    reconstruct_shared,
    simplest_in_interval,
    sufficient_digits,
    sufficient_digits_for_denominators,
)


def test_observation_cell_is_the_exact_preimage():
    """A fraction f lies in the observation cell iff decimal_digits(f)
    regenerates the observed string.  Both digit rules and both signs are
    covered, including the half-open ends of the cell and signed zero.
    The uniqueness certificates of identify() and identify_shared() are
    computed from this cell."""
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
    """The strings "0.00" and "-0.00" are different observations, with base
    cells [0, h) and (-h, 0), where h is half an ulp when rounding.  Noise
    widening must start from the base cell with the right sign, so that
    -1/200 stays outside the widened "0.00" cell and 1/200 outside the
    widened "-0.00" cell."""
    pos = observation_interval(parse_decimal("0.00"), "round", "0.1")
    neg = observation_interval(parse_decimal("-0.00"), "round", "0.1")
    assert (pos.lo, pos.hi) == (Fraction(-1, 1000), Fraction(3, 500))
    assert (neg.lo, neg.hi) == (Fraction(-3, 500), Fraction(1, 1000))
    assert Fraction(-1, 200) not in pos and Fraction(1, 200) not in neg


def test_farey_enumeration_matches_brute_force():
    """The Farey neighbour walk should produce every fraction with q <= Q in
    an interval.  Its output is compared with a full double loop over q
    and p."""
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


def test_threshold_is_verified_exhaustively():
    """k digits determine a fraction with q <= Q whenever 10^k >= Q(Q-1)
    (THEORY 2.1).  The test checks every proper fraction at
    k = sufficient_digits(Q) for Q = 100 (k = 4) and for Q = 32 (k = 3, one
    digit fewer than 10^k >= Q^2 would ask for), in both digit modes.  It
    also checks that one digit fewer leaves some fraction ambiguous, and
    that Q(Q-1) is the largest pairwise lcm of 1, ..., Q (THEORY 7.1)."""
    for qmax, k in ((100, 4), (32, 3)):
        assert sufficient_digits(qmax) == k
        assert minimum_spacing_denominator(range(1, qmax + 1)) == qmax * (qmax - 1)
        fracs = [Fraction(p, q) for q in range(2, qmax + 1) for p in range(1, q)
                 if math.gcd(p, q) == 1]
        for x in fracs:
            for mode in ("round", "truncate"):
                rep = identify(decimal_digits(x, k, mode), qmax=qmax, mode=mode)
                assert rep.identifiable and rep.witness == x, (x, mode)
        assert not all(identify(decimal_digits(x, k - 1), qmax=qmax).identifiable
                       for x in fracs)


def test_constrained_grid_threshold_uses_largest_pairwise_lcm():
    """On a denominator family the minimum spacing is 1/M, where M is the
    largest pairwise lcm.  For {6, 10, 14}, M = lcm(10, 14) = 70, so the
    minimum spacing is 1/70 and two digits suffice.  Without the family
    constraint, the bound 10^k >= Q(Q-1) with Q = 14 gives three digits."""
    fams = [6, 10, 14]
    assert sufficient_digits_for_denominators(fams) == 2
    for q in fams:
        for p in range(1, q):
            x = Fraction(p, q)
            rep = identify(decimal_digits(x, 2), qmax=14, denominators=fams)
            assert rep.identifiable and rep.witness == x, (x,)


def test_ranking_is_prior_dominated_below_threshold():
    """With prior exponent s = 2 (Occam) and s = 0 (pure residual), the same
    ambiguous observation is ranked differently, and the s = 0 winner is the
    zero-residual fraction."""
    occam = reconstruct("0.3", qmax=1000, max_candidates=1)[0].fraction
    closest = reconstruct("0.3", qmax=1000, prior_exponent=0.0, max_candidates=1)[0].fraction
    assert occam == Fraction(1, 3)  # simplest consistent
    assert closest == Fraction(3, 10)  # closest consistent (residual exactly 0)


def test_periodicity_is_verified_and_survives_rounded_last_digit():
    # 67/74 = 0.9(054).
    # 2/3 rounds to 0.66667, so the last digit breaks the literal period.
    top = reconstruct("0.9054054", qmax=100)[0]
    assert top.fraction == Fraction(67, 74) and top.periodic is not None
    top = reconstruct("0.66667", qmax=10)[0]
    assert top.fraction == Fraction(2, 3) and top.periodic == (0, 1)


def _gram_schmidt(b):
    star, mu = [], [[Fraction(0)] * len(b) for _ in b]
    for i, row in enumerate(b):
        v = [Fraction(x) for x in row]
        for j in range(i):
            mu[i][j] = sum(a * c for a, c in zip(row, star[j])) / sum(c * c for c in star[j])
            v = [a - mu[i][j] * c for a, c in zip(v, star[j])]
        star.append(v)
    return star, mu


def test_lll_output_is_reduced_and_lattice_refinement_survives_noise():
    """lll_reduce returns a size-reduced basis (|mu_ij| <= 1/2) that meets
    the Lovasz condition with delta = 3/4, and lattice_refine recovers x
    from an observation that is off by up to three units in the last place,
    six times the largest rounding error."""
    rng = random.Random(9)
    for _ in range(60):
        n = rng.choice([2, 3, 4, 5])
        basis = [[rng.randint(-50, 50) for _ in range(n)] for _ in range(n)]
        try:
            b = lll_reduce(basis)
        except ValueError:  # a random basis can be singular
            continue
        star, mu = _gram_schmidt(b)
        norm = [sum(c * c for c in v) for v in star]
        assert all(abs(mu[i][j]) <= Fraction(1, 2) for i in range(n) for j in range(i))
        assert all(norm[k] >= (Fraction(3, 4) - mu[k][k - 1] ** 2) * norm[k - 1]
                   for k in range(1, n))
    for _ in range(40):
        q = rng.randrange(2, 500)
        x = Fraction(rng.randrange(1, q), q)
        obs = x + Fraction(rng.randrange(-3000, 3000), 1000) * Fraction(1, 10 ** 9)
        assert x in lattice_refine(obs, 9, 500, noise_ulp=3)


def test_shared_reconstruction_and_uniqueness():
    """Cases for reconstruct_shared and identify_shared.
    (a) Jointly, four 3-digit decimals identify q = 61 uniquely, but
        ranked one at a time none of them has its generating fraction as
        top candidate.
    (b) If LLL does not find the denominator, an exhaustive scan is used.
        For 0.750 and 0.293 the scan finds q = 92, with 27/92 as the second
        fraction.
    (c) With the scan disabled, the same call cannot decide the case and
        raises SearchIncomplete.
    (d) The uniqueness check counts every numerator for each q, so '1.0' at
        qmax=20 has two consistent tuples, 1 and 19/20."""
    decs = ["0.377", "0.656", "0.180", "0.852"]
    truths = [Fraction(p, 61) for p in (23, 40, 11, 52)]
    assert all(reconstruct(d, qmax=200, max_candidates=1)[0].fraction != t
               for d, t in zip(decs, truths))
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
    """The candidate pool is built up to a score bound, so asking for one
    answer or twenty must not change which fraction ranks first.  The first
    case uses a small prior exponent, s = 0.1, since the spare-doubling
    margin ceil(0.5 / (s ln 2)) is largest for small s.  The second case is
    a wide noisy cell, where the Farey enumeration reaches its size cap
    before that margin is used up, and the top answer must still be the
    exhaustive winner.  The third is a cell holding many integers, where
    the cap is reached at the first doubling."""
    kwargs = dict(
        qmax=352, mode="truncate", noise_ulp="0.1", prior_exponent=0.1,
        include_periodic=False, include_lattice=False,
    )
    tops = {reconstruct("-0.0", max_candidates=n, **kwargs)[0].fraction
            for n in (1, 2, 5, 20)}
    assert tops == {Fraction(-1, 18)}  # the exhaustive-ranking winner

    wide = dict(qmax=500, noise_ulp=2, prior_exponent=0.001,
                include_periodic=False, include_lattice=False)
    tops = {reconstruct("0.0", max_candidates=n, **wide)[0].fraction
            for n in (1, 20)}
    assert tops == {Fraction(1, 37)}  # the exhaustive-ranking winner

    tops = {reconstruct("12", qmax=1000, mode="truncate", noise_ulp=300,
                        max_candidates=n)[0].fraction for n in (1, 20)}
    assert tops == {12}  # 12 and 13 tie nearest the centre 25/2


def test_constraint_semantics():
    # with denominators=[10**9] the admissible q are the divisors of 10^9
    # up to qmax
    assert reconstruct("0.5", qmax=10, denominators=[10 ** 9])[0].fraction == Fraction(1, 2)
    # an empty family admits no fractions and is rejected with ValueError,
    # as is a zero denominator
    with pytest.raises(ValueError):
        identify("0.5", qmax=10, denominators=[])
    with pytest.raises(ValueError):
        reconstruct("0.5", qmax=10, denominators=[0])
