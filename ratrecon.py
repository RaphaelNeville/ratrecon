"""ratrecon -- exact rational reconstruction from finite decimal expansions.

A finite decimal is an *observation* of an unknown rational p/q: k digits pin
the value into a cell of width 10^-k, and the whole subject is what that cell
does and does not determine.  This module is the complete pipeline in one
file, ordered the way the mathematics builds:

  1. Observation cells  -- the exact preimage of the digit rule (half-open,
                           signed zero, exact noise widening).
  2. Thresholds         -- when do k digits determine q <= Q at all?
  3. Continued fractions-- convergents, best bounded approximation, Legendre.
  4. Stern-Brocot/Farey -- simplest fraction in a cell; complete enumeration
                           of the ambiguity set at O(1) per element.
  5. Periodicity        -- repeating-digit hypotheses, verified exactly.
  6. Lattices (LLL)     -- noise-robust refinement and multi-decimal pooling.
  7. reconstruct/identify -- ranked candidates under an explicit prior, and
                           exact identifiability certificates.
  8. Shared denominators-- joint recovery with three graded strengths of
                           claim: verified / smallest / unique.

Design rules that everything below obeys:

* Exact arithmetic (`fractions.Fraction`) for every consistency or
  identifiability decision; floats appear only in ranking scores.
* Consistency with the cell is HARD evidence: a candidate outside it has
  likelihood zero and is discarded, never down-weighted.
* An undecided search raises :class:`SearchIncomplete`; only an exhaustive
  scan may report non-existence.

References: Khinchin, *Continued Fractions* (1964); Hardy & Wright ch. III,
X-XI; Richards, "Continued fractions without tears", Math. Mag. 54 (1981);
Graham-Knuth-Patashnik, *Concrete Mathematics* section 4.5; Lenstra-Lenstra-
Lovasz, Math. Ann. 261 (1982).  The accompanying THEORY.md/pdf derives every
bound used here.
"""
from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import NamedTuple

#: reconstruct_shared falls back to the exact O(qmax) scan up to this bound.
SHARED_EXHAUSTIVE_LIMIT = 20_000

#: step cap for exact constrained (denominator-family) enumeration.
SEARCH_BUDGET = 1_000_000


class SearchIncomplete(RuntimeError):
    """The configured search budget was insufficient to decide the question.

    Raised instead of returning "not found", so that an undecided search is
    never mistaken for a proof of non-existence.
    """


class EmptyIntervalError(ValueError):
    """The requested interval is empty."""


# ---------------------------------------------------------------------------
# 1. Observation cells: the exact preimage of the digit rule
# ---------------------------------------------------------------------------

_DEC_RE = re.compile(r"^\s*([+-]?)(\d+)?(?:\.(\d*))?\s*$")


class RationalCell(NamedTuple):
    """An interval with explicit endpoint semantics.

    Endpoint openness is the difference between a sound and an unsound
    certificate: 0.5 truncates to "0.5", not "0.4", so the truncation cell
    [0.4, 0.5) must exclude its right endpoint.  Unpacks as
    ``lo, hi, include_lo, include_hi``.
    """

    lo: Fraction
    hi: Fraction
    include_lo: bool
    include_hi: bool

    def __contains__(self, x) -> bool:
        x = Fraction(x)
        if x < self.lo or x > self.hi:
            return False
        if x == self.lo and not self.include_lo:
            return False
        if x == self.hi and not self.include_hi:
            return False
        return True

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    @property
    def center(self) -> Fraction:
        return (self.lo + self.hi) / 2

    def __str__(self) -> str:
        return (
            f"{'[' if self.include_lo else '('}{self.lo}, "
            f"{self.hi}{']' if self.include_hi else ')'}"
        )


@dataclass(frozen=True)
class Observation:
    """A parsed finite decimal: sign, integer part, fractional digit string."""

    negative: bool
    int_part: int
    frac_digits: str

    @property
    def k(self) -> int:
        return len(self.frac_digits)

    @property
    def ulp(self) -> Fraction:
        """One unit in the last place, 10**-k."""
        return Fraction(1, 10 ** self.k)

    @property
    def value(self) -> Fraction:
        v = self.int_part + Fraction(int(self.frac_digits or "0"), 10 ** self.k)
        return -v if self.negative else v

    def text(self) -> str:
        """Canonical string form (matches :func:`decimal_digits` output)."""
        sign = "-" if self.negative else ""
        return f"{sign}{self.int_part}.{self.frac_digits}" if self.k else f"{sign}{self.int_part}"


def parse_decimal(s: str) -> Observation:
    """Parse a plain decimal literal like ``-3.14159``, ``.25`` or ``42``.

    Scientific notation is rejected on purpose: the number of fractional
    digits IS the information content of the observation, so it must be
    explicit in the string.
    """
    m = _DEC_RE.match(s)
    if not m or (m.group(2) is None and not m.group(3)):
        raise ValueError(f"not a plain decimal literal: {s!r}")
    return Observation(m.group(1) == "-", int(m.group(2) or "0"), m.group(3) or "")


def decimal_digits(x: Fraction, k: int, mode: str = "round") -> str:
    """First ``k`` fractional decimal digits of ``x``, computed exactly.

    'round' = half away from zero; 'truncate' = toward zero.  Canonical
    output: optional ``-``, exactly k fractional digits (no dot if k = 0).
    """
    if k < 0:
        raise ValueError("k (number of fractional digits) must be >= 0")
    x = Fraction(x)
    sign = "-" if x < 0 else ""
    scaled = abs(x) * 10 ** k
    if mode == "round":
        n = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
    elif mode == "truncate":
        n = scaled.numerator // scaled.denominator
    else:
        raise ValueError(f"mode must be 'round' or 'truncate', got {mode!r}")
    if k == 0:
        return f"{sign}{n}"
    s = str(n).rjust(k + 1, "0")
    return f"{sign}{s[:-k]}.{s[-k:]}"


def observation_interval(obs: Observation, mode: str = "round", noise_ulp=0) -> RationalCell:
    """The EXACT preimage of the observation under the stated digit rule.

    A real x produces exactly this digit string iff x lies in the returned
    cell (property-tested against :func:`decimal_digits`).  With u = ulp:

    * round (half away from zero):
        v > 0:   [v - u/2, v + u/2)        v < 0:   (v - u/2, v + u/2]
        "0.00":  [0, u/2)                  "-0.00": (-u/2, 0)
    * truncate (toward zero):
        v > 0:   [v, v + u)                v < 0:   (v - u, v]
        "0.00":  [0, u)                    "-0.00": (-u, 0)

    The signed-zero rows follow from :func:`decimal_digits` giving negative
    inputs a "-" sign: "-0.00" is produced only by strictly negative x.

    ``noise_ulp`` widens the interval by that many ulps per side (strings
    like "0.1" parse exactly as 1/10).  The exact base cell is built first
    and THEN widened -- widening a generic symmetric interval instead would
    silently discard the signed-zero asymmetry, so that -1/200 would count
    as consistent with a *positive* zero observation.
    """
    v, u = obs.value, obs.ulp
    slack = Fraction(noise_ulp) * u
    if slack < 0:
        raise ValueError("noise_ulp must be >= 0")
    if mode not in ("round", "truncate"):
        raise ValueError(f"mode must be 'round' or 'truncate', got {mode!r}")
    h = u / 2 if mode == "round" else u
    if v == 0:
        base = (
            RationalCell(Fraction(0), h, True, False)
            if not obs.negative
            else RationalCell(-h, Fraction(0), False, False)
        )
    elif mode == "round":
        base = (
            RationalCell(v - h, v + h, True, False)
            if not obs.negative
            else RationalCell(v - h, v + h, False, True)
        )
    elif not obs.negative:
        base = RationalCell(v, v + u, True, False)
    else:
        base = RationalCell(v - u, v, False, True)
    if slack > 0:
        return RationalCell(base.lo - slack, base.hi + slack, True, True)
    return base


def interval_center(obs: Observation, mode: str = "round") -> Fraction:
    """Midpoint of the noise-free observation cell (used for scoring)."""
    lo, hi, _, _ = observation_interval(obs, mode, 0)
    return (lo + hi) / 2


def _in_interval(f: Fraction, lo, hi, inc_lo: bool, inc_hi: bool) -> bool:
    if f < lo or f > hi:
        return False
    if f == lo and not inc_lo:
        return False
    if f == hi and not inc_hi:
        return False
    return True


# ---------------------------------------------------------------------------
# 2. Information thresholds
# ---------------------------------------------------------------------------


def sufficient_digits(qmax: int) -> int:
    """Smallest k such that k digits determine q <= Q = ``qmax`` uniquely.

    Two reals producing the same k-digit string differ by strictly less than
    10^-k, so it suffices that 10^-k not exceed the minimum gap between
    distinct fractions of height <= Q.  That gap is NOT 1/Q^2: the bound
    |p/q - p'/q'| >= 1/(qq') is attained only when qq' is maximal, and
    qq' = Q^2 forces q = q' = Q -- but two distinct fractions over the same
    denominator differ by at least 1/Q.  Hence the true minimum gap is
    >= 1/(Q(Q-1)) > 1/Q^2 and **10^k >= Q^2 already suffices** (THEORY 7.1).
    So sufficient_digits(100) == 4, verified exhaustively in the tests.
    Sufficient, not necessary: 0.333 -> 1/3 is identifiable at k = 3.
    """
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    k = 0
    while 10 ** k < qmax * qmax:
        k += 1
    return k


def minimum_spacing_denominator(denominators: Iterable[int]) -> int:
    """Exact minimum spacing of a denominator family, as its reciprocal M.

    |p/q - p'/q'| = |pq' - p'q|/(qq'), and by Bezout |pq' - p'q| ranges over
    the nonzero multiples of gcd(q, q'), so the closest distinct pair is
    exactly 1/lcm(q, q') -- attained.  Hence the family's minimum spacing is
    1/M with M = max PAIRWISE lcm (q = q' included), NOT lcm of the whole
    family: {6, 10, 14} has total lcm 210, but the worst pair is (10, 14)
    with lcm 70, so the true spacing is 1/70.
    """
    fams = _validated_denominators(denominators)
    return max(a * b // math.gcd(a, b) for a in fams for b in fams)


def sufficient_digits_for_denominators(denominators: Iterable[int]) -> int:
    """Digits sufficient to identify any fraction over a denominator family.

    Smallest k with 10^k >= M (M as in :func:`minimum_spacing_denominator`):
    *linear* in the grid where the unconstrained threshold is quadratic in Q.
    Basis points (1/10000 grid) need 4 digits, not the 8 that q <= 10^4
    would demand unrestricted; dyadic ticks to 1/64 need 2.
    """
    m = minimum_spacing_denominator(denominators)
    k = 0
    while 10 ** k < m:
        k += 1
    return k


def _validated_denominators(values: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalise a family: positive ints, deduplicated, sorted.

    Rejects zero and negative entries (``0 % d == 0`` holds for every d, so
    a dropped 0 would silently disable the divisibility test) and the empty
    family (an empty hypothesis class admits nothing).
    """
    out = set()
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"denominators must be positive integers, got {v!r}")
        if v < 1:
            raise ValueError(f"denominators must be positive, got {v}")
        out.add(v)
    if not out:
        raise ValueError("denominators must not be empty (pass None for unrestricted)")
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# 3. Continued fractions
# ---------------------------------------------------------------------------


def cf_terms(x: Fraction) -> list[int]:
    """Canonical continued fraction [a0; a1, ..., an] of a rational
    (final term >= 2 unless the expansion is a single term, making it unique;
    a0 = floor(x) may be negative, all later terms are positive)."""
    p, q = Fraction(x).numerator, Fraction(x).denominator
    terms: list[int] = []
    while q:
        a, r = divmod(p, q)
        terms.append(a)
        p, q = q, r
    if len(terms) > 1 and terms[-1] == 1:
        terms.pop()
        terms[-1] += 1
    return terms


def convergents(terms: Iterable[int]) -> list[Fraction]:
    """Convergents p_i/q_i via p_i = a_i p_{i-1} + p_{i-2} (same for q)."""
    p0, q0, p1, q1 = 0, 1, 1, 0
    out = []
    for a in terms:
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        out.append(Fraction(p1, q1))
    return out


def best_rational_bounded(x: Fraction, qmax: int) -> Fraction:
    """Closest rational to ``x`` with denominator <= qmax, in O(len cf).

    By best-approximation theory the minimiser is a convergent or a
    semiconvergent obtained by clipping the last partial quotient, so no
    search over denominators is needed.
    """
    x = Fraction(x)
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if x.denominator <= qmax:
        return x
    terms = cf_terms(x)
    p0, q0 = 1, 0
    p1, q1 = terms[0], 1
    for a in terms[1:]:
        p2, q2 = a * p1 + p0, a * q1 + q0
        if q2 > qmax:
            t = (qmax - q0) // q1  # largest admissible semiconvergent step
            cands = [Fraction(p1, q1)]
            if t >= 1:
                cands.append(Fraction(t * p1 + p0, t * q1 + q0))
            return min(cands, key=lambda c: abs(x - c))
        p0, q0, p1, q1 = p1, q1, p2, q2
    return Fraction(p1, q1)


def legendre_certified(x: Fraction, cand: Fraction) -> bool:
    """Legendre: |x - p/q| < 1/(2q^2) forces p/q to be a convergent of x.

    Passing certifies *convergent-hood* and nothing more -- several
    convergents can pass at once (both 22/7 and 355/113 do against pi).
    Uniqueness of a reconstruction comes only from :func:`identify`.
    """
    q = Fraction(cand).denominator
    return abs(Fraction(x) - cand) * 2 * q * q < 1


# ---------------------------------------------------------------------------
# 4. Stern-Brocot descent and Farey enumeration
# ---------------------------------------------------------------------------


def simplest_in_interval(lo, hi, include_lo: bool = True, include_hi: bool = True) -> Fraction:
    """The fraction with the smallest denominator in the interval.

    Stern-Brocot descent, one continued-fraction term per level.  For minimal
    denominator q >= 2 the minimiser is unique: two fractions with the same
    denominator can never be Farey-adjacent (adjacency forces q(c - a) = 1),
    so something strictly simpler always lies between them.  Ties among
    *integers* go to the one closest to zero.
    """
    lo, hi = Fraction(lo), Fraction(hi)
    if lo > hi or (lo == hi and not (include_lo and include_hi)):
        raise EmptyIntervalError(f"empty interval ({lo}, {hi})")
    return _simplest(lo, hi, include_lo, include_hi)


def _simplest(lo: Fraction, hi: Fraction, inc_lo: bool, inc_hi: bool) -> Fraction:
    # Any integer inside beats every non-integer.
    a = math.ceil(lo)
    if a == lo and not inc_lo:
        a += 1
    b = math.floor(hi)
    if b == hi and not inc_hi:
        b -= 1
    if a <= b:
        if a <= 0 <= b:
            return Fraction(0)
        return Fraction(a if a > 0 else b)
    # No integer inside: the interval sits in (n, n+1] with n = floor(lo).
    # x = n + 1/y maps it onto y in [1/hi2, 1/lo2] with inclusivities
    # swapped; recursing peels one continued-fraction term per level.
    n = math.floor(lo)
    lo2, hi2 = lo - n, hi - n
    if lo2 == 0:
        # lo was an excluded integer endpoint: y ranges over [1/hi2, +inf),
        # so the simplest y is the smallest admissible integer.
        new_lo = 1 / hi2
        m = math.floor(new_lo)
        y = Fraction(m) if (new_lo == m and inc_hi) else Fraction(m + 1)
    else:
        y = _simplest(1 / hi2, 1 / lo2, inc_hi, inc_lo)
    return n + 1 / y


def farey_left_neighbor(f: Fraction, qmax: int) -> Fraction:
    """Predecessor of ``f`` in the Farey sequence of order ``qmax``
    (extended to all rationals; the identity p*b - a*q = 1 holds throughout
    the Stern-Brocot tree).  The result a/b is the unique fraction with
    p*b - a*q = 1 and the largest b <= qmax, which makes a/b, f adjacent."""
    p, q = f.numerator, f.denominator
    if q > qmax:
        raise ValueError(f"denominator {q} exceeds Farey order {qmax}")
    if q == 1:
        return Fraction(p * qmax - 1, qmax)
    b = pow(p, -1, q)  # 0 < b < q with p*b = 1 (mod q)
    b += (qmax - b) // q * q  # lift to the largest solution <= qmax
    return Fraction((p * b - 1) // q, b)


def farey_in_interval(
    lo, hi, qmax: int, include_lo: bool = True, include_hi: bool = True,
    max_count: int | None = None,
) -> list[Fraction]:
    """ALL fractions with denominator <= qmax in the interval, ascending.

    Starts from the simplest fraction and walks outward with the Farey
    neighbour recurrence: if a/b < c/d are adjacent in F_Q, the next term is
    (kc - a)/(kd - b) with k = floor((Q + b)/d), symmetrically leftward.
    O(1) big-int work per element, so total cost is linear in the OUTPUT --
    this is what makes exact ambiguity counts affordable.  Raises
    ``OverflowError`` past ``max_count`` (callers report a truncated count
    rather than hanging).
    """
    lo, hi = Fraction(lo), Fraction(hi)
    try:
        s = simplest_in_interval(lo, hi, include_lo, include_hi)
    except EmptyIntervalError:
        return []
    if s.denominator > qmax:
        return []

    total = 0

    def bump() -> None:
        nonlocal total
        total += 1
        if max_count is not None and total > max_count:
            raise OverflowError(f"more than {max_count} fractions with q <= {qmax}")

    right: list[Fraction] = []
    a, c = farey_left_neighbor(s, qmax), s
    while _in_interval(c, lo, hi, include_lo, include_hi):
        bump()
        right.append(c)
        k = (qmax + a.denominator) // c.denominator
        a, c = c, Fraction(k * c.numerator - a.numerator, k * c.denominator - a.denominator)
    left: list[Fraction] = []
    a, c = farey_left_neighbor(s, qmax), s
    while _in_interval(a, lo, hi, include_lo, include_hi):
        bump()
        left.append(a)
        k = (qmax + c.denominator) // a.denominator
        a, c = Fraction(k * a.numerator - c.numerator, k * a.denominator - c.denominator), a
    left.reverse()
    return left + right


# ---------------------------------------------------------------------------
# 5. Periodic-pattern hypotheses (verified, never certified)
# ---------------------------------------------------------------------------


def find_periodic_patterns(digits: str, min_repeats: float = 2.0, max_patterns: int = 8):
    """Candidate (preperiod s, period t) pairs consistent with the digits.

    A pair is reported only if the window shows >= ``min_repeats`` full
    periods after the preperiod -- the period must be *observed* to repeat,
    not extrapolated.  Ordered by description length s + t (Occam).
    """
    n = len(digits)
    found = []
    for t in range(1, int(n / min_repeats) + 1):
        s = 0
        for i in range(n - t - 1, -1, -1):
            if digits[i] != digits[i + t]:
                s = i + 1
                break
        if n - s >= min_repeats * t:
            found.append((s, t))
    found.sort(key=lambda st: (st[0] + st[1], st[1]))
    return found[:max_patterns]


def pattern_value(int_part: int, negative: bool, digits: str, s: int, t: int) -> Fraction:
    """Exact value of ``int_part . digits[:s] (repeating t-block)``:
    A + B/10^s + C/(10^s (10^t - 1))."""
    pre = int(digits[:s]) if s else 0
    per = int(digits[s : s + t])
    v = int_part + Fraction(pre, 10 ** s) + Fraction(per, 10 ** s * (10 ** t - 1))
    return -v if negative else v


def periodic_candidates(obs: Observation, mode: str = "round", min_repeats: float = 2.0):
    """Exact rationals implied by periodic structure in the observed digits.

    Rounding the final digit is the one legitimate way a periodic expansion
    appears aperiodic (2/3 -> 0.66667), and a round-up can cascade through
    trailing 9s, so detection retries with up to three trailing digits
    dropped.  Every proposal is then verified to regenerate the full k-digit
    observation -- a finite window witnesses a period, it never proves the
    unseen continuation, so nothing here is a certificate.
    """
    out = {}
    variants = [
        obs.frac_digits[: obs.k - j] for j in range(min(3, obs.k - 2) + 1) if obs.k >= 2
    ] or [obs.frac_digits]
    target = obs.text()
    for digs in variants:
        for s, t in find_periodic_patterns(digs, min_repeats):
            v = pattern_value(obs.int_part, obs.negative, digs, s, t)
            if v not in out and decimal_digits(v, obs.k, mode) == target:
                out[v] = (s, t)
    return list(out.items())


# ---------------------------------------------------------------------------
# 6. Exact LLL and the reconstruction lattice
# ---------------------------------------------------------------------------
# If x is observed to accuracy delta and x ~ p/q with q <= Q, the lattice
# spanned by (1, x/delta) and (0, 1/delta) contains q*b0 - p*b1 =
# (q, (qx - p)/delta) with both entries O(Q); when 1/delta >> Q^2 that vector
# is far below the Gaussian heuristic, so reduction finds it.  This is the
# noise-robust twin of the continued-fraction algorithm, and it generalises
# to several decimals sharing one denominator (one extra dimension each).


def _dot(u, v) -> Fraction:
    return sum(a * b for a, b in zip(u, v))


def nearest_int(x: Fraction) -> int:
    x = Fraction(x)
    return (2 * x.numerator + x.denominator) // (2 * x.denominator)


def lll_reduce(basis: Sequence[Sequence], delta: Fraction = Fraction(3, 4)) -> list:
    """Textbook LLL over exact Fractions (dimensions here are 2-6, so
    clarity beats speed: Gram-Schmidt is recomputed after each change).

    Rejects ragged or linearly dependent bases and delta outside (1/4, 1),
    where LLL's termination and approximation guarantees do not hold.
    """
    b = [[Fraction(x) for x in row] for row in basis]
    n = len(b)
    if n == 0:
        raise ValueError("basis must be non-empty")
    if len({len(row) for row in b}) != 1:
        raise ValueError("basis rows must all have the same length")
    if len(b[0]) < n:
        raise ValueError(f"{n} vectors cannot be independent in dimension {len(b[0])}")
    delta = Fraction(delta)
    if not Fraction(1, 4) < delta < 1:
        raise ValueError(f"LLL delta must satisfy 1/4 < delta < 1, got {delta}")

    def gso():
        star, mu = [], [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            v = list(b[i])
            for j in range(i):
                norm = _dot(star[j], star[j])
                if norm == 0:
                    raise ValueError("basis vectors are linearly dependent")
                mu[i][j] = _dot(b[i], star[j]) / norm
                v = [x - mu[i][j] * y for x, y in zip(v, star[j])]
            star.append(v)
        if any(_dot(v, v) == 0 for v in star):
            raise ValueError("basis vectors are linearly dependent")
        return star, mu

    star, mu = gso()
    k = 1
    while k < n:
        changed = False
        for j in range(k - 1, -1, -1):  # size reduction
            r = nearest_int(mu[k][j])
            if r:
                b[k] = [x - r * y for x, y in zip(b[k], b[j])]
                changed = True
        if changed:
            star, mu = gso()
        if _dot(star[k], star[k]) >= (delta - mu[k][k - 1] ** 2) * _dot(star[k - 1], star[k - 1]):
            k += 1  # Lovasz condition holds
        else:
            b[k - 1], b[k] = b[k], b[k - 1]
            star, mu = gso()
            k = max(k - 1, 1)
    return b


def lattice_refine(x, k: int, qmax: int, noise_ulp=0) -> list[Fraction]:
    """Candidate rationals for ``x`` known to +-delta = (1/2 + noise)*10^-k,
    via 2D lattice reduction; sorted by |x - p/q|.  Degrades continuously as
    delta grows, which is exactly where interval bisection is brittle."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if Fraction(noise_ulp) < 0:
        raise ValueError("noise_ulp must be >= 0")
    x = Fraction(x)
    delta = (Fraction(1, 2) + Fraction(noise_ulp)) * Fraction(1, 10 ** k)
    inv = 1 / delta
    r0, r1 = lll_reduce([[Fraction(1), x * inv], [Fraction(0), inv]])
    seen = {}
    for v in (r0, r1, [a + b for a, b in zip(r0, r1)], [a - b for a, b in zip(r0, r1)]):
        if v[0] == 0:
            continue
        q = int(v[0])
        p = nearest_int(q * x)
        if q < 0:
            q, p = -q, -p
        if q <= qmax:
            f = Fraction(p, q)
            seen.setdefault(f, abs(x - f))
    return sorted(seen, key=lambda f: (seen[f], f.denominator))


def shared_denominator(values: Sequence, deltas: Sequence, qmax: int):
    """LLL fast path for a COMMON denominator across several decimals.

    Pooling intuition, stated carefully: n decimals of k digits behave like
    one decimal of ~ 2nk/(n+1) digits for pinning down q (NOT n*k -- the
    success condition Q^(n+1) delta^n <~ 1 versus Q^2 10^-K <~ 1 saturates
    at 2k as n grows), so the joint problem can succeed where every
    component alone is ambiguous.  Returns (q, [p_i/q, ...]) satisfying the
    exact bound |q x_i - p_i| <= q delta_i, or None.  A None result does not
    establish non-existence (the Gaussian heuristic can simply miss); the
    decision procedure lives in :func:`reconstruct_shared`.
    """
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if len(values) != len(deltas):
        raise ValueError("values and deltas must have equal length")
    if not values:
        raise ValueError("need at least one observation")
    if any(Fraction(d) <= 0 for d in deltas):
        raise ValueError("deltas must be positive")
    xs = [Fraction(v) for v in values]
    ds = [Fraction(d) for d in deltas]
    n = len(xs)
    basis = [[Fraction(1)] + [x / d for x, d in zip(xs, ds)]]
    for i in range(n):
        row = [Fraction(0)] * (n + 1)
        row[1 + i] = 1 / ds[i]
        basis.append(row)
    for v in sorted(lll_reduce(basis), key=lambda v: _dot(v, v)):
        if v[0] == 0:
            continue
        q = abs(int(v[0]))
        if not 1 <= q <= qmax:
            continue
        ps = [nearest_int(q * x) for x in xs]
        if all(abs(q * x - p) <= q * d for x, p, d in zip(xs, ps, ds)):
            return q, [Fraction(p, q) for p in ps]
    return None


# ---------------------------------------------------------------------------
# 7. Single-observation pipeline: identify (certificates) + reconstruct (rank)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One ranked reconstruction hypothesis."""

    fraction: Fraction
    residual: Fraction  # |cell centre - fraction|, exact
    residual_ulp: float
    log_score: float  # -s ln q - (residual_ulp / sigma)^2 / 2
    sources: tuple[str, ...]  # subset of ('constrained','interval','lattice','periodic')
    legendre: bool
    periodic: tuple[int, int] | None = None  # (preperiod, period) if detected

    @property
    def numerator(self) -> int:
        return self.fraction.numerator

    @property
    def denominator(self) -> int:
        return self.fraction.denominator

    def __str__(self) -> str:
        return (
            f"{self.fraction} (q={self.denominator}, resid={self.residual_ulp:.3g} ulp, "
            f"score={self.log_score:.3f}, {'+'.join(self.sources)})"
        )


@dataclass(frozen=True)
class IdentifyReport:
    """Exact identifiability: when ``identifiable``, the answer is a
    certificate, and NO algorithm -- classical or learned -- could do better
    from these digits.  A truncated enumeration sets ``exhaustive`` False,
    makes ``consistent_count`` a lower bound, and forces ``identifiable``
    False (conservative, never optimistic)."""

    consistent_count: int
    exhaustive: bool
    identifiable: bool
    witness: Fraction | None
    cell: RationalCell


def _family_fractions(cell: RationalCell, qmax: int, families: tuple[int, ...]) -> list[Fraction]:
    """EVERY cell fraction whose reduced denominator divides a family member.

    Admissibility is divisor-closed, so enumerate the *divisors* of family
    members that are <= qmax (an O(sqrt m) scan each), never the members
    themselves: denominators=[10**9] with qmax=10 means the six divisors
    {1, 2, 4, 5, 8, 10}, not a sweep of 10^9.  Exhaustive by construction,
    or SearchIncomplete when the admissible set is genuinely enormous --
    a truncated scan cannot back an exhaustiveness claim.
    """
    lo, hi, inc_lo, inc_hi = cell
    qs = set()
    work = 0
    for m in families:
        limit = min(math.isqrt(m), qmax)
        work += limit
        for d in range(1, limit + 1):
            if m % d:
                continue
            if d <= qmax:
                qs.add(d)
            if m // d <= qmax:
                qs.add(m // d)
    out = set()
    for q in sorted(qs):
        p_lo, p_hi = math.ceil(lo * q), math.floor(hi * q)
        work += max(0, p_hi - p_lo + 1)
        if work > SEARCH_BUDGET:
            raise SearchIncomplete(
                f"family enumeration would exceed {SEARCH_BUDGET} steps at q = {q}; "
                "tighten the bounds or supply more digits"
            )
        for p in range(p_lo, p_hi + 1):
            f = Fraction(p, q)  # reduces; the reduced denominator divides q
            if _in_interval(f, lo, hi, inc_lo, inc_hi):
                out.add(f)
    return sorted(out)


def identify(
    decimal, qmax: int, *, mode: str = "round", noise_ulp=0,
    max_count: int = 100_000, denominators: Iterable[int] | None = None,
) -> IdentifyReport:
    """Does the observation determine a *unique* rational within the bounds?

    This separates information-theoretic ambiguity from ranking error: when
    the report is identifiable, :func:`reconstruct`'s top answer is a
    certificate, not a guess; when it is not, the digits themselves are
    insufficient and no method can be sure.  ``denominators`` restricts the
    hypothesis class (divisor-closed), answering questions like "uniquely
    identifiable *among dyadic ticks*?" -- and constraints are applied
    DURING enumeration, so counts always refer to the constrained class.
    """
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if max_count < 1:
        raise ValueError("max_count must be >= 1")
    if isinstance(decimal, float):
        decimal = repr(decimal)
    obs = parse_decimal(decimal)
    cell = observation_interval(obs, mode, noise_ulp)
    if denominators is not None:
        pool = _family_fractions(cell, qmax, _validated_denominators(denominators))
    else:
        try:
            pool = farey_in_interval(
                cell.lo, cell.hi, qmax, cell.include_lo, cell.include_hi, max_count=max_count
            )
        except OverflowError:
            return IdentifyReport(max_count, False, False, None, cell)
    unique = len(pool) == 1
    return IdentifyReport(len(pool), True, unique, pool[0] if unique else None, cell)


def _interval_candidates(cell, qmax, want, prior_exponent) -> list[Fraction]:
    """Smallest-denominator consistent fractions, by adaptive Farey doubling.

    Sound stopping rule: under a prior q^-s the score falls by s ln 2 per
    doubling of q, while the residual term can improve by at most 1/2, so
    once ``want`` candidates are collected, ceil(0.5/(s ln 2)) spare
    doublings make the stop sound for any s > 0 (one for the default s = 2).
    At s = 0 the bound is meaningless -- residual-ordered completeness comes
    from :func:`_closest_candidates` instead.
    """
    lo, hi, inc_lo, inc_hi = cell
    try:
        s = simplest_in_interval(lo, hi, inc_lo, inc_hi)
    except EmptyIntervalError:
        return []
    if s.denominator > qmax:
        return []
    pool = [s]
    q = s.denominator
    cap = max(8 * want, 512)
    # Implement the bound stated above exactly: ceil(0.5 / (s ln 2)) spare
    # doublings (0 at s = 0, where the residual-ordered walk is the exact
    # mechanism).  For the default s = 2 this is one doubling.
    spare_doublings = (
        0 if prior_exponent == 0 else math.ceil(0.5 / (prior_exponent * math.log(2)))
    )
    while q < qmax:
        if len(pool) >= want:
            if spare_doublings == 0:
                break
            spare_doublings -= 1
        q = min(qmax, 2 * q)
        try:
            pool = farey_in_interval(lo, hi, q, inc_lo, inc_hi, max_count=cap)
        except OverflowError:
            break  # keep the last complete pool
    return pool


def _closest_candidates(center, cell, qmax, want) -> list[Fraction]:
    """The ``want`` consistent fractions closest to the cell centre: start at
    the best bounded approximation (O(k) via continued fractions) and walk
    outward along F_qmax both ways, merging by distance.  Guarantees the
    maximum-likelihood candidate is in the pool wherever the prior-ordered
    enumeration stopped -- this is what makes ``prior_exponent=0`` sound."""
    lo, hi, inc_lo, inc_hi = cell
    g = best_rational_bounded(Fraction(center), qmax)

    def walk_left():
        a, b = farey_left_neighbor(g, qmax), g
        while True:
            yield b
            k = (qmax + b.denominator) // a.denominator
            a, b = Fraction(k * a.numerator - b.numerator, k * a.denominator - b.denominator), a

    def walk_right():
        a, b = farey_left_neighbor(g, qmax), g
        while True:
            k = (qmax + a.denominator) // b.denominator
            a, b = b, Fraction(k * b.numerator - a.numerator, k * b.denominator - a.denominator)
            yield b

    left, right = walk_left(), walk_right()
    lf, rf = next(left), next(right)
    out: list[Fraction] = []
    while len(out) < want:
        if lf is not None and lf < lo:
            lf = None  # every further left value is smaller still
        if rf is not None and rf > hi:
            rf = None
        if lf is None and rf is None:
            break
        take_left = rf is None or (lf is not None and abs(center - lf) <= abs(center - rf))
        f = lf if take_left else rf
        if _in_interval(f, lo, hi, inc_lo, inc_hi):
            out.append(f)
        if take_left:
            lf = next(left)
        else:
            rf = next(right)
    return out


def reconstruct(
    decimal, qmax: int, *, mode: str = "round", noise_ulp=0, max_candidates: int = 20,
    include_periodic: bool = True, include_lattice: bool = True,
    denominators: Iterable[int] | None = None, prior_exponent: float = 2.0,
) -> list[Candidate]:
    """Ranked rational candidates for a finite decimal observation.

    Candidates come from Farey enumeration (or exact family enumeration when
    ``denominators`` is given), periodic-pattern proposals, and LLL lattice
    refinement; every one is hard-filtered by the observation cell, then
    ranked by

        log score = -s ln q - (residual_ulp / sigma)^2 / 2.

    The prior P(p/q) ~ q^-s is an EXPLICIT choice: s = 2 (default) is the
    Occam/description-length prior -- not the Farey-uniform density, which is
    ~ phi(q) and *increasing* -- and below the information threshold the
    prior dominates, so measured top-1 accuracy reflects prior/source match
    rather than algorithm quality.  s = 0 ranks purely by residual.  The
    residual term is a centre-preference surrogate, not a consequence of
    deterministic rounding (inside the cell all candidates are equally
    compatible); it is motivated by the ``noise_ulp`` regime, where digits
    quantise an already-noisy value, without claiming to be the exact
    quantised-noise likelihood.

    Pool guarantee: the pool holds both the simplest consistent fractions
    (enumerated past the point where the prior can be overcome, with
    ceil(0.5 / (s ln 2)) spare doublings) and the ``max_candidates`` closest
    ones, so the top answer is exact for every s >= 0; with ``denominators``
    the pool is the entire admissible class, so the whole ranking is exact
    there.  ``max_candidates`` changes how many answers you get, never which
    ranks first.
    """
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if not math.isfinite(prior_exponent) or prior_exponent < 0:
        raise ValueError("prior_exponent must be a finite number >= 0")
    noise = Fraction(noise_ulp)
    if isinstance(decimal, float):
        decimal = repr(decimal)
    obs = parse_decimal(decimal)
    cell = observation_interval(obs, mode, noise)
    center = interval_center(obs, mode)
    ulp = obs.ulp
    sigma = 0.5 + float(noise)
    families = None if denominators is None else _validated_denominators(denominators)

    sources: dict = {}
    periodic_info: dict = {}
    if families is None:
        for f in _interval_candidates(cell, qmax, max_candidates, prior_exponent):
            sources.setdefault(f, set()).add("interval")
        for f in _closest_candidates(center, cell, qmax, max_candidates):
            sources.setdefault(f, set()).add("interval")
    else:
        for f in _family_fractions(cell, qmax, families):
            sources.setdefault(f, set()).add("constrained")
    if include_periodic:
        for f, st in periodic_candidates(obs, mode):
            sources.setdefault(f, set()).add("periodic")
            periodic_info[f] = st
    if include_lattice and obs.k > 0:
        for f in lattice_refine(center, obs.k, qmax, noise):
            sources.setdefault(f, set()).add("lattice")

    out: list[Candidate] = []
    for f, src in sources.items():
        if f not in cell:  # consistency is hard evidence: zero likelihood outside
            continue
        if f.denominator > qmax:
            continue
        if families is not None and not any(m % f.denominator == 0 for m in families):
            continue  # no source can widen an explicit constraint
        resid = abs(center - f)
        r_ulp = float(resid / ulp)
        score = -prior_exponent * math.log(f.denominator) - 0.5 * (r_ulp / sigma) ** 2
        out.append(
            Candidate(f, resid, r_ulp, score, tuple(sorted(src)),
                      legendre_certified(center, f), periodic_info.get(f))
        )
    out.sort(key=lambda c: (-c.log_score, c.denominator, c.fraction))
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# 8. Shared denominators: three graded strengths of claim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedResult:
    denominator: int
    fractions: tuple[Fraction, ...]
    residual_ulps: tuple[float, ...] = field(default=())
    method: str = "lll"  # 'lll' (heuristic fast path) or 'exhaustive' (exact)


def _parse_shared(decimals: Sequence, mode: str, noise: Fraction):
    decimals = list(decimals)
    if not decimals:
        raise ValueError("need at least one observation")
    if noise < 0:
        raise ValueError("noise_ulp must be >= 0")
    obs = [parse_decimal(repr(d) if isinstance(d, float) else d) for d in decimals]
    centers = [interval_center(o, mode) for o in obs]
    deltas = [(Fraction(1, 2) + noise) * o.ulp for o in obs]
    cells = [observation_interval(o, mode, noise) for o in obs]

    def digit_ok(fracs) -> bool:
        if noise:
            return True  # the widened bound *is* the model once noise is declared
        return all(decimal_digits(f, o.k, mode) == o.text() for f, o in zip(fracs, obs))

    return obs, centers, deltas, cells, digit_ok


def _numerators_in_cell(cell: RationalCell, q: int) -> range:
    """Every integer p with p/q inside the observation cell (exact)."""
    lo, hi, inc_lo, inc_hi = cell
    p_lo = math.ceil(lo * q)
    if not inc_lo and p_lo == lo * q:
        p_lo += 1
    p_hi = math.floor(hi * q)
    if not inc_hi and p_hi == hi * q:
        p_hi -= 1
    return range(p_lo, p_hi + 1)


def _iter_shared_feasible(cells, centers, qmax: int):
    """Yield (q, fracs) for each feasible common denominator, ascending,
    taking the centre-nearest numerator per observation.  Sufficient for
    existence/minimality (if any numerator works, the nearest does) -- but
    NOT for counting hypotheses; that is :func:`_iter_shared_tuples`."""
    for q in range(1, qmax + 1):
        picks = []
        for cell, c in zip(cells, centers):
            ps = _numerators_in_cell(cell, q)
            if not ps:
                break
            # Nearest admissible numerator in O(1): round c*q half-down and
            # clamp into the range.  This keeps the whole scan O(Q n).
            p = math.ceil(c * q - Fraction(1, 2))
            picks.append(min(max(p, ps.start), ps.stop - 1))
        else:
            yield q, [Fraction(p, q) for p in picks]


def _iter_shared_tuples(cells, qmax: int, budget: int):
    """Yield every DISTINCT reduced tuple consistent with all cells for some
    common q <= qmax (smallest representing q first).  Takes the FULL
    Cartesian product of admissible numerators at each q: a cell can admit
    two numerators over one q ("1.0" at qmax=20 contains both 1 and 19/20),
    so uniqueness requires enumerating them all."""
    seen = set()
    for q in range(1, qmax + 1):
        ranges = []
        for cell in cells:
            ps = _numerators_in_cell(cell, q)
            if not ps:
                break
            ranges.append(ps)
        else:
            for combo in itertools.product(*ranges):
                t = tuple(Fraction(p, q) for p in combo)
                if t in seen:
                    continue
                if len(seen) >= budget:
                    raise OverflowError(f"more than {budget} consistent tuples")
                seen.add(t)
                yield q, t


def reconstruct_shared(
    decimals: Sequence, qmax: int, *, mode: str = "round", noise_ulp=0,
    exhaustive_limit: int = SHARED_EXHAUSTIVE_LIMIT, minimize: bool = False,
) -> SharedResult | None:
    """Reconstruct several decimals assumed to share one denominator.

    LLL fast path first (pooling can identify q when every component alone
    is ambiguous); if it misses and ``qmax <= exhaustive_limit``, an exact
    O(qmax) scan decides.  Claims, precisely graded: a result is always a
    *verified* representation but the LLL q may be non-minimal;
    ``minimize=True`` scans exhaustively and guarantees the SMALLEST q;
    uniqueness of the tuple is :func:`identify_shared`'s job.  ``None``
    means PROVED non-existence (scan ran); an undecided lattice miss raises
    :class:`SearchIncomplete` instead of inventing a "no".  At noise 0 every
    result regenerates the observed digit strings exactly.
    """
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if exhaustive_limit < 0:
        raise ValueError("exhaustive_limit must be >= 0")
    noise = Fraction(noise_ulp)
    obs, centers, deltas, cells, digit_ok = _parse_shared(decimals, mode, noise)

    method, hit = "lll", None
    if not minimize:
        hit = shared_denominator(centers, deltas, qmax)
        if hit is not None and not digit_ok(hit[1]):
            hit = None
    decided = minimize or qmax <= exhaustive_limit
    if hit is None and decided:
        method = "exhaustive"
        hit = next(_iter_shared_feasible(cells, centers, qmax), None)
    if hit is None:
        if not decided:
            raise SearchIncomplete(
                f"the lattice heuristic found no common denominator q <= {qmax} and "
                f"the exhaustive scan was skipped (qmax > {exhaustive_limit}); pass "
                "minimize=True to decide the question"
            )
        return None
    q, fracs = hit
    resid = tuple(float(abs(c - f) / o.ulp) for c, f, o in zip(centers, fracs, obs))
    return SharedResult(q, tuple(fracs), resid, method)


@dataclass(frozen=True)
class SharedIdentifyReport:
    """``consistent_count`` counts distinct REDUCED TUPLES over all
    consistent q <= qmax: a q and its multiples describe one tuple (counted
    once); two numerator choices at the same q are two (counted twice).
    Truncation makes the count a lower bound and forces identifiable False."""

    consistent_count: int
    exhaustive: bool
    identifiable: bool
    witness: SharedResult | None


def identify_shared(
    decimals: Sequence, qmax: int, *, mode: str = "round", noise_ulp=0,
    max_count: int = 100_000,
) -> SharedIdentifyReport:
    """Exhaustively decide whether decimals sharing a denominator determine
    a unique tuple of rationals -- the joint analogue of :func:`identify`,
    and the only routine entitled to say "unique" about a shared answer.
    With one observation it agrees with :func:`identify` by construction."""
    if qmax < 1:
        raise ValueError("qmax must be >= 1")
    if max_count < 1:
        raise ValueError("max_count must be >= 1")
    noise = Fraction(noise_ulp)
    obs, centers, _deltas, cells, _ok = _parse_shared(decimals, mode, noise)
    tuples: dict = {}
    exhaustive = True
    try:
        for q, fracs in _iter_shared_tuples(cells, qmax, max_count):
            tuples.setdefault(fracs, q)
    except OverflowError:
        exhaustive = False
    identifiable = exhaustive and len(tuples) == 1
    witness = None
    if identifiable:
        ((fracs, q),) = tuples.items()
        resid = tuple(float(abs(c - f) / o.ulp) for c, f, o in zip(centers, fracs, obs))
        witness = SharedResult(q, fracs, resid, "exhaustive")
    return SharedIdentifyReport(len(tuples), exhaustive, identifiable, witness)
