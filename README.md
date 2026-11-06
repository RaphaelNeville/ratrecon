# Rational Reconstruction from Noisy Decimal Data

Recover exact rationals `p/q` from truncated or corrupted decimals, using
continued-fraction convergents, Stern–Brocot descent, Farey enumeration and
LLL lattice reduction in exact arithmetic. Candidates are ranked under a
prior with a tunable exponent. Optionally, a GRU proposes denominators and
each proposal is checked exactly. Proofs of the identifiability thresholds
are in `THEORY.md`.

The core module and demo are standard-library Python, with
`fractions.Fraction` arithmetic for every consistency or identifiability
decision. The tests run under pytest, and the optional GRU experiment needs
PyTorch.

```
ratrecon.py       the full pipeline in a single module
test_ratrecon.py  tests of the main guarantees, several against brute force
demo.py           a walkthrough in 7 sections (reproduces the benchmark below)
hybrid_gru.py     proposer-verifier experiment (GRU proposals, exact checks)
THEORY.md / .pdf  the mathematics, as notes and as a typeset paper
results/          committed experiment artefacts (JSON)
```

## Quick start

```bash
python demo.py               # walkthrough, a few seconds, stdlib only
pytest test_ratrecon.py      # 12 tests, < 1 s
python hybrid_gru.py         # the ML experiment (needs torch, ~2 min)
```

```python
from ratrecon import identify, reconstruct, reconstruct_shared

reconstruct("3.14159292", qmax=1000)[0].fraction     # Fraction(355, 113)
identify("0.1", qmax=20, mode="truncate").identifiable
# False: many fractions with q <= 20 truncate to "0.1"

reconstruct_shared(["0.377", "0.656", "0.180", "0.852"], qmax=200).denominator
# 61, found by LLL from four 3-digit decimals, each ambiguous on its own
```

## Identifiability and ranking

`identify` decides whether the digits contain enough information to
determine a unique rational of bounded denominator. It counts every
consistent fraction exactly (by Farey-neighbour enumeration, O(1) per
element) and certifies uniqueness when it holds, so information-theoretic
ambiguity is kept separate from ranking.

The threshold (THEORY §2, sharpened in §7.1) is that `k` rounded digits
identify any fraction with `q <= Q` as soon as `10^k >= Q^2`. In particular
`sufficient_digits(100) = 4`, which the test suite checks on all 3043 fractions
in (0, 1) with `q <= 100`. If the denominators are restricted to a given
family, such as the divisors of `N` (a grid with step `1/N`), the bound becomes
linear in `N`. The minimum spacing is then exactly `1/max lcm(q, q')`, with the
maximum taken over all pairs in the family, including pairs with `q = q'`.
Basis points (`N = 10^4`) therefore need four digits, whereas general
denominators up to `10^4` would need eight.

Below the threshold, candidates are ranked by
`-s*ln(q) - residual^2/(2*sigma^2)` with prior exponent `s`. The default
`s = 2` is the Occam (description-length) prior and `s = 0` ranks by
closeness to the observation alone. The Occam prior differs from the uniform
prior on the Farey set, under which the total weight on denominator `q` is
proportional to φ(q) (THEORY §3). In this regime top-1 accuracy reflects how
well the prior suits the source of the fractions and says little about the
algorithm, which is why `s` is left for the user to set.

Fractions outside the observation cell (the interval of reals consistent
with the observed digits) are discarded. If a heuristic search misses and no
exhaustive scan follows, `SearchIncomplete` is raised, so non-existence is
reported only after an exhaustive scan. For several decimals with a common
denominator, `reconstruct_shared` returns a verified denominator (LLL fast
path), or the smallest one when called with `minimize=True`.
`identify_shared` decides uniqueness by counting all consistent numerators
for each denominator, since one denominator can admit two of them. For
example, `"1.0"` at `qmax=20` fits both `1` and `19/20`.

## Results (committed in `results/`, reproducible here)

| experiment | result |
|---|---|
| k=4 digits, q ≤ 500, 1000 cases | 3.8% uniquely identifiable, top-1 33.6% overall and 100% on the identifiable cases (`demo.py` §7 reproduces this) |
| 10% digit corruption, k=12, q ≤ 100, 3 seeds | exact-fraction top-1: classical 0.685 ± 0.008, hybrid 0.703 ± 0.007 (`hybrid_gru.py`) |
| clean digits, same protocol | classical exact (1.000), hybrid lower by ≈ 0.003 |

The 3.8% identifiability rate in the first row is set by the information in
four digits, and no method can certify the remaining cases. In the corruption
experiment each digit is replaced with probability 0.1 by a uniformly random
digit, which can equal the original, so a digit actually changes with
probability 0.09. The hybrid uses a learned prior to narrow the exact search.
Its gain over the classical baseline averages 1.8 percentage points and is
positive for each of the three seeds, but this is a small pilot, and three
seeds on one data split are too few to establish the gain statistically.

## Scope

The ingredients (continued fractions, Farey and Stern–Brocot theory, Legendre's
criterion, LLL) are classical. The contribution is to combine them in one
system built around exact observation cells. Identifiability is decided from
the digits alone, which fix the admissible candidates, and a tunable prior then
ranks those candidates. The system also gives certificates of different
strengths for shared denominators and verifies learned proposals. In this
repository "rational reconstruction" means recovery from finite decimal
observations. In computer algebra the same name refers to a related modular
problem, which recovers `p/q` from residues.

See [THEORY.pdf](THEORY.pdf) for the full development, including what
provably cannot be recovered. `THEORY.pdf` is typeset from `THEORY.md` by
[`build_pdf.sh`](build_pdf.sh) with pandoc and XeLaTeX. MIT licensed
([LICENSE](LICENSE)).
