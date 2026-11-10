# Rational Reconstruction from Noisy Decimal Data

Recover exact rationals `p/q` from rounded, truncated or noisy decimals, using
continued-fraction convergents, Stern–Brocot descent, Farey enumeration and
LLL lattice reduction in exact arithmetic. Candidates are ranked under a
prior with a tunable exponent. A pilot experiment lets a GRU propose
denominators, and on its test set a nearest-codeword decoder, which needs no
training, does much better (see Results). Proofs of the identifiability
thresholds are in `THEORY.md`.

The core module and demo are standard-library Python, with
`fractions.Fraction` arithmetic for every consistency or identifiability
decision. The tests run under pytest, and the optional GRU experiment needs
PyTorch.

```
ratrecon.py       the full pipeline in a single module
test_ratrecon.py  tests of the main guarantees, several against brute force
demo.py           a walkthrough in 7 sections (reproduces the benchmark below)
hybrid_gru.py     digit-corruption experiment (GRU proposals and baselines)
THEORY.md / .pdf  notes on the mathematics (Markdown source and typeset PDF)
results/          committed experiment artefacts (JSON)
```

## Quick start

```bash
python demo.py               # walkthrough, a few seconds, stdlib only
pytest test_ratrecon.py      # 12 tests, < 1 s
python hybrid_gru.py         # GRU experiment, seed 0 (needs torch, < 1 min)
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
element, up to the budget `max_count`) and certifies uniqueness when it
holds, so what the digits determine is kept separate from how candidates are
ranked.

The threshold (THEORY §2.1) is that `k` rounded or truncated digits identify
any fraction with `q <= Q` as soon as `10^k >= Q(Q-1)`, because distinct such
fractions are at least `1/(Q(Q-1))` apart. In particular
`sufficient_digits(100) = 4`, and `sufficient_digits(32) = 3` where the cruder
bound `10^k >= Q^2` would give 4. The test suite checks both on all fractions
in (0, 1) with `q <= 100` and `q <= 32`. If the denominators are restricted to
a given family, the minimum spacing is exactly `1/M`, where `M` is the largest
`lcm(q, q')` over all pairs in the family, including pairs with `q = q'`, and
`10^k >= M` suffices. For the divisors of `N` (a grid with step `1/N`) this
gives `M = N`, so basis points (`N = 10^4`) need four digits, whereas general
denominators up to `10^4` would need eight.

Below the threshold, candidates are ranked by
`-s*ln(q) - residual^2/(2*sigma^2)`, where `s` is the prior exponent
(`prior_exponent`), the residual is measured in units of the last digit and
`sigma = 1/2 + noise_ulp`. The default `s = 2` is the Occam
(description-length) prior and `s = 0` ranks by closeness to the observation
alone. The Occam prior differs from the uniform prior on the Farey set, under
which the total weight on denominator `q` is proportional to φ(q) (THEORY §3).
In this regime top-1 accuracy reflects how well the prior suits the source of
the fractions and says little about the algorithm, which is why `s` is left
for the user to set.

Fractions outside the observation cell (the interval of reals consistent
with the observed digits) are discarded. If a heuristic search misses and no
exhaustive scan follows, `SearchIncomplete` is raised, so non-existence is
reported only after an exhaustive scan. For several decimals with a common
denominator, `reconstruct_shared` returns a verified denominator (LLL fast
path), or the smallest one when called with `minimize=True`.
`identify_shared` decides uniqueness by counting all consistent numerators
for each denominator, since one denominator can admit two of them. For
example, `"1.0"` at `qmax=20` fits both `19/20` and `20/20`.

## Results (committed in `results/`, reproducible here)

| experiment | result |
|---|---|
| k=4 digits, q ≤ 500, 1000 cases | 3.8% uniquely identifiable. Top-1 33.6% with the default prior s = 2 (`demo.py` §7 reproduces this) |
| 10% digit corruption, k=12, q ≤ 100, 3 seeds | exact-fraction top-1 (mean ± sample sd): classical 0.685 ± 0.008, hybrid 0.703 ± 0.007, nearest-codeword decoder 0.945 ± 0.004 (`hybrid_gru.py`) |
| clean digits, same protocol | classical 1.000, hybrid 0.997 |

The first row was measured on one sample: `q` uniform on 2 to 500, then `p`
uniform among numerators coprime to `q`, with the 1000 cases split between
truncation and rounding. The 3.8% identifiability rate is what four digits
determine for that sample, and no method can certify the remaining cases. The
top-1 rate also depends on how well the prior fits the sampling (THEORY §3).
On the identifiable cases it is 100% by construction.

In the corruption experiment each digit is replaced with probability 0.1 by a
uniformly random digit, which can equal the original, so a digit actually
changes with probability 0.09. Since four digits suffice for `q <= 100`,
clean inputs at twelve digits are never ambiguous, and the task is to correct
digit errors. The hybrid lets a GRU propose denominators and checks them
against the observation cell, widened until a candidate fits. Its mean gain
over the classical baseline is 1.8 percentage points (3.4, 1.3 and 0.6 per
seed), which three seeds do not establish statistically. The nearest-codeword
decoder returns the fraction whose digits differ from the observed ones in the
fewest places. It needs no training, and for this noise model and a uniform
prior it is the maximum a posteriori decoder. The classical baseline instead
treats noise as a shift in value, and a changed digit in any of the first
three places almost always makes that shift too large for it to correct (it
recovers 1 of the 1169 such inputs). The nearest-codeword decoder recovers
94.5% of the fractions, about 24 percentage points more than the hybrid, so
for the learned component the experiment is a negative result at this scale.
THEORY §7.2 gives the full table, including a matched-prior comparison and the
hybrid given the true denominator, and discusses where learned proposals might
still help.

## Scope

The ingredients (continued fractions, Farey and Stern–Brocot theory, Legendre's
criterion, LLL) are classical, and recovering a fraction from its leading
decimal digits with the extended Euclidean algorithm is a textbook application
(Shoup, *A Computational Introduction to Number Theory and Algebra*, 2008,
§4.6.1). `ratrecon` is a library built around exact half-open observation
cells for both rounding and truncation. It decides identifiability from the
digits alone, enumerates the whole ambiguity set when the digits do not settle
the answer, and ranks that set under an explicit prior. It also gives
certificates of different strengths for shared denominators. In this
repository "rational reconstruction" means recovery from finite decimal
observations. In computer algebra the same name refers to recovering `p/q`
from a residue modulo `m`. The two are closely related. If `p/q > 0` and `D`
is the integer formed by its `k` rounded digits, the relation
`qD - 10^k p = r` with `-q/2 < r <= q/2` gives a congruence modulo `10^k` of
the kind modular reconstruction solves, and the extended Euclidean algorithm
used there computes the continued fraction of `D/10^k` that `ratrecon` uses.
THEORY §2.1 gives the connection and its limits (decimal denominators need not
be invertible modulo `10^k`, and a modular solution must still be checked
against the digits).

## Use of AI tools

AI tools were used for programming and for checking computations.

## Further reading

See [THEORY.pdf](THEORY.pdf) for the full development, including what
provably cannot be recovered. `THEORY.pdf` is typeset from `THEORY.md` by
[`build_pdf.sh`](build_pdf.sh) with pandoc and XeLaTeX. MIT licensed
([LICENSE](LICENSE)).
