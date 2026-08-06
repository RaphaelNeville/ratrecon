# Rational Reconstruction from Noisy Decimal Data

Recover exact rationals `p/q` from truncated or corrupted decimals —
continued-fraction convergents, Stern–Brocot descent, Farey enumeration and
exact-arithmetic LLL lattice reduction, with proven identifiability
thresholds, an explicit tunable prior, and a learned proposer whose output is
always verified by exact arithmetic.

Everything is standard-library Python with exact `fractions.Fraction`
arithmetic for every consistency or identifiability decision (the optional
GRU experiment needs PyTorch). The repository is deliberately small: **one
module, one test file, one demo, one experiment, one paper.**

```
ratrecon.py       the whole pipeline, ordered the way the mathematics builds
test_ratrecon.py  one test per core guarantee; brute force as the judge
demo.py           7-section guided tour (reproduces the benchmark below)
hybrid_gru.py     proposer–verifier experiment (GRU + exact verification)
THEORY.md / .pdf  the mathematics, as notes and as a typeset paper
results/          committed experiment artifacts (JSON)
```

## Quick start

```bash
python demo.py               # guided tour, a few seconds, stdlib only
pytest test_ratrecon.py      # 12 tests, < 1 s
python hybrid_gru.py         # the ML experiment (needs torch, ~2 min)
```

```python
from ratrecon import identify, reconstruct, reconstruct_shared

reconstruct("3.14159292", qmax=1000)[0].fraction     # Fraction(355, 113)
identify("0.1", qmax=20, mode="truncate").identifiable
# False — and then no algorithm, classical or learned, can be sure

reconstruct_shared(["0.377", "0.656", "0.180", "0.852"], qmax=200).denominator
# 61 — four 3-digit decimals pooled by LLL, though each alone is ambiguous
```

## The central question

Not *"which fraction is closest?"* but **"do these digits contain enough
information to identify a unique bounded rational at all?"** The library
exposes this question explicitly: `identify()` counts the complete consistent
set exactly (Farey-neighbour enumeration, O(1) per element) and certifies
uniqueness, separating information-theoretic ambiguity from ranking.

**The threshold** (THEORY §2, sharpened in §7.1): `k` rounded digits
determine `q ≤ Q` as soon as `10^k ≥ Q²` — equality included, because the
extremal Farey gap `1/Q²` would force `q = q′ = Q`, while same-denominator
fractions are `1/Q` apart; so the true minimum gap is `≥ 1/(Q(Q−1))` and
`sufficient_digits(100) = 4`, verified exhaustively over all 3043 fractions
in the test suite. Under an explicit denominator family the bound turns
**linear** in the grid: spacing is exactly `1/max lcm(q,q′)` (worst *pair*,
not the family lcm), so basis points need 4 digits, not 8.

**Ranking is a modelling choice, made explicit.** Below the threshold,
candidates are ranked by `−s·ln q − residual²/2σ²` with the prior exponent
`s` exposed: `s = 2` is the Occam/description-length prior (not the
Farey-uniform density, which grows like φ(q)), `s = 0` is pure maximum
likelihood. Measured top-1 accuracy reflects prior/source match rather than
algorithm quality — which is why it is exposed rather than fixed.

**Failure is never ambiguous.** Consistency with the observation cell is
hard evidence; a heuristic miss raises `SearchIncomplete` rather than being
reported as non-existence; and shared reconstruction grades its claims —
*verified* (LLL fast path), *smallest* (`minimize=True`), *unique*
(`identify_shared`, which counts all numerators per denominator: `"1.0"` at
`qmax=20` admits both `1` and `19/20`, so sampling one per q would certify
uniqueness falsely).

## Results (committed in `results/`, reproducible here)

| experiment | result |
|---|---|
| k=4 digits, q ≤ 500, 1000 cases | **3.8%** uniquely identifiable; top-1 33.6% overall but **100% whenever identifiable** (`demo.py` §7 reproduces this live) |
| 10% digit corruption, k=12, q ≤ 100, 3 seeds | exact-fraction top-1: classical **0.685 ± 0.008** → hybrid **0.703 ± 0.007** (`hybrid_gru.py`) |
| clean digits, same protocol | classical exact (1.000); the hybrid concedes ≈ 0.003 |

The low-precision row is the headline: the 3.8% identifiability rate
is an *information limit*. Overall top-1 additionally depends on the ranking
prior; whenever the observation is identifiable, recovery is exact. The hybrid row
shows the proposer–verifier design working: a learned prior narrows an exact
search under noise, and no learned guess is ever presented as a certificate.

## Scope

The individual ingredients are classical — continued fractions, Farey/
Stern–Brocot theory, Legendre's criterion, LLL — and the contribution is
the integrated system: exact observation cells,
identifiability before ranking, explicit priors, graded shared certificates,
and learned proposals subordinated to exact verification, organised as
**information → admissible candidates → prior → certificate or heuristic**.
(Here "rational reconstruction" means recovery from finite *decimal
observations* — related to, but distinct from, the modular
rational-reconstruction problem of computer algebra, which recovers p/q from
residues.)
See [THEORY.pdf](THEORY.pdf) for the full development, including what is
provably lost (`THEORY.pdf` is typeset from `THEORY.md` by
[`build_pdf.sh`](build_pdf.sh), pandoc + XeLaTeX). MIT licensed
([LICENSE](LICENSE)).
