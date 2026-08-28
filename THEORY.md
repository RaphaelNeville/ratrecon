# How much arithmetic survives a decimal expansion?

*Notes on the information content of finite decimal observations of rationals —
what can be recovered, at what cost, and what is inevitably lost. These are
expository notes: the theorems are classical; the framing, and the way the
implementation in `ratrecon` realises each bound, is what the notes are for.*

---

## 1. Setup

Fix a height bound $Q$ and write
$\mathcal{F}_Q = \{\,p/q \in \mathbb{Q} : 1 \le q \le Q\,\}$
(numerator bounds enter trivially and are ignored here). The *observation map*

$$\rho_k : x \mapsto \text{first } k \text{ decimal digits of } x \text{ (rounded)}$$

sends a rational to a finite string. Two questions:

1. **Exact regime.** When is $\rho_k$ injective on $\mathcal{F}_Q$, and how
   cheaply can it be inverted?
2. **Lossy regime.** When it is not injective — or when the digits carry noise
   — what structure remains, and how should candidates be ranked?

Everything in `ratrecon` is an algorithmic shadow of the answers below.

*Experimental results quoted below are stored in `results/`; the
identifiability benchmark is also reproduced live by `demo.py`, and the
learned-prior experiment by `hybrid_gru.py`.*

## 2. The exact regime

### 2.1 The uniqueness threshold

**Theorem 1.** *If $10^k > Q^2$ then $\rho_k$ is injective on
$\mathcal{F}_Q$: $k$ rounded digits determine $p/q$ with $q \le Q$ uniquely.*

*Proof.* Distinct $p/q \ne p'/q'$ with $q, q' \le Q$ satisfy

$$\left|\frac{p}{q} - \frac{p'}{q'}\right| = \frac{|pq' - p'q|}{qq'} \ge \frac{1}{qq'} \ge \frac{1}{Q^2},$$

since the integer $|pq'-p'q|$ is nonzero. Two reals producing the same
$k$-digit rounding lie in a common interval of length $10^{-k} < 1/Q^2$. ∎

The bound is essentially sharp: consecutive Farey fractions of order $Q$ have
gaps as small as $\sim 1/Q^2$, so with $10^k \ll Q^2$ collisions are not just
possible but generic. `ratrecon.sufficient_digits` computes the threshold with
integer arithmetic; the end-to-end tests exercise exactness at the threshold.

### 2.2 The encoding is near-optimal

$|\mathcal{F}_Q \cap (0,1]| = \sum_{q \le Q} \varphi(q) = \tfrac{3}{\pi^2}Q^2 + O(Q\log Q)$,
so identifying a member costs $\log_2(\tfrac{3}{\pi^2}Q^2) = 2\log_2 Q - 1.72 + o(1)$
bits, while $k$ digits carry $k\log_2 10$ bits. Theorem 1 needs
$k\log_2 10 \ge 2\log_2 Q$, so the *continuous* separation requirement sits
$1.72$ bits above the set entropy.

Two distinct overheads are easy to conflate here, so state both: on top of
those $1.72$ bits, $k$ must be a whole number of digits, which costs up to a
further $\log_2 10 \approx 3.32$ bits of granularity — worst case just under
five bits total, not two. (At $Q = 100$: entropy $\approx 11.6$ bits,
continuous requirement $13.3$, and $k = 4$ digits carry $13.3$ — here the
rounding costs nothing, which is exactly the equality case of §7.1.) Digits
are still a near-optimal — but *arithmetically scrambled* — encoding of
$(p,q)$; the entire subject is the cost of unscrambling.

### 2.3 Inversion in logarithmic time

Brute force over $\mathcal{F}_Q$ costs $\Theta(Q^2)$. Continued fractions
invert in $O(k)$ arithmetic steps (on integers of $O(k)$ digits, so
$O(k^2)$ bit operations with schoolbook arithmetic):

**Theorem 2 (Legendre).** *If $|x - p/q| < 1/(2q^2)$ then $p/q$ is a
convergent of the continued fraction of $x$.*

Above the threshold, the true $p/q$ satisfies Legendre's inequality with
respect to the observed value, so it appears among the $O(\log Q)$ convergents
of the digit string — this is why `cf_terms` + `convergents` suffices, and why
`ratrecon` marks candidates passing the Legendre test as *certified
convergents*. Note the precise scope: the criterion certifies membership in
the convergent sequence, nothing more — several convergents can pass at once
(both $22/7$ and $355/113$ do against $\pi$), and uniqueness of a
reconstruction is established only by exhaustive bounded enumeration
(`identify`).

The interval refinement of this idea is the **Stern–Brocot descent**
(`simplest_in_interval`): the observed digits pin $x$ into an interval $I$,
and the smallest-denominator fraction in $I$ is found by descending the
Stern–Brocot tree, one continued-fraction term per step (Richards 1981). A
useful uniqueness fact justifies "the" simplest fraction: two fractions with
the same denominator $q \ge 2$ can never be adjacent in a Farey sequence
(adjacency forces $q(c-a)=1$), so the minimal denominator in an interval is
attained exactly once.

## 3. Below the threshold: the ambiguity set

When $10^k \le Q^2$ the fibre $\rho_k^{-1}(\text{digits}) \cap \mathcal{F}_Q$
is the set of Farey fractions in an interval $I$ of length $10^{-k}$; its
expected size is

$$\mathbb{E}\,\#(\mathcal{F}_Q \cap I) \approx \frac{3}{\pi^2}\,Q^2\,10^{-k},$$

which the `farey_in_interval` neighbour-walk enumerates at $O(1)$ amortised
cost per element. Consistency with the interval is treated as *hard* evidence
(zero likelihood outside it); surviving candidates are ranked by the MAP score

$$\log \text{score}(p/q) = -s\ln q \;-\; \frac{1}{2}\left(\frac{|x_{\text{obs}} - p/q|}{\sigma \cdot 10^{-k}}\right)^{\!2},$$

with a prior $P(p/q) \propto q^{-s}$ and a Gaussian term in the residual
measured in ulps.

**What the Gaussian term is, and is not.** Under pure deterministic
quantisation the likelihood is already exhausted by the cell: it is $1$ for
every fraction inside and $0$ outside, so no residual term follows from the
rounding model — inside the cell, all candidates are equally compatible. The
Gaussian is therefore a *centre-preference surrogate*: it is motivated by an
analogue-noise model (the regime `noise_ulp` is designed for, where $\sigma$
grows and the cell widens) but is not the exact quantised-data likelihood,
which would integrate the noise density over the cell. Stated plainly: **candidates are hard-filtered by
the quantisation cell, then optionally ranked by a Gaussian centre
preference.** With $s = 2$ and exact digits the residual term is nearly
inert anyway — the prior does the discriminating.

**On the choice of prior — a point worth being precise about.** The default
$s = 2$ is a *complexity (Occam) prior*: it charges each candidate its
description length $2\log q$ and makes the simplest consistent fraction win.
It is **not** the density of the bounded Farey set: among reduced fractions in
$(0,1]$ with $q \le Q$ there are $\varphi(q)$ with denominator exactly $q$, so
sampling that set uniformly gives $P(q) \propto \varphi(q)$ — an *increasing*
function of $q$. The two priors give opposite advice, and below the threshold
the prior dominates the ranking — so whichever prior better matches the law
generating the truths wins the benchmark, and swapping the source law swaps
the winner. Measured top-1 accuracy below the threshold therefore reflects
prior/source match, not algorithmic quality, which is why `reconstruct` exposes
the exponent $s$ ($s = 0$ recovers maximum-likelihood, closest-to-observation
ranking) instead of hard-coding a philosophy. For "naturally occurring"
constants — quoted probabilities, tick prices, human-chosen ratios — small
denominators really are more probable, and $s = 2$ is the sensible default.

Section 5 of `demo.py` walks the same fraction through
$k = 3,\dots,6$ digits and watches the fibre shrink from 49 candidates to 1
at $k = 5$, one digit before the universal worst-case threshold —
illustrating that the bound is sufficient, not necessary.

## 4. Graph representation: the Stern–Brocot tree

The Stern–Brocot tree is a bijection between positive rationals and finite
words in $\{L, R\}^*$; the run-lengths of the word are precisely the continued
fraction terms $[a_0; a_1, a_2, \dots]$. This is the natural *graph-based
representation* of a rational: a vertex of an infinite binary tree, or
equivalently a vertex of the Farey graph (fractions joined when $|pq'-p'q|=1$).

What does a decimal prefix know about the tree position? The digit interval
$I$ selects the set of tree nodes lying in $I$, and the longest $L/R$ prefix
shared by every candidate — the common Stern–Brocot cylinder — is exactly the
part of the word that $k$ digits have irrevocably determined. For a closed
interval the deepest common ancestor of that set is the simplest fraction in
$I$; when an endpoint is excluded (as in the half-open observation cells) the
ancestor can be the excluded endpoint itself, with the simplest *admissible*
fraction sitting just inside at the pruning frontier
(`simplest_in_interval`). Digit refinement is monotone tree pruning.

So translating "digit knowledge" into "graph knowledge" preserves a *prefix*
of the algebraic data $(a_0, a_1, \dots)$ and destroys the rest. How long is
the preserved prefix? That is a theorem:

**Theorem 3 (Lochs, 1964).** *For almost every $x$, the number $m(k)$ of
continued-fraction terms determined by the first $k$ decimal digits satisfies*

$$\frac{m(k)}{k} \;\longrightarrow\; \frac{6\ln 2\,\ln 10}{\pi^2} \;=\; 0.9702\ldots$$

One decimal digit buys (almost surely, asymptotically) just under one
continued-fraction term — the two representations have nearly equal entropy
rates, decimal's being $\ln 10$ per digit and the CF's being the entropy
$\pi^2/(6\ln 2)$ of the Gauss map per term. The constant is not $1$: the
representations are *incommensurable*, and the conversion cost is precisely
the ratio of the two entropies. For rationals (finite words) the statement
degrades gracefully: the prefix determined is governed by the same rate until
the expansion terminates.

Worst cases are far from the average: near a high branch of small height
(e.g. $x \approx 1/2$), many digits can be spent confirming a single huge CF
term $a_{n}$ — digits determine tree *depth* at rate $\log_2 10$, but CF
*terms* only at the Lochs rate, and a single run can hoard the depth budget.

## 5. Geometric representation: the Farey tessellation

The hyperbolic plane $\mathbb{H}$ carries the Farey tessellation: ideal
triangles on the vertices $\mathbb{Q} \cup \{\infty\}$, with $p/q$ and
$p'/q'$ joined by a geodesic iff $|pq'-p'q| = 1$ — the Farey graph of §4 made
of geodesics. Attach to each $p/q$ its **Ford circle**, the horocycle of
Euclidean radius $1/(2q^2)$ tangent at $p/q$: *the denominator is literally
the geometric scale at which the rational is visible.*

A real $x$ becomes the vertical geodesic $\gamma_x$ ending at $x$; the
sequence of tessellation edges crossed by $\gamma_x$ (its *cutting sequence*)
is an $L/R$ word whose run-lengths are again the continued-fraction terms
(Series 1985). The three representations —

* digit string (analytic),
* Stern–Brocot / Farey-graph path (combinatorial/graph),
* cutting sequence of a geodesic on the modular surface (geometric)

— are three coordinatisations of the same algebraic object, and the
information exchange rates between them are governed by one mechanism (the
statements below are classical; the point is the unified reading):

* Knowing $x$ to precision $\varepsilon$ perturbs the endpoint of $\gamma_x$
  by $\varepsilon$; the cutting sequence is unchanged until the geodesic
  enters Ford circles of radius $\lesssim \varepsilon$, i.e. until convergent
  denominators reach $q_n^2 \gtrsim 1/\varepsilon$. With
  $\varepsilon = 10^{-k}$ this recovers Theorem 1's threshold $q^2 < 10^k$,
  now as a statement about horoball depth.
* Denominator growth is exponential along the geodesic
  ($q_n^{1/n} \to e^{\pi^2/(12\ln 2)}$ a.e., Lévy), which converts the
  geometric statement into the Lochs rate of §4: depth $\sim$ time $\sim$
  digits, terms $\sim$ time $\times$ entropy ratio.

This is the sense in which "translation into graph-based or geometric
representations" preserves algebraic information: *everything above scale
$\varepsilon$ survives every one of these translations; nothing below it
survives any of them* — a slogan, not a new theorem, where "survives" means
"is determined by the $\varepsilon$-neighbourhood" exactly as quantified by
Theorem 1 (worst case) and Theorem 3 (almost-everywhere rate). The
representation changes which operations are cheap (comparison for digits,
mediants for the tree, $SL_2(\mathbb{Z})$-action for geodesics), not what is
knowable.

## 6. Noise, lattices, and pooling

Real observations are corrupted beyond rounding. With accuracy
$|x_{\text{obs}} - p/q| \le \delta$, consider the planar lattice generated by
$(1, x_{\text{obs}}/\delta)$ and $(0, 1/\delta)$. It contains
$v = (q, (q\,x_{\text{obs}} - p)/\delta)$ with $\|v\| \le \sqrt{2}\,Q$, while
the Gaussian heuristic puts the shortest *irrelevant* vector near
$\sqrt{\det}/\text{const} \sim \delta^{-1/2}$. Heuristically, then, lattice
reduction finds the truth when

$$Q^2\,\delta \;\lesssim\; 1,$$

which at $\delta = 10^{-k}/2$ is Theorem 1 again — but the lattice
formulation *degrades continuously* in $\delta$ where interval bisection is
brittle, and it generalises:

**Pooling (heuristic + verified).** For $n$ decimals $x_1, \dots, x_n$
sharing one denominator (components of a probability vector, entries of a
rational matrix row), the $(n+1)$-dimensional analogue has determinant
$\delta^{-n}$ and the same Gaussian-heuristic argument predicts success when

$$Q^{n+1}\,\delta^{n} \;\lesssim\; 1.$$

Converting that into digits needs care, because the natural reading
overstates it. With $\delta = 10^{-k}$ the condition reads
$(n+1)\log_{10} Q \le nk$, while a *single* observation of $K$ digits
requires $2\log_{10} Q \le K$. Equating the two gives the effective
precision of pooling:

$$K_{\text{eff}} \;=\; \frac{2n}{n+1}\,k \;\xrightarrow[n \to \infty]{}\; 2k,$$

**not** $nk$. Pooling is subject to diminishing returns: it buys at most a
doubling of effective precision no matter how many components share the
denominator, because each new component adds one lattice dimension as well as
$k$ digits of evidence. For four 3-digit observations that is
$K_{\text{eff}} = 4.8$ digits — comfortably enough to pin $q = 61$ (which
needs $2\log_{10} 61 \approx 3.6$), and a good illustration that a modest
constant-factor gain can still flip a problem from ambiguous to identifiable.
These success conditions are heuristics, not
theorems — the Gaussian heuristic can fail for structured lattices — but the
implementation checks every extracted candidate against the exact consistency
bound $|q x_i - p_i| \le q\delta_i$ and, when no extra noise is declared,
additionally verifies that the observed digit strings are regenerated
exactly, so a heuristic failure costs recall, never soundness; randomized
tests in the suite probe the claimed regime. Recall itself is
then repaired exactly: when the lattice finds nothing at moderate bounds,
`reconstruct_shared` falls back to an exact $O(Qn)$ scan over the observation
cells, so "no common denominator exists with $q \le Q$" is only ever asserted
as a *theorem about the enumeration*, never as a lattice guess (an LLL miss
at $Q = 100$ that the scan repairs: rounded
$0.750, 0.293 \mapsto (3/4,\, 27/92)$).

The API grades its claims, and the levels are not interchangeable — each
needs strictly more enumeration than the last:

1. **Verified** (LLL fast path): the returned $q$ is consistent, but a short
   lattice vector need not be the shortest, so $q$ may be non-minimal.
2. **Smallest** (`minimize=True`): scan $q = 1, 2, \dots$ and stop at the
   first feasible denominator. Testing the *nearest* numerator suffices here,
   because the nearest integer minimises $|qx_i - p_i|$: if any numerator
   works at $q$, the nearest one does.
3. **Unique** (`identify_shared`): this is where the nearest-numerator
   shortcut becomes unsound. A cell of width $w$ admits $\lfloor wq \rfloor$
   or so numerators, and as soon as $wq \ge 1$ two distinct hypotheses share
   one denominator — e.g. the cell $[0.95, 1.05)$ of the observation "1.0"
   contains both $1 = 20/20$ and $19/20$. Sampling one numerator per
   denominator therefore *undercounts the fibre and can certify uniqueness
   falsely*; the implementation takes the full Cartesian product of the
   admissible numerator ranges and counts distinct reduced tuples, so a
   consistent $q$ and its multiples collapse to one hypothesis while two
   numerators at one $q$ count twice. With a single observation it agrees
   with `identify` by construction (randomised test in
   the test suite). Four 3-digit decimals identify a $q = 61$
that no single 3-digit decimal can (each observation's top-ranked individual
candidate differs from the generating fraction — see `demo.py`, section 4). Implementation: exact-arithmetic LLL in
`ratrecon.py`.

## 7. Constrained denominators and learned priors

### 7.1 Restricted denominator sets

Real sources rarely draw q uniformly: prices move in ticks ($q \in \{2^a\}$
or $\{100, 10000\}$), probabilities are quoted in basis points, musical and
calendrical ratios favour smooth q. Restricting to an admissible set
$S$ (with divisor-closure: $q' \mid q \in S$ is admissible, since $p'/q'$ is
representable over q) changes the information budget twice over:

* **Entropy:** identifying a member of $\{p/q : q \in S\}$ costs about
  $\log_2\sum_{q\in S}\varphi(q)$ bits instead of $2\log_2 Q - 1.72$ — for
  sparse $S$, dramatically fewer digits suffice.
* **Geometry:** the separation improves, and the right invariant is the
  *common grid*, not the product of denominators. Restricting to
  $q, q' \le Q$ within $S$ and reusing $1/(qq')$ would give
  $\max\{qq'\} = Q^2$ again — no sharpening at all. But every admissible
  fraction with denominator dividing some $m \in S$ lies on the grid
  $\tfrac{1}{L}\mathbb{Z}$ with $L = \mathop{\mathrm{lcm}}(S)$, so distinct
  admissible fractions satisfy

  $$\left|\frac{p}{q} - \frac{p'}{q'}\right| \;=\; \frac{|pq' - p'q|}{qq'} \;\ge\; \frac{\gcd(q,q')}{qq'} \;=\; \frac{1}{\mathop{\mathrm{lcm}}(q,q')},$$

  with equality attained (Bézout makes $|pq' - p'q|$ range over the nonzero
  multiples of $\gcd(q,q')$). So the family's minimum spacing is *exactly*
  $1/M$ with

  $$M \;=\; \max_{q,q' \in S} \mathop{\mathrm{lcm}}(q,q'),$$

  and $k$ digits suffice as soon as $10^k \ge M$ — **linear** in the grid
  where the unconstrained bound is quadratic in $Q$. Two refinements matter
  here. First, $M$ is the worst *pair*, not $\mathop{\mathrm{lcm}}(S)$: for
  $S = \{6, 10, 14\}$ the total lcm is $210$, but no two members reach it
  (the worst pair is $(10,14)$ with lcm $70$), so two digits suffice where
  the cruder bound would demand three. Second, this is *sufficient* and based
  on the exact spacing, but not proven minimal — identifiability can survive
  a coarser grid when the extremal pair never shares a cell, just as
  individual fractions beat the unconstrained bound. Basis points
  ($S = \{10^4\}$) need $k = 4$ rather than the $8$ that $Q = 10^4$ would
  demand unrestricted; dyadic ticks to $1/64$ need $k = 2$. Implemented as
  `minimum_spacing_denominator` / `sufficient_digits_for_denominators`.
  Enumeration changes shape too. Admissibility is divisor-closed, so the
  search runs over the *divisors* of the family members that are at most
  $Q$ — found in $O\!\left(\sum_{m \in S}\min(\sqrt{m}, Q)\right)$ — and
  then over the cell's numerators for each such denominator, replacing the
  Farey walk. Scanning the generators themselves would be exponentially
  wasteful: $S = \{10^9\}$ with $Q = 10$ admits just
  $\{1,2,4,5,8,10\}$.

A sharpening of Theorem 1 falls out of the same argument even without
constraints. Ambiguity at exactly the threshold width $10^{-k} = Q^{-2}$
would need two admissible fractions at the extremal gap $1/(qq') = Q^{-2}$,
which forces $q = q' = Q$; but two *distinct* fractions with the same
denominator $Q$ differ by at least $1/Q \gg 1/Q^2$. So the true minimum gap
is $\ge 1/(Q(Q-1))$, strictly bigger than $1/Q^2$, and

$$10^k \;\ge\; Q^2 \quad\text{already suffices.}$$

`ratrecon.sufficient_digits` implements this sharp form — `sufficient_digits(100) = 4`,
not 5 — and the claim is verified exhaustively in the test suite: at $k = 4$,
$Q = 100$, every one of the 3043 proper reduced fractions is recovered
exactly, in both rounding and truncation modes. (The bound remains
*sufficient, not necessary*: individual fractions are often identifiable
with fewer digits — $0.333 \mapsto 1/3$ at $k = 3$ — but not all of them.)

### 7.2 Learned priors: propose, then verify

Below the threshold the ranking problem is pure prior selection (section 3),
and a prior can be *learned*: train a sequence model on the source
distribution and use its denominator posterior as proposals. The architecture
that respects the mathematics is **proposer–verifier**: the network proposes
a shortlist of denominator grids (a proposed value admits the reduced
denominators dividing it); exact arithmetic keeps only consistent fractions
over those grids, and ranks them. The network can
raise recall but can never overrule hard evidence — a learned guess is never
presented as a certificate.

Empirically (`results/`, reproducible via `hybrid_gru.py`): results are
mean ± sample sd over three runs on one fixed fraction split; each run varies
model initialisation and the training and evaluation corruptions, and every
method is evaluated on the same corrupted rows within a run. With 12-digit
inputs and 10% per-digit corruption, hybrid exact-fraction top-1 is
0.703 ± 0.007 against 0.685 ± 0.008 for the classical baseline; at
denominator level the hybrid reaches 0.725 vs 0.694, and the GRU alone
0.490 ± 0.002. The aggregate gain suggests the learned and classical
proposals make some complementary errors; per-example overlap,
corruption-position effects and other noise levels are not analysed in this
pilot (one bound, one split, one architecture, three runs). On clean inputs the same hybrid nearly
matches the classical optimum (0.997 vs 1.000), i.e. the learned proposals
cost almost nothing when they are not needed. The scope of the idea is
sharply bounded, and section 8 says by what: learned proposals help against
**noise**, where the information is present but scrambled, not against
**missing information** — below the threshold the fibre is genuinely
many-to-one, and no proposer, learned or otherwise, can beat its limits.

## 8. What is inevitably lost

Four distinct mechanisms, in decreasing order of severity:

1. **Cardinality.** Below threshold, $\rho_k$ is $\sim \frac{3}{\pi^2}Q^2 10^{-k}$-to-one;
   no algorithm recovers what the fibre does not determine. Only priors
   (§3) or extra structure (§6) help.
2. **Discontinuity.** The denominator map $p/q \mapsto q$ is unbounded on
   every interval, so no *continuous* function of the observed value can
   compute it globally, and an exact reconstructor over unbounded height is
   discontinuous everywhere. Bounded to $q \le Q$ with finite digit strings
   the map becomes a finite classification problem a network can in
   principle represent — the operative difficulty is generalisation to
   unseen digit sequences, not topological impossibility (the framing for
   the experiment in `hybrid_gru.py`).
3. **Certification gaps.** Determining $p/q$ needs $O(\log q)$ digits, but
   even *witnessing* its period twice — the evidence bar `periodic_candidates`
   demands before proposing — needs $s + 2t \le k$ with
   $t = \mathrm{ord}_{\tilde q}(10)$ as large as $q - 1$ (full-reptend
   primes 7, 17, 19, 23, 29, 47, 59, 61, 97, …): exponentially more digits
   than the value itself requires. And no finite window certifies that the
   unseen continuation stays periodic at all, which is why proposals are
   verified against the window and never promoted to certificates.
4. **Representation friction.** Even above threshold, converting between
   coordinatisations is not free: the Lochs constant $0.9702\ldots < 1$ is a
   fixed conversion loss between decimal and continued-fraction
   information, with a.e.-vs-worst-case gaps on top (§4).

## References

* A. Ya. Khinchin, *Continued Fractions*, Univ. of Chicago Press, 1964.
* G. H. Hardy, E. M. Wright, *An Introduction to the Theory of Numbers*, ch. III, X–XI.
* I. Richards, "Continued fractions without tears", *Math. Magazine* 54 (1981) 163–171.
* G. Lochs, "Vergleich der Genauigkeit von Dezimalbruch und Kettenbruch", *Abh. Math. Sem. Hamburg* 27 (1964) 142–144.
* C. Series, "The modular surface and continued fractions", *J. London Math. Soc.* 31 (1985) 69–80.
* A. K. Lenstra, H. W. Lenstra, L. Lovász, "Factoring polynomials with rational coefficients", *Math. Ann.* 261 (1982) 515–534.
* P. S. Wang, "A p-adic algorithm for univariate partial fractions", *SYMSAC* 1981 (modular rational reconstruction).
* Graham, Knuth, Patashnik, *Concrete Mathematics*, §4.5 (Stern–Brocot).
