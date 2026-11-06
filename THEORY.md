# How much arithmetic survives a decimal expansion?

*Notes on the information content of finite decimal observations of
rationals. We describe what can be recovered from such observations and how
efficiently, and what information is lost. The notes are expository and
their theorems are classical. They keep the question of what the digits
determine separate from the question of how candidates are ranked, and they
describe how the implementation in `ratrecon` realises each bound.*

---

## 1. Setup

Fix a height bound $Q$ and write
$\mathcal{F}_Q = \{\,p/q \in \mathbb{Q} : 1 \le q \le Q\,\}$
(numerator bounds enter trivially and are ignored here). The *observation map*

$$\rho_k : x \mapsto \text{first } k \text{ decimal digits of } x \text{ (rounded)}$$

sends a rational to a finite string. We study it in two regimes.

1. In the *exact regime* we ask when $\rho_k$ is injective on
   $\mathcal{F}_Q$, and how quickly it can be inverted.
2. In the *lossy regime* $\rho_k$ is not injective, or the digits carry
   noise, and we ask what structure remains and how candidates should be
   ranked.

The algorithms in `ratrecon` are built on the results below, and we name the
relevant routines as we go.

Experimental results quoted below are stored in `results/`. The
identifiability benchmark can also be reproduced by running `demo.py`, and
the learned-prior experiment by running `hybrid_gru.py`.

## 2. The exact regime

### 2.1 The uniqueness threshold

**Theorem 1.** *If $10^k > Q^2$ then $\rho_k$ is injective on
$\mathcal{F}_Q$: $k$ rounded digits determine $p/q$ with $q \le Q$ uniquely.*

*Proof.* Distinct $p/q \ne p'/q'$ with $q, q' \le Q$ satisfy

$$\left|\frac{p}{q} - \frac{p'}{q'}\right| = \frac{|pq' - p'q|}{qq'} \ge \frac{1}{qq'} \ge \frac{1}{Q^2},$$

since the integer $|pq'-p'q|$ is nonzero. Two reals producing the same
$k$-digit rounding lie in a common interval of length $10^{-k} < 1/Q^2$. ∎

The bound is essentially sharp: consecutive Farey fractions of order $Q$ have
gaps as small as $\sim 1/Q^2$, so with $10^k \ll Q^2$ collisions are generic.
`ratrecon.sufficient_digits` computes the threshold in integer arithmetic,
using the sharpened form of §7.1.

### 2.2 The encoding is near-optimal

$|\mathcal{F}_Q \cap (0,1]| = \sum_{q \le Q} \varphi(q) = \tfrac{3}{\pi^2}Q^2 + O(Q\log Q)$,
so identifying a member requires $\log_2(\tfrac{3}{\pi^2}Q^2) = 2\log_2 Q - 1.72 + o(1)$
bits, while $k$ digits carry $k\log_2 10$ bits. Theorem 1 needs
$k\log_2 10 \ge 2\log_2 Q$, so the continuous separation requirement sits
$1.72$ bits above the set entropy.

There is a second overhead on top of those $1.72$ bits. Since $k$ must be a
whole number of digits, rounding it up adds as much as a further
$\log_2 10 \approx 3.32$ bits of granularity, so in the worst case the total
overhead is about five bits. At $Q = 100$, for example, the entropy is
$\approx 11.6$ bits, the continuous requirement is $13.3$ and $k = 4$ digits
carry $13.3$. Rounding adds nothing here, which is the equality case of §7.1.
Up to these overheads, decimal digits are a near-optimal encoding of $(p,q)$.
They hide its arithmetic structure, however, and the remaining sections study
the cost of recovering it.

### 2.3 Inversion in logarithmic time

Brute force over $\mathcal{F}_Q$ costs $\Theta(Q^2)$, whereas inversion by
continued fractions takes $O(k)$ arithmetic steps (on integers of $O(k)$
digits, so $O(k^2)$ bit operations with schoolbook arithmetic).

**Theorem 2 (Legendre).** *If $|x - p/q| < 1/(2q^2)$ then $p/q$ is a
convergent of the continued fraction of $x$.*

Above the threshold the true $p/q$ satisfies Legendre's inequality with
respect to the observed value, so it appears among the $O(\log Q)$
convergents of the digit string. It is therefore enough to compute
`cf_terms` followed by `convergents`, and `ratrecon` labels the candidates
that pass the Legendre test *certified convergents*. The criterion certifies
only membership in the convergent sequence. Several convergents can pass at
once (both $22/7$ and $355/113$ do against $\pi$), so uniqueness of a
reconstruction is established only by exhaustive bounded enumeration
(`identify`).

The interval refinement of this idea is the *Stern–Brocot descent*
(`simplest_in_interval`). The observed digits pin $x$ into an interval $I$,
and the smallest-denominator fraction in $I$ is found by descending the
Stern–Brocot tree, one continued-fraction term per step (Richards 1981). We
may speak of "the" simplest fraction because two fractions with the same
denominator $q \ge 2$ are never adjacent in a Farey sequence (adjacency
forces $q(c-a)=1$), so the minimal denominator in an interval is attained
exactly once.

## 3. Below the threshold: the ambiguity set

When $10^k < Q^2$ the fibre $\rho_k^{-1}(\text{digits}) \cap \mathcal{F}_Q$
is the set of Farey fractions in an interval $I$ of length $10^{-k}$, and its
expected size is

$$\mathbb{E}\,\#(\mathcal{F}_Q \cap I) \approx \frac{3}{\pi^2}\,Q^2\,10^{-k}.$$

The neighbour walk in `farey_in_interval` enumerates this set at $O(1)$
amortised cost per element. We treat consistency with the interval as a hard
constraint, with zero likelihood outside it, and rank the surviving candidates
by the score

$$\log \text{score}(p/q) = -s\ln q \;-\; \frac{1}{2}\left(\frac{|x_{\text{obs}} - p/q|}{\sigma \cdot 10^{-k}}\right)^{\!2},$$

with a prior $P(p/q) \propto q^{-s}$, $x_{\text{obs}}$ the centre of the cell,
and a Gaussian term of width $\sigma = 1/2 + n$ in the residual, where $n$ is
`noise_ulp`, both measured in units in the last place (ulps).

Under pure deterministic quantisation the cell already determines the
likelihood completely. It is $1$ for every fraction inside and $0$ outside, so
no residual term follows from the rounding model. The Gaussian is therefore a
*centre-preference surrogate*, motivated by an analogue-noise model of the kind
`noise_ulp` is designed for, in which $\sigma$ grows and the cell widens. Under
such a model the exact likelihood of $p/q$ would be the noise density centred
at $p/q$, integrated over the cell. With $s = 2$ and exact digits the residual
term lies between $-1/2$ and $0$, so it can reverse the order set by the prior
only between denominators within a factor $e^{1/4} \approx 1.28$ of each other.
With `0.856` and $q \le 100$, for example, $83/97$ (residual $0.33$ ulp)
outranks the simplest consistent fraction $77/90$ (residual $0.44$ ulp).

The default $s = 2$ is a *complexity (Occam) prior*. It penalises each
candidate by its description length $2\log q$ and so favours small
denominators. It differs from the uniform distribution on the bounded Farey
set. Among reduced fractions in $(0,1]$ with $q \le Q$ there are $\varphi(q)$
with denominator exactly $q$, so sampling that set uniformly gives
$P(q) \propto \varphi(q)$, which on average increases with $q$ and favours
large denominators. Below the threshold the prior dominates the ranking, so a
benchmark there is won by whichever prior better matches the law generating the
test fractions, and its top-1 accuracy does not measure the quality of the
algorithm. For this reason `reconstruct` takes the exponent $s$ as a parameter,
and $s = 0$ ranks by the residual alone, closest to the observation first. For
naturally occurring constants (quoted probabilities, tick prices, ratios chosen
by people) small denominators are more probable, which makes $s = 2$ a sensible
default.

Section 5 of `demo.py` follows one fraction through $k = 3,\dots,6$ digits.
Its fibre shrinks from 49 candidates to 1 at $k = 5$, one digit before the
universal worst-case threshold.

## 4. Graph representation: the Stern–Brocot tree

The Stern–Brocot tree gives a bijection between positive rationals and finite
words in $\{L, R\}^*$, and the run-lengths of the word are the
continued-fraction terms $[a_0; a_1, a_2, \dots]$ (for a rational the last run
is $a_n - 1$). This gives a natural graph-based representation of a rational,
as a vertex of an infinite binary tree or, equivalently, a vertex of the Farey
graph (fractions joined when $|pq'-p'q|=1$).

The digit interval $I$ selects the set of tree nodes lying in $I$. The
longest $L/R$ prefix shared by every candidate (the common Stern–Brocot
cylinder) is the part of the word that $k$ digits determine. For a closed
interval the deepest common ancestor of that set is the simplest fraction in
$I$. When an endpoint is excluded, as in the half-open observation cells, the
ancestor can be the excluded endpoint itself, and the simplest admissible
fraction is then the shallowest tree node lying inside the interval
(`simplest_in_interval`).

In passing from the digits to the tree we therefore keep an initial segment
of the algebraic data $(a_0, a_1, \dots)$ and lose the rest. The length of that
segment is described by a theorem of Lochs.

**Theorem 3 (Lochs, 1964).** *For almost every $x$, the number $m(k)$ of
continued-fraction terms determined by the first $k$ decimal digits satisfies*

$$\frac{m(k)}{k} \;\longrightarrow\; \frac{6\ln 2\,\ln 10}{\pi^2} \;=\; 0.9702\ldots$$

Almost surely and asymptotically, then, one decimal digit determines just under
one continued-fraction term. The two representations have nearly equal entropy
rates, $\ln 10$ per digit for decimals and the entropy $\pi^2/(6\ln 2)$ of the
Gauss map per term for continued fractions. The constant in Theorem 3 is their
ratio, slightly below $1$ because a continued-fraction term carries a little
more information than a decimal digit. Rationals form a null set, so Theorem 3
says nothing about them directly. Heuristically, a rational with a large
denominator behaves like a typical real until its continued-fraction expansion
terminates.

Worst cases are far from the average. Near a fraction of small height, which
lies high in the tree (e.g. $x \approx 1/2$), many digits may be needed to
confirm a single large continued-fraction term $a_n$, because the path to $x$
is dominated by one run of length $a_n$. About $\log_{10} a_n$ digits already
fix a depth of order $a_n$, while confirming $a_n$ itself takes about
$2\log_{10} a_n$, so tree depth is not a fixed multiple of the number of
digits.

## 5. Geometric representation: the Farey tessellation

The hyperbolic plane $\mathbb{H}$ carries the Farey tessellation: ideal
triangles on the vertices $\mathbb{Q} \cup \{\infty\}$, with $p/q$ and
$p'/q'$ joined by a geodesic iff $|pq'-p'q| = 1$. This is the Farey graph of
§4, drawn with geodesic edges. To each $p/q$ we attach its *Ford circle*,
the horocycle of Euclidean radius $1/(2q^2)$ tangent at $p/q$, so the
denominator sets the geometric scale at which the rational is visible.

A real $x$ corresponds to the vertical geodesic $\gamma_x$ ending at $x$. The
sequence of tessellation edges crossed by $\gamma_x$ (its *cutting sequence*)
is an $L/R$ word whose run-lengths are again the continued-fraction terms
(Series 1985). This gives three representations of $x$:

* the digit string (analytic),
* the Stern–Brocot or Farey-graph path (combinatorial),
* the cutting sequence of a geodesic on the modular surface (geometric).

Each of them determines $x$, and the rates at which information passes between
them follow from two classical facts:

* An error of $\varepsilon$ in $x$ moves the endpoint of $\gamma_x$ by
  $\varepsilon$. The cutting sequence is unchanged until the geodesic
  enters Ford circles of radius $\lesssim \varepsilon$, that is, until
  convergent denominators reach $q_n^2 \gtrsim 1/\varepsilon$. With
  $\varepsilon = 10^{-k}$ this recovers Theorem 1's threshold $q^2 < 10^k$,
  now as a statement about horoball depth.
* Denominator growth is exponential along the geodesic
  ($q_n^{1/n} \to e^{\pi^2/(12\ln 2)}$ a.e., Lévy), which converts the
  geometric statement into the Lochs rate of §4: hyperbolic depth $\sim$
  time $\sim$ digits, terms $\sim$ time $\times$ entropy ratio.

Informally, then, translating into the graph-based or geometric representations
preserves algebraic information. Every one of these translations keeps the data
determined by the $\varepsilon$-neighbourhood of $x$, and none of them recovers
anything finer, as quantified by Theorem 1 (worst case) and Theorem 3
(almost-everywhere rate). For a given $x$ the three representations determine
one another, and they differ in which operations are efficient (comparison for
digits, mediants for the tree, the $SL_2(\mathbb{Z})$-action for geodesics).

## 6. Noise, lattices, and pooling

Real observations are corrupted beyond rounding. With accuracy
$|x_{\text{obs}} - p/q| \le \delta$, consider the planar lattice generated by
$(1, x_{\text{obs}}/\delta)$ and $(0, 1/\delta)$. It contains
$v = (q, (q\,x_{\text{obs}} - p)/\delta)$ with $\|v\| \le \sqrt{2}\,Q$, while
the Gaussian heuristic puts the shortest irrelevant vector near
$\sqrt{\det}/\text{const} \sim \delta^{-1/2}$. Heuristically, then, lattice
reduction recovers $p/q$ when

$$Q^2\,\delta \;\lesssim\; 1,$$

which at $\delta = 10^{-k}/2$ is Theorem 1 again. Interval bisection is
brittle, but the lattice formulation degrades continuously in $\delta$ and
generalises to several observations with a common denominator.

*Pooling (heuristic, with verification).* For $n$ decimals $x_1, \dots, x_n$
sharing one denominator (components of a probability vector, entries of a
rational matrix row), the $(n+1)$-dimensional analogue has determinant
$\delta^{-n}$ and the same Gaussian-heuristic argument predicts success when

$$Q^{n+1}\,\delta^{n} \;\lesssim\; 1.$$

With $\delta = 10^{-k}$ the condition reads $(n+1)\log_{10} Q \le nk$, while
a single observation of $K$ digits requires $2\log_{10} Q \le K$. Equating
the two gives the effective precision of pooling:

$$K_{\text{eff}} \;=\; \frac{2n}{n+1}\,k \;\xrightarrow[n \to \infty]{}\; 2k,$$

well short of the $nk$ digits observed in total when $n$ is large. Pooling
has diminishing returns because each new component adds one lattice
dimension as well as $k$ digits of evidence. However many components share
the denominator, pooling at most doubles the effective precision. For four
3-digit observations this gives $K_{\text{eff}} = 4.8$ digits, more than the
$2\log_{10} 61 \approx 3.6$ needed to pin down $q = 61$, which no single
3-digit decimal can identify. Section 4 of `demo.py` works through this
example. Ranked separately, each of the four decimals puts some other
fraction first, and together they determine $q = 61$.

Success under these conditions is not guaranteed, since the Gaussian
heuristic can fail for structured lattices. The implementation in
`ratrecon.py` runs LLL in exact arithmetic and checks every extracted
candidate against the consistency bound $|q x_i - p_i| \le q\delta_i$. When no
extra noise is declared, it also checks that the candidate regenerates the
observed digit strings exactly. A failure of the heuristic can therefore only
lower recall. Randomised tests in the suite probe the regime in which success
is claimed. When the lattice finds nothing at moderate bounds,
`reconstruct_shared` falls back to an exhaustive $O(Qn)$ scan over the
observation cells, and the claim that no common denominator $q \le Q$ exists
always rests on that scan. For example, with rounded digits at $Q = 100$,
the LLL step misses the reconstruction $0.750, 0.293 \mapsto (3/4,\, 27/92)$,
and the scan recovers it.

`reconstruct_shared` and `identify_shared` give three levels of guarantee,
and each needs strictly more enumeration than the last.

1. On the LLL fast path the returned $q$ is verified to be consistent, but a
   short lattice vector need not be the shortest, so $q$ may be non-minimal.
2. With `minimize=True` the returned $q$ is the smallest feasible
   denominator. We scan $q = 1, 2, \dots$ and stop at the first feasible
   one. Testing only the nearest numerator is enough here. Since the
   nearest integer minimises $|qx_i - p_i|$, if any numerator works at $q$
   then the nearest one does.
3. `identify_shared` decides whether the reconstruction is unique, and
   here the nearest-numerator shortcut is no longer sound. A cell of width
   $w$ admits roughly $\lfloor wq \rfloor$ numerators, and as soon as
   $wq \ge 1$ two distinct hypotheses can share one denominator. For
   example, the cell $[0.95, 1.05)$ of the observation "1.0" contains both
   $1 = 20/20$ and $19/20$. Sampling one numerator per denominator therefore
   undercounts the fibre and can falsely certify uniqueness. The
   implementation takes the full Cartesian product of the admissible
   numerator ranges and counts distinct reduced tuples. A consistent $q$
   and its multiples then collapse to one hypothesis, while two numerators
   at one $q$ count twice. With a single observation `identify_shared`
   agrees with `identify` by construction, and a randomised test in the
   suite checks this.

## 7. Constrained denominators and learned priors

### 7.1 Restricted denominator sets

Real sources rarely draw $q$ uniformly. Prices move in ticks
($q \in \{2^a\}$ or $\{100, 10000\}$), probabilities are quoted in basis
points, and musical and calendrical ratios favour smooth $q$. Restricting to
an admissible set $S$ of denominators lowers the entropy and improves the
separation. Admissibility is closed under divisors: $q' \mid q \in S$ is
admissible, since $p'/q'$ is representable over $q$.

* Identifying a member of $\{p/q : q \in S\}$ requires about
  $\log_2\sum_{q\in S}\varphi(q)$ bits, compared with $2\log_2 Q - 1.72$
  without the restriction, so for sparse $S$ far fewer digits suffice.
* The separation is governed by the common grid of the denominators.
  Restricting to $q, q' \le Q$ within $S$ and reusing $1/(qq')$ would give
  $\max\{qq'\} = Q^2$ again, which is no improvement. But every admissible
  fraction with denominator dividing some $m \in S$ lies on the grid
  $\tfrac{1}{L}\mathbb{Z}$ with $L = \mathop{\mathrm{lcm}}(S)$, so distinct
  admissible fractions satisfy

  $$\left|\frac{p}{q} - \frac{p'}{q'}\right| \;=\; \frac{|pq' - p'q|}{qq'} \;\ge\; \frac{\gcd(q,q')}{qq'} \;=\; \frac{1}{\mathop{\mathrm{lcm}}(q,q')},$$

  with equality attained (by Bézout, $|pq' - p'q|$ ranges over the nonzero
  multiples of $\gcd(q,q')$). Hence the minimum spacing of the family is
  exactly $1/M$ with

  $$M \;=\; \max_{q,q' \in S} \mathop{\mathrm{lcm}}(q,q'),$$

  and $k$ digits suffice as soon as $10^k \ge M$. This condition is linear
  in $M$, whereas the unconstrained one is quadratic in $Q$.

  The maximum defining $M$ is taken over pairs, and $M$ can be smaller than
  $\mathop{\mathrm{lcm}}(S)$. For $S = \{6, 10, 14\}$ the total lcm is
  $210$, but no two members reach it (the worst pair is $(10,14)$ with lcm
  $70$), so two digits suffice where the cruder bound would demand three.
  The condition $10^k \ge M$ comes from the exact spacing and is sufficient,
  but we have not shown that it is necessary. Identifiability can survive a
  coarser grid when the extremal pair never shares a cell. Basis points
  ($S = \{10^4\}$) need $k = 4$, compared with $8$ for unrestricted
  denominators up to $Q = 10^4$, and dyadic ticks to $1/64$ need $k = 2$.
  These bounds are computed by `minimum_spacing_denominator` and
  `sufficient_digits_for_denominators`.

  Divisor-closure also changes the enumeration. The search runs over those
  divisors of members of $S$ that are at most $Q$, which can be found in
  $O\!\left(\sum_{m \in S}\min(\sqrt{m}, Q)\right)$ time, and then over the
  cell's numerators for each such denominator. This replaces the Farey walk
  and is exponentially cheaper than scanning the members of $S$ themselves.
  For example, $S = \{10^9\}$ with $Q = 10$ admits only
  $\{1,2,4,5,8,10\}$.

The same argument gives a sharper form of Theorem 1, even without
constraints. Ambiguity at exactly the threshold width $10^{-k} = Q^{-2}$
would need two admissible fractions at the extremal gap $1/(qq') = Q^{-2}$,
which forces $q = q' = Q$. But two distinct fractions with the same
denominator $Q$ differ by at least $1/Q \gg 1/Q^2$. So the minimum gap is
$\ge 1/(Q(Q-1))$, strictly greater than $1/Q^2$, and

$$10^k \;\ge\; Q^2 \quad\text{already suffices.}$$

`ratrecon.sufficient_digits` implements this sharpened form, so
`sufficient_digits(100) = 4`, whereas the strict inequality of Theorem 1
would give 5. The test suite verifies the claim exhaustively: at $k = 4$ and
$Q = 100$, all 3043 proper reduced fractions are recovered exactly, in both
rounding and truncation modes. The bound is sufficient but not necessary,
since many individual fractions are identifiable with fewer digits (for
example $0.333 \mapsto 1/3$ at $k = 3$), though not all are.

### 7.2 Learned priors with exact verification

Below the threshold the ranking is decided by the prior (§3), and a prior can
also be learned. We train a sequence model (a GRU) on the source distribution
and use its posterior over denominators to generate proposals, in a
*proposer–verifier* scheme that we call the hybrid. The network proposes a
shortlist of grid denominators $N$, and each proposal covers every reduced
denominator dividing $N$. The fractions over those grids that are consistent
with the observation are then selected and ranked in exact arithmetic, so the
network can raise recall while every fraction returned is checked against the
cell.

The numbers below come from `hybrid_gru.py`, whose docstring describes the full
protocol, and the outputs are stored in `results/`. They are means ± sample
standard deviation over three seeds on one fixed fraction split. The seed sets
the network initialisation and the corruptions used in training and evaluation,
and within a seed every method is scored on the same corrupted inputs. Each
digit of a 12-digit input is replaced with probability 0.1 by a uniformly
random digit, which can equal the original, so a digit actually changes with
probability 0.09. Under this corruption the hybrid's exact-fraction top-1
accuracy is 0.703 ± 0.007, against 0.685 ± 0.008 for the classical baseline,
and on clean inputs the same hybrid scores 0.997 against 1.000. At the
denominator level the hybrid reaches 0.725 ± 0.009 against 0.694 ± 0.006, and
the GRU alone 0.490 ± 0.002, which suggests that the learned and classical
proposals make partly complementary errors. The exact-fraction gain is positive
for each of the three seeds (3.4, 1.3 and 0.6 percentage points, mean 1.8), but
this is a small pilot with one architecture, one denominator bound and one data
split, and three seeds are too few to establish the gain statistically. We have
not analysed per-example overlap, the effect of corruption position or the
behaviour at other noise levels. Learned proposals can help only when the
corrupted digits still contain the information. Below the threshold the
observation map is many-to-one (§8), and a prior, whether learned or not, can
only reorder the fibre.

## 8. Unavoidable losses

1. *Cardinality.* Below the threshold, $\rho_k$ is
   $\sim \frac{3}{\pi^2}Q^2 10^{-k}$-to-one, and the digits alone cannot
   determine $p/q$. Only a prior (§3) or extra structure (§6) can help.
2. *Discontinuity.* The denominator map $p/q \mapsto q$ is unbounded on
   every interval, so no continuous function of the observed value can
   compute it globally, and an exact reconstructor over unbounded height is
   discontinuous everywhere. Restricted to $q \le Q$ and finite digit
   strings, computing the map becomes a finite classification problem, and
   a network can in principle represent the resulting classifier. There is
   then no topological obstruction, and the difficulty that remains is
   generalisation to unseen digit sequences, which is the setting of the
   experiment in `hybrid_gru.py` (§7.2).
3. *Certification gaps.* Determining $p/q$ needs $O(\log q)$ digits, but
   even seeing its period twice needs $s + 2t \le k$, where $s$ is the
   length of the preperiod and $t = \mathrm{ord}_{\tilde q}(10)$, with
   $\tilde q$ the part of $q$ prime to $10$. The period $t$ can be as large
   as $q - 1$ (full-reptend primes 7, 17, 19, 23, 29, 47, 59, 61, 97,
   $\dots$), exponentially more digits than the value itself requires.
   Before proposing a candidate, `periodic_candidates` checks that the
   period appears twice. No finite window can certify that the unseen
   continuation stays periodic, so these proposals are verified only against
   the window.
4. *Change of representation.* Even above the threshold, the Lochs
   constant $0.9702\ldots < 1$ means that decimal digits determine
   continued-fraction terms at a fixed rate of less than one term per digit,
   and worst cases can be far from this almost-everywhere rate (§4).

## References

* A. Ya. Khinchin, *Continued Fractions*, Univ. of Chicago Press, 1964.
* G. H. Hardy, E. M. Wright, *An Introduction to the Theory of Numbers*, ch. III, X–XI.
* I. Richards, "Continued fractions without tears", *Math. Magazine* 54 (1981) 163–171.
* G. Lochs, "Vergleich der Genauigkeit von Dezimalbruch und Kettenbruch", *Abh. Math. Sem. Hamburg* 27 (1964) 142–144.
* C. Series, "The modular surface and continued fractions", *J. London Math. Soc.* 31 (1985) 69–80.
* A. K. Lenstra, H. W. Lenstra, L. Lovász, "Factoring polynomials with rational coefficients", *Math. Ann.* 261 (1982) 515–534.
* P. S. Wang, "A p-adic algorithm for univariate partial fractions", *SYMSAC* 1981 (modular rational reconstruction).
* R. L. Graham, D. E. Knuth, O. Patashnik, *Concrete Mathematics*, §4.5 (Stern–Brocot).
