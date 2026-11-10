# How much arithmetic survives a decimal expansion?

*Notes on the information content of finite decimal observations of
rationals. We describe what can be recovered from such observations and how
efficiently, and what information is lost. The notes are expository and
their theorems are classical. In particular, recovering a fraction from its
leading decimal digits with the extended Euclidean algorithm is a textbook
application (Shoup 2008, §4.6.1). The notes keep the question of what the
digits determine separate from the question of how candidates are ranked,
and they describe how the implementation in `ratrecon` realises each bound.*

---

## 1. Setup

Fix a height bound $Q$ and write
$\mathcal F_Q = \lbrace p/q \in \mathbb{Q} : 1 \le q \le Q \rbrace$
(numerator bounds enter trivially and are ignored here). The *observation map*

$$\rho_k : x \mapsto \text{decimal string of } x \text{ rounded to } k \text{ decimal places}$$

sends a rational to a finite string. We study it in two regimes.

1. In the *exact regime* we ask when $\rho_k$ is injective on
   $\mathcal F_Q$, and how quickly it can be inverted.
2. In the *lossy regime* $\rho_k$ is not injective, or the digits carry
   noise, and we ask what structure remains and how candidates should be
   ranked.

The algorithms in `ratrecon` are built on the results below, and we name the
relevant routines as we go.

Experimental results quoted below are stored in `results/`, and the
digit-corruption experiment of §7.2 can be reproduced by running
`hybrid_gru.py`.

## 2. The exact regime

### 2.1 The uniqueness threshold

**Theorem 1.** Let $Q \ge 2$. If $10^k \ge Q(Q-1)$ then $\rho_k$ is
injective on $\mathcal F_Q$: $k$ rounded digits determine $p/q$ with
$q \le Q$ uniquely. The same holds for truncated digits.

*Proof.* Let $p/q \ne p'/q'$ with $q, q' \le Q$. If $q \ne q'$ then
$qq' \le Q(Q-1)$, and

$$\left|\frac{p}{q} - \frac{p'}{q'}\right| = \frac{|pq' - p'q|}{qq'} \ge \frac{1}{qq'} \ge \frac{1}{Q(Q-1)},$$

since the integer $|pq'-p'q|$ is nonzero. If $q = q'$ the two fractions
differ by at least $1/q \ge 1/Q \ge 1/(Q(Q-1))$. Two reals producing the
same $k$-digit string lie in one half-open cell of length $10^{-k}$, so
they are less than $10^{-k} \le 1/(Q(Q-1))$ apart. ∎

The minimum gap $1/(Q(Q-1))$ is attained by $1/Q$ and $1/(Q-1)$. Zhang and
Feng (2007) use the same separation to recover rationals from numerical
approximations. The cruder condition $10^k \ge Q^2$, from $qq' \le Q^2$, asks
for one more digit only at rare $Q$ (below $10^6$, only at $Q = 32$, $31623$
and $316228$). The threshold is sharp to within about half a digit. The
$\tfrac{3}{\pi^2}Q^2 + O(Q\log Q)$ members of $\mathcal F_Q \cap (0,1]$ need
distinct strings, and only $10^k + 1$ strings describe values in $(0,1]$, so
injectivity requires $10^k \gtrsim \tfrac{3}{\pi^2}Q^2$, a factor
$\pi^2/3 \approx 3.3$ below $Q(Q-1)$. `ratrecon.sufficient_digits` returns the
smallest $k$ with $10^k \ge Q(Q-1)$, and §7.1 recovers the same bound as a
special case of a result on restricted denominator sets.

*Relation to modular rational reconstruction.* For rounded digits the
problem implies a congruence. Let $p/q > 0$ and let $D$ be the integer
formed by its $k$ rounded digits, so that the observed value is $D/10^k$.
Rounding gives

$$qD - 10^k p = r, \qquad -q/2 < r \le q/2,$$

so $qD \equiv r \pmod{10^k}$ with $|r| \le Q/2$ and $0 < q \le Q$. This has
the form of the congruence solved by modular rational reconstruction (Wang
1981, and Wang, Guy and Davenport 1982), with modulus $10^k$, numerator bound
$Q/2$ and denominator bound $Q$. Two fractions with the same rounded digits
give pairs $(r, q)$ and $(r', q')$, with $r$ and $r'$ in the corresponding
ranges, so $q'r \equiv qr' \pmod{10^k}$ and $|q'r - qr'| < qq' \le Q^2$. If
$10^k \ge Q^2$ then $r/q = r'/q'$, and $p/q = (D - r/q)/10^k$ is determined.
Up to the boundary case this is the standard uniqueness condition of modular
reconstruction, that twice the product of the numerator and denominator bounds
be less than the modulus (Shoup 2008, Theorem 4.8). This argument gives a
slightly weaker result than Theorem 1, which needs only $10^k \ge Q(Q-1)$.

The extended Euclidean algorithm that solves the modular problem, applied to
$10^k$ and $D$, produces the continued-fraction expansion of $D/10^k$, which
is the computation behind `cf_terms` and `convergents` (§2.3). Shoup (2008,
§4.6.1) gives exactly this application for truncated digits. He recovers
$s/t$ with $0 \le s < t \le M$ from its first $k$ digits whenever
$10^k > 2M^2$, and notes that when $10^k$ is much smaller than $M^2$ the
digits do not in general determine $s/t$, since there are $\Omega(M^2)$ such
fractions.

The correspondence has limits. Wang's algorithm and the 1982 note look for a
fraction $u/v$ with $uv^{-1} \equiv c \pmod m$, which needs $v$ invertible
modulo $m$, whereas decimal denominators are often divisible by 2 or 5. For
example $1/2$ gives $D = 5000$ at $k = 4$, with $(r, q) = (0, 2)$, and 2 is
not invertible modulo $10^4$. The congruence form used above, as in Shoup
(2008), needs no such condition. The implication also runs one way only. The
modular bound $|r| \le Q/2$ is looser than $-q/2 < r \le q/2$, so a modular
solution need not fit the digits. For $D = 3$, $k = 2$ and $Q = 10$ the only
modular solution with $|r| \le 5$ and $0 < q \le 10$ is $(r, q) = (3, 1)$,
which gives $p = 0$, and no fraction with $q \le 10$ rounds to $0.03$. A
modular reconstruction must therefore still be checked against the observation
cell, which the congruence does not record. Its output is the rounding error
$r/q$, measured in units of the last digit, from which $p/q$ follows as above.
Truncated digits give $qD - 10^k p = -r$ with $0 \le r < q$ instead.

### 2.2 The encoding is near-optimal

The set $\mathcal F_Q \cap (0,1]$ has
$\sum_{q \le Q} \varphi(q) = \tfrac{3}{\pi^2}Q^2 + O(Q\log Q)$ members, so
identifying one requires
$\log_2(\tfrac{3}{\pi^2}Q^2) = 2\log_2 Q - 1.72 + o(1)$ bits, while $k$ digits
carry $k\log_2 10$ bits. Theorem 1 needs
$k\log_2 10 \ge \log_2\big(Q(Q-1)\big) = 2\log_2 Q - o(1)$, so the continuous
separation requirement sits $\log_2(\pi^2/3) \approx 1.72$ bits above the set
entropy (the half-digit gap noted in §2.1).

There is a second overhead on top of those $1.72$ bits. Since $k$ must be a
whole number of digits, rounding it up adds less than a further
$\log_2 10 \approx 3.32$ bits, for a worst-case total overhead of about five
bits. At $Q = 100$, for example, the entropy is
$\approx 11.6$ bits, the continuous requirement is
$\log_2 9900 \approx 13.27$ and $k = 4$ digits carry $13.29$, so rounding
$k$ up costs almost nothing here.
Up to these overheads, decimal digits are a near-optimal encoding of $(p,q)$.
They hide its arithmetic structure, however, and the remaining sections study
the cost of recovering it.

### 2.3 Inversion in logarithmic time

A scan over the denominators $q \le Q$ costs $\Theta(Q)$ arithmetic steps,
whereas inversion by continued fractions takes $O(k)$ arithmetic steps (on
integers of $O(k)$ digits, so $O(k^2)$ bit operations with schoolbook
arithmetic).

**Theorem 2 (Legendre).** If $|x - p/q| < 1/(2q^2)$ then $p/q$ is a
convergent of the continued fraction of $x$.

The true $p/q$ lies within $10^{-k}/2$ of the centre of the observation
cell, so when $q^2 < 10^k$ it satisfies Legendre's inequality with $x$ equal
to that centre. By Theorem 2, $p/q$ is then one of the $O(\log Q)$
convergents of the centre with denominator at most $Q$. For rounded digits
the centre is the value of the digit string itself. It is therefore enough to
compute `cf_terms` followed by `convergents`, and `ratrecon` marks each
candidate that passes the Legendre test (`Candidate.legendre`). The criterion
certifies only membership in the convergent sequence. Several convergents can
pass at once (both $22/7$ and $355/113$ do against $\pi$), so uniqueness of a
reconstruction comes from Theorem 1 above the threshold and otherwise from
exhaustive bounded enumeration (`identify`).

The interval refinement of this idea is the *Stern–Brocot descent*
(`simplest_in_interval`). The observed digits pin $x$ into an interval $I$,
and the smallest-denominator fraction in $I$ is found by descending the
Stern–Brocot tree, one continued-fraction term per step (Richards 1981). We
may speak of "the" simplest fraction because two fractions $p/q < p'/q$ with
the same denominator $q \ge 2$ are never adjacent in a Farey sequence
(adjacency would force $q(p'-p)=1$), so a fraction of smaller denominator lies
between them. In an interval containing at most one integer, such as a
noise-free observation cell, the minimal denominator is therefore attained
exactly once.

## 3. Below the threshold: the ambiguity set

Below the threshold the fibre $\rho_k^{-1}(\text{digits}) \cap \mathcal F_Q$
is the set of Farey fractions in an interval $I$ of length $10^{-k}$, and its
expected size is

$$\mathbb{E} \lvert \mathcal F_Q \cap I \rvert \approx \frac{3}{\pi^2} Q^2 \cdot 10^{-k}.$$

The neighbour walk in `farey_in_interval` starts from the simplest fraction in
$I$ and enumerates this set at $O(1)$ cost per element. We treat consistency
with the interval as a hard constraint, with zero likelihood outside it, and
rank the surviving candidates by the score

$$\ln \text{score}(p/q) = -s\ln q - \frac{1}{2}\left(\frac{|x_{\text{obs}} - p/q|}{\sigma \cdot 10^{-k}}\right)^{2},$$

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

Section 5 of `demo.py` follows $219/352$ with $Q = 400$ through
$k = 3,\dots,6$ digits. Its fibre shrinks from 49 candidates to 1 at $k = 5$,
one digit before the threshold $k = 6$ of Theorem 1.

## 4. Graph representation: the Stern–Brocot tree

The Stern–Brocot tree gives a bijection between positive rationals and finite
words in $\lbrace L, R \rbrace^*$, and the run-lengths of the word are the
continued-fraction terms $[a_0; a_1, a_2, \dots]$ (for a rational the last run
is $a_n - 1$). A rational is thus a vertex of an infinite binary tree, and
every edge of the tree is an edge of the Farey graph (fractions joined when
$|pq'-p'q|=1$).

The digit interval $I$ selects the set of tree nodes lying in $I$. The
longest $L/R$ prefix shared by every node in $I$ is the part of the word that
$k$ digits determine. For any interval $I$ of positive reals, whether closed,
open or half-open, the deepest common ancestor of that set is the shallowest
node $f$ in $I$, which is the simplest fraction in $I$. The subtree below $f$
consists of the rationals strictly between the two fractions whose mediant is
$f$. Each of those two is a shallower node or one of the bounds $0/1$ and
$1/0$, so neither lies in $I$, and since $I$ is an interval containing $f$,
every node in $I$ descends from $f$. A cell that contains 0, from the digits
$0.0\ldots0$, is not an interval of positive reals, and its simplest fraction
0 is not a node. Excluded endpoints matter only for the descent itself. Run
on the closure of a half-open cell, the descent can stop at the excluded
endpoint (for the truncation cell $[0.4, 0.5)$ of `0.4` it would return
$1/2$), so `simplest_in_interval` skips such an endpoint and returns the
shallowest node inside the cell ($2/5$ in this example).

In passing from the digits to the tree we therefore keep an initial segment of
the continued-fraction terms $(a_0, a_1, \dots)$ and lose the rest. The length
of that segment is described by a theorem of Lochs.

**Theorem 3 (Lochs, 1964).** For almost every $x$, the number $m(k)$ of
continued-fraction terms determined by the first $k$ decimal digits satisfies

$$\frac{m(k)}{k} \longrightarrow \frac{6 \ln 2 \cdot \ln 10}{\pi^2} = 0.9702\ldots$$

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
triangles on the vertices $\mathbb{Q} \cup \lbrace \infty \rbrace$, with $p/q$
and $p'/q'$ joined by a geodesic iff $|pq'-p'q| = 1$. This is the Farey graph
of §4, drawn with geodesic edges. To each $p/q$ we attach its *Ford circle*,
the horocycle of Euclidean radius $1/(2q^2)$ tangent to $\mathbb{R}$ at $p/q$,
so the denominator sets the geometric scale at which the rational is visible.

A real $x$ corresponds to the vertical geodesic $\gamma_x$ ending at $x$. The
sequence of tessellation edges crossed by $\gamma_x$ (its *cutting sequence*)
is an $L/R$ word whose run-lengths are again the continued-fraction terms
(Series 1985). A real $x \in (0,1)$ therefore has three representations:

* its digit string,
* its Stern–Brocot or Farey-graph path,
* the cutting sequence of a geodesic on the modular surface.

Each of them determines $x$. Two classical facts give a heuristic account of
the rates at which information passes between them (the precise statements are
Theorems 1 and 3):

* An error of $\varepsilon$ in $x$ moves the endpoint of $\gamma_x$ by
  $\varepsilon$. The cutting sequence is unchanged until the geodesic
  enters Ford circles of radius $\lesssim \varepsilon$, that is, until
  convergent denominators reach $q_n^2 \gtrsim 1/\varepsilon$. With
  $\varepsilon = 10^{-k}$ this recovers the scale $q^2 \approx 10^k$ of
  Theorem 1, now as a statement about horoball depth.
* Denominator growth is exponential along the geodesic
  ($q_n^{1/n} \to e^{\pi^2/(12\ln 2)}$ a.e., Lévy 1936). The Ford circle
  at $p_n/q_n$ lies at hyperbolic depth about
  $2\ln q_n \approx n\pi^2/(6\ln 2)$, and $k$ digits resolve depth about
  $k\ln 10$. Equating the two gives the Lochs rate
  $n/k \to 6 \ln 2 \cdot \ln 10/\pi^2$ of §4.

Passing from a finite observation to the tree or the tessellation keeps the
data determined by the $\varepsilon$-neighbourhood of $x$ and nothing finer, as
quantified by Theorem 1 (worst case) and Theorem 3 (almost-everywhere rate).
The three representations differ in which operations are efficient
(comparison for digits, mediants for the tree, the
$\mathrm{SL}_2(\mathbb{Z})$-action for geodesics).

## 6. Noise, lattices and pooling

Real observations are corrupted beyond rounding. With accuracy
$|x_{\text{obs}} - p/q| \le \delta$, consider the planar lattice generated by
$(1, x_{\text{obs}}/\delta)$ and $(0, 1/\delta)$. It contains
$v = (q, (q x_{\text{obs}} - p)/\delta)$ with $\lVert v \rVert \le Q\sqrt{2}$,
while the Gaussian heuristic puts the shortest irrelevant vector near
$\sqrt{\det}/\text{const} \sim \delta^{-1/2}$. Heuristically, then, lattice
reduction recovers $p/q$ when

$$Q^2 \delta \lesssim 1,$$

which at $\delta = 10^{-k}/2$ matches Theorem 1 up to a constant factor.
The interval methods of §2.3 and §3 fail once the error puts the true value
outside the observation cell. The lattice formulation degrades continuously
in $\delta$ and generalises to several observations with a common
denominator. That generalisation is simultaneous Diophantine approximation,
and Lagarias (1985) analysed its complexity and its lattice-reduction
algorithms.

*Pooling (heuristic, with verification).* For $n$ decimals $x_1, \dots, x_n$
sharing one denominator (components of a probability vector, entries of a
rational matrix row), the $(n+1)$-dimensional analogue has determinant
$\delta^{-n}$ and the same Gaussian-heuristic argument predicts success when

$$Q^{n+1} \delta^{n} \lesssim 1.$$

Equivalently $\delta^{-1} \gtrsim Q^{1+1/n}$, against $Q^2$ for a single
observation. The modular analogue, vector rational number reconstruction,
shows a similar saving. Bright and Storjohann (2011) prove that a modulus of
order $N^{1+1/c}$, for a small constant $c$, already confines every solution
of size at most $N$ to a lattice of dimension at most $c$, which they compute
efficiently, whereas reconstruction one entry at a time needs a modulus of
order $N^2$.

Ignoring constant factors and taking $\delta = 10^{-k}$, the condition reads
$(n+1)\log_{10} Q \lesssim nk$, while a single observation of $K$ digits
requires $2\log_{10} Q \lesssim K$. Equating
the two gives the effective precision of pooling:

$$K_{\text{eff}} = \frac{2n}{n+1} k \xrightarrow[n \to \infty]{} 2k,$$

well short of the $nk$ digits observed in total when $n$ is large. Pooling has
diminishing returns because each new component adds one lattice dimension as
well as $k$ digits of evidence. For four 3-digit observations this gives
$K_{\text{eff}} = 4.8$ digits, just more than the $2\log_{10} 200 \approx 4.6$
that a single observation needs when $Q = 200$. Section 4 of `demo.py` works
through this example. With $q \le 200$ no single 3-digit decimal identifies a
fraction with denominator 61, and ranked separately, each of the four decimals
puts some other fraction first. Together they determine $q = 61$.

Success under these conditions is not guaranteed, since the Gaussian heuristic
can fail for structured lattices. The implementation in `ratrecon.py` runs LLL
(Lenstra, Lenstra and Lovász 1982) in exact arithmetic and checks every
extracted candidate against the consistency bound
$|q x_i - p_i| \le q\delta_i$. When no extra noise is declared, it also checks
that the candidate regenerates the observed digit strings exactly. A failure
of the heuristic can therefore only lower recall. The suite tests the pooled
path on the examples in this section. When the lattice finds nothing and $Q$
is at most `exhaustive_limit` (20000 by default), `reconstruct_shared` falls
back to an exhaustive $O(Qn)$ scan over the observation cells, and the claim
that no common denominator $q \le Q$ exists always rests on that scan. For
example, with rounded digits at $Q = 100$, the LLL step misses the
reconstruction $0.750, 0.293 \mapsto (3/4, 27/92)$, and the scan recovers
it.

`reconstruct_shared` and `identify_shared` give three levels of guarantee,
and each needs at least as much enumeration as the last.

1. On the LLL fast path the returned $q$ is verified to be consistent, but a
   short lattice vector need not have the smallest first coordinate, so $q$
   may be non-minimal.
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
   numerator ranges and counts distinct reduced tuples. A tuple represented
   over $q$ and over its multiples then counts once, while two numerators at
   one $q$ count twice. With a single observation `identify_shared` agrees
   with `identify` by construction.

## 7. Constrained denominators and learned proposals

### 7.1 Restricted denominator sets

Real sources rarely draw $q$ uniformly. Prices move in ticks
($q \in \lbrace 2^a \rbrace$ or $\lbrace 100, 10000 \rbrace$), probabilities
are quoted in basis points, and the frequency ratios of just intonation
($3/2$, $5/4$, $16/15$) have smooth $q$. Restricting to an admissible set $S$
of denominators lowers the entropy and improves the separation. Admissibility
is closed under divisors: $q' \mid q \in S$ is admissible, since $p'/q'$ is
representable over $q$.

* Identifying an admissible fraction in a unit interval requires about
  $\log_2 \sum_{d} \varphi(d)$ bits, where $d$ runs over the divisor closure
  of $S$ (the divisors of members of $S$, with $d \le Q$ when a height bound
  is also imposed). For $S = \lbrace m \rbrace$ the sum is $m$, the number of
  grid points $j/m$ in $[0, 1)$. This compares with $2\log_2 Q - 1.72$ bits
  without the restriction, so for sparse $S$ far less information is needed.
* The separation is governed by the common grid of the denominators.
  Restricting to $q, q' \le Q$ within $S$ and reusing $1/(qq')$ would give
  a bound of order $Q^2$ again, which is no improvement. But every admissible
  fraction with denominator dividing some $m \in S$ lies on the grid
  $\tfrac{1}{L}\mathbb{Z}$ with $L = \mathop{\mathrm{lcm}}(S)$, so distinct
  admissible fractions are at least $1/L$ apart. For two denominators $q$
  and $q'$ the bound sharpens to

  $$\left|\frac{p}{q} - \frac{p'}{q'}\right| = \frac{|pq' - p'q|}{qq'} \ge \frac{\gcd(q,q')}{qq'} = \frac{1}{\mathop{\mathrm{lcm}}(q,q')},$$

  with equality attained, because by Bézout $|pq' - p'q|$ ranges over the
  nonzero multiples of $\gcd(q, q')$. Hence the minimum spacing of the family
  is exactly $1/M$ with

  $$M = \max_{q,q' \in S} \mathop{\mathrm{lcm}}(q,q'),$$

  and $k$ digits suffice as soon as $10^k \ge M$. When $S$ is sparse, $M$
  can be far smaller than $Q^2$.

  The maximum defining $M$ is taken over pairs, and $M$ can be smaller than
  $\mathop{\mathrm{lcm}}(S)$. For $S = \lbrace 6, 10, 14 \rbrace$ the total
  lcm is $210$, but no two members reach it (the worst pair is $(10,14)$ with
  lcm $70$), so two digits suffice where the cruder bound would demand three.
  The condition $10^k \ge M$ comes from the exact spacing. It is sufficient
  but not always necessary, since identifiability can survive a coarser grid
  when the extremal pair never shares a cell. For example, $M = 110$ for
  $S = \lbrace 1, \dots, 11 \rbrace$, but two rounded digits already suffice.
  Basis points ($S = \lbrace 10^4 \rbrace$) need $k = 4$, compared with $8$
  for unrestricted denominators up to $Q = 10^4$, and dyadic ticks to $1/64$
  need $k = 2$. These bounds are computed by `minimum_spacing_denominator` and
  `sufficient_digits_for_denominators`.

  Divisor closure also changes the enumeration. The search runs over those
  divisors of members of $S$ that are at most $Q$, which can be found in
  $O\left(\sum_{m \in S}\min(\sqrt{m}, Q)\right)$ time, and then over the
  cell's numerators for each such denominator. This replaces the Farey walk,
  and for a member $m \ge Q^2$ the divisor search takes $O(Q)$ steps,
  independent of $m$.
  For example, $S = \lbrace 10^9 \rbrace$ with $Q = 10$ admits only
  $\lbrace 1,2,4,5,8,10 \rbrace$.

Without constraints, that is for $S = \lbrace 1, \dots, Q \rbrace$ with
$Q \ge 2$, the largest pairwise lcm is
$\mathop{\mathrm{lcm}}(Q, Q-1) = Q(Q-1)$, because
$\mathop{\mathrm{lcm}}(q, q') \le qq' \le Q(Q-1)$ when $q \ne q'$ and
$\mathop{\mathrm{lcm}}(q, q) = q$. The condition $10^k \ge M$ is then
Theorem 1. `ratrecon.sufficient_digits` implements it, so
`sufficient_digits(100) = 4` and `sufficient_digits(32) = 3`, where
$10^k \ge Q^2$ would give 4. The test
suite verifies both values exhaustively. All 3043 proper reduced fractions
with $q \le 100$ are recovered exactly at $k = 4$, and all 323 with $q \le 32$
at $k = 3$, in both rounding and truncation modes, and one digit fewer leaves
some fraction ambiguous in each case. Particular fractions can need fewer
digits. At $Q = 100$, for example, $0.333$ already determines $1/3$.

### 7.2 Learned proposals under digit corruption

Above the threshold clean digits determine $p/q$ (§2.1), but a corrupted
digit usually moves the observation into a cell that holds no fraction with
$q \le Q$, and occasionally into the cell of another fraction. A sequence
model (a GRU) trained on corrupted digits can propose denominators for such
inputs, in a *proposer–verifier* scheme that we call the hybrid. The network
proposes a shortlist of grid denominators $N$, and each proposal covers every
reduced denominator dividing $N$. The fractions over those grids that are
consistent with the observation are then selected in exact arithmetic and
ranked by the score of §3.

The experiment in `hybrid_gru.py`, whose docstring gives the full protocol,
tests this scheme. The fractions are all $p/q \in (0,1)$ with $q \le 100$,
divided 80/10/10 into training, validation and test sets with no fraction in
more than one set, and each is observed through $k = 12$ rounded digits.
Theorem 1 needs only $k = 4$ here, so the experiment runs eight digits above
the threshold. Clean inputs are never ambiguous, and the task is to correct
digit errors rather than to rank an ambiguous fibre. Each digit is replaced
with probability 0.1 by a uniformly random digit, which can equal the
original, so a digit actually changes with probability 0.09.

The table gives exact-fraction top-1 accuracy as mean ± sample standard
deviation over three seeds on one fixed split. The seed sets the network
initialisation and the corruptions, and within a seed every method is scored
on the same 1525 corrupted inputs (five corruptions of each of the 305 test
fractions). The per-seed scores are stored in `results/`, together with the
counts quoted below.

| method | top-1 |
|---|---|
| classical baseline, prior $s = 2$ | 0.685 ± 0.008 |
| hybrid, $s = 2$ | 0.703 ± 0.007 |
| classical baseline, $s = 0$ | 0.691 ± 0.004 |
| hybrid, $s = 0$ | 0.709 ± 0.009 |
| hybrid given only the true denominator, $s = 2$ | 0.798 ± 0.011 |
| hybrid given only the true denominator, $s = 0$ | 0.817 ± 0.011 |
| nearest codeword, ties to the smallest denominator | 0.945 ± 0.004 |
| nearest codeword, every tie counted as a miss | 0.920 ± 0.001 |

On clean inputs the classical baseline scores 1.000 and the hybrid 0.997.
At $s = 2$ the hybrid's gain over the classical baseline is 3.4, 1.3 and 0.6
percentage points in the three seeds. At the denominator
level the hybrid reaches 0.725 ± 0.009 against 0.694 ± 0.006 for the
classical baseline, and the GRU alone 0.490 ± 0.002.

Under this corruption the probability of the observed string, given a
candidate whose 12 digits differ from it in $d$ places, is
$0.91^{12-d} \cdot 0.01^{d}$, which decreases with $d$. The test fractions are
drawn uniformly, so the maximum a posteriori decoder returns the candidate at
the smallest Hamming distance, at a cost of $3043 \times 12$ digit comparisons
per input. About 5% of inputs have more than one nearest candidate, which is
why the tie rule moves the score. This decoder recovers 94.5% of the
fractions, against 70.3% for the hybrid.

The classical baseline falls short because its noise model is wrong for this
corruption. It treats corruption as a shift in value. Its slack ladder widens
the cell step by step, up to $2 \times 10^{11}$ ulps ($\pm 0.2$), until
candidates appear, and the baseline then ranks them by the prior and by
distance to the corrupted value. A changed digit in place $j$ moves the value
by up to $9 \cdot 10^{-j}$, so the cell must widen to about that size, and the
correct digits after place $j$ then carry almost no weight. Pooled over the
three seeds there are 4575 test inputs, and 1169 of them have a changed digit
in the first three places. The classical baseline recovers 1 of these. The
nearest-codeword decoder recovers 918, or 804 with every tie counted as a
miss.

Three further points concern the hybrid.

1. With the prior that matches the uniform draw of test fractions ($s = 0$
   for both methods) the classical baseline scores 0.691 and the hybrid
   0.709, so the hybrid's gain of about 1.8 percentage points over the
   classical baseline does not come from the choice of prior.
2. The hybrid's exact check is made against the widened cell, which can
   extend to $\pm 0.2$ around the observed value. It certifies only that the
   returned fraction lies in that cell, so under digit substitution the exact
   verification available for clean digits does not carry over.
3. Given only the true denominator, the hybrid recovers 79.8% at $s = 2$ and
   81.7% at $s = 0$. It loses the rest because the widened cell either misses
   the true fraction or contains other fractions that rank above it. At
   $s = 2$ any fraction in the cell whose denominator is a proper divisor of
   $q$ ranks above $p/q$. The proposal is exact here, so every miss comes
   from the verifier.

In this pilot the noise model matters more than the proposer. At $q \le 100$
exhaustive nearest-codeword decoding is cheap and optimal under a uniform
prior, and it beats the hybrid by about 24 percentage points, so for the
learned component the experiment is a negative result. Learned proposals
might help where the noise model is unknown, or where $Q$ is large enough
that exhaustive decoding over about $\frac{3}{\pi^2}Q^2$ candidates is
infeasible. In the second case the exact posterior, computed at a smaller $Q$
where enumeration is feasible, would give a Bayes reference to measure them
against. The experiment uses one architecture, one denominator bound and one
noise rate. Three seeds on one data split are too few to establish the
difference between the hybrid and the classical baseline statistically, but
the gap to the nearest-codeword decoder is far larger than the spread across
seeds.

## 8. Unavoidable losses

1. *Cardinality.* Below the threshold, $\rho_k$ is on average
   $\sim \frac{3}{\pi^2}Q^2 10^{-k}$-to-one, and the digits alone cannot
   determine $p/q$. Only a prior (§3) or extra structure (§6, §7.1) can
   help.
2. *Discontinuity.* The denominator map $p/q \mapsto q$ is unbounded on
   every interval, so no continuous function of the observed value can
   compute it globally, and an exact reconstructor over unbounded height is
   discontinuous everywhere. Restricted to $q \le Q$ and finite digit
   strings, computing the map becomes a finite classification problem, and
   a network can in principle represent the resulting classifier. There is
   then no topological obstruction, and the difficulty that remains is
   generalisation to unseen digit sequences, which is the setting of the
   experiment in `hybrid_gru.py` (§7.2).
3. *Certification gaps.* Determining $p/q$ needs $O(\log q)$ digits, but even
   seeing its period twice needs $s + 2t \le k$, where $s$ is the length of
   the preperiod and $t = \mathrm{ord}_{\tilde q}(10)$, with $\tilde q$ the
   part of $q$ prime to $10$. The period $t$ can be as large as $q - 1$
   (full-reptend primes 7, 17, 19, 23, 29, 47, 59, 61, 97, $\dots$),
   exponentially more digits than the value itself requires, a comparison
   Shoup (2008, §4.6.1) also makes. Before proposing a candidate,
   `periodic_candidates` checks that the period appears twice. No finite
   window can certify that the unseen continuation stays periodic, so these
   proposals are verified only against the window.
4. *Change of representation.* Without a height bound, the first $k$ digits
   fix only an initial segment of the continued-fraction expansion (§4). For
   almost every real $x$ it has about $0.97k$ terms when $k$ is large
   (Theorem 3), and near a fraction of small height it can be much shorter.
   With the bound $q \le Q$ and $k$ above the threshold nothing is lost, since
   the digits then determine $p/q$ and with it every term.

## Use of AI tools

AI tools were used for programming and for checking computations.

## References

* C. Bright, A. Storjohann, "Vector rational number reconstruction", *Proc. ISSAC 2011*, ACM (2011) 51–58.
* R. L. Graham, D. E. Knuth, O. Patashnik, *Concrete Mathematics*, 2nd ed., Addison-Wesley, 1994, §4.5.
* G. H. Hardy, E. M. Wright, *An Introduction to the Theory of Numbers*, 6th ed., Oxford Univ. Press, 2008, ch. III, X–XI.
* A. Ya. Khinchin, *Continued Fractions*, Univ. of Chicago Press, 1964.
* J. C. Lagarias, "The computational complexity of simultaneous Diophantine approximation problems", *SIAM J. Comput.* 14 (1985) 196–209.
* A. K. Lenstra, H. W. Lenstra Jr., L. Lovász, "Factoring polynomials with rational coefficients", *Math. Ann.* 261 (1982) 515–534.
* P. Lévy, "Sur le développement en fraction continue d'un nombre choisi au hasard", *Compositio Math.* 3 (1936) 286–303.
* G. Lochs, "Vergleich der Genauigkeit von Dezimalbruch und Kettenbruch", *Abh. Math. Sem. Univ. Hamburg* 27 (1964) 142–144.
* I. Richards, "Continued fractions without tears", *Math. Magazine* 54 (1981) 163–171.
* C. Series, "The modular surface and continued fractions", *J. London Math. Soc.* (2) 31 (1985) 69–80.
* V. Shoup, *A Computational Introduction to Number Theory and Algebra*, 2nd ed., Cambridge Univ. Press, 2008, §4.6.
* P. S. Wang, "A p-adic algorithm for univariate partial fractions", *Proc. SYMSAC '81*, ACM (1981) 212–217.
* P. S. Wang, M. J. T. Guy, J. H. Davenport, "P-adic reconstruction of rational numbers", *ACM SIGSAM Bulletin* 16(2) (1982) 2–3.
* J. Zhang, Y. Feng, "An improved algorithm for recovering exact value from its approximation", arXiv:math/0701672 (2007).
