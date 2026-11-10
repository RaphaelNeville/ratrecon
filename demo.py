"""Examples of the main functions in ratrecon.

Run:  python demo.py   (stdlib only, a few seconds)
"""
import math
import random
from fractions import Fraction

from ratrecon import (
    decimal_digits,
    identify,
    identify_shared,
    lattice_refine,
    reconstruct,
    reconstruct_shared,
    sufficient_digits,
    sufficient_digits_for_denominators,
)


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


section("1. Continued fractions: 3.14159292")
for c in reconstruct("3.14159292", qmax=1000, max_candidates=3):
    print("   ", c)
print("    355/113, the classical convergent of pi, passes Legendre's test, and")
print("    identify() finds no other consistent fraction with q <= 1000.")

section("2. Periodicity: 0.9054054")
for c in reconstruct("0.9054054", qmax=100, max_candidates=3):
    print("   ", c)
print("    detected 0.9(054) = 9/10 + 54/(10*999) = 67/74, then verified")

section("3. Noise: one corrupted digit in 355/113 to 10 places")
true = Fraction(355, 113)
clean = decimal_digits(true, 10)
noisy = clean[:7] + "1" + clean[8:]  # 6th fractional digit changed from 2 to 1
print(f"    clean {clean}   noisy {noisy}")
for slack in (0.0, 1e3, 1e5):
    cands = reconstruct(noisy, qmax=1000, noise_ulp=slack, max_candidates=4)
    tops = ", ".join(str(c.fraction) for c in cands) or "(none)"
    mark = "   <- includes truth" if any(c.fraction == true for c in cands) else ""
    print(f"    interval, slack {slack:>7g} ulp: {tops}{mark}")
obs = Fraction(int(noisy.replace(".", "")), 10 ** 10)
lat = lattice_refine(obs, 10, 1000, noise_ulp=1e5)[:3]
print("    lattice_refine at the same noise:", ", ".join(map(str, lat)))

section("4. Several decimals with a common denominator")
decs = ["0.377", "0.656", "0.180", "0.852"]
print("    observations:", decs)
print("    each decimal's own top candidate differs from its generating fraction:")
for d in decs:
    print(f"      {d} -> {reconstruct(d, qmax=200, max_candidates=1)[0].fraction}")
joint = reconstruct_shared(decs, qmax=200)
fracs = ", ".join(map(str, joint.fractions))
print(f"    jointly: q = {joint.denominator}, {fracs}  [{joint.method}]")
rep = identify_shared(decs, qmax=200)
print(f"    identify_shared: identifiable={rep.identifiable} ({rep.consistent_count} tuple)")
print("    what each call guarantees, weakest first:")
print("      reconstruct_shared(...)                -> a verified q (maybe not minimal)")
print("      reconstruct_shared(..., minimize=True) -> the smallest q")
print("      identify_shared(...)                   -> uniqueness of the tuple")
weak = identify_shared(["1.0"], qmax=20)
print(f"    a non-unique case: '1.0' at qmax=20 has {weak.consistent_count} tuples")
print("      (both 1 and 19/20 round to '1.0', so uniqueness requires")
print("       enumerating every numerator at each denominator)")

section("5. The information threshold 10^k >= qmax(qmax-1)")
qmax = 400
print(f"    qmax = {qmax}: need k >= {sufficient_digits(qmax)} digits for guaranteed uniqueness")
x = Fraction(219, 352)
for k in range(3, 7):
    s = decimal_digits(x, k)
    rep = identify(s, qmax=qmax)
    top = reconstruct(s, qmax=qmax, max_candidates=1)[0].fraction
    print(
        f"      k={k}: {s:<9} -> {rep.consistent_count:>3} candidate(s), "
        f"identifiable={rep.identifiable!s:<5} top = {top}"
        + ("   <- exact" if top == x else "")
    )

section("6. Constrained denominators: tick grids")
print("    '0.6875' among dyadic ticks:", reconstruct(
    "0.6875", qmax=1000, denominators=[2, 4, 8, 16, 32, 64], max_candidates=1)[0].fraction)
print("    for a set of denominators, 10^k >= M suffices (M = largest pairwise lcm):")
print("      dyadic ticks to 1/64 :",
      sufficient_digits_for_denominators([2, 4, 8, 16, 32, 64]), "digits")
print("      basis points (1/10^4):", sufficient_digits_for_denominators([10000]), "digits",
      " (unconstrained q <= 10^4 would need 8)")
print("      {6,10,14}            :", sufficient_digits_for_denominators([6, 10, 14]),
      "digits  (M = lcm(10,14) = 70)")

section("7. Reproducing the benchmark: identifiability at low precision")
# Same protocol and seed as results/classical_benchmark_low_precision.json:
# 1000 distinct cases with k = 4 digits, alternating truncate and round.
# q is drawn uniformly from [2, 500], then p uniformly from the residues
# coprime to q.  Resampling q whenever gcd(p, q) > 1 would bias q.
rng = random.Random(11)
qm, k = 500, 4
coprimes = {q: [p for p in range(1, q) if math.gcd(p, q) == 1] for q in range(2, qm + 1)}
cases, seen = [], set()
while len(cases) < 1000:
    q = rng.choice(range(2, qm + 1))
    p = rng.choice(coprimes[q])
    mode = "truncate" if len(cases) % 2 == 0 else "round"
    if (p, q, mode) in seen:
        continue
    seen.add((p, q, mode))
    cases.append((Fraction(p, q), decimal_digits(Fraction(p, q), k, mode), mode))
ident = hits = top1 = 0
for truth, dec, mode in cases:
    rep = identify(dec, qmax=qm, mode=mode)
    top = reconstruct(dec, qmax=qm, mode=mode, max_candidates=1)[0].fraction
    top1 += top == truth
    if rep.identifiable:
        ident += 1
        hits += top == truth
print(f"    k={k}, q<={qm}, 1000 cases: identifiable {ident / 10:.1f}%  "
      f"top-1 overall {top1 / 10:.1f}%  "
      f"top-1 when identifiable {100 * hits / max(1, ident):.0f}%")
print("    With 4 digits most cases have more than one consistent fraction, so the")
print("    digits alone cannot settle them and the prior picks the top candidate.")
print("    An identifiable case has a single consistent fraction, so its top-1 is")
print("    100% by construction. The overall top-1 depends on the prior (s = 2")
print("    here) and on how the fractions were drawn (THEORY 3).")
