"""Proposer-verifier experiment: a GRU proposes denominators, exact
arithmetic verifies.  Needs PyTorch (everything else in this repo is stdlib).

    python hybrid_gru.py --noise 0.1 --seed 0        # one run
    for s in 0 1 2; do python hybrid_gru.py --noise 0.1 --seed "$s"; done

Protocol (regenerates results/*.json):
* Population: all reduced p/q with 0 < p < q <= qmax, split 80/10/10 BY
  FRACTION -- test fractions never appear in training, so evaluation measures
  generalisation to unseen fraction-digit pairs (denominator classes remain
  represented across the population).
* ``--split-seed`` (data split) is separate from ``--seed``, which varies
  model initialisation and the training and evaluation corruptions; holding
  the split fixed keeps clean-vs-noisy comparisons paired, and the reported
  spread is run-to-run variation on that one split.
* Within each run, every method is evaluated on the SAME corrupted rows.
  The classical baseline widens its cell until candidates appear (slack
  ladder).  The hybrid restricts exact reconstruction to the GRU's top-5
  denominator-grid proposals -- a proposed value q admits the reduced
  denominators dividing q -- falling back to pure classical when no proposal
  is consistent.  A learned guess is never reported without exact
  verification.

Result this reproduces (3 seeds, split 0): under 10% per-digit corruption,
exact-fraction top-1 goes classical 0.685 +/- 0.008 -> hybrid 0.703 +/- 0.007,
while on clean digits the hybrid gives up only ~0.003 to the exact optimum.
The aggregate gain suggests the learned and classical proposals make some
complementary errors; per-example error overlap, corruption-position effects
and other noise levels are not analysed here.  This is a pilot demonstration
rather than a general benchmark: one denominator bound, one fraction split,
one architecture, three runs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("this experiment needs PyTorch:  pip install torch") from exc

from ratrecon import decimal_digits, reconstruct, sufficient_digits

SLACK_LADDER = [0.0] + [2.0 * 10 ** i for i in range(12)]


class DenominatorGRU(nn.Module):
    """~45k parameters: digits -> class q - 2.  The map factors through the
    residue p*10^i mod q that drives the digit stream, so this is a hard,
    structured sequence task for a small recurrent model."""

    def __init__(self, qmax: int, emb: int = 24, hidden: int = 96):
        super().__init__()
        self.embed = nn.Embedding(10, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, qmax - 1)

    def forward(self, digits):
        h, _ = self.gru(self.embed(digits))
        return self.head(h[:, -1])


def make_splits(qmax: int, k: int, seed: int):
    fracs = [
        (p, q) for q in range(2, qmax + 1) for p in range(1, q) if math.gcd(p, q) == 1
    ]
    random.Random(seed).shuffle(fracs)
    n = len(fracs)
    parts = {
        "train": fracs[: int(0.8 * n)],
        "val": fracs[int(0.8 * n) : int(0.9 * n)],
        "test": fracs[int(0.9 * n) :],
    }
    out = {}
    for name, fs in parts.items():
        rows = [[int(c) for c in decimal_digits(Fraction(p, q), k)[2:]] for p, q in fs]
        out[name] = {
            "fracs": fs,
            "x": torch.tensor(rows, dtype=torch.int64),
            "y": torch.tensor([q - 2 for _, q in fs], dtype=torch.int64),
        }
    return out


def corrupt(x, rho: float, gen):
    """Each digit independently replaced by a uniform digit with prob rho."""
    if rho <= 0:
        return x
    mask = torch.rand(x.shape, generator=gen) < rho
    return torch.where(mask, torch.randint(0, 10, x.shape, generator=gen), x)


def classical_predict(row, qmax: int) -> Fraction:
    s = "0." + "".join(str(int(d)) for d in row)
    for slack in SLACK_LADDER:
        cands = reconstruct(s, qmax=qmax, noise_ulp=slack, max_candidates=8,
                            include_lattice=False)
        if cands:
            return cands[0].fraction
    return Fraction(0)


def hybrid_predict(row, hint_qs, qmax: int) -> Fraction:
    s = "0." + "".join(str(int(d)) for d in row)
    for slack in SLACK_LADDER:
        cands = reconstruct(s, qmax=qmax, noise_ulp=slack, max_candidates=5,
                            include_lattice=False, include_periodic=False,
                            denominators=hint_qs)
        if cands:
            return cands[0].fraction
    return classical_predict(row, qmax)


def exact_eval(x, hints_per_row, true_fracs, qmax: int):
    cls_f = hyb_f = cls_q = hyb_q = 0
    for row, hints, (p, q) in zip(x.tolist(), hints_per_row, true_fracs):
        truth = Fraction(p, q)
        c, h = classical_predict(row, qmax), hybrid_predict(row, hints, qmax)
        cls_f += c == truth
        hyb_f += h == truth
        cls_q += c.denominator == q
        hyb_q += h.denominator == q
    n = len(true_fracs)
    return {"classical_q_top1": cls_q / n, "classical_fraction_top1": cls_f / n,
            "hybrid_q_top1": hyb_q / n, "hybrid_fraction_top1": hyb_f / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=100)
    ap.add_argument("--digits", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--aug", type=int, default=6)
    ap.add_argument("--eval-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="weights + corruption (NOT the split)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    if args.qmax < 8:
        ap.error("--qmax must be >= 8 (smaller empties a split)")
    if not 0.0 <= args.noise <= 1.0:
        ap.error("--noise is a probability")

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    sp = make_splits(args.qmax, args.digits, args.split_seed)
    x_tr, y_tr = sp["train"]["x"], sp["train"]["y"]
    x_va, y_va = sp["val"]["x"], sp["val"]["y"]
    x_te, y_te = sp["test"]["x"], sp["test"]["y"]
    print(f"train={len(y_tr)} val={len(y_va)} test={len(y_te)} fractions  "
          f"(threshold k* = {sufficient_digits(args.qmax)}, k = {args.digits})")

    model = DenominatorGRU(args.qmax)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        xs = corrupt(x_tr.repeat(args.aug, 1), args.noise, gen)  # fresh corruption = augmentation
        ys = y_tr.repeat(args.aug)
        perm = torch.randperm(len(ys), generator=gen)
        xs, ys = xs[perm], ys[perm]
        for i in range(0, len(ys), args.batch):
            opt.zero_grad()
            loss = loss_fn(model(xs[i : i + args.batch]), ys[i : i + args.batch])
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                va = (model(corrupt(x_va, args.noise, gen)).argmax(-1) == y_va).float().mean()
            print(f"  epoch {epoch}: train loss {loss.item():.3f}  val acc {va.item():.3f}")

    eval_gen = torch.Generator().manual_seed(args.seed + 12345)
    draws = [corrupt(x_te, args.noise, eval_gen) for _ in range(args.eval_draws)]
    model.eval()
    with torch.no_grad():
        lg = model(x_te)
        top_k = min(5, lg.shape[-1])
        metrics = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "rnn": {"clean_top1": (lg.argmax(-1) == y_te).float().mean().item(),
                    "clean_top5": (lg.topk(top_k, -1).indices == y_te[:, None]).any(-1).float().mean().item()},
            "exact_clean": exact_eval(
                x_te, (lg.topk(top_k, -1).indices + 2).tolist(), sp["test"]["fracs"], args.qmax),
        }
        noisy = {k: 0.0 for k in metrics["exact_clean"]}
        r1 = r5 = 0.0
        for xd in draws:
            lgd = model(xd)
            r1 += (lgd.argmax(-1) == y_te).float().mean().item() / len(draws)
            r5 += (lgd.topk(top_k, -1).indices == y_te[:, None]).any(-1).float().mean().item() / len(draws)
            ev = exact_eval(xd, (lgd.topk(top_k, -1).indices + 2).tolist(),
                            sp["test"]["fracs"], args.qmax)
            for k in noisy:
                noisy[k] += ev[k] / len(draws)
        metrics["rnn"]["noisy_top1"], metrics["rnn"]["noisy_top5"] = r1, r5
        metrics["exact_noisy"] = noisy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (
        f"q{args.qmax}_k{args.digits}_noise{args.noise}_seed{args.seed}_split{args.split_seed}.json"
    )
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("rnn", "exact_clean", "exact_noisy")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
