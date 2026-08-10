#!/usr/bin/env python3
"""Assemble the real chain-KKT matrix set that supersedes the proxies.

`ab_run.py` and `arm_run.py` take a `--mtx-dir` of `.mtx` files. This
builds that directory as symlinks into `data/matrices/kkt-mittelmann`,
picking one iterate per family, so the same paired protocol runs on the
real matrices behind pounce#552 instead of on the generated proxies.

Selection rules, and why each one:

* One iterate per family. Iterates within a family share sparsity after
  the first, so extra ones cost run time without adding geometry.
* `_0001` over `_0000` where both exist: `_0000` is the first KKT and
  is consistently sparser (dtoc1nd 77,854 nnz vs 217,270; marine_1600
  342,399 vs 414,399), i.e. not representative of the iterates a solve
  actually spends its time in.
* `dtoc2` is the exception and uses `_0000`. `dtoc2_0001` and `_0002`
  are singular: feral reports `inertia_zero = 4` and MA57 reports 103,
  both with `rel_res = NaN`. Two independent solvers agreeing the
  matrix is rank-deficient means the matrix, not the solver, so it
  cannot carry a timing. `dtoc2_0000` factors clean (`inertia_zero = 0`,
  `rel_res = 3.1e-16`).
* `steering_12800` had no `.mtx` in the corpus until this session --
  only an empty `.solver.log`. The original harvest path is dead:
  ripopt commit 76d3575 deleted the KKT dump while leaving the
  `kkt_dump_dir=` option parsing in place, so it accepts the flag and
  writes nothing. Regenerated with pounce instead
  (`pounce steering_12800.nl --dump kkt:1-3`) and converted by
  `scripts/harvest-pounce-kkt.py`. The conversion is checked against an
  external oracle: pounce reports `num_neg_evals_actual = 51201` and
  feral and MA57 independently agree (`inertia_neg = 51201`,
  `inertia_zero = 0`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SP = Path(__file__).resolve().parent
ROOT = SP.parents[1]
SRC = ROOT / "data" / "matrices" / "kkt-mittelmann"

# family -> iterate stem. See the module docstring for why each.
SELECTED = {
    "clnlbeam": "clnlbeam_0000",
    "dtoc1nd": "dtoc1nd_0001",
    "dtoc2": "dtoc2_0000",
    "marine_1600": "marine_1600_0001",
    "rocket_12800": "rocket_12800_0001",
    "steering_12800": "steering_12800_0001",
}
UNAVAILABLE = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to populate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing = []
    for fam, stem in sorted(SELECTED.items()):
        src = SRC / fam / f"{stem}.mtx"
        if not src.exists():
            missing.append(str(src))
            continue
        link = out / f"{stem}.mtx"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        with src.open() as f:
            f.readline()
            dims = f.readline().split()
        print(f"  {stem:<24} n={dims[0]:>7} nnz={dims[2]:>8}")

    for fam, why in UNAVAILABLE.items():
        print(f"  {fam:<24} SKIPPED -- {why}")

    if missing:
        print("\nmissing matrices:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"\n{len(SELECTED)} matrices linked into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
