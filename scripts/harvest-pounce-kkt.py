#!/usr/bin/env python3
"""Convert a pounce `--dump kkt:...` tree into corpus .mtx/.json pairs.

Replaces the ripopt path in `harvest-mittelmann-kkt.sh`, which no
longer works: ripopt commit 76d3575 (2026-04-28) deleted the dump
implementation (`dump_kkt_matrix`, `write_kkt_mtx_file`, ...) while
leaving `SolverOptions::kkt_dump_dir` and its CLI wiring in place, so
`kkt_dump_dir=` is still accepted and silently writes nothing.

pounce dumps the same systems in its own JSONL schema. Generate them
with:

    pounce <problem>.nl --dump kkt:all --dump-dir <dumpdir>

which writes `<dumpdir>/iter_NNN/kkt_solve_NNN.jsonl`, one JSON object
per line with `n`, `irn`, `jcn`, `vals`, `rhs` and the inertia fields.
Then:

    scripts/harvest-pounce-kkt.py --dump-dir <dumpdir> --name steering_12800

writes `data/matrices/kkt-mittelmann/<name>/<name>_<iter:04>.{mtx,json}`
in the layout the rest of the corpus tooling already reads.

Only the first solve of each iteration is taken. Later `kkt_solve_NNN`
records in the same iteration are inertia-correction retries of the
same system with a different `delta_w`, so keeping them would weight
hard iterations more heavily in any corpus statistic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "matrices" / "kkt-mittelmann"


def write_mtx(path: Path, n: int, irn, jcn, vals) -> int:
    """Write the lower triangle in Matrix Market symmetric coordinate form.

    pounce emits Fortran-style 1-based triplets for the whole symmetric
    structure. Mirroring the entries into the lower triangle here means
    a duplicated (i, j) would be written twice, so entries are summed
    into a dict first -- Matrix Market has no defined semantics for a
    repeated coordinate and readers disagree about it.
    """
    acc = {}
    for i, j, v in zip(irn, jcn, vals):
        if i < j:
            i, j = j, i
        acc[(i, j)] = acc.get((i, j), 0.0) + v
    with path.open("w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {len(acc)}\n")
        for (i, j), v in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            f.write(f"{i} {j} {v:.17e}\n")
    return len(acc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True,
                    help="pounce --dump-dir root containing iter_NNN/")
    ap.add_argument("--name", required=True, help="problem name / file stem")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help=f"corpus root (default: {DEFAULT_OUT})")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="keep at most this many iterations (0 = all)")
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    iters = sorted(d for d in dump.glob("iter_*") if d.is_dir())
    if not iters:
        print(f"no iter_* directories under {dump}", file=sys.stderr)
        return 1
    if args.max_iters:
        iters = iters[:args.max_iters]

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for k, d in enumerate(iters):
        solves = sorted(d.glob("kkt_solve_*.jsonl"))
        if not solves:
            continue
        line = solves[0].read_text().splitlines()
        if not line:
            continue
        rec = json.loads(line[0])
        n = int(rec["n"])
        stem = f"{args.name}_{k:04d}"
        nnz = write_mtx(out / f"{stem}.mtx", n,
                        rec["irn"], rec["jcn"], rec["vals"])
        neg = int(rec.get("num_neg_evals_actual", -1))
        (out / f"{stem}.json").write_text(json.dumps({
            "problem_name": args.name,
            "iteration": k,
            "n": n,
            "rhs": rec.get("rhs", []),
            "inertia": {"negative": neg,
                        "positive": n - neg if neg >= 0 else -1,
                        "zero": 0},
            "status": rec.get("status", "unknown"),
            "source": "pounce --dump kkt",
        }))
        written += 1
        print(f"  {stem}  n={n}  nnz={nnz}")

    print(f"\n{written} iterations written to {out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
