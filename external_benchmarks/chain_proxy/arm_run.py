#!/usr/bin/env python3
"""Paired alternating A/B over an arbitrary number of arms.

`ab_run.py` fixes three arms (new / base / ma57). The bisect and the
thread sweep need five and four, so this generalizes it: arms are
declared on the command line, each one a binary plus optional
environment overrides, and every arm is timed once per pair.

Protocol is unchanged from `ab_run.py` and from feral
`dev/decisions.md` (2026-08-09): one timing per arm per pair so drift
hits all arms equally, `min` over pairs as the per-arm statistic, exact
two-sided sign test over the pairs for significance. Medians collected
at different times are never compared. All arms read the same manifest
format (`<mtx> <rhs> <out>`), so the matrices, the RHS and the residual
definition are identical across arms.

`--mtx-dir` points at any directory of `.mtx` files, which is how the
same protocol runs on the generated proxies and on the real corpus
(see `real_corpus_mtx.py`).

Arm syntax:

    --arm 'NAME=/path/to/bin'
    --arm 'NAME=/path/to/bin|RAYON_NUM_THREADS=1,FERAL_PACKED_SIMD=0'

Example -- bisect five feral builds against MA57, ratios against 0.14.0:

    python3 arm_run.py --mtx-dir $WORK/mtx --pairs 15 --ref v0.14.0 \
      --arm 'v0.14.0=/tmp/wt/v0140/target/release/bench_one_matrix' \
      --arm 'main=target/release/bench_one_matrix' \
      --arm 'ma57=external_benchmarks/ma57_oracle/ma57_bench'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ab_run import parse_sidecar, read_mtx_lower, sign_test_p, synth_rhs


def parse_arm(spec: str) -> tuple[str, Path, dict]:
    """`NAME=BIN` or `NAME=BIN|K=V,K2=V2` -> (name, bin, env overrides)."""
    if "=" not in spec:
        raise ValueError(f"arm {spec!r}: expected NAME=BIN")
    name, rest = spec.split("=", 1)
    env = {}
    if "|" in rest:
        rest, envspec = rest.split("|", 1)
        for kv in envspec.split(","):
            if not kv.strip():
                continue
            if "=" not in kv:
                raise ValueError(f"arm {name}: env {kv!r} is not K=V")
            k, v = kv.split("=", 1)
            env[k.strip()] = v.strip()
    return name.strip(), Path(rest.strip()), env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtx-dir", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=BIN[|K=V,...]; repeatable")
    ap.add_argument("--ref", help="arm every ratio is taken against "
                                  "(default: the first arm)")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--rhs-dir", help="where synthesized RHS live "
                                      "(default: <mtx-dir>/../rhs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arm]
    names = [a[0] for a in arms]
    if len(set(names)) != len(names):
        print("duplicate arm names", file=sys.stderr)
        return 1
    ref = args.ref or names[0]
    if ref not in names:
        print(f"--ref {ref} is not one of {names}", file=sys.stderr)
        return 1
    for name, binp, _ in arms:
        if not binp.exists():
            print(f"arm {name}: no binary at {binp}", file=sys.stderr)
            return 1

    mtx_dir = Path(args.mtx_dir)
    mats = sorted(mtx_dir.glob("*.mtx"))
    if not mats:
        print(f"no matrices in {mtx_dir}", file=sys.stderr)
        return 1

    rhs_dir = Path(args.rhs_dir) if args.rhs_dir else mtx_dir.parent / "rhs"
    rhs_dir.mkdir(parents=True, exist_ok=True)
    print("=== synthesizing RHS ===", flush=True)
    for m in mats:
        rp = rhs_dir / (m.stem + ".rhs")
        if rp.exists():
            continue
        n, trips = read_mtx_lower(m)
        with rp.open("w") as f:
            for v in synth_rhs(n, trips):
                f.write(f"{v:.17e}\n")
        print(f"  {m.stem}  n={n}  nnz={len(trips)}", flush=True)

    out_root = Path(args.out).parent / (Path(args.out).stem + "_out")
    manifests = {}
    for name, _, _ in arms:
        (out_root / name).mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=f"_{name}.manifest", text=True)
        with os.fdopen(fd, "w") as f:
            for m in mats:
                f.write(f"{m} {rhs_dir / (m.stem + '.rhs')} "
                        f"{out_root / name / (m.stem + '.out')}\n")
        manifests[name] = Path(path)

    samples = {n: {m.stem: [] for m in mats} for n in names}
    meta = {n: {} for n in names}

    print(f"\n=== {args.pairs} pairs, {len(arms)} arms, "
          f"{len(mats)} matrices ===", flush=True)
    for p in range(args.pairs):
        for name, binp, envover in arms:
            env = dict(os.environ)
            env.update(envover)
            subprocess.run([str(binp), str(manifests[name])], env=env,
                           check=False, capture_output=True, timeout=3600)
            for m in mats:
                d = parse_sidecar(out_root / name / (m.stem + ".out"))
                if d.get("status") == "ok" and "factor_us" in d:
                    samples[name][m.stem].append(int(d["factor_us"]))
                    meta[name][m.stem] = {
                        "solver": d.get("solver", name),
                        "n": int(d.get("n", 0)),
                        "rel_res": float(d.get("rel_res", "nan")),
                        "inertia_pos": d.get("inertia_pos"),
                        "inertia_neg": d.get("inertia_neg"),
                        "inertia_zero": d.get("inertia_zero"),
                    }
        print(f"  pair {p + 1}/{args.pairs} done", flush=True)

    Path(args.out).write_text(json.dumps(
        {"pairs": args.pairs, "ref": ref,
         "arms": [{"name": n, "bin": str(b), "env": e} for n, b, e in arms],
         "samples": samples, "meta": meta}, indent=2))

    # ---- correctness gate, before any timing is read ----
    print("\n=== correctness (inertia_zero / rel_res) ===", flush=True)
    suspect = []
    for m in mats:
        bad = []
        for name in names:
            d = meta[name].get(m.stem)
            if d is None:
                bad.append(f"{name}: no result")
                continue
            rr = d["rel_res"]
            if d["inertia_zero"] not in (None, "0"):
                bad.append(f"{name}: inertia_zero={d['inertia_zero']}")
            if not (rr == rr) or rr > 1e-8:      # NaN or too large
                bad.append(f"{name}: rel_res={rr:.2e}")
        if bad:
            suspect.append(m.stem)
            print(f"  !! {m.stem}: " + "; ".join(bad))
    if not suspect:
        print("  all arms: inertia_zero=0, rel_res <= 1e-8 on every matrix")
    else:
        print("  timings on the above are NOT trustworthy -- fix or drop "
              "the matrix before quoting a ratio")

    # ---- report ----
    print("\n=== min factor_us over pairs ===", flush=True)
    print(f"{'matrix':<22}{'n':>8}" + "".join(f"{n:>16}" for n in names))
    for m in mats:
        n = next((meta[a][m.stem]["n"] for a in names if m.stem in meta[a]), 0)
        row = f"{m.stem:<22}{n:>8}"
        for name in names:
            s = samples[name][m.stem]
            row += f"{min(s) if s else -1:>16}"
        print(row)

    print(f"\n=== ratio vs {ref} (>1 means the arm is faster "
          f"than {ref}) ===", flush=True)
    for name in names:
        if name == ref:
            continue
        print(f"\n--- {name} vs {ref}")
        print(f"{'matrix':<22}{'ratio':>9}{'wins':>9}{'p':>10}")
        for m in mats:
            sa, sb = samples[name][m.stem], samples[ref][m.stem]
            if not sa or not sb:
                print(f"{m.stem:<22}{'n/a':>9}")
                continue
            k = min(len(sa), len(sb))
            wins = sum(1 for i in range(k) if sa[i] < sb[i])
            print(f"{m.stem:<22}{min(sb) / min(sa):>9.3f}"
                  f"{f'{wins}/{k}':>9}{sign_test_p(wins, k):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
