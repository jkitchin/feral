#!/usr/bin/env python3
"""Paired alternating A/B: two feral builds vs HSL MA57.

Protocol per feral dev/decisions.md (2026-08-09): every arm is timed
once per pair so drift hits all arms equally, `min` over pairs is the
per-arm statistic, and the sign test over pairs is the significance
check. Medians collected at different times are NOT compared.

Each arm is an external binary reading the same manifest format
(<mtx> <rhs> <out>), so the matrices, the RHS and the residual
definition are identical across arms.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SP = Path(__file__).resolve().parent
ROOT = SP.parents[1]
WORK = Path(os.environ.get("CHAIN_PROXY_WORK", SP / "work"))

# Arm binaries. The "new" arm defaults to this checkout's build; the
# baseline arm is whatever you built in a worktree at the older tag.
# Override any of them by environment variable.
#
#   git worktree add /tmp/feral-0140 v0.14.0
#   (cd /tmp/feral-0140 && cargo build --release -p feral-diagnostics \
#        --bin bench_one_matrix)
#   FERAL_BIN_BASE=/tmp/feral-0140/target/release/bench_one_matrix \
#       python3 ab_run.py
DRIVER = "target/release/bench_one_matrix"
FERAL_NEW = Path(os.environ.get("FERAL_BIN_NEW", ROOT / DRIVER))
FERAL_BASE = Path(os.environ.get("FERAL_BIN_BASE", WORK / "feral-base" / DRIVER))
MA57 = Path(os.environ.get(
    "MA57_BIN", ROOT / "external_benchmarks" / "ma57_oracle" / "ma57_bench"))

NEW_LABEL = os.environ.get("FERAL_LABEL_NEW", "feral-new")
BASE_LABEL = os.environ.get("FERAL_LABEL_BASE", "feral-base")

ARMS = [(NEW_LABEL, FERAL_NEW), (BASE_LABEL, FERAL_BASE), ("ma57", MA57)]


def read_mtx_lower(path: Path):
    trips = []
    n = 0
    saw = False
    with path.open() as f:
        for line in f:
            if line.startswith("%"):
                continue
            p = line.split()
            if not saw:
                n = int(p[1])
                saw = True
                continue
            if len(p) < 3:
                continue
            r, c, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
            if r < c:
                r, c = c, r
            trips.append((r, c, v))
    return n, trips


def synth_rhs(n, trips):
    x = [1.0 + i / n for i in range(n)]
    b = [0.0] * n
    for r, c, v in trips:
        b[r] += v * x[c]
        if r != c:
            b[c] += v * x[r]
    return b


def parse_sidecar(path: Path) -> dict:
    d = {}
    if not path.exists():
        return d
    for line in path.read_text().splitlines():
        p = line.split(None, 1)
        if len(p) == 2:
            d[p[0]] = p[1].strip()
    return d


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p at q=0.5."""
    from math import comb
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtx-dir", default=str(WORK / "mtx"))
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--out", default=str(WORK / "ab_results.json"))
    args = ap.parse_args()

    mtx_dir = Path(args.mtx_dir)
    mats = sorted(mtx_dir.glob("*.mtx"))
    if not mats:
        print(f"no matrices in {mtx_dir}", file=sys.stderr)
        return 1

    rhs_dir = WORK / "rhs"
    rhs_dir.mkdir(parents=True, exist_ok=True)
    print("=== synthesizing RHS ===", flush=True)
    for m in mats:
        rp = rhs_dir / (m.stem + ".rhs")
        if not rp.exists():
            n, trips = read_mtx_lower(m)
            with rp.open("w") as f:
                for v in synth_rhs(n, trips):
                    f.write(f"{v:.17e}\n")
            print(f"  {m.stem}  n={n}  nnz={len(trips)}", flush=True)

    out_root = WORK / "ab_out"
    manifests = {}
    for arm, _ in ARMS:
        (out_root / arm).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(suffix=f"_{arm}.manifest", text=True)
        with os.fdopen(fd, "w") as f:
            for m in mats:
                f.write(f"{m} {rhs_dir / (m.stem + '.rhs')} "
                        f"{out_root / arm / (m.stem + '.out')}\n")
        manifests[arm] = Path(name)

    # samples[arm][matrix] = [factor_us per pair]
    samples = {arm: {m.stem: [] for m in mats} for arm, _ in ARMS}
    meta = {arm: {} for arm, _ in ARMS}

    print(f"\n=== {args.pairs} pairs, {len(ARMS)} arms, "
          f"{len(mats)} matrices ===", flush=True)
    for p in range(args.pairs):
        for arm, binp in ARMS:
            subprocess.run([str(binp), str(manifests[arm])],
                           check=False, capture_output=True, timeout=1800)
            for m in mats:
                d = parse_sidecar(out_root / arm / (m.stem + ".out"))
                if d.get("status") == "ok" and "factor_us" in d:
                    samples[arm][m.stem].append(int(d["factor_us"]))
                    meta[arm][m.stem] = {
                        "solver": d.get("solver", arm),
                        "n": int(d.get("n", 0)),
                        "rel_res": float(d.get("rel_res", "nan")),
                        "inertia_pos": d.get("inertia_pos"),
                        "inertia_neg": d.get("inertia_neg"),
                        "inertia_zero": d.get("inertia_zero"),
                    }
        print(f"  pair {p + 1}/{args.pairs} done", flush=True)

    results = {"pairs": args.pairs, "samples": samples, "meta": meta}
    Path(args.out).write_text(json.dumps(results, indent=2))

    # ---- report ----
    print("\n=== min factor_us over pairs ===", flush=True)
    hdr = f"{'matrix':<22}{'n':>7}" + "".join(f"{a:>16}" for a, _ in ARMS)
    print(hdr)
    for m in mats:
        row = f"{m.stem:<22}"
        n = next((meta[a][m.stem]["n"] for a, _ in ARMS
                  if m.stem in meta[a]), 0)
        row += f"{n:>7}"
        for arm, _ in ARMS:
            s = samples[arm][m.stem]
            row += f"{min(s) if s else -1:>16}"
        print(row)

    def compare(a: str, b: str, label: str) -> None:
        print(f"\n=== {label} ({a} vs {b}) ===", flush=True)
        print(f"{'matrix':<22}{'ratio':>9}{'wins':>9}{'p':>10}   "
              f"{'rel_res ' + a:>14}{'rel_res ' + b:>14}")
        for m in mats:
            sa, sb = samples[a][m.stem], samples[b][m.stem]
            if not sa or not sb:
                print(f"{m.stem:<22}{'n/a':>9}")
                continue
            k = min(len(sa), len(sb))
            wins = sum(1 for i in range(k) if sa[i] < sb[i])
            ratio = min(sb) / min(sa)  # >1 means a is faster
            ra = meta[a].get(m.stem, {}).get("rel_res", float("nan"))
            rb = meta[b].get(m.stem, {}).get("rel_res", float("nan"))
            print(f"{m.stem:<22}{ratio:>9.3f}{f'{wins}/{k}':>9}"
                  f"{sign_test_p(wins, k):>10.4f}{ra:>14.2e}{rb:>14.2e}")

    compare(NEW_LABEL, BASE_LABEL, "release delta (validity control)")
    compare(NEW_LABEL, "ma57", f"THE GAP: {NEW_LABEL} vs HSL MA57")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
