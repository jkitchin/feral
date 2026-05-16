#!/usr/bin/env python3
"""Analyze IR trajectory sidecars produced by `probe_ir_trajectory`.

For each matrix, compute:
  * `rel_res_0` — unrefined relative residual (IR-OFF proxy)
  * `rel_res_best` — best relative residual across the trajectory (IR-ON
    end-state, mirrors what `solve_sparse_refined` returns)
  * `improvement` — log10(rel_res_0 / rel_res_best). Larger = IR helped.
  * `kappa_1_est` — Hager-Higham 1-norm condition estimate.
  * `n_useful_steps` — number of `improved=1` flags after step 0
    (how many refinement passes actually reduced ||r||).

Emits a TSV table and a category roll-up. The point of the table is
to find a κ̂ (or per-matrix cheap-to-compute) threshold above which IR
strictly helps, and below which it's a no-op.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

STRESS_DIR = Path(__file__).resolve().parent
PROBE_DIR = STRESS_DIR / "out" / "ir_probe"
MANIFEST = STRESS_DIR / "manifest.tsv"


def parse_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))
    for r in rows:
        r["n"] = int(r["n"])
        r["nnz"] = int(r["nnz"])
    return rows


def parse_sidecar(path: Path) -> dict | None:
    if not path.exists():
        return None
    d: dict = {"steps": []}
    step_re = re.compile(
        r"^ir_step_(\d+) res2=(\S+) rel_res=(\S+) fwd_bound=(\S+) improved=(\d+)$"
    )
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = step_re.match(line)
            if m:
                d["steps"].append({
                    "i": int(m.group(1)),
                    "res2": float(m.group(2)),
                    "rel_res": float(m.group(3)),
                    "fwd_bound": float(m.group(4)),
                    "improved": int(m.group(5)) == 1,
                })
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def fnum(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def main() -> int:
    rows = parse_manifest(MANIFEST)
    records = []
    for r in rows:
        side = parse_sidecar(PROBE_DIR / f"{r['group']}__{r['name']}.out")
        if side is None or side.get("status") != "ok":
            continue
        steps = side["steps"]
        if not steps:
            continue
        rel0 = steps[0]["rel_res"]
        rel_best = min(s["rel_res"] for s in steps)
        n_useful = sum(1 for s in steps[1:] if s["improved"])
        kappa = fnum(side.get("kappa_1_est")) or 0.0
        n_dim = int(side.get("n", r["n"]))
        # Floor noise threshold from the existing IR loop: eps * sqrt(n).
        threshold = 2.220446049250313e-16 * math.sqrt(n_dim)
        # "Strictly helps" = improvement of at least 1 order of magnitude
        # AND ends below the threshold (or starts above it and drops 10x).
        improvement = (math.log10(rel0 / rel_best)
                       if rel_best > 0 and rel0 > 0 else 0.0)
        records.append({
            "category": r.get("category", "?"),
            "name": r["name"],
            "n": n_dim,
            "kappa": kappa,
            "rel0": rel0,
            "rel_best": rel_best,
            "improvement_log10": improvement,
            "n_useful": n_useful,
            "n_steps": len(steps),
            "threshold": threshold,
            "starts_below_threshold": rel0 < threshold,
            "ends_below_threshold": rel_best < threshold,
        })

    records.sort(key=lambda r: r["kappa"])

    print(f"{'category':<10} {'n':>7} {'name':<25} "
          f"{'kappa_1_est':>11} {'rel0':>10} {'rel_best':>10} "
          f"{'gain':>6} {'useful':>6} {'steps':>5} "
          f"{'b0':>3} {'bN':>3}")
    print("-" * 110)
    for r in records:
        print(f"{r['category']:<10} {r['n']:>7} {r['name'][:25]:<25} "
              f"{r['kappa']:>11.2e} {r['rel0']:>10.2e} "
              f"{r['rel_best']:>10.2e} "
              f"{r['improvement_log10']:>6.2f} "
              f"{r['n_useful']:>6} {r['n_steps']:>5} "
              f"{'Y' if r['starts_below_threshold'] else 'N':>3} "
              f"{'Y' if r['ends_below_threshold'] else 'N':>3}")
    print("-" * 110)

    # Buckets
    print("\n=== buckets ===")
    # Bucket 1: matrices where unrefined solve is already below threshold.
    # IR offers no useful work on these.
    b1 = [r for r in records if r["starts_below_threshold"]]
    print(f"\n[bucket A] starts below eps*sqrt(n) threshold "
          f"(IR is a no-op): {len(b1)} matrices")
    for r in b1:
        print(f"  {r['name']:<25} kappa={r['kappa']:.2e} rel0={r['rel0']:.2e} "
              f"useful_steps={r['n_useful']} gain={r['improvement_log10']:.2f}")

    # Bucket 2: IR moves residual below threshold (strictly helps).
    b2 = [r for r in records if not r["starts_below_threshold"]
          and r["ends_below_threshold"]]
    print(f"\n[bucket B] IR moves residual below threshold "
          f"(strictly helps): {len(b2)} matrices")
    for r in b2:
        print(f"  {r['name']:<25} kappa={r['kappa']:.2e} "
              f"rel0={r['rel0']:.2e} rel_best={r['rel_best']:.2e} "
              f"useful_steps={r['n_useful']} gain={r['improvement_log10']:.2f}")

    # Bucket 3: IR runs but never reaches threshold (stagnates above).
    b3 = [r for r in records if not r["starts_below_threshold"]
          and not r["ends_below_threshold"]]
    print(f"\n[bucket C] IR runs but never reaches threshold "
          f"(stagnates): {len(b3)} matrices")
    for r in b3:
        print(f"  {r['name']:<25} kappa={r['kappa']:.2e} "
              f"rel0={r['rel0']:.2e} rel_best={r['rel_best']:.2e} "
              f"useful_steps={r['n_useful']} gain={r['improvement_log10']:.2f}")

    # Per-bucket kappa distribution
    print("\n=== kappa_1_est distribution per bucket ===")
    for label, b in [("A (no-op)", b1), ("B (helps)", b2),
                     ("C (stagnant)", b3)]:
        if not b:
            print(f"  {label}: (empty)")
            continue
        ks = sorted(r["kappa"] for r in b)
        print(f"  {label}: n={len(b)} "
              f"min={ks[0]:.2e} median={ks[len(ks)//2]:.2e} "
              f"max={ks[-1]:.2e}")

    # Cost analysis: count extra solves on bucket A (wasted work).
    waste_solves = sum(r["n_steps"] - 1 for r in b1)
    helpful_solves = sum(r["n_steps"] - 1 for r in b2)
    print(f"\n=== solve-call accounting ===")
    print(f"bucket A: {waste_solves} extra solves did not improve residual")
    print(f"bucket B: {helpful_solves} extra solves were necessary")
    print(f"bucket C: {sum(r['n_steps']-1 for r in b3)} extra solves "
          f"reduced but did not eliminate residual")

    return 0


if __name__ == "__main__":
    sys.exit(main())
