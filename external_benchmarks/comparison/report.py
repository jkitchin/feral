#!/usr/bin/env python3
"""Render comparison.json as a markdown report.

Sections:
  1. Overview (sample composition, solver versions, environment).
  2. Per-matrix table (n, nnz, density, per-solver factor_us + rel_res).
  3. Status summary (pass/fail counts per solver).
  4. Speed: per-size-bucket geomean factor_us; head-to-head ratios.
  5. Accuracy: distribution of rel_res; ill-conditioned outliers.
  6. Inertia agreement across solvers.
  7. Failures table.
  8. Notable matrices (largest feral/mumps ratio wins/losses).

Usage:  python3 report.py  [--out REPORT.md] [--data comparison.json]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path

COMP_DIR = Path(__file__).resolve().parent
SOLVERS = ["feral", "mumps", "ma97"]


def bucket(n: int) -> str:
    if n < 100: return "tiny (n<100)"
    if n < 1000: return "small (100-1k)"
    if n < 10000: return "medium (1k-10k)"
    if n < 100000: return "large (10k-100k)"
    return "xl (>=100k)"


def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and x > 0]
    if not xs: return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def fmt_us(us: int | None) -> str:
    if us is None: return "—"
    if us < 1000: return f"{us}μs"
    if us < 1_000_000: return f"{us/1000:.1f}ms"
    return f"{us/1_000_000:.2f}s"


def fmt_res(r: float | None) -> str:
    if r is None or not math.isfinite(r): return "—"
    return f"{r:.1e}"


def collect(records, key):
    """{solver: [value or None per record]}"""
    out = {s: [] for s in SOLVERS}
    for rec in records:
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            v = e.get(key) if e.get("status") == "ok" else None
            out[s].append(v)
    return out


def status_counts(records):
    out = {s: {"ok": 0, "fail": 0, "missing": 0} for s in SOLVERS}
    for rec in records:
        for s in SOLVERS:
            st = rec["solvers"].get(s, {}).get("status", "missing")
            if st == "ok": out[s]["ok"] += 1
            elif st == "missing": out[s]["missing"] += 1
            else: out[s]["fail"] += 1
    return out


def render(records) -> str:
    lines: list[str] = []

    # --- 1. Overview ---
    lines.append("# FERAL vs MUMPS vs HSL MA97 — KKT solver comparison")
    lines.append("")
    lines.append(f"Total matrices: **{len(records)}**, drawn from the FERAL CUTEst")
    lines.append("KKT corpus and Mittelmann large-scale KKT corpus.")
    lines.append("Sampling spans 5 size buckets and 63 distinct CUTEst/Mittelmann")
    lines.append("families. RHS is synthetic: `b = A · x_true` with")
    lines.append("`x_true[i] = 1 + i/n`. Same RHS is fed to all three solvers.")
    lines.append("")
    lines.append("**Solvers** — each is configured with its recommended")
    lines.append("high-accuracy defaults: shape-aware scaling on, iterative")
    lines.append("refinement on. The goal is an apples-to-apples comparison")
    lines.append("of what a real consumer would get from each library when")
    lines.append("accuracy matters; the bare back-substitution defaults are")
    lines.append("*not* what these tables show.")
    lines.append("")
    versions = {s: None for s in SOLVERS}
    for rec in records:
        for s in SOLVERS:
            v = rec["solvers"].get(s, {}).get("solver_version")
            if v: versions[s] = v
    lines.append("| Solver | Version | Driver | Configuration |")
    lines.append("|---|---|---|---|")
    lines.append(f"| feral | {versions['feral'] or '?'} | "
                 "`factorize_multifrontal_parallel` + `solve_sparse_refined` | "
                 "`ScalingStrategy::Auto` (MC64-symmetric or inf-norm by shape, "
                 "threshold tuned 2026-04-19); BK pivot threshold `1e-8` (MA27 "
                 "default); refinement loop runs up to 10 steps with "
                 "stagnation-based exit. rayon-parallel multifrontal; falls "
                 "through to sequential below 32 supernodes. |")
    lines.append(f"| MUMPS | {versions['mumps'] or '?'} | "
                 "`dmumps SYM=2` | "
                 "`ICNTL(10) = 2` (two iterative-refinement steps; MUMPS "
                 "default is `0` = no refinement); `ICNTL(11) = 1` (full "
                 "error analysis); `ICNTL(24) = 1` (null pivot detection). "
                 "Sequential build, no MC64 scaling by default. |")
    lines.append(f"| MA97  | {versions['ma97']  or '?'} | "
                 "`ma97_factor matrix_type=4` + Richardson loop around "
                 "`ma97_solve_d` | "
                 "`scaling = 1` (MC64 enabled, the recommended HSL default); "
                 "`ordering = 5` (auto AMD/METIS); `action = 1` (continue past "
                 "singular pivots). MA97 has no built-in residual-based "
                 "refinement entry point, so the driver wraps `ma97_solve_d` "
                 "in a 4-step Richardson loop (stagnation exit) to match what "
                 "MUMPS+ICNTL(10) and feral+`solve_sparse_refined` deliver. "
                 "CoinHSL 2023.11.17, OpenMP. |")
    lines.append("")
    lines.append("> Change any of these settings and the timing/accuracy")
    lines.append("> columns will move. The bench captures each library's")
    lines.append("> *best-effort* mode, not its raw defaults.")
    lines.append("")
    lines.append(f"**Host**: {platform.platform()}, {platform.machine()}, "
                 f"Python {platform.python_version()}.")
    lines.append("")

    # --- 2. Size composition ---
    lines.append("## Sample composition")
    lines.append("")
    from collections import Counter
    c = Counter(bucket(r["matrix"]["n"]) for r in records)
    lines.append("| Bucket | Count |")
    lines.append("|---|---:|")
    for b in ["tiny (n<100)", "small (100-1k)", "medium (1k-10k)",
              "large (10k-100k)", "xl (>=100k)"]:
        lines.append(f"| {b} | {c.get(b, 0)} |")
    lines.append("")

    # --- 3. Status summary ---
    lines.append("## Status summary")
    lines.append("")
    sc = status_counts(records)
    lines.append("| Solver | OK | Fail | Missing |")
    lines.append("|---|---:|---:|---:|")
    for s in SOLVERS:
        lines.append(f"| {s} | {sc[s]['ok']} | {sc[s]['fail']} | {sc[s]['missing']} |")
    lines.append("")

    # --- 4. Speed by bucket ---
    lines.append("## Factor time by size bucket (geomean μs)")
    lines.append("")
    by_bucket: dict[str, list] = {}
    for rec in records:
        b = bucket(rec["matrix"]["n"])
        by_bucket.setdefault(b, []).append(rec)
    lines.append("| Bucket | n range |  feral |  MUMPS |  MA97  | feral/MUMPS | feral/MA97 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for b in ["tiny (n<100)", "small (100-1k)", "medium (1k-10k)",
              "large (10k-100k)", "xl (>=100k)"]:
        recs = by_bucket.get(b, [])
        if not recs: continue
        ns = [r["matrix"]["n"] for r in recs]
        nrange = f"{min(ns)}–{max(ns)}"
        gm = {s: geomean([r["solvers"].get(s, {}).get("factor_us")
                          for r in recs if r["solvers"].get(s, {}).get("status") == "ok"])
              for s in SOLVERS}
        ratio_mu = gm["feral"]/gm["mumps"] if gm["feral"] and gm["mumps"] else None
        ratio_ma = gm["feral"]/gm["ma97"]  if gm["feral"] and gm["ma97"]  else None
        def f(x): return f"{x:,.0f}" if x else "—"
        def fr(x): return f"{x:.2f}×" if x else "—"
        lines.append(f"| {b} | {nrange} | {f(gm['feral'])} | {f(gm['mumps'])} | "
                     f"{f(gm['ma97'])} | {fr(ratio_mu)} | {fr(ratio_ma)} |")
    lines.append("")
    lines.append("> Ratios < 1.0 mean **feral is faster**. Geomean is over the")
    lines.append("> matrices in the bucket where the named solver succeeded.")
    lines.append("")

    # --- 5. Accuracy distribution ---
    lines.append("## Accuracy: ‖Ax − b‖₂ / ‖b‖₂ distribution")
    lines.append("")
    lines.append("| Solver | min | median | p90 | max | # > 1e-8 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    res_by_solver = collect(records, "rel_res")
    for s in SOLVERS:
        vs = sorted(v for v in res_by_solver[s] if v is not None and math.isfinite(v))
        if not vs: continue
        p90 = vs[int(0.9 * (len(vs)-1))]
        bad = sum(1 for v in vs if v > 1e-8)
        lines.append(f"| {s} | {fmt_res(vs[0])} | {fmt_res(statistics.median(vs))} | "
                     f"{fmt_res(p90)} | {fmt_res(vs[-1])} | {bad} |")
    lines.append("")

    # --- 6. Inertia agreement ---
    lines.append("## Inertia agreement")
    lines.append("")
    agree = 0
    disagree = []
    for rec in records:
        ins = [tuple(rec["solvers"].get(s, {}).get("inertia") or ())
               for s in SOLVERS]
        if all(i for i in ins) and len(set(ins)) == 1:
            agree += 1
        elif all(i for i in ins):
            disagree.append((rec, ins))
    lines.append(f"All three solvers report identical inertia on **{agree}** "
                 f"of {len(records)} matrices.")
    if disagree:
        lines.append("")
        lines.append("Inertia mismatches:")
        lines.append("")
        lines.append("| Matrix | n | feral | MUMPS | MA97 |")
        lines.append("|---|---:|---|---|---|")
        for rec, ins in disagree:
            m = rec["matrix"]
            f_, m_, a_ = (f"{x[0]}+{x[1]}+{x[2]}" if x else "—" for x in ins)
            lines.append(f"| {m['family']}/{m['id']} | {m['n']} | {f_} | {m_} | {a_} |")
    lines.append("")

    # --- 7. Failures table ---
    lines.append("## Failures")
    lines.append("")
    failed_rows = []
    for rec in records:
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            if e.get("status") not in (None, "ok", "missing"):
                failed_rows.append((rec["matrix"], s, e.get("fail_reason", e.get("status", "?"))))
    if not failed_rows:
        lines.append("None.")
    else:
        lines.append("| Matrix | n | nnz | Solver | Reason |")
        lines.append("|---|---:|---:|---|---|")
        for m, s, why in sorted(failed_rows, key=lambda t: t[0]["n"]):
            lines.append(f"| {m['family']}/{m['id']} | {m['n']} | {m['nnz']} | {s} | `{why}` |")
    lines.append("")

    # --- 8. Notable matrices ---
    lines.append("## Notable matrices (feral / MUMPS factor-time ratio)")
    lines.append("")
    ratios = []
    for rec in records:
        f = rec["solvers"].get("feral", {})
        m = rec["solvers"].get("mumps", {})
        if f.get("status") == "ok" and m.get("status") == "ok":
            fu = f.get("factor_us"); mu = m.get("factor_us")
            if fu and mu:
                ratios.append((fu / mu, rec))
    ratios.sort(key=lambda t: t[0])
    def fmt_row(rec, ratio):
        mx = rec["matrix"]
        f = rec["solvers"]["feral"]; m = rec["solvers"]["mumps"]; a = rec["solvers"].get("ma97", {})
        return (f"| {mx['family']}/{mx['id']} | {mx['n']:,} | {mx['nnz']:,} | "
                f"{fmt_us(f['factor_us'])} | {fmt_us(m['factor_us'])} | "
                f"{fmt_us(a.get('factor_us'))} | {ratio:.2f}× |")
    lines.append("### Top 10 feral wins vs MUMPS")
    lines.append("")
    lines.append("| Matrix | n | nnz | feral | MUMPS | MA97 | feral/MUMPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ratio, rec in ratios[:10]:
        lines.append(fmt_row(rec, ratio))
    lines.append("")
    lines.append("### Top 10 feral losses vs MUMPS")
    lines.append("")
    lines.append("| Matrix | n | nnz | feral | MUMPS | MA97 | feral/MUMPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ratio, rec in ratios[-10:][::-1]:
        lines.append(fmt_row(rec, ratio))
    lines.append("")

    # --- 9. Full per-matrix table ---
    lines.append("## Full per-matrix table")
    lines.append("")
    lines.append("| Matrix | n | nnz | density | "
                 "feral factor / rel\\_res | MUMPS factor / rel\\_res | "
                 "MA97 factor / rel\\_res |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    for rec in records:
        mx = rec["matrix"]
        cells = []
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            if e.get("status") == "ok":
                cells.append(f"{fmt_us(e.get('factor_us'))} / {fmt_res(e.get('rel_res'))}")
            else:
                cells.append(f"`{e.get('status', '?')}`")
        lines.append(f"| {mx['family']}/{mx['id']} | {mx['n']:,} | {mx['nnz']:,} | "
                     f"{mx['density']:.1%} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `external_benchmarks/comparison/report.py` from "
                 "`comparison.json`. Reproduce with `python3 run.py && python3 aggregate.py "
                 "&& python3 report.py`.*")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(COMP_DIR / "comparison.json"))
    ap.add_argument("--out",  default=str(COMP_DIR / "REPORT.md"))
    args = ap.parse_args()
    records = json.loads(Path(args.data).read_text())
    records.sort(key=lambda r: (r["matrix"]["n"], r["matrix"]["family"]))
    md = render(records)
    Path(args.out).write_text(md)
    print(f"wrote {args.out} ({len(md):,} bytes, {len(records)} matrices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
