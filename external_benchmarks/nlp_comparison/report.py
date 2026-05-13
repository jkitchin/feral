#!/usr/bin/env python3
"""Render a comparison report from `results.json` (produced by run.py).

Captures, for each (problem, solver):
- status (Optimal / Infeasible / iteration_limit / ...)
- iterations
- total seconds inside Ipopt
- final objective (unscaled)
- final NLP error

Sections:
1. Solver configuration
2. Success summary (counts of "Optimal Solution Found" per solver)
3. Iteration counts (problems where IPM trajectory differs)
4. Timing comparison (geomean total seconds; per-problem rows)
5. Full per-problem table
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVERS = ["mumps", "ma57", "feral"]


def fmt_secs(s):
    if s is None:
        return "-"
    if s < 1e-3:
        return f"{s*1e6:.0f}μs"
    if s < 1.0:
        return f"{s*1e3:.1f}ms"
    if s < 60:
        return f"{s:.2f}s"
    return f"{s/60:.1f}m"


def fmt_obj(o):
    if o is None or (isinstance(o, float) and not math.isfinite(o)):
        return "-"
    return f"{o:.4e}"


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out",     default=str(HERE / "REPORT.md"))
    args = ap.parse_args()

    with open(args.results) as f:
        records = json.load(f)

    lines: list[str] = []
    lines.append("# Ipopt 3.14.20 × {feral, MUMPS, MA57} on the Scalable NLP suite")
    lines.append("")
    lines.append(f"Total problems: **{len(records)}** from "
                 "`ref/Ipopt/examples/ScalableProblems` "
                 "(LuksanVlcek 1–7 in equality + inequality flavors, "
                 "Mittelmann boundary/distributed/3D control suites, "
                 "and MittelmannParaCntrl).  ")
    lines.append("")
    lines.append("Each problem is fed to **three different Ipopt 3.14.20 "
                 "binaries**, each linked to a single sparse direct linear "
                 "solver and otherwise identical:")
    lines.append("")
    lines.append("| Binary | Linear solver | Build dir | Version |")
    lines.append("|---|---|---|---|")
    lines.append("| `build-mumps`  | MUMPS 5.8.2 (sequential) "
                 "| `ref/Ipopt/build-mumps` "
                 "| `--with-mumps` against vendored `ref/mumps` |")
    lines.append("| `build-ma57`   | HSL MA57 (CoinHSL 2023.11.17, "
                 "sequential) | `ref/Ipopt/build-ma57` "
                 "| `--with-hsl` against `libcoinhsl.a` |")
    lines.append("| `build-feral`  | feral 0.2.0 (Rust, multifrontal "
                 "parallel) | `ref/Ipopt/build-feral` "
                 "| `feral-ipopt-shim` C-ABI patch onto Ipopt |")
    lines.append("")
    lines.append("All three binaries use Ipopt's stock defaults — only "
                 "`linear_solver` is overridden, written to an `ipopt.opt` "
                 "in the working directory. Each run is fresh "
                 "(separate cwd) so no state leaks between solvers.")
    lines.append("")
    lines.append(f"**Host**: {platform.platform()}, {platform.machine()}.")
    lines.append("")

    # ---- 1. Success summary ----
    lines.append("## 1. Success / failure summary")
    lines.append("")
    counts = {s: {"optimal": 0, "acceptable": 0, "iter_limit": 0,
                  "infeasible": 0, "restoration_failed": 0,
                  "other_fail": 0, "missing": 0, "timeout": 0}
              for s in SOLVERS}
    for rec in records:
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            st = (e.get("exit_status") or e.get("status") or "missing").lower()
            if "optimal" in st:
                counts[s]["optimal"] += 1
            elif "acceptable" in st:
                counts[s]["acceptable"] += 1
            elif "iteration" in st or "iter_limit" in st:
                counts[s]["iter_limit"] += 1
            elif "infeasible" in st:
                counts[s]["infeasible"] += 1
            elif "restoration" in st:
                counts[s]["restoration_failed"] += 1
            elif "timeout" in st:
                counts[s]["timeout"] += 1
            elif "missing" in st:
                counts[s]["missing"] += 1
            else:
                counts[s]["other_fail"] += 1
    lines.append("| Solver | Optimal | Acceptable | Iter-limit | Infeasible | "
                 "Restoration-failed | Timeout | Other |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SOLVERS:
        c = counts[s]
        lines.append(f"| {s} | {c['optimal']} | {c['acceptable']} | "
                     f"{c['iter_limit']} | {c['infeasible']} | "
                     f"{c['restoration_failed']} | {c['timeout']} | "
                     f"{c['other_fail'] + c['missing']} |")
    lines.append("")

    # ---- 2. Iteration counts ----
    lines.append("## 2. IPM iteration counts (where they differ)")
    lines.append("")
    lines.append("Same NLP → same KKT system at each iterate → expect "
                 "identical iteration counts when each solver delivers a "
                 "comparable Newton step. Discrepancies indicate one of the "
                 "linear solvers is giving subtly different residuals "
                 "(scaling, inertia, refinement).")
    lines.append("")
    diff_rows = []
    same_count = 0
    for rec in records:
        iters = {s: rec["solvers"].get(s, {}).get("iterations") for s in SOLVERS}
        vals = [v for v in iters.values() if v is not None]
        if len(vals) >= 2 and len(set(vals)) > 1:
            diff_rows.append((rec["problem"], rec["N"], iters))
        elif vals and len(set(vals)) == 1:
            same_count += 1
    lines.append(f"**{same_count}/{len(records)}** problems have identical "
                 "iteration count across solvers.")
    lines.append("")
    if diff_rows:
        lines.append("| Problem | N | MUMPS | MA57 | feral |")
        lines.append("|---|---:|---:|---:|---:|")
        for problem, N, iters in diff_rows:
            row = f"| {problem} | {N} |"
            for s in SOLVERS:
                v = iters[s]
                row += f" {v if v is not None else '-'} |"
            lines.append(row)
    else:
        lines.append("(All problems agree.)")
    lines.append("")

    # ---- 3. Total time summary ----
    lines.append("## 3. Total Ipopt time (geomean, only over problems "
                 "where ALL solvers reached *Optimal Solution Found*)")
    lines.append("")
    triple_optimal = []
    for rec in records:
        ok = True
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            if "optimal" not in (e.get("exit_status") or "").lower():
                ok = False
        if ok:
            triple_optimal.append(rec)
    lines.append(f"({len(triple_optimal)}/{len(records)} problems converge "
                 "on all three solvers — only these contribute to the "
                 "geometric mean.)")
    lines.append("")
    gmeans = {}
    for s in SOLVERS:
        gmeans[s] = geomean([r["solvers"][s].get("total_seconds")
                             for r in triple_optimal])
    lines.append("| Solver | geomean Ipopt seconds |")
    lines.append("|---|---:|")
    for s in SOLVERS:
        lines.append(f"| {s} | {fmt_secs(gmeans[s])} |")
    lines.append("")
    if gmeans["mumps"] and gmeans["feral"]:
        lines.append(f"Speedup feral vs MUMPS: **{gmeans['mumps']/gmeans['feral']:.2f}×** "
                     "(geomean over triple-optimal subset).")
        lines.append("")
    if gmeans["ma57"] and gmeans["feral"]:
        lines.append(f"Speedup feral vs MA57: **{gmeans['ma57']/gmeans['feral']:.2f}×** "
                     "(geomean over triple-optimal subset).")
        lines.append("")

    # ---- 4. Per-problem timing ----
    lines.append("## 4. Per-problem detail")
    lines.append("")
    lines.append("Status legend: ✓ = Optimal Solution Found; ~ = Solved to "
                 "Acceptable Level; ✗ = other (see exit_status column).")
    lines.append("")
    lines.append("| Problem | N | n_vars | MUMPS iter / sec / status | "
                 "MA57 iter / sec / status | feral iter / sec / status | "
                 "Objective |")
    lines.append("|---|---:|---:|---|---|---|---|")
    for rec in records:
        problem = rec["problem"]
        N = rec["N"]
        any_solver = next(iter(rec["solvers"].values()), {})
        nv = any_solver.get("n_vars", "?")
        # Pull first available objective for the reference column.
        obj = None
        for s in SOLVERS:
            o = rec["solvers"].get(s, {}).get("obj_unscaled")
            if o is not None and math.isfinite(o):
                obj = o
                break
        cells = []
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            it = e.get("iterations", "-")
            sec = fmt_secs(e.get("total_seconds"))
            st_raw = (e.get("exit_status") or e.get("status") or "?").strip()
            st_low = st_raw.lower()
            if "optimal" in st_low:
                tag = "✓"
            elif "acceptable" in st_low:
                tag = "~"
            elif "iter" in st_low:
                tag = "iter_lim"
            elif "infeasible" in st_low:
                tag = "infeas"
            elif "restoration" in st_low:
                tag = "rest_fail"
            elif "timeout" in st_low:
                tag = "timeout"
            else:
                tag = st_raw[:18]
            cells.append(f"{it} / {sec} / {tag}")
        lines.append(f"| {problem} | {N} | {nv} | "
                     f"{cells[0]} | {cells[1]} | {cells[2]} | "
                     f"{fmt_obj(obj)} |")
    lines.append("")

    # ---- 5. Cross-solver objective agreement ----
    lines.append("## 5. Objective-value cross-check")
    lines.append("")
    lines.append("For problems where all three solvers reach Optimal, the "
                 "final objective should agree to several digits. Large "
                 "spreads indicate one solver found a different local "
                 "minimum or terminated early.")
    lines.append("")
    spreads = []
    for rec in triple_optimal:
        objs = [rec["solvers"][s].get("obj_unscaled") for s in SOLVERS]
        objs = [o for o in objs if o is not None and math.isfinite(o)]
        if len(objs) >= 2:
            base = max(abs(o) for o in objs)
            if base > 0:
                spread = (max(objs) - min(objs)) / base
                spreads.append((rec["problem"], rec["N"], spread, objs))
    spreads.sort(key=lambda t: -t[2])
    lines.append("Top 10 by relative objective spread:")
    lines.append("")
    lines.append("| Problem | N | rel spread | MUMPS obj | MA57 obj | feral obj |")
    lines.append("|---|---:|---:|---|---|---|")
    for problem, N, spread, _ in spreads[:10]:
        recmap = next(r for r in triple_optimal
                      if r["problem"] == problem and r["N"] == N)
        objs = [recmap["solvers"][s].get("obj_unscaled") for s in SOLVERS]
        lines.append(f"| {problem} | {N} | {spread:.2e} | "
                     f"{fmt_obj(objs[0])} | {fmt_obj(objs[1])} | {fmt_obj(objs[2])} |")
    lines.append("")

    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text)} bytes, {len(records)} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
