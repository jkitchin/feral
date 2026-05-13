#!/usr/bin/env python3
"""Run three Ipopt binaries (linked to MUMPS, MA57, feral) on the
ScalableProblems NLPs from Ipopt and capture per-problem metrics.

Each binary is invoked as `solve_problem <name> <N>`. We set
`linear_solver` via an `ipopt.opt` file written into a per-run cwd
so the three runs cannot bleed state into each other.

Per-(problem,solver) we capture iterations, total Ipopt seconds,
objective value, NLP error, exit status. Sidecars land at
`out/<solver>/<problem>__<N>.txt` (raw Ipopt log) plus a parsed
`.json`. Use `aggregate.py` to combine into `comparison.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

BINARIES = {
    "mumps": ROOT / "ref/Ipopt/build-mumps/examples/ScalableProblems/solve_problem",
    "ma57":  ROOT / "ref/Ipopt/build-ma57/examples/ScalableProblems/solve_problem",
    "feral": ROOT / "ref/Ipopt/build-feral/examples/ScalableProblems/solve_problem",
}


def parse_log(text: str) -> dict:
    """Pull the metrics we care about out of an Ipopt log."""
    d: dict = {"exit_status": "unknown"}

    m = re.search(r"^EXIT: (.+?)\.?\s*$", text, re.MULTILINE)
    if m:
        d["exit_status"] = m.group(1).strip()

    m = re.search(r"^Number of Iterations\.+:\s*(\d+)", text, re.MULTILINE)
    if m:
        d["iterations"] = int(m.group(1))

    m = re.search(r"^Total seconds in IPOPT\s+=\s+([0-9.eE+-]+)", text, re.MULTILINE)
    if m:
        d["total_seconds"] = float(m.group(1))

    # "Objective...............:   <scaled>    <unscaled>"
    m = re.search(r"^Objective\.+:\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)", text, re.MULTILINE)
    if m:
        d["obj_scaled"] = float(m.group(1))
        d["obj_unscaled"] = float(m.group(2))

    m = re.search(r"^Overall NLP error\.+:\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)", text, re.MULTILINE)
    if m:
        d["nlp_error_scaled"] = float(m.group(1))
        d["nlp_error_unscaled"] = float(m.group(2))

    m = re.search(r"^Dual infeasibility\.+:\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)", text, re.MULTILINE)
    if m:
        d["dual_inf"] = float(m.group(2))

    m = re.search(r"^Constraint violation\.+:\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)", text, re.MULTILINE)
    if m:
        d["constr_viol"] = float(m.group(2))

    # Number of {function,gradient,Jacobian,Hessian} evaluations
    for label, key in [
        ("objective function", "n_f"),
        ("objective gradient", "n_g"),
        ("equality constraint", "n_eq"),
        ("equality constraint Jacobian", "n_jac"),
        ("Lagrangian Hessian", "n_hess"),
    ]:
        m = re.search(rf"Number of {label} evaluations\s+=\s+(\d+)", text)
        if m and key not in d:
            d[key] = int(m.group(1))

    # Variable / constraint counts (from problem print)
    m = re.search(r"Total number of variables\.+:\s+(\d+)", text)
    if m:
        d["n_vars"] = int(m.group(1))
    m = re.search(r"Total number of equality constraints\.+:\s+(\d+)", text)
    if m:
        d["n_eq_cons"] = int(m.group(1))
    m = re.search(r"Total number of inequality constraints\.+:\s+(\d+)", text)
    if m:
        d["n_ineq_cons"] = int(m.group(1))

    return d


def run_one(solver: str, problem: str, N: int, time_limit_s: int) -> dict:
    binary = BINARIES[solver]
    if not binary.exists():
        return {"status": "missing_binary"}
    workdir = HERE / "logs" / solver
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "ipopt.opt").write_text(f"linear_solver {solver}\nprint_level 5\n")

    log_path = workdir / f"{problem}__N{N}.log"
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [str(binary), problem, str(N)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=time_limit_s,
        )
        elapsed = time.monotonic() - t0
        log_path.write_text(proc.stdout + "\n--STDERR--\n" + proc.stderr)
        parsed = parse_log(proc.stdout)
        parsed["wall_seconds"] = elapsed
        parsed["returncode"] = proc.returncode
        if "exit_status" not in parsed or parsed["exit_status"] == "unknown":
            parsed["status"] = "crashed_or_unparseable"
        else:
            parsed["status"] = "ok"
        return parsed
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT")
        return {"status": "timeout", "wall_seconds": time.monotonic() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(HERE / "problems.tsv"))
    ap.add_argument("--solvers", default="mumps,ma57,feral")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=120,
                    help="seconds per (solver, problem)")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    rows = []
    with open(args.problems) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            rows.append(dict(zip(header, parts)))
    if args.limit is not None:
        rows = rows[: args.limit]

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    print(f"problems={len(rows)} solvers={solvers}", flush=True)

    results = []
    for row in rows:
        problem = row["problem"]
        N = int(row["N"])
        rec = {"problem": problem, "N": N, "solvers": {}}
        for s in solvers:
            print(f"  {s:5s}  {problem} N={N} ...", end="", flush=True)
            r = run_one(s, problem, N, args.time_limit)
            iters = r.get("iterations", "?")
            secs = r.get("total_seconds", "?")
            print(f"  iters={iters} sec={secs} status={r.get('status','?')}",
                  flush=True)
            rec["solvers"][s] = r
        results.append(rec)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out} ({len(results)} problems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
