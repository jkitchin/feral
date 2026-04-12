#!/usr/bin/env python3
"""Compute multi-source consensus verdicts for the feral KKT corpus.

For each matrix in `data/matrices/kkt`, reads up to four sidecars:

  <id>.json          ipopt sidecar (rmumps inertia)
  <id>.feral.json    feral output (TBD — produced by `cargo run --bin bench --emit-sidecars`)
  <id>.mumps.json    canonical Fortran MUMPS 5.8.2
  <id>.ssids.json    canonical SPRAL/SSIDS

Applies the consensus rules from dev/plans/phase-1b-consensus-exit.md:

  Inertia consensus:
    - Strong: at least 3 of N inertia triples are equal.
    - Weak:   2 of N agree, others differ by ≤(±1, ∓1, ∓1) component-wise.
    - None:   no agreement.

  Residual consensus:
    A matrix is "consensus solvable" if at least 3 of N solvers produce
    a passing residual on the same RHS, where passing = r ≤ n·eps·1e6.

  Per-matrix verdict:
    Definitive            strong inertia + solvable
    Borderline            weak inertia + solvable
    NumericallyIntractable strong/weak inertia + not solvable
    Excluded              none + anything

For each matrix writes a `<id>.verdict.json` next to the `.mtx` file with
the consensus inertia (when defined), the verdict, and which solvers
dissented from the consensus on inertia and residual.

Usage:
    python3 compute_consensus.py data/matrices/kkt
    python3 compute_consensus.py data/matrices/kkt --report report.txt
    python3 compute_consensus.py data/matrices/kkt --verdict-summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

EPS = 2.220446049250313e-16


@dataclass
class SolverResult:
    name: str
    inertia: tuple[int, int, int] | None  # (positive, negative, zero)
    residual: float | None
    n: int | None
    factor_us: int | None = None
    solve_us: int | None = None
    available: bool = True

    @classmethod
    def missing(cls, name: str) -> "SolverResult":
        return cls(name=name, inertia=None, residual=None, n=None, available=False)


def load_rmumps(json_path: Path) -> SolverResult:
    if not json_path.exists():
        return SolverResult.missing("rmumps")
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return SolverResult.missing("rmumps")
    inertia = data.get("inertia")
    if not isinstance(inertia, dict):
        return SolverResult.missing("rmumps")
    pos = inertia.get("positive")
    neg = inertia.get("negative")
    zero = inertia.get("zero")
    n = inertia.get("n") or (pos + neg + zero if pos is not None and neg is not None and zero is not None else None)
    if pos is None or neg is None or zero is None:
        return SolverResult.missing("rmumps")
    # rmumps sidecars don't store a residual; use sentinel
    return SolverResult(
        name="rmumps",
        inertia=(int(pos), int(neg), int(zero)),
        residual=None,
        n=int(n) if n is not None else None,
    )


def load_canonical(json_path: Path, label: str) -> SolverResult:
    if not json_path.exists():
        return SolverResult.missing(label)
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return SolverResult.missing(label)
    if data.get("factorization_status") != "ok":
        return SolverResult.missing(label)
    inertia = data.get("inertia")
    if not isinstance(inertia, dict):
        return SolverResult.missing(label)
    pos = inertia.get("positive")
    neg = inertia.get("negative")
    zero = inertia.get("zero")
    if pos is None or neg is None or zero is None:
        return SolverResult.missing(label)
    res = data.get("residual_2norm_relative")
    return SolverResult(
        name=label,
        inertia=(int(pos), int(neg), int(zero)),
        residual=float(res) if res is not None and math.isfinite(float(res)) else None,
        n=int(data.get("n", 0)) or None,
        factor_us=int(data.get("factor_us", 0)) or None,
        solve_us=int(data.get("solve_us", 0)) or None,
    )


def residual_passes(n: int, residual: float | None) -> bool:
    """Mirrors the feral bench tolerance: r ≤ n·eps·1e6."""
    if residual is None or not math.isfinite(residual):
        return False
    tol = n * EPS * 1e6
    return residual <= tol


# Voting oracles: these are the solvers whose inertia and residual count
# toward the Definitive / Borderline / NumericallyIntractable / Excluded
# classification. rmumps is deliberately NOT in this set — see
# dev/decisions.md (2026-04-12, "rmumps deprecated as a validation oracle").
VOTING_ORACLES = {"feral", "mumps", "ssids"}


def inertia_consensus(
    results: list[SolverResult],
) -> tuple[tuple[int, int, int] | None, str, list[str]]:
    """Return (consensus_inertia, agreement_level, dissenters).

    Only oracles in VOTING_ORACLES participate. rmumps is ignored here
    and reported separately as informational metadata.

    agreement_level ∈ {"strong", "weak", "none"}.
    dissenters lists solver names whose inertia disagrees with the consensus.
    """
    inertias = [
        (r.name, r.inertia)
        for r in results
        if r.available and r.inertia is not None and r.name in VOTING_ORACLES
    ]
    if not inertias:
        return None, "none", []

    counts = Counter(inert for _, inert in inertias)
    most_common, most_count = counts.most_common(1)[0]
    n_total = len(inertias)

    # Strong: unanimous agreement among available voting oracles.
    # (With three voting oracles, unanimous is 3/3; with fewer — e.g.
    # MUMPS failed the workspace check — we require whatever is
    # available to all agree.)
    if most_count == n_total:
        consensus = most_common
        return consensus, "strong", []

    # Weak: majority agrees AND every dissenter is within ±1 per component.
    # For N=3 this is 2/3 with the third differing by ≤1. For N=2 there is
    # no "weak" state — either unanimous or no consensus.
    if most_count >= 2 and n_total >= 3:
        consensus = most_common
        ok = True
        dissenters = []
        for name, inert in inertias:
            if inert == consensus:
                continue
            diff = tuple(abs(a - b) for a, b in zip(inert, consensus))
            if all(d <= 1 for d in diff):
                dissenters.append(name)
            else:
                ok = False
                dissenters.append(name)
        if ok:
            return consensus, "weak", dissenters

    # Otherwise no consensus
    return None, "none", [name for name, _ in inertias]


def residual_consensus(results: list[SolverResult]) -> tuple[bool, list[str]]:
    """Return (consensus_solvable, residual_dissenters).

    Only VOTING_ORACLES count toward the solvable verdict. A matrix is
    "consensus solvable" when a majority of the available voting oracles
    produce a residual below the feral bench tolerance. For N=3 this is
    ≥2/3, consistent with the prior 3-of-4 rule's spirit (most oracles
    can solve it, not all of them).
    """
    have_residual = [
        r
        for r in results
        if r.available
        and r.residual is not None
        and r.n is not None
        and r.name in VOTING_ORACLES
    ]
    if not have_residual:
        return False, []
    passes = [residual_passes(r.n, r.residual) for r in have_residual]
    n_pass = sum(passes)
    n_total = len(have_residual)
    # Strict majority for N≥3 (e.g. 2/3), unanimous for N≤2.
    solvable = n_pass > n_total // 2 if n_total >= 3 else n_pass == n_total
    dissenters = [r.name for r, p in zip(have_residual, passes) if not p]
    return solvable, dissenters


def verdict_for(inertia_level: str, solvable: bool) -> str:
    if inertia_level == "strong" and solvable:
        return "definitive"
    if inertia_level == "weak" and solvable:
        return "borderline"
    if inertia_level in ("strong", "weak") and not solvable:
        return "numerically_intractable"
    return "excluded"


def compute_one(mtx_path: Path) -> dict | None:
    """Read all available oracles and compute the verdict for one matrix."""
    base = mtx_path.with_suffix("")
    rmumps = load_rmumps(mtx_path.with_suffix(".json"))
    feral = load_canonical(mtx_path.with_suffix(".feral.json"), "feral")
    mumps = load_canonical(mtx_path.with_suffix(".mumps.json"), "mumps")
    ssids = load_canonical(mtx_path.with_suffix(".ssids.json"), "ssids")

    results = [rmumps, feral, mumps, ssids]
    n_avail = sum(1 for r in results if r.available)
    if n_avail == 0:
        return None

    consensus_inertia, level, inertia_dissenters = inertia_consensus(results)
    solvable, residual_dissenters = residual_consensus(results)
    verdict = verdict_for(level, solvable)

    if feral.available:
        feral_match_inertia = (
            consensus_inertia is not None and feral.inertia == consensus_inertia
        )
        feral_residual_pass = (
            feral.n is not None and residual_passes(feral.n, feral.residual)
        )
    else:
        # Feral hasn't run yet; mark as unknown so the aggregator skips
        feral_match_inertia = None
        feral_residual_pass = None

    return {
        "matrix": mtx_path.stem,
        "n_oracles": n_avail,
        "consensus_inertia": (
            {
                "positive": consensus_inertia[0],
                "negative": consensus_inertia[1],
                "zero": consensus_inertia[2],
            }
            if consensus_inertia is not None
            else None
        ),
        "inertia_agreement": level,
        "inertia_dissenters": inertia_dissenters,
        "consensus_solvable": solvable,
        "residual_dissenters": residual_dissenters,
        "verdict": verdict,
        "feral_match_inertia": feral_match_inertia,
        "feral_residual_pass": feral_residual_pass,
        "oracles": {
            r.name: (
                {
                    "inertia": list(r.inertia) if r.inertia is not None else None,
                    "residual": r.residual,
                }
                if r.available
                else None
            )
            for r in results
        },
    }


def aggregate(verdicts: Iterable[dict]) -> dict:
    counts: Counter[str] = Counter()
    feral_fail_definitive: list[str] = []
    feral_fail_borderline: list[str] = []
    pairwise: dict[tuple[str, str], int] = {}
    pairs = [
        ("feral", "rmumps"),
        ("feral", "mumps"),
        ("feral", "ssids"),
        ("rmumps", "mumps"),
        ("rmumps", "ssids"),
        ("mumps", "ssids"),
    ]
    pair_total: dict[tuple[str, str], int] = {p: 0 for p in pairs}
    pair_match: dict[tuple[str, str], int] = {p: 0 for p in pairs}

    n_total = 0
    for v in verdicts:
        n_total += 1
        counts[v["verdict"]] += 1
        oracles = v["oracles"]
        if v["verdict"] == "definitive":
            # Only count as a failure if feral actually has data and disagrees
            if v["feral_match_inertia"] is False or v["feral_residual_pass"] is False:
                feral_fail_definitive.append(v["matrix"])
        elif v["verdict"] == "borderline":
            if v["feral_residual_pass"] is False:
                feral_fail_borderline.append(v["matrix"])

        # Skip rows where feral hasn't run yet
        if v["feral_match_inertia"] is None:
            pass  # not counted

        for a, b in pairs:
            ra = oracles.get(a)
            rb = oracles.get(b)
            if ra is None or rb is None:
                continue
            ia = tuple(ra["inertia"]) if ra["inertia"] else None
            ib = tuple(rb["inertia"]) if rb["inertia"] else None
            if ia is None or ib is None:
                continue
            pair_total[(a, b)] += 1
            if ia == ib:
                pair_match[(a, b)] += 1

    return {
        "n_total": n_total,
        "verdict_counts": dict(counts),
        "feral_fail_definitive": feral_fail_definitive,
        "feral_fail_borderline": feral_fail_borderline,
        "pairwise_inertia_agreement": {
            f"{a} vs {b}": {
                "matches": pair_match[(a, b)],
                "total": pair_total[(a, b)],
                "pct": (100.0 * pair_match[(a, b)] / pair_total[(a, b)]) if pair_total[(a, b)] else None,
            }
            for a, b in pairs
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="data/matrices/kkt root")
    ap.add_argument("--write-verdicts", action="store_true",
                    help="write per-matrix .verdict.json files alongside .mtx")
    ap.add_argument("--report", type=Path, default=None,
                    help="write aggregate summary to this file (default: stdout)")
    args = ap.parse_args()

    matrices = sorted(args.root.rglob("*.mtx"))
    if not matrices:
        print(f"no .mtx files under {args.root}", file=sys.stderr)
        return 1

    print(f"computing consensus for {len(matrices)} matrices...", file=sys.stderr)

    verdicts: list[dict] = []
    for mtx in matrices:
        v = compute_one(mtx)
        if v is None:
            continue
        verdicts.append(v)
        if args.write_verdicts:
            out = mtx.with_suffix(".verdict.json")
            out.write_text(json.dumps(v) + "\n")

    summary = aggregate(verdicts)

    out_lines = []
    out_lines.append("=== Consensus summary ===")
    out_lines.append(f"matrices analyzed: {summary['n_total']}")
    out_lines.append("voting oracles:    feral, mumps, ssids")
    out_lines.append("informational:     rmumps (reported but not counted)")
    out_lines.append("")
    out_lines.append("Verdict counts:")
    for verdict in ["definitive", "borderline", "numerically_intractable", "excluded"]:
        c = summary["verdict_counts"].get(verdict, 0)
        out_lines.append(f"  {verdict:24s} {c:6d}  ({100.0*c/summary['n_total']:.2f}%)")
    out_lines.append("")
    out_lines.append("Feral failures on Definitive matrices:")
    out_lines.append(f"  {len(summary['feral_fail_definitive'])} matrices")
    for name in summary['feral_fail_definitive'][:30]:
        out_lines.append(f"    {name}")
    if len(summary['feral_fail_definitive']) > 30:
        out_lines.append(f"    ... and {len(summary['feral_fail_definitive']) - 30} more")
    out_lines.append("")
    out_lines.append("Feral failures on Borderline matrices:")
    out_lines.append(f"  {len(summary['feral_fail_borderline'])} matrices")
    out_lines.append("")
    out_lines.append("Pairwise inertia agreement (voting oracles):")
    voting_pairs = [("feral", "mumps"), ("feral", "ssids"), ("mumps", "ssids")]
    for pair, stats in summary["pairwise_inertia_agreement"].items():
        pair_tuple = tuple(pair.split(" vs "))
        if pair_tuple not in voting_pairs:
            continue
        if stats["total"] == 0:
            out_lines.append(f"  {pair:20s} no overlap")
        else:
            out_lines.append(
                f"  {pair:20s} {stats['matches']:6d} / {stats['total']:6d}  "
                f"({stats['pct']:.2f}%)"
            )
    out_lines.append("")
    out_lines.append("Pairwise inertia agreement (rmumps — informational only):")
    for pair, stats in summary["pairwise_inertia_agreement"].items():
        pair_tuple = tuple(pair.split(" vs "))
        if "rmumps" not in pair_tuple:
            continue
        if stats["total"] == 0:
            out_lines.append(f"  {pair:20s} no overlap")
        else:
            out_lines.append(
                f"  {pair:20s} {stats['matches']:6d} / {stats['total']:6d}  "
                f"({stats['pct']:.2f}%)"
            )

    text = "\n".join(out_lines)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
