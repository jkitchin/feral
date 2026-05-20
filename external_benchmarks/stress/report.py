#!/usr/bin/env python3
"""Analyze stress-suite sidecars and flag pathologies.

Reads `out/feral/<group>__<name>.out`, joins against `manifest.tsv` for
category metadata and against `oracles.json` for the frozen solver
inertia of the rank-deficient synthetics, and emits:

  * a per-matrix table (n, status, factor_us, rel_res, inertia)
  * a summary table by category
  * a "flagged" section listing matrices that fail any acceptance rule:
      - status != ok
      - rel_res > REL_RES_THRESHOLD (default 1e-6)
      - inertia.zero matches no canonical oracle (MUMPS/SSIDS) for a
        rank-deficient synthetic — see oracles.json
      - inertia components do not sum to n

Exit code: 0 if no flags, 1 if any matrix is flagged (CI gate friendly).

Usage: python3 report.py  [--rel-res 1e-6]  [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

STRESS_DIR = Path(__file__).resolve().parent
OUT_DIR = STRESS_DIR / "out" / "feral"
SYNTH_DIR = STRESS_DIR / "matrices" / "synth"
ORACLES_JSON = STRESS_DIR / "oracles.json"

DEFAULT_REL_RES = 1e-6


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
    d: dict = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            d[parts[0]] = parts[1]
    return d


def fnum(d: dict, k: str) -> float | None:
    v = d.get(k)
    if v is None:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def inum(d: dict, k: str) -> int | None:
    v = d.get(k)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def load_oracles(path: Path) -> dict[str, dict]:
    """Load oracles.json → {matrix_name: oracle_entry}.

    Each oracle_entry carries n, constructed_k, mtx_sha256, and an
    `oracles` sub-dict of solver → inertia triple. See
    `dev/research/stress-consensus-oracle.md` and gen_oracles.py.

    Returns {} if the file is absent or unparseable; classify() then
    skips the consensus check, degrading to status + sum checks only
    (a missing oracle file must not crash the gate).
    """
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    matrices = doc.get("matrices", {})
    return matrices if isinstance(matrices, dict) else {}


# Per-matrix allowlist: matrices whose `classify` flags are known
# pre-existing divergences, kept here to unblock local runs while the
# underlying issue is tracked. Each entry must cite a GH issue and a
# short reason. Remove the entry when the issue closes.
#
# Currently empty. Issues #40 and #42 — the borderline rank-deficient
# synthetics rankdef_10_3 / rankdef_50_5 / rankdef_exact_50_5, whose
# `zero` inertia count diverged from the canonical oracles (no
# consensus match on #42, cross-arch divergence on #40) — were
# resolved by Option A: feral now counts every pivot by sign, so the
# `zero` component is structurally 0 under ForceAccept on every
# architecture. feral reports the SSIDS/MA57 consensus triple on all
# three matrices. See dev/decisions.md and
# dev/research/f01-rankdef-underreporting.md.
#
# Format: matrix_name -> (issue_url_or_number, reason).
ALLOWLIST: dict[str, tuple[str, str]] = {}


def classify(row: dict, side: dict | None, rel_res_threshold: float,
             oracles: dict[str, dict]) -> list[str]:
    """Return a list of flag strings for this matrix (empty = clean)."""
    flags: list[str] = []
    if side is None:
        flags.append("missing")
        return flags
    status = side.get("status", "missing")
    if status != "ok":
        reason = side.get("fail_reason", "no_reason")
        # For any category whose oracle expects nonzero null-space dim,
        # refusing to factor with NumericallyRankDeficient is *correct*
        # behavior — the matrix really is rank-deficient. Covers
        # rankdef, saddle_rankdef, stokes (constant+checkerboard
        # pressure modes).
        rankdef_like_cats = {"rankdef", "saddle_rankdef", "stokes"}
        if (row.get("category") in rankdef_like_cats
                and "RankDeficient" in reason):
            return flags
        flags.append(f"status={status}:{reason}")
        return flags
    rel = fnum(side, "rel_res")
    if rel is None:
        flags.append("rel_res=NaN")
    elif rel > rel_res_threshold:
        flags.append(f"rel_res={rel:.2e}>{rel_res_threshold:.0e}")
    pos = inum(side, "inertia_pos")
    neg = inum(side, "inertia_neg")
    zer = inum(side, "inertia_zero")
    if pos is None or neg is None or zer is None:
        flags.append("inertia=missing")
        return flags
    if pos + neg + zer != row["n"]:
        flags.append(f"inertia_sum={pos+neg+zer}!=n={row['n']}")
    # Consensus inertia check for the rank-deficient synthetics.
    # Borderline-singular matrices have no unique `zero`: canonical
    # solvers disagree by design (a near-null pivot counted by sign vs.
    # detected as null). CLAUDE.md defines "correct" as agreeing with
    # at least one of {MUMPS, SSIDS}, so the gate checks feral.zero
    # against the frozen per-matrix oracle in oracles.json rather than
    # the constructed null-space label. See
    # dev/research/stress-consensus-oracle.md.
    oracle = oracles.get(row["name"])
    if oracle is not None:
        mtx = SYNTH_DIR / f"{row['name']}.mtx"
        if mtx.exists():
            live_sha = hashlib.sha256(mtx.read_bytes()).hexdigest()
            if live_sha != oracle.get("mtx_sha256"):
                flags.append("oracle_stale (matrix bytes changed; "
                             "rerun gen_oracles.py)")
                return flags
        canonical = {
            name: o["zero"]
            for name, o in oracle.get("oracles", {}).items()
            if name in ("mumps", "ssids")
        }
        if canonical and zer not in canonical.values():
            pairs = ", ".join(f"{k}={v}"
                              for k, v in sorted(canonical.items()))
            flags.append(
                f"zero={zer} matches no canonical oracle ({pairs})")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(STRESS_DIR / "manifest.tsv"))
    ap.add_argument("--rel-res", type=float, default=DEFAULT_REL_RES,
                    help=f"max acceptable relative residual "
                         f"(default {DEFAULT_REL_RES:.0e})")
    ap.add_argument("--json", default=None,
                    help="write full results JSON to this path")
    args = ap.parse_args()

    rows = parse_manifest(Path(args.manifest))
    rows.sort(key=lambda r: (r.get("category", ""), r["n"], r["name"]))
    oracles = load_oracles(ORACLES_JSON)

    records = []
    for r in rows:
        sidecar = OUT_DIR / f"{r['group']}__{r['name']}.out"
        side = parse_sidecar(sidecar)
        flags = classify(r, side, args.rel_res, oracles)
        records.append({"row": r, "side": side, "flags": flags})

    print(f"{'category':<11} {'n':>7} {'name':<30} {'status':<8} "
          f"{'fac_us':>10} {'rel_res':>10} {'inertia':>16}  flags")
    print("-" * 110)
    n_ok = 0
    n_flag = 0
    n_missing = 0
    for rec in records:
        r = rec["row"]
        s = rec["side"] or {}
        status = s.get("status", "missing")
        fac = s.get("factor_us", "-")
        rel = fnum(s, "rel_res")
        rel_str = f"{rel:.2e}" if rel is not None else "-"
        pos = inum(s, "inertia_pos")
        neg = inum(s, "inertia_neg")
        zer = inum(s, "inertia_zero")
        inertia_str = (f"({pos},{neg},{zer})"
                       if pos is not None and neg is not None and zer is not None
                       else "-")
        is_allowlisted = (
            r["name"] in ALLOWLIST
            and rec["flags"]
            and rec["flags"] != ["missing"]
        )
        flag_str = ",".join(rec["flags"]) if rec["flags"] else ""
        if is_allowlisted:
            flag_str = f"ALLOWLISTED({ALLOWLIST[r['name']][0]}): {flag_str}"
        print(f"{r.get('category', '?'):<11} {r['n']:>7} {r['name'][:30]:<30} "
              f"{status:<8} {str(fac):>10} {rel_str:>10} "
              f"{inertia_str:>16}  {flag_str}")
        if rec["flags"] == ["missing"]:
            n_missing += 1
        elif is_allowlisted:
            n_ok += 1
        elif rec["flags"]:
            n_flag += 1
        elif status == "ok":
            n_ok += 1
        elif (rec["row"].get("category") == "rankdef"
              and not rec["flags"]):
            # Correct refusal to factor a rankdef matrix.
            n_ok += 1

    print("-" * 110)
    print(f"total {len(records)}: ok={n_ok}, flagged={n_flag}, "
          f"missing={n_missing}, "
          f"other={len(records) - n_ok - n_flag - n_missing}")
    if n_missing:
        print(f"  ({n_missing} matrices not downloaded — "
              f"run fetch.py to include them)")

    # Per-category roll-up
    cats: dict[str, dict[str, int]] = {}
    for rec in records:
        cat = rec["row"].get("category", "?")
        c = cats.setdefault(cat,
                            {"total": 0, "ok": 0, "flagged": 0, "missing": 0})
        c["total"] += 1
        name = rec["row"]["name"]
        is_allowlisted = (
            name in ALLOWLIST
            and rec["flags"]
            and rec["flags"] != ["missing"]
        )
        if rec["flags"] == ["missing"]:
            c["missing"] += 1
        elif is_allowlisted:
            c["ok"] += 1
        elif rec["flags"]:
            c["flagged"] += 1
        else:
            c["ok"] += 1
    print("\n=== by category ===")
    print(f"{'category':<14} {'total':>6} {'ok':>6} {'flag':>6} {'miss':>6}")
    for cat in sorted(cats):
        c = cats[cat]
        print(f"{cat:<14} {c['total']:>6} {c['ok']:>6} "
              f"{c['flagged']:>6} {c['missing']:>6}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\nwrote {args.json}")

    return 0 if n_flag == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
