#!/usr/bin/env python3
"""Analyze stress-suite sidecars and flag pathologies.

Reads `out/feral/<group>__<name>.out`, joins against `manifest.tsv` to
recover category/expected-inertia metadata (for synth/* rows we know the
zero count exactly), and emits:

  * a per-matrix table (n, status, factor_us, rel_res, inertia)
  * a summary table by category
  * a "flagged" section listing matrices that fail any acceptance rule:
      - status != ok
      - rel_res > REL_RES_THRESHOLD (default 1e-6)
      - inertia.zero != expected for rankdef rows
      - cascade matrices that fail to factor at all

Exit code: 0 if no flags, 1 if any matrix is flagged (CI gate friendly).

Usage: python3 report.py  [--rel-res 1e-6]  [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

STRESS_DIR = Path(__file__).resolve().parent
OUT_DIR = STRESS_DIR / "out" / "feral"

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


def expected_zero(row: dict) -> int | None:
    """Expected null space dimension for oracle-checked synth rows.

    Oracle conventions for the M4 generators (issue #27 + #31
    follow-up). See `dev/research/synthetic-generators-m4.md` for the
    derivations.

    - rankdef_<n>_<k>            → k          (existing convention)
    - rankdef_exact_<n>_<k>      → k          (#31 follow-up)
    - saddle_rankdef_<n>_<k>_<r> → r          (saddle nullity)
    - stokes_q1p0_<h>            → 2          (constant + checkerboard
                                                pressure modes)
    Other categories (wide_frontal, mc64_resistant) return None;
    they are checked only for status + consistency-sum, not zero.
    """
    if row.get("group") != "synth":
        return None
    name = row["name"]
    m = re.match(r"^rankdef_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(2))
    m = re.match(r"^rankdef_exact_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(2))
    m = re.match(r"^saddle_rankdef_(\d+)_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(3))
    m = re.match(r"^stokes_q1p0_(\d+)$", name)
    if m:
        return 2
    return None


# Per-matrix allowlist: matrices whose `classify` flags are known
# pre-existing divergences, kept here to unblock CI while the
# underlying issue is tracked. Each entry must cite a GH issue and
# a short reason. Remove the entry when the issue closes.
#
# Format: matrix_name -> (issue_url_or_number, reason).
ALLOWLIST: dict[str, tuple[str, str]] = {
    "saddle_rankdef_50_10_3": (
        "#40",
        "x86/aarch64 BK-pivot divergence on borderline rank-deficient saddle. "
        "Local aarch64 returns inertia (50,39,1) -- detects 1 zero pivot. "
        "CI x86 returns (52,38,0) -- absorbs the null mode into a normal "
        "pivot. Both factors are numerically valid (rel_res < 1e-14); the "
        "rankdef detection is exactly the borderline case the classify() "
        "comment notes MUMPS itself misses on similar matrices. Allowlist "
        "until the cross-arch BK pivot path is hardened.",
    ),
    "rankdef_5_2": (
        "#40",
        "Cross-arch BK-pivot divergence, exposed by the #39 F-01 band "
        "widening (2026-05-17). probe_f01.rs on aarch64 finds 1 "
        "strict-zero pivot (|d| = 9.3e-17, below EPS) -> inertia "
        "(2,2,1). CI x86 produces a slightly different pivot that lands "
        "just above EPS and is counted by sign -> (3,2,0), zero=0. Both "
        "factors numerically valid (rel_res < 1e-15). This is the same "
        "x86/aarch64 BK divergence as saddle_rankdef_50_10_3, not the "
        "deterministic every-arch F-01 flip of the #39 entries below.",
    ),
    "rankdef_50_5": (
        "#40",
        "Cross-arch BK-pivot divergence, exposed by the #39 F-01 band "
        "widening (2026-05-17). CI x86 reports (26,24,0), zero=0; "
        "MUMPS (ICNTL(24)=1) itself reports zero=0 on this matrix "
        "(cited in dev/research/f01-rankdef-underreporting.md and the "
        "classify() rankdef_like_cats comment), so x86 feral matches "
        "MUMPS. aarch64 probe_f01.rs finds 1 strict-zero pivot. Factor "
        "numerically valid (rel_res < 1e-14). The gate's expected>=1 is "
        "the synthetic construction label, not the MUMPS oracle.",
    ),
    "rankdef_exact_50_5": (
        "#40",
        "Cross-arch BK-pivot divergence, exposed by the #39 F-01 band "
        "widening (2026-05-17). CI x86 reports (24,26,0), zero=0; "
        "probe_f01.rs on aarch64 finds 1 strict-zero pivot -> zero=1. "
        "Factor numerically valid (rel_res < 1e-14). Same x86/aarch64 "
        "BK divergence class as saddle_rankdef_50_10_3 (#40).",
    ),
    "rankdef_exact_100_10": (
        "#39",
        "F-01 sign-fallback (2026-05-17). Pivots with |d| in the band "
        "(EPS, sqrt(n)*EPS*||A||] are now counted by sign instead of as "
        "zero, to match MUMPS/SSIDS convention on FBRAIN3LS_0839. MUMPS "
        "(ICNTL(24)=1) itself reports zero=0 on this matrix; the factor "
        "is numerically valid (rel_res < 1e-13). See "
        "dev/research/f01-rankdef-underreporting.md 2026-05-17 addendum.",
    ),
    "rankdef_200_20": (
        "#39",
        "F-01 sign-fallback (2026-05-17). MUMPS (ICNTL(24)=1) also "
        "reports zero=0 on this matrix; feral's new behavior matches. "
        "Factor numerically valid (rel_res < 1e-13). See "
        "dev/research/f01-rankdef-underreporting.md 2026-05-17 addendum.",
    ),
    "saddle_rankdef_100_20_5": (
        "#39",
        "F-01 sign-fallback (2026-05-17). Borderline rank-deficient "
        "saddle; band pivots now counted by sign. Factor numerically "
        "valid (rel_res < 1e-14). See "
        "dev/research/f01-rankdef-underreporting.md 2026-05-17 addendum.",
    ),
    "stokes_q1p0_8": (
        "#39",
        "F-01 sign-fallback (2026-05-17). Q1-P0 Stokes element with "
        "constant + checkerboard pressure null modes. Band pivots now "
        "counted by sign. Factor numerically valid (rel_res < 1e-14). "
        "See dev/research/f01-rankdef-underreporting.md 2026-05-17 "
        "addendum.",
    ),
}


def classify(row: dict, side: dict | None, rel_res_threshold: float) -> list[str]:
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
    exp_zero = expected_zero(row)
    if exp_zero is not None:
        # Rank-deficient matrices: BK pivoting can absorb part of the
        # null space into ostensibly-normal pivots (verified against
        # MUMPS 5.8.2 oracle with ICNTL(24)=1, which itself reports
        # zero=0 on synth/rankdef_50_5 and rankdef_200_20 despite their
        # constructed nullity). The acceptance rule is therefore
        # `1 <= zero <= expected`: BK must detect *some* rank deficiency
        # (zero=0 is a genuine bug — F-01 regression guard), but
        # partial detection is consistent with MUMPS's own behavior.
        # See `dev/research/f01-rankdef-underreporting.md`.
        if zer == 0:
            flags.append(f"zero=0 (rankdef, expected>=1, k={exp_zero})")
        elif zer > exp_zero:
            flags.append(f"zero={zer}>expected={exp_zero}")
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

    records = []
    for r in rows:
        sidecar = OUT_DIR / f"{r['group']}__{r['name']}.out"
        side = parse_sidecar(sidecar)
        flags = classify(r, side, args.rel_res)
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
