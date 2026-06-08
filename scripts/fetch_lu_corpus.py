#!/usr/bin/env python3
"""Fetch real-world square *unsymmetric* matrices from the SuiteSparse Matrix
Collection for the LU validation harness (issue #81, epic Phase 7).

Downloads a curated set of square, real, structurally-unsymmetric matrices —
including circuit-simulation, chemical-process, economic, and (crucially)
``bp_*`` *simplex basis* matrices — as Matrix Market ``.mtx`` files into
``data/matrices/lu-corpus/`` (gitignored). Then run the Rust harness:

    cargo run -p feral-diagnostics --release --bin lu_reference

Requires ``ssgetpy`` (``pip install ssgetpy``). Network access to
``sparse.tamu.edu`` is needed; if the environment blocks it, download the
matrices manually and drop the ``.mtx`` files into the corpus dir.

The ``bp_*`` group is the most relevant: those ARE basis matrices extracted
from a linear program, i.e. exactly the kind of matrix this engine factors.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

# Curated square unsymmetric matrices. Mix of sizes and domains; the bp_*
# entries are real LP simplex bases (Harwell-Boeing "basis problem" set).
CURATED = [
    # --- LP simplex bases (the real target) ---
    "bp_200", "bp_600", "bp_1000", "bp_1400", "bp_1600",
    # --- chemical process (square, very unsymmetric) ---
    "west0067", "west0479", "west2021", "fs_183_1", "fs_541_1",
    # --- circuit simulation ---
    "add20", "add32", "memplus",
    # --- oil reservoir / fluid (unsymmetric square) ---
    "sherman1", "sherman2", "sherman3", "sherman5",
    "saylr1", "saylr3", "saylr4", "pores_2", "pores_3",
    "lns_131", "lns_511", "lnsp_511",
    # --- economic / power flow ---
    "orani678", "mahindas", "gre_115", "gre_343", "gre_1107",
    "gemat11", "gemat12", "watt_1", "watt_2",
]

CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "matrices", "lu-corpus",
)
RAW = os.path.join(CORPUS, "_raw")


def main() -> int:
    try:
        import ssgetpy
    except ImportError:
        print("ssgetpy not installed. Run: pip install ssgetpy", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — ssgetpy refreshes its index on import
        print(
            f"ssgetpy could not initialize ({e!r}).\n"
            "This usually means no network access to sparse.tamu.edu. Run this "
            "script where outbound HTTPS is allowed, or download the .mtx files "
            "manually into data/matrices/lu-corpus/.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(RAW, exist_ok=True)
    fetched, skipped = 0, 0
    for name in CURATED:
        dest = os.path.join(CORPUS, f"{name}.mtx")
        if os.path.exists(dest):
            print(f"  have   {name}")
            fetched += 1
            continue
        try:
            results = ssgetpy.search(name=name, limit=5)
        except Exception as e:  # noqa: BLE001 — network/lookup best-effort
            print(f"  search-fail {name}: {e}", file=sys.stderr)
            skipped += 1
            continue
        # Prefer an exact square match.
        match = None
        for m in results:
            if m.name == name and m.rows == m.cols:
                match = m
                break
        if match is None:
            match = next((m for m in results if m.rows == m.cols), None)
        if match is None:
            print(f"  no-square {name}")
            skipped += 1
            continue
        try:
            match.download(format="MM", destpath=RAW, extract=True)
        except Exception as e:  # noqa: BLE001
            print(f"  download-fail {name}: {e}", file=sys.stderr)
            skipped += 1
            continue
        # ssgetpy extracts to RAW/<name>/<name>.mtx — find and flatten it.
        hits = glob.glob(os.path.join(RAW, "**", f"{match.name}.mtx"), recursive=True)
        if not hits:
            print(f"  no-mtx {name}")
            skipped += 1
            continue
        shutil.copyfile(hits[0], dest)
        print(f"  fetched {name}  ({match.rows}x{match.cols}, nnz={match.nnz})")
        fetched += 1

    print(f"\n{fetched} matrices in {CORPUS} ({skipped} skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
