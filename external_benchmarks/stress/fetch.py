#!/usr/bin/env python3
"""Fetch SuiteSparse matrices listed in manifest.tsv.

Downloads each (group, name) pair from the SuiteSparse Matrix Collection
Matrix-Market mirror to `external_benchmarks/stress/matrices/<group>/<name>.mtx`.

Skips downloads whose `.mtx` already exists. Synthetic rows (group=='synth')
are emitted by `synth.py` and skipped here.

Usage:
    python3 fetch.py                # all real SuiteSparse rows
    python3 fetch.py --limit 5      # smoke
    python3 fetch.py --group GHS_indef
    python3 fetch.py --force        # re-download even if present
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

STRESS_DIR = Path(__file__).resolve().parent
MATRICES_DIR = STRESS_DIR / "matrices"
REPO_ROOT = STRESS_DIR.parent.parent

BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"

# `group=cuter_kkt` rows are sourced from the local CUTEst KKT dump
# (produced by ripopt's `collect_kkt`). The `name` field encodes
# "<family>/<sample>" so the lookup path is
#     data/matrices/kkt/<family>/<sample>.mtx
# Used for FBRAIN3LS borderline samples (issue #29) which are not in
# SuiteSparse but are part of feral's regression coverage.
CUTER_KKT_ROOT = REPO_ROOT / "data" / "matrices" / "kkt"


def parse_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def target_path(group: str, name: str) -> Path:
    return MATRICES_DIR / group / f"{name}.mtx"


def fetch_one(group: str, name: str, force: bool) -> bool:
    tgt = target_path(group, name)
    if tgt.exists() and not force:
        return False
    tgt.parent.mkdir(parents=True, exist_ok=True)

    if group == "cuter_kkt":
        # name = "<family>__<sample>", path = data/matrices/kkt/<family>/<sample>.mtx
        parts = name.split("__", 1)
        if len(parts) != 2:
            print(f"    FAIL cuter_kkt: bad name {name!r} "
                  f"(expected <family>__<sample>)", flush=True)
            return False
        family, sample = parts
        src = CUTER_KKT_ROOT / family / f"{sample}.mtx"
        if not src.exists():
            print(f"    FAIL cuter_kkt: missing {src}", flush=True)
            return False
        shutil.copy(src, tgt)
        print(f"    ok -> {tgt.relative_to(REPO_ROOT)} (from local CUTEst dump)",
              flush=True)
        return True

    url = f"{BASE_URL}/{group}/{name}.tar.gz"
    print(f"  fetching {group}/{name} ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        tar_path = tdp / f"{name}.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                with tar_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
        except Exception as e:
            print(f"    FAIL download: {e}", flush=True)
            return False
        try:
            with tarfile.open(tar_path) as tf:
                tf.extractall(tdp)
        except Exception as e:
            print(f"    FAIL extract: {e}", flush=True)
            return False
        # SuiteSparse layout: <name>/<name>.mtx (+ optional b/x sidecars)
        extracted = tdp / name / f"{name}.mtx"
        if not extracted.exists():
            # Some archives nest differently — search.
            candidates = list(tdp.rglob(f"{name}.mtx"))
            if not candidates:
                print(f"    FAIL: no {name}.mtx in archive", flush=True)
                return False
            extracted = candidates[0]
        shutil.copy(extracted, tgt)
        print(f"    ok -> {tgt.relative_to(STRESS_DIR.parent.parent)}",
              flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(STRESS_DIR / "manifest.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--group", default=None,
                    help="filter to a single SuiteSparse group")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = parse_manifest(Path(args.manifest))
    real = [r for r in rows if r.get("group") != "synth"]
    if args.group:
        real = [r for r in real if r["group"] == args.group]
    if args.limit is not None:
        real = real[: args.limit]

    print(f"fetching {len(real)} matrices from {BASE_URL}", flush=True)
    n_new = 0
    for r in real:
        try:
            if fetch_one(r["group"], r["name"], args.force):
                n_new += 1
        except KeyboardInterrupt:
            print("\ninterrupted", flush=True)
            return 130
    print(f"\ndone: {n_new} newly downloaded, "
          f"{len(real) - n_new} already present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
