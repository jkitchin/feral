#!/usr/bin/env python3
"""Pin the 0.15.0-vs-0.14.0 chain regression to a mechanism.

The 15-pair A/B showed feral 0.15.0 significantly slower than 0.14.0 on
the two largest chain-structured proxies (prommis_sx_like 0.833,
p=0.0074). Two candidate mechanisms shipped in 0.15.0:

  #149  explicit SIMD packed trailing update + x86 pulp dispatch fix
  #150  one rayon task per subtree instead of one per supernode

This runs the same paired protocol over arms that isolate them:

  0.14.0                    baseline
  0.15.0 default            the regression
  0.15.0 PAR_MIN_SEEDS=max  forces the sequential fallback, so the
                            task-graph path is never taken
  0.15.0 RAYON=1            single-threaded
  0.15.0 PACKED_SIMD=0      restores the scalar tile walk

If forcing the sequential fallback recovers the loss, the regression is
in the parallel driver (#150). If disabling the packed SIMD kernel
recovers it, it is the kernel (#149).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from math import comb
from pathlib import Path

SP = Path(__file__).resolve().parent
ROOT = SP.parents[1]
WORK = Path(os.environ.get("CHAIN_PROXY_WORK", SP / "work"))
DRIVER = "target/release/bench_one_matrix"

F15 = os.environ.get("FERAL_BIN_NEW", str(ROOT / DRIVER))
F14 = os.environ.get("FERAL_BIN_BASE", str(WORK / "feral-base" / DRIVER))

U64_MAX = "18446744073709551615"

ARMS = [
    ("base", F14, {}),
    ("new-default", F15, {}),
    ("new-seq-fallback", F15, {"FERAL_PAR_MIN_SEEDS": U64_MAX}),
    ("new-rayon1", F15, {"RAYON_NUM_THREADS": "1"}),
    ("new-nosimd", F15, {"FERAL_PACKED_SIMD": "0"}),
]

MATS = ["prommis_sx_like", "double_column_like"]
PAIRS = 15


def sign_test_p(wins: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def main() -> int:
    rhs_dir = WORK / "rhs"
    out_root = WORK / "probe_out"
    manifests = {}
    for arm, _, _ in ARMS:
        (out_root / arm).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(suffix=".manifest", text=True)
        with os.fdopen(fd, "w") as f:
            for m in MATS:
                f.write(f"{WORK / 'mtx' / (m + '.mtx')} {rhs_dir / (m + '.rhs')} "
                        f"{out_root / arm / (m + '.out')}\n")
        manifests[arm] = Path(name)

    samples = {arm: {m: [] for m in MATS} for arm, _, _ in ARMS}
    for p in range(PAIRS):
        for arm, binp, env in ARMS:
            e = dict(os.environ)
            e.update(env)
            subprocess.run([binp, str(manifests[arm])], check=False,
                           capture_output=True, timeout=1800, env=e)
            for m in MATS:
                d = {}
                for line in (out_root / arm / (m + ".out")).read_text().splitlines():
                    q = line.split(None, 1)
                    if len(q) == 2:
                        d[q[0]] = q[1].strip()
                if d.get("status") == "ok":
                    samples[arm][m].append(int(d["factor_us"]))
        print(f"  pair {p + 1}/{PAIRS}", flush=True)

    (WORK / "probe_results.json").write_text(json.dumps(samples, indent=2))

    for m in MATS:
        print(f"\n=== {m} ===")
        base = min(samples["base"][m])
        print(f"{'arm':<24}{'min_us':>10}{'vs base':>12}{'wins/15':>10}{'p':>9}")
        for arm, _, _ in ARMS:
            s = samples[arm][m]
            if not s:
                print(f"{arm:<24}{'FAIL':>10}")
                continue
            b = samples["base"][m]
            k = min(len(s), len(b))
            wins = sum(1 for i in range(k) if s[i] < b[i])
            ratio = base / min(s)  # >1 means this arm is faster than 0.14.0
            tag = "" if arm == "base" else f"{wins}/{k}"
            pv = "" if arm == "base" else f"{sign_test_p(wins, k):.4f}"
            print(f"{arm:<24}{min(s):>10}{ratio:>12.3f}{tag:>10}{pv:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
