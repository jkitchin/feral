#!/usr/bin/env python3
"""Characterize the FERAL KKT validation corpus.

Walks data/matrices/kkt/, data/matrices/kkt-expansion/, and
data/matrices/kkt-mittelmann/, parses the matrix-market header
(n, nnz) of each .mtx and the per-matrix sidecar JSON
(inertia, iteration), and emits:

  dev/corpus-characterization/summary.csv  -- one row per matrix
  manuscript/figures/corpus-size.png       -- size histogram
  manuscript/figures/corpus-nnz-vs-n.png   -- nnz vs n scatter
  manuscript/figures/corpus-families.png   -- top-30 family counts
  manuscript/figures/corpus-inertia.png    -- inertia structure
  manuscript/figures/corpus-iteration.png  -- IPM-iteration histogram

Reads only the first non-comment line of each .mtx, so the I/O pass
is bounded by stat + small reads.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KKT_ROOTS = [
    ROOT / "data" / "matrices" / "kkt",
    ROOT / "data" / "matrices" / "kkt-expansion",
    ROOT / "data" / "matrices" / "kkt-mittelmann",
]
OUT_CSV = ROOT / "dev" / "corpus-characterization" / "summary.csv"
FIG_DIR = ROOT / "manuscript" / "figures"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_mtx_header(path: Path) -> tuple[int, int] | None:
    """Return (n, nnz) from the first non-comment line of an mtx file."""
    try:
        with path.open("rb") as f:
            for raw in f:
                line = raw.decode("ascii", errors="ignore").strip()
                if not line or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    return None
                return int(parts[0]), int(parts[2])
    except OSError:
        return None
    return None


def scan_corpus() -> list[dict]:
    rows: list[dict] = []
    seen = 0
    for kkt_root in KKT_ROOTS:
        if not kkt_root.exists():
            continue
        for fam_dir in sorted(kkt_root.iterdir()):
            if not fam_dir.is_dir():
                continue
            family = fam_dir.name
            for mtx in fam_dir.glob("*.mtx"):
                stem = mtx.stem
                sidecar = fam_dir / f"{stem}.json"
                hdr = read_mtx_header(mtx)
                if hdr is None:
                    continue
                n, nnz = hdr
                inertia = (None, None, None)
                iteration = None
                if sidecar.exists():
                    try:
                        with sidecar.open() as f:
                            d = json.load(f)
                        inert = d.get("inertia") or {}
                        inertia = (
                            inert.get("positive"),
                            inert.get("negative"),
                            inert.get("zero"),
                        )
                        iteration = d.get("iteration")
                    except (OSError, json.JSONDecodeError):
                        pass
                rows.append(
                    {
                        "family": family,
                        "name": stem,
                        "iteration": iteration,
                        "n": n,
                        "nnz": nnz,
                        "n_pos": inertia[0],
                        "n_neg": inertia[1],
                        "n_zero": inertia[2],
                    }
                )
                seen += 1
                if seen % 20000 == 0:
                    print(f"  scanned {seen}", file=sys.stderr)
    return rows


def write_csv(rows: list[dict]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["family", "name", "iteration", "n", "nnz", "n_pos", "n_neg", "n_zero"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fig_size_hist(ns: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bins = np.logspace(np.log10(max(ns.min(), 1)), np.log10(ns.max()), 60)
    ax.hist(ns, bins=bins, color="#2E7D9A", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix dimension $n$")
    ax.set_ylabel("number of matrices")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    median = int(np.median(ns))
    p99 = int(np.quantile(ns, 0.99))
    ax.axvline(median, color="#C44536", linestyle="--", linewidth=1, label=f"median {median}")
    ax.axvline(p99, color="#888", linestyle=":", linewidth=1, label=f"p99 {p99}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corpus-size.png", dpi=160)
    plt.close(fig)


def fig_nnz_scatter(ns: np.ndarray, nnzs: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    density = nnzs / np.maximum(ns.astype(float), 1.0)
    sc = ax.scatter(ns, nnzs, c=np.log10(density), cmap="viridis", s=2.5, alpha=0.4)
    diag = np.array([ns.min(), ns.max()])
    ax.plot(diag, diag, color="#888", linestyle=":", linewidth=1, label="$\\mathrm{nnz}=n$")
    ax.plot(diag, diag * np.log2(diag), color="#C44536", linestyle="--", linewidth=1, label="$n \\log_2 n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix dimension $n$")
    ax.set_ylabel("number of nonzeros (lower triangle)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("$\\log_{10}(\\mathrm{nnz}/n)$ row density")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corpus-nnz-vs-n.png", dpi=160)
    plt.close(fig)


def fig_families(families: list[str]) -> None:
    counts = Counter(families)
    top = counts.most_common(30)
    other_total = sum(counts.values()) - sum(c for _, c in top)
    labels = [name for name, _ in top] + ["(other)"]
    values = [c for _, c in top] + [other_total]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=["#2E7D9A"] * 30 + ["#888"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("matrix count")
    ax.set_xscale("log")
    for i, v in enumerate(values):
        ax.text(v * 1.05, i, f"{v:,}", va="center", fontsize=7)
    ax.grid(True, axis="x", which="both", linestyle=":", alpha=0.4)
    ax.set_title(f"top 30 families of {len(set(families))} (total {sum(counts.values()):,} matrices)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corpus-families.png", dpi=160)
    plt.close(fig)


def fig_inertia(rows: list[dict]) -> None:
    have = [
        r
        for r in rows
        if r["n_pos"] is not None and r["n_neg"] is not None and r["n_zero"] is not None
    ]
    pos_def = sum(1 for r in have if (r["n_neg"] or 0) == 0 and (r["n_pos"] or 0) > 0)
    neg_def = sum(1 for r in have if (r["n_pos"] or 0) == 0 and (r["n_neg"] or 0) > 0)
    indef = sum(1 for r in have if (r["n_pos"] or 0) > 0 and (r["n_neg"] or 0) > 0)
    neg_frac = np.array(
        [(r["n_neg"] or 0) / max(r["n"], 1) for r in have if (r["n_neg"] or 0) > 0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.1, 1.4]})

    ax = axes[0]
    labels = [
        "indefinite\n($n_+, n_- > 0$)",
        "pos. definite\n($n_- = 0$)",
        "neg. definite\n($n_+ = 0$)",
    ]
    values = [indef, pos_def, neg_def]
    colors = ["#C44536", "#2E7D9A", "#888"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        pct = 100.0 * v / max(len(have), 1)
        ax.text(i, max(v, 1), f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("matrix count")
    ax.set_yscale("log")
    ax.set_ylim(top=max(values) * 3)
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax.set_title("sign class", fontsize=10)

    ax = axes[1]
    ax.hist(neg_frac, bins=50, color="#C44536", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("negative-eigenvalue fraction $n_- / n$ (indefinite subset)")
    ax.set_ylabel("matrix count")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    med = float(np.median(neg_frac))
    ax.axvline(med, color="#222", linestyle="--", linewidth=1, label=f"median {med:.2f}")
    ax.legend(frameon=False)
    ax.set_title("KKT indefiniteness signature", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "corpus-inertia.png", dpi=160)
    plt.close(fig)


def fig_iteration(rows: list[dict]) -> None:
    iters = np.array([r["iteration"] for r in rows if r["iteration"] is not None], dtype=int)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if len(iters) == 0:
        ax.text(0.5, 0.5, "no iteration data", ha="center", va="center", transform=ax.transAxes)
    else:
        max_iter = int(np.quantile(iters, 0.99))
        bins = np.arange(0, max_iter + 2) - 0.5
        ax.hist(iters, bins=bins, color="#2E7D9A", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("IPM outer iteration index")
        ax.set_ylabel("matrix count")
        ax.set_xlim(-0.5, max_iter + 0.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        med = int(np.median(iters))
        ax.axvline(med, color="#C44536", linestyle="--", linewidth=1, label=f"median {med}")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corpus-iteration.png", dpi=160)
    plt.close(fig)


def main() -> int:
    print("scanning corpus ...", file=sys.stderr)
    rows = scan_corpus()
    if not rows:
        print("no matrices found under data/matrices/kkt*", file=sys.stderr)
        return 1
    print(f"  total {len(rows):,}", file=sys.stderr)
    write_csv(rows)
    print(f"  wrote {OUT_CSV}", file=sys.stderr)

    ns = np.array([r["n"] for r in rows], dtype=int)
    nnzs = np.array([r["nnz"] for r in rows], dtype=int)
    families = [r["family"] for r in rows]

    fig_size_hist(ns)
    fig_nnz_scatter(ns, nnzs)
    fig_families(families)
    fig_inertia(rows)
    fig_iteration(rows)
    print(f"  wrote 5 figures to {FIG_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
