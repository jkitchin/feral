#!/usr/bin/env python3
"""Generate scaling plots from external_benchmarks/scaling/scaling.tsv.

Produces, under external_benchmarks/scaling/plots/:
  - factor_<family>.png         log-log factor_us vs n, one curve per solver
  - ratio_<family>.png           factor_us / mumps_factor_us, feral and ma57 vs n
  - residual_<family>.png        rel_res vs n (log-y)
  - overview.png                  4-panel grid: all families, factor_us only

Usage:
    python3 plot.py                    # plot all families found in scaling.tsv
    python3 plot.py --families dense_si,laplace2d
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALING_DIR = Path(__file__).resolve().parent
TSV_PATH = SCALING_DIR / "scaling.tsv"
PLOT_DIR = SCALING_DIR / "plots"

SOLVER_STYLE = {
    "feral": {"color": "#d62728", "marker": "o", "label": "feral"},
    "mumps": {"color": "#1f77b4", "marker": "s", "label": "MUMPS 5.8.2"},
    "ma57":  {"color": "#2ca02c", "marker": "^", "label": "HSL MA57"},
}

FAMILY_TITLE = {
    "dense_si":   "Dense symmetric indefinite",
    "banded_spd": "Banded SPD (bandwidth=10)",
    "laplace2d":  "2D 5-point Laplacian",
    "saddle_kkt": "Saddle-point KKT",
}

FAMILY_EXPECTED_SLOPE = {
    "dense_si":   3.0,
    "banded_spd": 1.0,
    "laplace2d":  1.5,
    "saddle_kkt": 2.0,  # nominal; depends on A density
}


def solver_key(name: str) -> str:
    """Map full solver name (e.g. 'feral-0.3.0') to short key ('feral')."""
    name = name.lower()
    for key in SOLVER_STYLE:
        if name.startswith(key):
            return key
    return name


def load(path: Path) -> list[dict]:
    """Load scaling.tsv and derive `total_factor_us` for fair comparison.

    Solver `factor_us` keys are NOT directly comparable:
      - feral:  analyse_us + factor_us (separate keys)
      - MA57:   factor_us only (MA57AD analysis is untimed in the bench)
      - MUMPS:  factor_us is JOB=4 = analyse + numeric factor combined.

    For an apples-to-apples comparison we use:
      total_factor_us = (analyse_us or 0) + factor_us

    For MUMPS, analyse_us is absent so total_factor_us == factor_us, which
    already includes both phases. For feral and MA57 we add their analyse
    cost (MA57's untimed analysis is treated as 0 — slight under-attribution
    that favours MA57; this is documented in the report).
    """
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                r["n"] = int(r["n"])
                analyse = float(r["analyse_us"]) if r.get("analyse_us") else 0.0
                factor = float(r["factor_us"]) if r["factor_us"] else None
                r["analyse_us"] = analyse
                r["factor_us"] = factor
                r["total_factor_us"] = (factor + analyse) if factor is not None else None
                r["solve_us"] = float(r["solve_us"]) if r["solve_us"] else None
                r["rel_res"] = float(r["rel_res"]) if r["rel_res"] else None
                r["solver_key"] = solver_key(r["solver"])
            except ValueError:
                continue
            rows.append(r)
    return rows


def loglog_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    num = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    den = sum((x - mx) ** 2 for x in lx)
    slope = num / den if den > 0 else 0.0
    intercept = my - slope * mx
    return slope, intercept


def plot_factor(rows: list[dict], family: str, out: Path) -> dict:
    """Single-family log-log plot: factor_us vs n. Returns fit summary."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fits: dict = {}
    for solver in ("mumps", "ma57", "feral"):
        pts = [(r["n"], r["total_factor_us"]) for r in rows
               if r["family"] == family and r["solver_key"] == solver
               and r["total_factor_us"] is not None and r["total_factor_us"] > 0]
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = SOLVER_STYLE[solver]
        ax.loglog(xs, ys, marker=style["marker"], color=style["color"],
                  linewidth=1.5, markersize=7, label=style["label"])
        if len(pts) >= 3:
            slope, intercept = loglog_fit(xs, ys)
            fits[solver] = {"slope": slope, "intercept": intercept,
                            "n": xs, "factor_us": ys}

    expected = FAMILY_EXPECTED_SLOPE.get(family)
    if expected is not None and "mumps" in fits:
        # Dashed reference line of expected slope, anchored at MUMPS's first point.
        x0 = fits["mumps"]["n"][0]
        y0 = fits["mumps"]["factor_us"][0]
        x1 = fits["mumps"]["n"][-1]
        y1 = y0 * (x1 / x0) ** expected
        ax.loglog([x0, x1], [y0, y1], "--", color="grey", linewidth=1.0,
                  alpha=0.5, label=f"$n^{{{expected}}}$ reference")

    ax.set_xlabel("matrix dimension $n$")
    ax.set_ylabel("factor time ($\\mu$s)")
    ax.set_title(f"{FAMILY_TITLE.get(family, family)} — factor time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return fits


def plot_ratio(rows: list[dict], family: str, out: Path) -> None:
    """factor_us(solver) / factor_us(mumps) vs n. MUMPS is the baseline (=1)."""
    by_n_mumps: dict[int, float] = {}
    for r in rows:
        if r["family"] == family and r["solver_key"] == "mumps" and r["total_factor_us"]:
            by_n_mumps[r["n"]] = r["total_factor_us"]
    if not by_n_mumps:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for solver in ("feral", "ma57"):
        pts: list[tuple[int, float]] = []
        for r in rows:
            if r["family"] == family and r["solver_key"] == solver and r["total_factor_us"]:
                mu = by_n_mumps.get(r["n"])
                if mu:
                    pts.append((r["n"], r["total_factor_us"] / mu))
        if not pts:
            continue
        pts.sort()
        style = SOLVER_STYLE[solver]
        ax.semilogx([p[0] for p in pts], [p[1] for p in pts],
                    marker=style["marker"], color=style["color"],
                    linewidth=1.5, markersize=7, label=style["label"] + " / MUMPS")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("matrix dimension $n$")
    ax.set_ylabel("factor time ratio (solver / MUMPS)")
    ax.set_title(f"{FAMILY_TITLE.get(family, family)} — factor time vs MUMPS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_residual(rows: list[dict], family: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    any_data = False
    for solver in ("feral", "mumps", "ma57"):
        pts = [(r["n"], r["rel_res"]) for r in rows
               if r["family"] == family and r["solver_key"] == solver
               and r["rel_res"] is not None and r["rel_res"] > 0]
        if not pts:
            continue
        any_data = True
        pts.sort()
        style = SOLVER_STYLE[solver]
        ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                  marker=style["marker"], color=style["color"],
                  linewidth=1.5, markersize=7, label=style["label"])
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("matrix dimension $n$")
    ax.set_ylabel("relative residual $\\|Ax - b\\|_2 / \\|b\\|_2$")
    ax.set_title(f"{FAMILY_TITLE.get(family, family)} — residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_overview(rows: list[dict], families: list[str], out: Path) -> None:
    n_fam = len(families)
    cols = 2
    rows_n = (n_fam + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 4.0 * rows_n))
    if rows_n == 1:
        axes = [axes] if n_fam == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for ax, family in zip(axes, families):
        for solver in ("mumps", "ma57", "feral"):
            pts = [(r["n"], r["total_factor_us"]) for r in rows
                   if r["family"] == family and r["solver_key"] == solver
                   and r["total_factor_us"] is not None and r["total_factor_us"] > 0]
            if not pts:
                continue
            pts.sort()
            style = SOLVER_STYLE[solver]
            ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                      marker=style["marker"], color=style["color"],
                      linewidth=1.4, markersize=6, label=style["label"])
        ax.set_xlabel("$n$")
        ax.set_ylabel("factor ($\\mu$s)")
        ax.set_title(FAMILY_TITLE.get(family, family), fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Hide unused panels.
    for ax in axes[n_fam:]:
        ax.set_visible(False)

    fig.suptitle("Factor time scaling — feral vs MUMPS vs MA57", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=None,
                    help="comma-separated subset (default: all in scaling.tsv)")
    args = ap.parse_args()

    rows = load(TSV_PATH)
    if not rows:
        print(f"no rows in {TSV_PATH}", file=sys.stderr)
        return 1

    if args.families:
        families = [f.strip() for f in args.families.split(",")]
    else:
        families = sorted({r["family"] for r in rows})

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    all_fits: dict[str, dict] = {}
    for family in families:
        f_out = PLOT_DIR / f"factor_{family}.png"
        r_out = PLOT_DIR / f"ratio_{family}.png"
        res_out = PLOT_DIR / f"residual_{family}.png"
        fits = plot_factor(rows, family, f_out)
        plot_ratio(rows, family, r_out)
        plot_residual(rows, family, res_out)
        all_fits[family] = fits
        print(f"  {family:<14} -> {f_out.name}, {r_out.name}, {res_out.name}", flush=True)

    plot_overview(rows, families, PLOT_DIR / "overview.png")
    print(f"  overview        -> overview.png", flush=True)

    # Print fit summary.
    print(f"\n{'family':<14}{'solver':<10}{'slope':>8}{'expected':>10}")
    for family in families:
        expected = FAMILY_EXPECTED_SLOPE.get(family, float("nan"))
        for solver, fit in sorted(all_fits.get(family, {}).items()):
            print(f"{family:<14}{solver:<10}{fit['slope']:>8.2f}{expected:>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
