# Phase 2.4.4 — `OrderingPreprocess::Auto` (shape-based dispatch)

## Problem

Phase 2.4.3 closed the rook-pivoting rescue path but left the dense-tail
investigation with a tension: unconditionally turning on
`OrderingPreprocess::LdltCompress` wins on HAHN1/GAUSS2 (2–5× numeric
speedups) but regresses the bench geomean 0.36 → 0.48 because the
compression symbolic overhead (~100–700μs) doesn't amortize on the
small-n bulk of the corpus.

Goal: a cheap, O(nnz) predicate that enables compression only when it
will pay off, by analogy to `scaling::pick_scaling_strategy`.

## Corpus shape

Bench corpus is 154,588 matrices × 566 families. Size distribution:

| n range          | pct    |
|------------------|--------|
| [0, 50)          | 80.8%  |
| [50, 100)        | 7.8%   |
| [100, 200)       | 7.4%   |
| [200, 500)       | 3.1%   |
| [500, 1000)      | 0.9%   |
| [1000, 5314]     | 0.05%  |

Median n = 8. This is small-n IPM-iterate land; the geomean is dominated
by tiny matrices where compression's symbolic cost dwarfs its numeric
benefit.

## Predicate

```rust
pub fn pick_ordering_preprocess(matrix: &CscMatrix) -> OrderingPreprocess {
    const MIN_N_FOR_COMPRESSION: usize = 128;
    const LOW_DEGREE_THRESHOLD: f64 = 0.30;

    let n = matrix.n;
    if n < MIN_N_FOR_COMPRESSION {
        return OrderingPreprocess::None;
    }
    let mut low_degree = 0usize;
    for j in 0..n {
        let nnz_col = matrix.col_ptr[j + 1] - matrix.col_ptr[j];
        if nnz_col <= 2 { low_degree += 1; }
    }
    if low_degree as f64 / n as f64 >= LOW_DEGREE_THRESHOLD {
        OrderingPreprocess::LdltCompress
    } else {
        OrderingPreprocess::None
    }
}
```

### Rationale for the two predicates

**Size floor `n >= 128`**. From the 154,588-matrix bench, 80.8% of
matrices have n<50. For those, numeric factor is sub-ms and the
compression symbolic overhead (dominated by MC64 Hungarian, 70–97%
of compression pipeline cost, `src/bin/diag_compress_profile.rs`)
becomes the loss. Calibrated against the unconditional-flip regression:
geomean 0.36 → 0.48 was driven entirely by sub-100 matrices. `n=128`
is the smallest power-of-two floor that excludes the bulk; 85% of the
corpus is cut by this one predicate alone.

**Low-degree ratio `low_degree/n >= 0.30`**. Columns with stored
degree ≤ 2 (diagonal + at most one off-diagonal) are the structural
fingerprint of IPM slack blocks. The augmented system Ipopt produces
(`IpStdAugSystemSolver.cpp`) has block

```
[ W+Σ_x+δ_x I   A^T   J^T       0     ]
[   A            0     0         0     ]
[   J            0    -Σ_c-δ_c I 0     ]
[   0            0     0     -Σ_s-δ_s I]
```

Slack rows `(-Σ_s-δ_s I)` are degree-2 (one identity + self), not
degree-1. This differs from `scaling::pick_scaling_strategy`, which
uses `diag_only/n >= 0.30` counting *exactly-degree-1* columns. The
broader degree-2 predicate is the right signal for compression: MC64
produces 2-cycles between slack rows and their coupling partners,
which is exactly the structure `build_supermap` exploits. Threshold
0.30 mirrors the scaling heuristic's empirically-calibrated value.

## Auto enum variant as default

`OrderingPreprocess::Auto` is added as a third variant, now the
`#[default]`. Resolution happens at the top of
`symbolic_factorize_with_method` before the existing match, so the
hot code path is unchanged — the extra step is a cheap scan of
`col_ptr` (len n+1) in the Auto case.

## Expert consultation (2026-04-23)

Ran three subagents against MUMPS, SPRAL/SSIDS, and Ipopt sources:

- **MUMPS** (`dana_aux.F`): *does* auto-dispatch. For SYM=2 (symmetric
  indefinite), the default `ICNTL(12)=0` resolves to compression mode
  unless three gates fire: excessive zero diagonal, quasi-dense rows,
  or post-MC64 residual. No explicit size floor in MUMPS, but the gates
  are consistent with "don't compress when it won't help."

- **SPRAL/SSIDS**: no compression at all — strictly user-gated ordering
  choice. Weak counterargument against shape dispatch.

- **Ipopt**: confirms the structural claim (slack columns are
  degree-2, not degree-1) and confirms Ipopt reuses the symbolic
  factorization across IPM iterations (`IpTSymLinearSolver.cpp:392-395`).
  One-time Auto resolution per analysis is cheap and amortizes.

Consensus: shape-based dispatch is architecturally sound and consistent
with the MUMPS reference.

## Calibration evidence

`diag_compress_profile` medians (μs, from session 2026-04-23-02):

| matrix         | n   | mc64 | smap | comp | exp | total |
|----------------|-----|------|------|------|-----|-------|
| HAHN1_0153     | 715 | 283  | 4    | 89   | 26  | 402   |
| HAHN1_0404     | 670 | 262  | 4    | 83   | 23  | 372   |
| GAUSS2_0029    | 758 | 260  | 5    | 101  | 27  | 393   |
| KIRBY2_0007    | 458 | 679  | 3    | 17   | 12  | 711   |
| HS118_0001     |  32 |   4  | 0    |  0   | 0   |   4   |
| HS92_0001      |   7 |   0  | 0    |  0   | 0   |   0   |

MC64 dominates; sub-100 matrices show it as pure overhead because
numeric factor for those is also sub-ms.

## Expected bench impact

Hypothesis: geomean stays at 0.36 (compression skipped on n<128 bulk),
p90 falls from 1.75 toward 1.61 (tail families like HAHN1 / GAUSS2
re-enter the compression path), max approaches 10.42 (ACOPR30 stays
non-compressed because its low-degree ratio is low).

To be confirmed by the bench run in progress.

## References

- `dev/sessions/2026-04-23-02.md` — session checkpoint
- `dev/journal/2026-04-23-02.org` — 22:50 entry, MC64 cache result
- `src/bin/diag_compress_profile.rs` — profiling harness
- `src/symbolic/mod.rs:221` — `pick_ordering_preprocess`
- `src/scaling/mod.rs` — `pick_scaling_strategy` (parallel precedent)
- MUMPS `dana_aux.F` — three-gate compression decision for SYM=2
- Ipopt `IpStdAugSystemSolver.cpp:250-305` — slack-block structure
- Ipopt `IpTSymLinearSolver.cpp:392-395` — symbolic reuse across IPM
