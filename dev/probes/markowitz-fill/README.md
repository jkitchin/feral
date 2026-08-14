# Markowitz fill oracle (discopt #1008)

Answers: **is the LU fill on discopt's LP bases intrinsic to the matrices, or an
artifact of choosing the pivot order statically and then pivoting for stability?**

feral picks a static fill-reducing column permutation (AMD on the AᵀA
column-intersection pattern, optionally after a Suhl–Suhl peel) and then factors
with partial pivoting. Production LP `INVERT` codes (LUSOL, HiGHS `HFactor`)
instead choose each pivot *dynamically* to minimize the Markowitz count
`(r_i-1)(c_j-1)` subject to a relative-magnitude threshold `|a_ij| >= u·max_k|a_kj|`.

`markowitz.py` is a right-looking threshold-Markowitz LU used **only as a fill
oracle**. It is slow Python and is not a kernel; its output is `nnz(L+U)` under
the same convention as feral's `SparseLu::factor_nnz()` (strict-lower `L` plus
`U` including the diagonal). Every reported number is gated on
`‖PBQ − LU‖∞/‖B‖∞ < 1e-10`, checked per factorization — a fill number from a
wrong factorization is worthless.

## Why SuperLU could not answer this

`scipy.sparse.linalg.splu` is the *same algorithm class* as feral: static column
order (COLAMD) plus partial pivoting. Agreement between it and feral is not
evidence about the dynamic alternative, and on this corpus both sit 2.3–2.6x
above what Markowitz reaches (see `table.txt`).

## Reproducing

```sh
# 1. dump real bases from the captured #1008 relaxation LPs (needs discopt)
python3 dump_many.py <outdir> 60 <path-to>/i1008/lps/*.npz
python3 npz2mtx.py '<outdir>/*.npz'
# 2. build feral's arm
cargo build --release --example basis_refactor
# 3. the table
python3 table.py <outdir>/*.mtx
# 4. stability of the Markowitz factor (growth, max|L|)
python3 quality.py <outdir>/*.mtx
```

`table.py` shells out to `examples/basis_refactor` for feral's two orderings, so
it must run from a tree where that example is built; the path is at the top of
the script.
