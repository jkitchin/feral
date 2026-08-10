# chain_proxy — chain-structured KKT harness vs HSL MA57

> **The proxies have been superseded.** The same protocol now runs on
> the real corpus (`real_corpus_mtx.py` + `arm_run.py`), and the real
> matrices contradict both conclusions the proxies produced — see
> `dev/research/chain-kkt-corpus-2026-08-09.md`. Prefer the real run.
> The generated proxies are kept as the portable smoke version for
> machines without `data/matrices/`, and as a worked example of
> geometry-matched stand-ins giving the wrong answer.

A paired A/B harness for the question behind [pounce#552]: how far is
feral's factorization from Harwell's on the **chain-structured** KKT
matrices that direct-transcription dynamic optimization produces?

The report's five models are Pyomo NMPC problems that are not in this
repo, and reproducing them needs idaes/prommis plus their initialization
chains. This harness substitutes block-tridiagonal proxies generated at
the *reported geometry* (time points, variables per time point, `n`), so
it can run anywhere without that stack. It is a stand-in, not the real
thing — see **Limits** before quoting a number from it.

## What it generates

`gen_chain_kkt.py` builds one symmetric indefinite KKT per model. Per
time point:

    [ H_t   A_t^T ]      H_t : nx x nx  banded SPD
    [ A_t  -delta*I ]    A_t : nc x nx  banded constraint Jacobian

coupled to the next point by state-continuity entries, which is what
makes the elimination tree a chain. Geometry:

| proxy | T | vars/pt | n | reported n |
|---|---|---|---|---|
| hicks_like | 301 | 6 | 1,806 | 1,706 |
| cart_pole_like | 301 | 9 | 2,709 | 2,810 |
| quad_tank_like | 301 | 10 | 3,010 | 2,910 |
| double_column_like | 31 | 819 | 25,389 | 25,377 |
| prommis_sx_like | 31 | 913 | 28,303 | 28,313 |

The `-delta*I` block makes the inertia analytically known: with `H` SPD
and `A` of full row rank the matrix has exactly `T*nx` positive and
`T*nc` negative eigenvalues and no zeros. **That is the correctness
oracle** — it does not depend on either solver being right. Both feral
and MA57 hit it exactly on all five, with residuals ~1e-15 or better.

Getting there took one fix worth remembering: an earlier generator
advanced each dual row's column start by a fixed stride and clamped it,
so when `nx` was close to `A_WIDTH` every row in a block got the same
columns and proportional values, making `A_t` rank 1. MA57 reported
`inertia_zero = 301` (one per time point) and feral returned a residual
of 5.9e+03. If you change the generator, re-check against the oracle
before trusting a timing.

## Running it

Build the MA57 oracle first (needs the CoinHSL bundle; see
`../ma57_oracle/Makefile` for the `HSL_ROOT` it expects):

```sh
make -C ../ma57_oracle
```

Build the driver for the arm under test, and a baseline in a worktree:

```sh
cargo build --release -p feral-diagnostics --bin bench_one_matrix
git worktree add /tmp/feral-base v0.14.0
(cd /tmp/feral-base && cargo build --release -p feral-diagnostics \
     --bin bench_one_matrix)
```

Then:

```sh
export CHAIN_PROXY_WORK=/tmp/chain-proxy      # matrices + results land here
python3 gen_chain_kkt.py --out $CHAIN_PROXY_WORK/mtx

FERAL_BIN_BASE=/tmp/feral-base/target/release/bench_one_matrix \
FERAL_LABEL_NEW=main FERAL_LABEL_BASE=v0.14.0 \
  python3 ab_run.py --pairs 15
```

`probe_regression.py` takes the same environment and re-runs the two
large proxies across `FERAL_PAR_MIN_SEEDS`, `RAYON_NUM_THREADS` and
`FERAL_PACKED_SIMD` arms, to attribute a difference to the parallel
driver or to the kernel.

Overrides: `FERAL_BIN_NEW`, `FERAL_BIN_BASE`, `MA57_BIN`,
`FERAL_LABEL_NEW`, `FERAL_LABEL_BASE`, `CHAIN_PROXY_WORK`.

## Protocol

Paired alternating A/B per `dev/decisions.md` (2026-08-09): every arm is
timed once per pair so drift hits all arms equally, `min` over pairs is
the per-arm statistic, and significance is an exact two-sided sign test
over the pairs. Medians collected at different times are never compared.
All arms read the same manifest, the same matrices and the same
synthesized RHS (`b = A x_true`, `x_true[i] = 1 + i/n`), so the residual
is defined identically across solvers.

## Limits

Read these before quoting a ratio.

- **These are proxies.** Geometry matches the reported models; sparsity
  and numerics are invented. Measured density is ~4 nnz/row, which is on
  the sparse side for a real flowsheet KKT. A conclusion from this
  harness is a hypothesis about the real matrices, not a measurement of
  them.
- **Cross-machine comparison is invalid.** The paired design cancels
  machine differences *within* a run, and that is the only thing it
  guarantees. Do not compare absolute microseconds here against numbers
  in the release notes or in pounce#552, which were taken on other
  hardware.
- **Heterogeneous cores matter.** On Apple silicon rayon sees P+E cores
  as equivalent. On a 4P+4E M2 a coarse task landing on an efficiency
  core stalls the whole factorization, so thread count is a real
  variable, not a detail. Record `sysctl -n hw.model` with any result.
- `factor_us` is numeric factorization only. Symbolic analysis and the
  solve are excluded, so this does not speak to end-to-end solve time,
  which is what pounce#552 actually reports.

## Running it on the real corpus

`real_corpus_mtx.py` builds the `--mtx-dir` as symlinks into
`data/matrices/kkt-mittelmann`, one iterate per family, with the
selection rules (and the `dtoc2` singularity exception) in its
docstring:

```sh
python3 real_corpus_mtx.py --out $WORK/mtx
```

`arm_run.py` is `ab_run.py` generalized to any number of arms, reusing
its protocol functions directly so the statistics are literally the
same code. Arms are `NAME=BIN`, optionally with environment overrides
after a `|`, which is what makes a thread sweep or a SIMD probe a
matter of arguments rather than another script:

```sh
python3 arm_run.py --mtx-dir $WORK/mtx --pairs 15 --ref v0.14.0 \
  --out $WORK/bisect.json \
  --arm "v0.14.0=/tmp/wt/v0140/target/release/bench_one_matrix" \
  --arm "main=target/release/bench_one_matrix" \
  --arm "main-t1=target/release/bench_one_matrix|RAYON_NUM_THREADS=1" \
  --arm "ma57=../ma57_oracle/ma57_bench"
```

It prints a correctness gate — any arm with `inertia_zero != 0` or a
NaN / `> 1e-8` residual — *before* the timing tables, and says plainly
that flagged matrices cannot carry a ratio. That gate is what caught
`dtoc2_0001` (singular by both solvers) and MA57's `4.53e-08` residual
on `steering_12800`.

`steering_12800` has no `.mtx` in the corpus as shipped. Regenerate it
with pounce (`--dump kkt:1-3`) and `scripts/harvest-pounce-kkt.py`; the
ripopt harvest path silently writes nothing. See the research note.

[pounce#552]: https://github.com/jkitchin/pounce/issues/552
