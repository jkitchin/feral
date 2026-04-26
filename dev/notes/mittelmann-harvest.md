# Harvesting KKT matrices from the Mittelmann ampl-nlp benchmark

The Mittelmann ampl-nlp set
(`../ripopt/benchmarks/mittelmann/`) is 47 medium-to-large NLP
instances (n = 500 to 261k) where ripopt does not currently do well.
Each *iteration* of those runs is a KKT system that we want in the
FERAL corpus: large, ill-conditioned, drawn from a workload where the
linear solver matters.

## One-time ripopt patch

ripopt already has the `kkt_dump_dir` and `kkt_dump_name` `SolverOptions`
fields and they fire on every IPM iteration regardless of whether the
solve converges (perfect for the failing-to-converge cases on this
set). What's missing is exposing them through the `ripopt_ampl` CLI
so they can be passed as `key=value` options when ripopt is driven
over `.nl` files.

Apply this patch to `../ripopt/src/bin/ripopt_ampl.rs`, in the match
arms inside `apply_option`, just before the catch-all `_ =>` arm:

```rust
        "kkt_dump_dir" => {
            opts.kkt_dump_dir = Some(std::path::PathBuf::from(value));
        }
        "kkt_dump_name" => {
            opts.kkt_dump_name = value.to_string();
        }
```

Then rebuild ripopt:

```sh
cd ../ripopt && cargo build --release --bin ripopt
```

## Run the harvest

```sh
# Translate .mod -> .nl once (cached)
(cd ../ripopt/benchmarks/mittelmann && make translate)

# Harvest every Mittelmann problem (5 min wall-time cap, 200 iter cap each)
scripts/harvest-mittelmann-kkt.sh

# Or a subset
scripts/harvest-mittelmann-kkt.sh nql180 marine_1600 cont5_1_l
```

Output lands in `data/matrices/kkt-mittelmann/<problem>/` as paired
`.mtx` + `.json` files, one pair per IPM iteration. The default caps
keep each problem bounded; set `PER_PROBLEM_TIMEOUT` and
`PER_PROBLEM_MAX_ITER` env vars to override.

## Adding the new directory to the validation pipeline

The corpus characterizer already walks both `data/matrices/kkt/` and
`data/matrices/kkt-expansion/`. Either:

- rename the harvest output to `kkt-expansion/mittelmann_<problem>/`
  (drop the `kkt-mittelmann` prefix), so `scripts/characterize-corpus.py`
  picks it up automatically, or
- extend `KKT_ROOTS` in `scripts/characterize-corpus.py` to include
  `data/matrices/kkt-mittelmann/`.

Same choice for the bench harness loader (`src/bin/bench.rs::load_kkt_dir`).

## Why this is worth doing

The current 183k corpus is dominated by tiny CUTEst Hessians
(median n = 9). The Mittelmann set is the opposite extreme: every
instance is at least n = 500, half are above n ≈ 10⁴, and several
exceed n ≈ 10⁵. Adding even a few thousand iteration dumps from this
set populates a region of the (n, density, condition) space the
current corpus barely covers, and each one is from a workload where
the IPM struggled — which is precisely where the linear solver's
robustness gets tested.
