# Getting started

Add feral to your `Cargo.toml`:

```toml
[dependencies]
feral = "0.7"
```

A minimal end-to-end factorization looks like:

```rust,ignore
use feral::{factor, solve, BunchKaufmanParams, SymmetricMatrix};

let a = SymmetricMatrix::from_dense_lower(/* ... */);
let factors = factor(&a, &BunchKaufmanParams::default())?;
let x = solve(&factors, &b)?;
```

See the [`examples/`](https://github.com/jkitchin/feral/tree/main/examples)
directory in the repository for runnable programs that exercise the dense and
sparse paths, scaling, and iterative refinement.
