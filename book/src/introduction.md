# Introduction

**feral** is a sparse symmetric indefinite direct solver written in pure Rust,
with certified inertia counts. It is a clean-room implementation from published
papers and BSD-licensed references, MIT-licensed, with no BLAS, LAPACK, or
Fortran dependencies in the core solver.

This book is the narrative companion to the [API reference](./api.md). The book
covers concepts, semantics, and design choices; the API reference covers types
and functions.

## Where to go next

- [Getting started](./getting-started.md) — install, link, run a small example.
- [Inertia semantics](./inertia.md) — what `Inertia` means and what feral
  guarantees on singular matrices.
- [API reference](./api.md) — the rustdoc output for the `feral` crate and
  workspace members.

## Project links

- Repository: <https://github.com/jkitchin/feral>
- Crate: <https://crates.io/crates/feral>
- Issue tracker: <https://github.com/jkitchin/feral/issues>
