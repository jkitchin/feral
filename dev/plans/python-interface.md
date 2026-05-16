# Python interface to feral — scoping plan (issue #20)

Status: scoping. Not approved for implementation.

Target consumer: the IPM in **discopt**. discopt needs to call feral
the way a Fortran IPM (Ipopt, KNITRO) would call MA57 or MUMPS —
symbolic analyse once, refactor with new values every Newton iteration,
solve one or several RHS, read inertia, and react to perturbations.
The interface should also be usable standalone (scipy.sparse users,
notebooks, teaching).

## 1. Binding technology

**Decision: PyO3 + maturin, in a new `python/` subdirectory of this
repo.**

Alternatives considered:

| approach | pros | cons | verdict |
| -------- | ---- | ---- | ------- |
| PyO3 + maturin (recommended) | zero-copy numpy (`rust-numpy`), real Python objects, proper exception hierarchy, GIL release for parallel factor, builds wheels on Linux/macOS/Windows | extra `Cargo.toml`, build-system is maturin not setuptools | **chosen** |
| cffi over the existing C API (`src/capi.rs`) | no new Rust code, smallest surface | C API only exposes ~10 calls and uses int return codes — too narrow for a thorough binding; no numpy integration; clumsy error story | rejected for the IPM use case (good enough for a smoke test) |
| ctypes over C API | zero build tooling | same problems as cffi plus worse type story | rejected |
| pyo3-asyncio / async | not needed; IPM Newton loop is sequential | adds complexity | rejected |

The existing C API stays — it's how the Ipopt shim talks to feral —
but the Python binding talks to the Rust crate directly so it can give
discopt a real `numpy.ndarray`-shaped interface rather than C pointers.

## 2. Repo layout

```
feral/                       # existing Rust crate at repo root
  src/                       # unchanged
  Cargo.toml                 # unchanged
python/
  Cargo.toml                 # new — crate "feral-python", cdylib
  src/lib.rs                 # PyO3 bindings (this plan)
  pyproject.toml             # maturin backend
  feral/                     # pure-Python wrapper package
    __init__.py              # re-exports + docstrings + scipy.sparse adapters
    _typing.pyi              # type stubs
    ipm.py                   # high-level IPM-oriented helper class
  tests/
    test_basic.py
    test_scipy_interop.py
    test_ipm_pattern.py      # symbolic-reuse / inertia-correction patterns
    test_refinement.py
  examples/
    quickstart.py
    discopt_ipm_kkt.py       # end-to-end IPM Newton step sketch
  README.md
  CHANGELOG.md (or shared with crate)
.github/workflows/
  python-wheels.yml          # cibuildwheel matrix: linux x86_64/aarch64, macos universal2, windows x86_64
```

Names:
- PyPI package: **`feral-solver`** (the bare name `feral` collides
  with a defunct PyPI project; check first, and if `feral` is
  available, prefer it).
- Import: `import feral` regardless of the PyPI name (configured via
  maturin `module-name`).

## 3. Public Python API

### 3.1 Matrix types

`feral.CscMatrix` — symmetric matrix in lower-triangular CSC.

Constructors:
- `CscMatrix.from_scipy(A: scipy.sparse.csc_matrix | csr_matrix | coo_matrix, *, symmetric: Literal["lower","upper","full"] = "lower")` —
  zero-copy when `A.indptr.dtype == int32` (or `int64`, depending on
  feral's index type) and `A.data.dtype == float64`; otherwise copy
  with a one-line warning.
- `CscMatrix.from_triplet(row, col, val, n, *, accumulate_duplicates=True)` —
  takes numpy arrays.
- `CscMatrix.from_mtx(path)` — wraps `feral::io::mtx::read_mtx`.
- `CscMatrix.from_dense(np.ndarray)` — convenience for small problems.

Accessors:
- `n: int`, `nnz: int`
- `indptr`, `row_idx`, `values` — numpy views (read-only by default;
  `values` writable if `mutable=True` was passed at construction, for
  the IPM refactor pattern, see §3.5).
- `to_scipy() -> scipy.sparse.csc_matrix` — symmetrized.
- `symv(x: np.ndarray) -> np.ndarray` — matrix-vector product
  (already on `CscMatrix`).

### 3.2 Solver class

`feral.Solver` — mirrors `feral::numeric::solver::Solver` 1:1.

```python
solver = feral.Solver(
    pivot_threshold: float = 1e-3,
    parallel: bool = True,
    fma: bool = True,
    static_pivoting: bool = False,
    cascade_break_ratio: float | None = None,   # opt-in (see decisions.md)
    cascade_break_eps: float | None = None,
    scaling: Literal["none","mc64","equilibration","auto"] = "auto",
    ordering: Literal["amd","metis","natural","auto"] = "auto",
)
```

Methods:
- `factor(A: CscMatrix, *, expected_inertia: Inertia | None = None) -> FactorStatus`
- `refactor(A: CscMatrix, *, expected_inertia=None) -> FactorStatus` —
  same pattern, new values (IPM hot path). Asserts the sparsity
  pattern hasn't changed vs the last `factor()`; raises
  `PatternMismatch` if it has.
- `solve(b: np.ndarray) -> np.ndarray` — accepts 1-D `(n,)` or 2-D
  `(n, nrhs)`.
- `solve_refined(A, b, *, max_iter: int = 5, tol: float = 1e-12) -> np.ndarray`
- `estimate_condition_1norm(A) -> float`
- `increase_quality() -> bool` — promotes through `QualityLevel`.

Properties (all read-only):
- `inertia: Inertia | None`
- `num_negative_eigenvalues: int`
- `min_diagonal: float | None`
- `quality_level: QualityLevel`
- `provides_inertia: bool`
- `symbolic_call_count: int`
- `factor_nnz: int`
- `needs_refinement: bool`

Context-manager support: `with feral.Solver(...) as s:` — frees the
factors deterministically on exit (matters for large problems where GC
delays cost memory).

### 3.3 Enums and dataclasses

```python
class FactorStatus(IntEnum):
    SUCCESS = 0
    SINGULAR = 1
    WRONG_INERTIA = 2
    NUMERIC_FAILURE = 3
    PATTERN_MISMATCH = 4   # python-only, from refactor()

class QualityLevel(IntEnum):
    DEFAULT = 0
    REFINED = 1
    AGGRESSIVE = 2

@dataclass(frozen=True)
class Inertia:
    n_pos: int
    n_neg: int
    n_zero: int
    def matches(self, other: Inertia) -> bool: ...
```

### 3.4 Exception hierarchy

```python
feral.FeralError              # base
  ├── feral.FactorError       # factor() / refactor() returned non-success
  │     ├── feral.SingularError
  │     ├── feral.WrongInertiaError       # carries (actual, expected)
  │     └── feral.NumericFailure
  ├── feral.SolveError        # solve() called before successful factor, dim mismatch
  ├── feral.PatternMismatch   # refactor() pattern drift
  └── feral.IOError           # mtx parse, etc.
```

Methods returning `FactorStatus` don't raise — callers check the
enum. Methods that *can't* return a status (`solve`, `refactor` when
asked to assert success, etc.) raise. The IPM caller wants explicit
status returns from factor so it can drive perturbation logic, so the
non-raising default matters.

### 3.5 IPM helper layer (`feral.ipm`)

This is the layer discopt actually wants. It captures the
analyse-once-refactor-many pattern:

```python
class KktSolver:
    """Owns a Solver and a symbolic factorization; supports fast
    refactor with updated KKT values and inertia-driven perturbation."""

    def __init__(
        self,
        A_pattern: CscMatrix,         # symbolic skeleton (values ignored)
        expected_inertia: Inertia,    # e.g. Inertia(n=n_vars, n_neg=n_constraints)
        *,
        params: Solver | dict | None = None,
        perturb_strategy: Literal["ipopt", "off"] = "ipopt",
        delta_w_min: float = 1e-20,
        delta_w_max: float = 1e+40,
        delta_w_0: float = 1e-4,
        kappa_w_plus: float = 8.0,
        kappa_w_plus_bar: float = 100.0,
        kappa_w_minus: float = 1.0 / 3.0,
        delta_c: float = 1e-8,
    ): ...

    def factor(self, values: np.ndarray) -> FactorReport:
        """Update KKT values, refactor; if inertia is wrong, perturb
        the (1,1) block and/or regularize constraints per Ipopt
        §3.1, retry up to ~5 times. Returns the perturbations that
        landed and the final inertia."""

    def solve(self, rhs: np.ndarray, *, refine: bool = True) -> np.ndarray: ...

    def solve_pair(self, rhs_aff: np.ndarray, rhs_corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictor-corrector: two RHS, one factorization, batched."""
```

`FactorReport` is a dataclass with `inertia`, `delta_w`, `delta_c`,
`n_attempts`, `factor_time_ms`, `needs_refinement`. discopt logs
these.

The perturbation strategy mirrors Ipopt's
`PDPerturbationHandler.cpp` and uses our `with_static_pivoting` knob
only as a last resort. Nothing in this layer is novel — it's a
straight Python implementation of the Wächter–Biegler 2006 escalation
rule, sitting on top of `Solver.refactor` + `Solver.inertia`.

### 3.6 numpy / scipy interop details

- f64 throughout. No f32 path.
- Index dtype: feral uses `usize` internally; the binding will pick
  `np.int32` or `np.int64` to match (likely `int64` on 64-bit hosts
  for safety, matching scipy.sparse's `int32` default by copying when
  necessary).
- Arrays passed to `solve()` are accepted contiguous or strided;
  strided gets one auto-copy with a debug-mode warning.
- Returned `numpy.ndarray` from `solve()` owns its memory (no
  back-reference to the Rust factor — survives `solver` going out of
  scope).

### 3.7 Threading / GIL

- `factor`, `refactor`, `solve`, `solve_refined`, `estimate_condition_1norm`
  release the GIL via `py.allow_threads()` so rayon-parallel
  multifrontal work runs concurrently with Python threads.
- The solver-owned rayon `ThreadPool` (commit `91e028a`) is reused
  across all `factor()` calls — Python doesn't see the pool, it's
  internal to the wrapped `Solver`.
- `Solver` is **not** thread-safe across `factor`/`solve` from
  multiple Python threads; document this. Concurrent solves on a
  single factored `Solver` are also unsafe today (the workspace is
  shared) — flag as a future improvement.

### 3.8 Pickling / serialization

Out of scope for v1. The IPM use case doesn't need it. Document as
`__reduce__` not implemented; raise `TypeError`. Future option:
serialize symbolic factorization (`SymbolicFactorization`) so an
analyzer can ship a precomputed pattern, but skip until asked.

## 4. Build, distribution, CI

- `pyproject.toml` with `build-backend = "maturin"`.
- `Cargo.toml` of `feral-python` depends on `feral = { path = ".." }`.
- Wheels via `cibuildwheel` driven from a new
  `.github/workflows/python-wheels.yml`:
  - cp310 / cp311 / cp312 / cp313 (drop py3.9 — too old for numpy 2.x).
  - linux: manylinux_2_28 x86_64, aarch64.
  - macos: universal2.
  - windows: x86_64.
- sdist always published; users on weird platforms can build from
  source.
- abi3: target `abi3-py310` so one wheel per platform/arch covers
  3.10+. This is well-supported by PyO3 today.
- PyPI publish from a tag-triggered workflow on the main repo. Use
  trusted publishing (no API tokens).

Note on numpy 2: rust-numpy supports numpy 2 from version 0.21+.
Pin floor at numpy 1.23 (the oldest still widely deployed) and let
upper float free.

## 4a. pip / uv compatibility

This drops out of maturin + PEP 517, but call out the specifics so
nothing slips:

- **`pyproject.toml` is the only build configuration.** PEP 517
  compliant with `build-backend = "maturin"`. No `setup.py`, no
  legacy bdist. This makes `pip install feral-solver`,
  `uv add feral-solver`, `uv pip install feral-solver`,
  `poetry add feral-solver`, and `hatch run pip install feral-solver`
  all work identically.
- **Wheels on PyPI**, sdist as fallback. The sdist builds from source
  via maturin and works on any platform with a stable Rust toolchain;
  users on niche platforms can still install. `pip install` and
  `uv pip install` both pick the wheel automatically when one matches
  their Python/ABI/platform tags.
- **abi3 wheels** (`abi3-py310` PyO3 feature) — one wheel per
  platform/arch covers Python 3.10 through whatever future 3.x lands,
  which keeps the PyPI footprint small and avoids re-publishing when
  a new CPython drops. uv resolves abi3 wheels the same way pip
  does.
- **uv-native fast path.** `uv` builds the sdist via the declared
  PEP 517 backend (maturin) without any extra config; no
  `uv`-specific metadata required. `uv lock` / `uv sync` resolve
  `feral-solver` from PyPI without source builds when wheels exist.
  Tested resolver target: `uv >= 0.4`.
- **Editable installs for development.** Two supported paths inside
  `python/`:
  - `pip install -e .` (PEP 660 editable, works with uv via
    `uv pip install -e .`).
  - `maturin develop` — faster iteration, builds the Rust extension
    in place. This is the recommended dev loop; document in
    `python/README.md`.
- **`requires-python = ">=3.10"`**, `numpy >= 1.23` as runtime
  dependency, `scipy >= 1.10` as optional dependency (only needed
  for the scipy.sparse adapters; `feral.CscMatrix` and `Solver` work
  without scipy installed). Both pip and uv honour the
  optional-dependency groups via extras:
  - `pip install 'feral-solver[scipy]'`
  - `uv add 'feral-solver[scipy]'`
- **Trusted publishing** from GitHub Actions (no API tokens
  committed). PyPI's project page surfaces "Built with Rust" / "PEP
  517" automatically.
- **`uv tool install`** is *not* a target — feral is a library, not
  a CLI. Document that explicitly so users don't try it.
- **CI smoke test**: after wheels are built, run
  `uv pip install --no-cache --find-links dist feral-solver` in a
  clean container and execute the quickstart example. This catches
  metadata mistakes (wrong Python tag, missing `Requires-Dist`)
  before publish.

## 5. Testing

- `pytest` test suite in `python/tests/`:
  - **Round-trip**: build a small KKT from a known Bunch-Kaufman
    example, factor, solve, verify `||Ax - b||_inf / ||b||_inf <
    1e-10`.
  - **scipy interop**: feed `scipy.sparse.random` symmetric matrix,
    compare solve against `scipy.sparse.linalg.spsolve`.
  - **IPM pattern**: build a fixed-pattern KKT, vary diagonal
    regularization, call `refactor()` 50× and confirm symbolic call
    count stays at 1.
  - **Inertia**: synthetic SPD → `Inertia(n,0,0)`; saddle-point →
    `(n,m,0)`; rank-deficient → `(<n, ?, >0)` or
    `WrongInertiaError` depending on `expected_inertia`.
  - **Pattern drift**: call `refactor()` with extra nonzero → raise
    `PatternMismatch`.
  - **Refinement**: ill-conditioned matrix where unrefined residual
    is ~1e-6 and refined is ~1e-14.
  - **GIL release**: spawn a Python thread that loops while a big
    factor is running, confirm it makes progress (proxy: wall clock
    < single-threaded time).
  - **Memory**: factor a 500k×500k synthetic, drop solver, confirm
    RSS drops (loose threshold).
- `mypy --strict` against the `.pyi` stubs.
- Run on the same matrix corpus that the Rust tests use (mtx files
  under `data/matrices/kkt`).

## 6. Documentation

- `python/README.md`: 30-line quickstart (load mtx → factor → solve).
- A `docs/` Sphinx site is optional for v1; defer until there's
  demand. README + docstrings carry v1.
- One worked example: `examples/discopt_ipm_kkt.py` showing a single
  Newton step against a small NLP (e.g. HS071), end to end. This is
  the artifact that demonstrates the interface is "thorough enough"
  for the discopt use case.
- API reference autogenerated from docstrings; ensure every public
  method has a docstring with: one-line summary, parameters, return
  type, raises, and a small example. The Rust docstrings are the
  source of truth — copy them into the Python bindings with light
  edits.

## 7. discopt-specific concerns

These are the questions discopt will actually ask of the binding:

1. **"Can I reuse the symbolic factor across 30 Newton iterations?"**
   Yes — `refactor()` does exactly this. Symbolic call count should
   stay at 1 across all iterations.
2. **"Can I check inertia and react?"** Yes — `factor()` returns
   `FactorStatus` (including `WRONG_INERTIA` when
   `expected_inertia` is set) and `solver.inertia` carries the actual.
3. **"Can I run inertia correction without throwing away
   ordering/symbolic?"** Yes — perturb `A.values`, call
   `refactor()`. The `KktSolver` helper bakes this loop in.
4. **"Can I solve two RHS sharing one factorization
   (predictor-corrector)?"** Yes — `solve(np.column_stack([b1, b2]))`
   or the `solve_pair` helper.
5. **"What does feral do that MA57/MUMPS don't?"** Iterative
   refinement is built in (`solve_refined`). Inertia is exposed as a
   first-class enum. The opt-in `cascade_break_*` knobs are
   non-standard — document clearly that they're off by default and
   why (see `dev/decisions.md` 2026-05-15).
6. **"What about static pivoting?"** Exposed via
   `static_pivoting=True` and per-call `cascade_break_eps`. Document
   the perturbation structure (see
   `dev/research/cascade-break-l-perturbation-2026-05-15.md`).
7. **"Licensing?"** MIT — same as the Rust crate.

## 8. Out of scope for v1

- f32 / mixed-precision path.
- Complex-valued matrices.
- GPU / accelerator backends.
- Async API.
- Sub-interpreter support (PEP 684).
- Pickling.
- Dask / distributed integration.
- A high-level "scipy.sparse.linalg.factorized"-style functional
  API (we have one, but it's wrapped behind `Solver` — fine).

## 9. Estimated effort

| phase | scope | rough effort |
| ----- | ----- | ------------ |
| 1 | Skeleton: `pyproject.toml`, `Cargo.toml`, PyO3 hello-world, `Solver` + `CscMatrix` minimal API, factor+solve round-trip test | 1 session |
| 2 | Full `Solver` surface, exception hierarchy, scipy interop, numpy zero-copy | 1 session |
| 3 | `feral.ipm.KktSolver`, IPM example, inertia-correction loop | 1 session |
| 4 | `.pyi` stubs, docstrings, README, examples polish | 0.5 session |
| 5 | CI wheels (cibuildwheel), publish to TestPyPI, verify install on Linux/macOS | 1 session |
| 6 | Test pass with discopt against a real NLP, fix integration friction, publish to PyPI | 1 session |

Total: ~5.5 sessions. Could be compressed but the wheel-CI step
historically eats a session on its own.

## 10. Open decisions to confirm before implementation

1. PyPI name — `feral-solver` or `feral`? (Check availability.)
2. Minimum Python version — 3.10 (rec) or 3.9?
3. Index dtype on the Python side — `int64` (safe) or `int32`
   (matches scipy default, requires copy on large problems)?
4. Where does the `feral.ipm` layer live — in this repo (single
   source of truth) or upstream in discopt? Recommend keeping the
   thin IPM helper here so it's tested against feral's own corpus,
   and discopt depends on it; but it's a judgement call.
5. Versioning — does `feral-solver==X.Y.Z` on PyPI track the Rust
   crate version 1:1? Recommend yes, with `X.Y` matching and `.Z`
   free to bump for binding-only fixes.

## 11. References

- Internal: `src/numeric/solver.rs` (the API we're wrapping),
  `src/capi.rs` (existing C surface, not used by this binding),
  `dev/decisions.md` (cascade-break default, scaling defaults),
  `dev/research/cascade-break-l-perturbation-2026-05-15.md`,
  `dev/plans/feral-ipopt-shim.md` (parallel work on the Ipopt C
  shim — the Python binding is independent but shares the IPM use
  case).
- External: PyO3 user guide
  (https://pyo3.rs), rust-numpy
  (https://github.com/PyO3/rust-numpy), maturin
  (https://www.maturin.rs/), Wächter & Biegler 2006 (Ipopt
  perturbation strategy).
