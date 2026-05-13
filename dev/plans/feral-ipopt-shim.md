# Plan — POC: build Ipopt with feral as `linear_solver=feral`

Date: 2026-05-13 (rewritten as POC scope)
Research: `dev/research/feral-ipopt-c-shim.md`
Scope: minimal proof-of-concept. Productionization deferred.

## Goal

Build a custom Ipopt binary against the vendored Ipopt source at
`ref/Ipopt/`, with the option `linear_solver=feral` selecting
feral as the symmetric-indefinite linear solver. Demonstrate
by running Ipopt's bundled `hs071` sample NLP to convergence.

## Acceptance

`hs071` solves to `Solve_Succeeded` with `linear_solver=feral`,
and the reported final objective is the textbook value
`17.014017` (Ipopt's own sample test value). No further bars.

## Steps

### 1. Add a C ABI to feral (~150 lines Rust, one file)

- Edit `Cargo.toml`: `[lib].crate-type = ["staticlib", "rlib"]`.
  No workspace conversion, no new crate.
- Create `src/capi.rs`. Hand-pick the minimal surface:
  ```c
  void*  feral_new(void);
  void   feral_free(void*);
  int    feral_set_structure(void*, int n, int nnz,
                             const int* ia, const int* ja);
  double* feral_values_ptr(void*);
  int    feral_factor(void*, int check_neg, int expected_neg);
  int    feral_solve(void*, int nrhs, double* rhs);
  int    feral_num_neg(const void*);
  ```
  Status codes: 0=SUCCESS, 1=SINGULAR, 2=WRONG_INERTIA,
  3=FATAL. Matches Ipopt's `ESymSolverStatus` (no CALL_AGAIN
  for POC).
- All `extern "C" fn` bodies wrapped in `catch_unwind`;
  panics become `FATAL`. No last-error string plumbing.
- Hardcode `Solver::new()` defaults — no options forwarding.
- Hand-write `feral-ipopt-shim/include/feral_capi.h` to
  mirror these signatures. ~15 lines.
- Test: `cargo build --release` produces
  `target/release/libferal.a`.

### 2. C++ shim (~150 lines, two files)

- `feral-ipopt-shim/include/FeralSolverInterface.hpp` —
  class declaration.
- `feral-ipopt-shim/src/FeralSolverInterface.cpp` —
  pure forwarding to the C ABI. `MatrixFormat()` returns
  `CSR_Format_0_Offset`. `ProvidesInertia()` returns
  `true`. `IncreaseQuality()` returns `false` (POC scope:
  no escalation). Option-reading in `InitializeImpl` is a
  no-op (`return true`).

### 3. Ipopt patch (~10 lines)

`feral-ipopt-shim/patches/ipopt-feral.patch` adds two
hunks to `ref/Ipopt/src/Algorithm/IpAlgBuilder.cpp`:

- One `AddStringValueToList("linear_solver", "feral", ...)`
  in `RegisterOptions`.
- One `else if (linear_solver == "feral")` branch in
  `SymLinearSolverFactory` constructing
  `new FeralSolverInterface()`.

### 4. Build glue (~30 lines Makefile)

`feral-ipopt-shim/Makefile`:

```
IPOPT_SRC   ?= ../ref/Ipopt
IPOPT_BUILD ?= $(IPOPT_SRC)/build-feral
FERAL_LIB   ?= ../target/release/libferal.a

# Step a: build feral as a staticlib.
$(FERAL_LIB):
	cd .. && cargo build --release

# Step b: drop shim into Ipopt source tree, apply patch.
$(IPOPT_SRC)/.feral-patched:
	cp include/FeralSolverInterface.hpp \
	   $(IPOPT_SRC)/src/Algorithm/LinearSolvers/
	cp src/FeralSolverInterface.cpp \
	   $(IPOPT_SRC)/src/Algorithm/LinearSolvers/
	cp include/feral_capi.h \
	   $(IPOPT_SRC)/src/Algorithm/LinearSolvers/
	cd $(IPOPT_SRC) && patch -p1 < ../../feral-ipopt-shim/patches/ipopt-feral.patch
	# Also need to add FeralSolverInterface.cpp to the Makefile.am
	# in src/Algorithm/LinearSolvers/ — handled by the patch.
	touch $@

# Step c: configure + build Ipopt with feral linked.
$(IPOPT_BUILD)/Makefile: $(IPOPT_SRC)/.feral-patched $(FERAL_LIB)
	mkdir -p $(IPOPT_BUILD)
	cd $(IPOPT_BUILD) && ../configure \
	    --without-hsl --without-spral --without-pardiso \
	    --with-mumps \
	    ADD_CFLAGS="-I$(CURDIR)/include" \
	    ADD_CXXFLAGS="-I$(CURDIR)/include" \
	    LIBS="$(CURDIR)/$(FERAL_LIB) -ldl -lm"
	cd $(IPOPT_BUILD) && make -j

# Step d: run hs071 with feral.
hs071-feral: $(IPOPT_BUILD)/Makefile
	cd $(IPOPT_BUILD) && make test \
	    || echo "(falling back to direct hs071 run)"
	# Direct: build the bundled hs071 example linking the new libipopt,
	# then run it with linear_solver=feral set in an ipopt.opt file.

.PHONY: hs071-feral
```

Manual fallback if any of the above resists automation:
copy the files in by hand, apply the patch with `patch -p1`,
run `./configure && make` in `ref/Ipopt`, then run hs071.

### 5. Verify

Run `make hs071-feral`. Expect:
- Build succeeds.
- `hs071` converges in <30 iterations.
- Final objective ≈ `17.014017`.

If iteration count looks pathological or inertia warnings
appear, that's a signal — investigate before declaring done.

## Anti-scope (deferred until after POC)

- Option forwarding (`feral_pivtol`, scaling, parallel).
- `SYMSOLVER_CALL_AGAIN` / `IncreaseQuality`.
- `feral_last_error_message`.
- cbindgen, CI integration, cross-platform builds.
- Workspace split / `feral-capi` separate crate.
- CUTEst harness.
- Performance comparison vs MUMPS.

These come back as separate work items only after the POC
demonstrates the integration is viable end-to-end.

## Risks (POC-scoped)

1. **Ipopt's autoconf machinery** may not like the
   `LIBS=...libferal.a` injection. Fallback: edit
   `src/Algorithm/LinearSolvers/Makefile.am` directly to
   add `libferal.a` to `libipopt_la_LIBADD` and run
   `autoreconf`.
2. **Static-lib symbol visibility on macOS.** `libferal.a`
   may need to be passed with `-force_load` to keep the C
   ABI symbols. Fallback: build a `cdylib` instead and
   link dynamically.
3. **hs071 needing IncreaseQuality.** If it does, the POC
   has to grow the `feral_increase_quality` entry —
   estimate +20 lines Rust, +5 lines C++.

## Sequencing

Single commit "feat: POC — Ipopt with feral linear solver"
covering steps 1–4. Verify (step 5) before committing.
If hs071 doesn't converge, iterate on the same uncommitted
patch — no plan revision needed.
