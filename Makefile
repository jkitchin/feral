# feral — top-level convenience Makefile.
#
# The canonical build system is cargo; this Makefile only wraps the
# most common invocations and the IPOPT shim build (which is itself
# a Makefile under feral-ipopt-shim/).
#
# Common targets:
#   make            — debug build of the workspace (cargo build)
#   make release    — release build of the workspace
#   make test       — cargo test (workspace)
#   make bench      — cargo run --bin bench --release
#   make lint       — cargo fmt --check + cargo clippy -D warnings
#   make fmt        — cargo fmt
#   make check      — cargo check --all-targets
#   make doc        — cargo doc --no-deps --open
#   make ipopt      — build Ipopt 3.14 with feral as linear_solver
#   make hs071      — run the hs071 sample under Ipopt+feral
#   make clean      — cargo clean (does not touch the Ipopt build)
#   make distclean  — cargo clean + clean the Ipopt shim build
#
# Override CARGO_FLAGS / TEST_FLAGS on the command line as needed.

CARGO       ?= cargo
CARGO_FLAGS ?=
TEST_FLAGS  ?=

SHIM_DIR    := feral-ipopt-shim

.PHONY: all build release test bench lint fmt fmt-check clippy check doc \
        ipopt hs071 shim-clean clean distclean help

all: build

build:
	$(CARGO) build $(CARGO_FLAGS)

release:
	$(CARGO) build --release $(CARGO_FLAGS)

test:
	$(CARGO) test $(CARGO_FLAGS) -- $(TEST_FLAGS)

bench:
	$(CARGO) run --bin bench --release $(CARGO_FLAGS)

check:
	$(CARGO) check --all-targets $(CARGO_FLAGS)

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --all-targets $(CARGO_FLAGS) -- -D warnings

lint: fmt-check clippy

doc:
	$(CARGO) doc --no-deps --open $(CARGO_FLAGS)

# IPOPT integration — delegates to feral-ipopt-shim/Makefile.
# That Makefile rebuilds the feral staticlib itself, so no explicit
# dependency on `release` is needed here.
ipopt:
	$(MAKE) -C $(SHIM_DIR) all

hs071:
	$(MAKE) -C $(SHIM_DIR) hs071-feral

shim-clean:
	$(MAKE) -C $(SHIM_DIR) clean

clean:
	$(CARGO) clean

distclean: clean shim-clean

help:
	@sed -n 's/^# \{0,1\}//p' Makefile | sed -n '1,30p'
