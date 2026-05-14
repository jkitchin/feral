# FERAL — Agent Protocol

## At Session Start

**First, check if this is a bootstrap session:**

```bash
test -f dev/context.md && echo "normal" || echo "bootstrap"
```

- If `dev/context.md` does **not** exist → this is Session 1. Follow **Bootstrap Protocol** below.
- If `dev/context.md` exists → follow **Normal Session Protocol** below.

---

## Bootstrap Protocol (Session 1 only)

Do NOT attempt to run `assemble-context.sh` — it does not exist yet.

1. Read `FERAL-PROJECT-SPEC.md` in full before writing any code
2. Read `dev/references.bib` to understand the literature foundation
3. Follow Section 13.1 to initialize the Cargo project and directory structure
4. Follow Section 13.2 first session goals in order:
   a. Set up CI (GitHub Actions: test, clippy, fmt, grep for unwrap)
   b. Implement core data structures: CSC sparse matrix, dense matrix, `Inertia` type
   c. Implement scalar (unblocked) dense LDLᵀ with Bunch-Kaufman pivoting
   d. Write exact tests using small matrices from the Bunch-Kaufman paper
   e. Write the benchmark harness skeleton (`cargo run --bin bench` runs, no matrices yet)
   f. Write `dev/assemble-context.sh` and run it to produce `dev/context.md`
5. Follow the normal **At Session End** protocol below to write the checkpoint

Before implementing the dense LDLᵀ, write the research note at `dev/research/dense-ldlt.md`
covering the items in Section 13.3 of the spec. This is mandatory — no implementation
without the research note first.

---

## Normal Session Protocol

### Orient

1. Run `./dev/assemble-context.sh`
2. Read `dev/context.md` — this is your orientation. It has a 350-line budget;
   lower-priority items (older tried-and-rejected) may be truncated.
3. Identify your goal from the "next session should" section at the top
4. If starting a new feature, **before writing any code**:
   - Read the relevant note in `dev/research/` and plan in `dev/plans/`
   - Search `dev/tried-and-rejected.md` for entries mentioning this feature —
     `context.md` only shows recent entries; the full history is in that file

### Work

- Follow the feature development lifecycle in the spec (Section 5.1): research →
  code inspection → plan → tests first → implement → benchmark
- Commit frequently and atomically — one commit per logical change
- Every commit message must have a body (what, why, evidence). No body = reject.
- Run `cargo test` before every commit
- **Install `pre-commit` once per clone: `pre-commit install`**. After that
  `cargo fmt --check` and `cargo clippy -- -D warnings` run automatically
  on every `git commit` and CI uses the identical hooks via
  `pre-commit/action`. Skip/override is not allowed; fix the offending
  code instead. See `.pre-commit-config.yaml`.
  - If `pre-commit install` errors with "Cowardly refusing to install hooks
    with `core.hooksPath` set", a global git config (often from another tool)
    is hijacking the hooks path. Override per-repo with
    `git config --local --unset core.hooksPath`, then re-run
    `pre-commit install`. Verify with `ls .git/hooks/pre-commit` (should
    exist and reference pre-commit). Without this, local commits silently
    skip fmt/clippy and CI will catch the drift — as happened on
    e8dab31 (cargo fmt fix-up).
  - Until hooks are confirmed installed, run
    `cargo fmt && cargo clippy --all-targets -- -D warnings` manually
    before every commit. Treat a missing hook as a bug to fix, not a step
    to live with.
- If you try something and abandon it, record it in `dev/tried-and-rejected.md`
  immediately — do not wait for the checkpoint

### Journal

Maintain a per-session journal at `dev/journal/YYYY-MM-DD-NN.org` (same numbering
as session files). Append an entry whenever something meaningful happens: a decision,
a finding, a failed attempt, a pivot, a benchmark result, a surprise.

Format (org-mode):

    * HH:MM :tag1:tag2:
    What was tried or discovered, what was observed (include evidence:
    error messages, test output, numbers), and what was concluded.
    Note any implications for future work.

Rules:
- Succinct but complete — a person or agent reading this cold should
  understand what happened and why
- Every claim needs evidence (test output, numbers, error text)
- Reuse existing tags when they fit; create new ones when they don't
- Tags emerge organically — do not prescribe a fixed set
- Write entries in real time as you work, not retroactively at session end
- The journal is append-only within a session — do not edit prior entries

The journal is an archive, not part of `context.md`. To query it, use the
journal agent (see below).

### Journal Agent

When you need historical context that isn't in `context.md`, spawn a
sub-agent to search the journal. The agent should:
- Search `dev/journal/*.org` for relevant tags, keywords, or date ranges
- Summarize findings relevant to the current task
- Flag prior failed attempts at the same problem
- Report benchmark trends if asked

Do not feed journal contents into `context.md` automatically. Query on demand.

### At Session End (MANDATORY)

1. Run `cargo run --bin bench --release` and record results
2. Run `./dev/assemble-context.sh` to refresh `dev/context.md`
3. Write session checkpoint to `dev/sessions/YYYY-MM-DD-NN.md` using the template
   in `dev/templates/session.md`
4. If anything was abandoned: append to `dev/tried-and-rejected.md`
5. If any architectural decisions were made: append to `dev/decisions.md`
6. If changes are user-visible: append to `CHANGELOG.md` Unreleased section
7. Commit the session file and all `dev/` changes as the final commit

---

## Hard Rules

- **NEVER** loosen a test tolerance without human approval. Record justification in
  the session checkpoint before asking.
- **NEVER** skip the checkpoint. A session without a checkpoint is lost work.
- **NEVER** modify existing entries in `decisions.md` or `tried-and-rejected.md`.
  These are append-only logs.
- **NEVER** use `unwrap()` or `expect()` in `src/`. Use proper `Result` error handling.
- **NEVER** introduce `unsafe` without a safety comment explaining the invariant.
- **NEVER** commit without running `cargo test` and `cargo clippy -- -D warnings`.
- **NEVER** write both the implementation and the test oracle in the same session
  without the oracle coming from an external source (hand calculation, reference
  solver, or the Bunch-Kaufman paper).
- When recording abandoned approaches: state symptoms, incorrect outputs, failing
  test cases. Do not reframe failure as a design choice.
- When benchmark numbers are worse than the previous session: report this explicitly
  at the top of the checkpoint. Do not omit unfavorable comparisons.

---

## Constraints (hard, do not change without recording in decisions.md)

- MIT license
- Pure Rust, stable toolchain
- Zero non-Rust dependencies in the core solver (no BLAS, LAPACK, Fortran)
- Clean-room implementation from published papers and BSD-licensed references
- Inertia must be exactly correct on non-singular matrices. On matrices where the
  canonical Fortran direct solvers (MUMPS 5.8.2 and SPRAL SSIDS) disagree on
  inertia, feral must agree with at least one of them. The corpus consensus
  framework (`external_benchmarks/consensus/compute_consensus.py`) tags matrices
  with no 3-of-4-oracle agreement as `excluded`; those matrices are not part of
  the inertia gate. See `dev/research/inertia-triage-2026-04-27.md` for the
  bucket analysis underlying this clarification.
- Correctness before performance, always
- rmumps (`../ripopt/rmumps`) is a testing reference only, not an architectural dependency
- Full BibTeX references: `dev/references.bib`
- Full project spec: `FERAL-PROJECT-SPEC.md`

<!-- crucible-project -->
## Crucible Knowledge Base

This project has a Crucible knowledge base in `.crucible/`.
Use the `crucible` CLI to ingest sources, search, and maintain the wiki.

Layout: `.crucible/sources/` (primary sources), `.crucible/wiki/` (distilled articles),
`.crucible/crucible.db` (graph database).

Conventions: org-mode with scimax, org-ref citations, narrative prose.
The LLM maintains the wiki; manual edits are the exception.
Run `crucible help all` for the full CLI reference.
<!-- crucible-project -->
