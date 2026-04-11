# Adversarial Stress-Testing with Competing Agents

**Status:** Exploratory research note
**Date:** 2026-04-02
**Related spec sections:** 4.1 (Tier 3: Constructed stress tests), 4.2 (Benchmark Harness), 4.3 (Metrics)

## 1. Motivation

FERAL's benchmark suite (Section 4.1) draws from three main sources: KKT matrices
extracted from ripopt/POUNCE solves (Tier 1), SuiteSparse matrices with MUMPS-generated
sidecars (Tier 2), and hand-constructed stress tests targeting known failure modes (Tier 3).
All three sources share a limitation: they reflect matrices that humans thought to construct
or that arose from real applications. The space of valid symmetric indefinite systems is
vastly larger than what any curated collection covers.

Adversarial stress-testing addresses this gap. A red-team agent systematically searches
for matrix instances that maximize difficulty for the blue-team agent (FERAL), producing
test cases that no human anticipated. This is the numerical linear algebra analogue of
fuzz testing in software engineering, or of generative adversarial training in machine
learning. The key difference from random fuzzing is that the red agent is *directed*:
it optimizes for difficulty rather than sampling uniformly.

This maps to the Evaluation Science pillar of AI-assisted scientific computing. The
artifact is not just a better-tested solver, but a reusable benchmark suite of hard
instances that the sparse solver community currently lacks. Existing collections like
SuiteSparse are invaluable but reflect application-driven sampling. Adversarially
generated instances sample the *difficulty frontier*, which is complementary information.


## 2. Architecture: Red Agent and Blue Agent

### 2.1 Blue Agent (Defender)

The blue agent is FERAL itself, run through the standard benchmark harness (Section 4.2).
It takes a matrix, factorizes it, solves a system, and reports:

- Inertia (positive, negative, zero)
- Residual norm: ||Ax - b||_2 / ||b||_2
- Factorization time
- Solve time
- Peak memory
- Element growth factor (max |L_ij| over factorization)

No modifications to FERAL are needed. The blue agent is purely passive.

### 2.2 Red Agent (Attacker)

The red agent constructs matrix instances by manipulating a parameterized generator
and evaluating FERAL's response. Its objective is to maximize a composite difficulty
score (Section 3). The red agent operates in one of two modes:

**Black-box mode.** The red agent sees only FERAL's external outputs (inertia, residual,
timing). This is the primary mode. It produces instances that stress any solver with the
same interface, making the resulting benchmarks portable.

**White-box mode.** The red agent additionally inspects FERAL's internal state: pivot
sequence, element growth at each step, delayed pivot count, number of 2x2 blocks.
This mode is more powerful for finding FERAL-specific weaknesses but couples the
generator to implementation details. Use white-box mode during development to find
bugs, then validate the hardest instances in black-box mode to confirm they are
inherently hard (not just hard for FERAL's current implementation).

### 2.3 Interaction Protocol

```
for round in 1..max_rounds:
    red_agent generates matrix A, rhs b, and declares ground-truth inertia
    blue_agent = FERAL.factor(A) then FERAL.solve(b)
    scorer evaluates (inertia_correct, residual, time, growth)
    red_agent receives score, updates its search strategy
    if score exceeds threshold:
        save (A, b, metadata) to candidate pool
    end
end
filter candidate pool for diversity (no near-duplicates)
promote hardest instances to Tier 3 benchmark set
```

The ground-truth inertia declared by the red agent must be verified independently.
For dense matrices, eigenvalue decomposition via a high-precision library (e.g.,
computing eigenvalues in f128 or using an arbitrary-precision library) provides
the oracle. For sparse matrices, MUMPS/rmumps serves as the reference, consistent
with the existing Tier 2 sidecar approach.


## 3. Difficulty Score

The red agent optimizes a composite score. Each component targets a distinct failure mode.

### 3.1 Inertia Misclassification (highest priority)

FERAL's spec requires exact inertia. The most valuable adversarial instances are those
where FERAL reports wrong inertia counts. The red agent targets this by constructing
matrices with eigenvalues clustered near zero, where floating-point rounding during
pivoting could cause a small positive eigenvalue to be classified as negative or vice
versa.

Score component: `S_inertia = 1.0` if inertia is wrong, `0.0` otherwise. Any instance
with `S_inertia = 1.0` is an automatic critical finding regardless of other scores.

### 3.2 Residual Blowup

Even with correct inertia, the solution may be inaccurate. The residual
`||Ax - b||_2 / ||b||_2` measures backward stability. The red agent tries to maximize
this, particularly seeking instances where the residual exceeds the condition-dependent
tolerance from Section 2.10 of the spec.

Score component: `S_residual = log10(residual / tolerance)`. Positive means the
tolerance was exceeded.

### 3.3 Element Growth

Bunch-Kaufman pivoting guarantees bounded element growth in theory. The bound is
`(1 + 1/alpha)^(n-1)` where `alpha = (1 + sqrt(17))/8`, which is exponential but
rarely approached in practice. The red agent searches for matrices that produce
large growth factors, testing whether the theoretical worst case is achievable.

Score component: `S_growth = log10(growth_factor) / n`. Normalized by dimension
so that different-sized matrices are comparable.

### 3.4 Performance Degradation

Timing is secondary to correctness but still valuable. The red agent can target
matrices where FERAL is disproportionately slow relative to dimension (e.g.,
O(n^3) with a large constant due to excessive pivot searches or 2x2 block
processing).

Score component: `S_time = log10(time / expected_time)` where `expected_time`
is calibrated from the median time for matrices of the same dimension.

### 3.5 Composite Score

```
S = 100 * S_inertia + 10 * max(S_residual, 0) + 5 * S_growth + S_time
```

The weights reflect priority: inertia correctness dominates, then numerical accuracy,
then growth control, then performance. The specific weights are tunable.


## 4. Matrix Generation Strategies

### 4.1 Eigenvalue-Based Construction (Dense, Stage 1)

For dense matrices, the most direct construction is:

```
A = Q * D * Q^T
```

where Q is a random orthogonal matrix (generated via QR of a random matrix) and D is
a diagonal matrix with prescribed eigenvalues. The red agent controls D directly.

Parameterization of D:
- `n_pos`, `n_neg`, `n_zero`: counts summing to n (defines target inertia)
- `lambda_min_pos`: smallest positive eigenvalue (controls near-zero clustering)
- `lambda_min_neg`: most negative eigenvalue closest to zero
- `spread`: ratio of largest to smallest eigenvalue magnitude (condition number proxy)
- `cluster_width`: how tightly eigenvalues cluster near zero

The hardest instances will have `lambda_min_pos` and `|lambda_min_neg|` near machine
epsilon times `spread`, creating eigenvalues that are barely distinguishable from zero
in finite precision.

### 4.2 Perturbation-Based Construction

Start from a known matrix (e.g., a Tier 1 KKT matrix) and perturb it:

```
A' = A + epsilon * E
```

where E is a structured perturbation (low-rank, diagonal, or sparse). The red agent
searches for perturbations that degrade FERAL's performance on A' relative to A.
This is particularly useful for finding sensitivity to small changes in matrix entries,
which matters for NLP solvers where the KKT matrix changes slightly at each iteration.

### 4.3 Structured Sparse Construction (Stage 2)

For sparse matrices, the red agent generates both the sparsity pattern and the values:

- **Arrow matrices:** Dense border + diagonal core. Common in decomposition-based NLP.
  The red agent varies the border width and diagonal conditioning.
- **Bordered block-diagonal:** Multiple dense diagonal blocks connected by coupling
  columns. Controls block size, number of blocks, and coupling density.
- **AMD-adversarial patterns:** Patterns designed to make AMD produce high fill, e.g.,
  matrices where the minimum degree heuristic makes systematically poor choices. These
  test ordering robustness.
- **KKT-structured:** Matrices of the form `[[H, A^T], [A, -delta*I]]` where H is
  the Hessian block and A is the constraint Jacobian. The red agent varies H's
  indefiniteness, A's rank deficiency, and delta's magnitude.

### 4.4 KKT Validity Constraint

For the generated matrices to be relevant to FERAL's target use case (NLP solving),
the red agent should optionally constrain matrices to have valid KKT structure. A
valid KKT matrix has the form:

```
K = [ H    J^T ]
    [ J   -D   ]
```

where H is n x n symmetric (not necessarily positive definite), J is m x n with
m <= n, and D is m x m diagonal with non-negative entries. The expected inertia for
a non-degenerate problem is (n, m, 0).

Enforcing this structure means the red agent searches within the space of matrices
that could actually arise from an optimization problem, producing benchmarks that
are directly relevant rather than purely pathological.


## 5. Search Strategies for the Red Agent

### 5.1 Random Search with Filtering

The simplest approach. Generate random parameterizations, evaluate the difficulty
score, keep instances above a threshold. Effective for finding some hard cases but
inefficient in high-dimensional parameter spaces.

### 5.2 Bayesian Optimization

Model the difficulty score as a function of the matrix parameters using a Gaussian
process surrogate. Use acquisition functions (expected improvement, upper confidence
bound) to select the next parameterization to evaluate. This is well-suited because
each FERAL evaluation is relatively expensive (factorization cost) and the parameter
space is continuous and moderate-dimensional (5-20 parameters).

### 5.3 Evolutionary Search

Maintain a population of matrix parameterizations. Evaluate difficulty scores,
select the hardest instances, mutate and recombine their parameters, repeat. This
naturally produces a diverse set of hard instances rather than converging to a
single optimum, which is desirable for benchmark diversity.

### 5.4 Gradient-Free Optimization

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is particularly well-suited
here. It handles non-convex, noisy objectives in continuous spaces of moderate
dimension. The difficulty score is non-differentiable (inertia correctness is
binary), ruling out gradient-based methods.

### 5.5 Curriculum: Increasing Difficulty

Start with small matrices (n=4 to n=20) where each evaluation is cheap and
eigenvalue verification is fast. Once hard instances are found at small scale,
use them as templates for larger constructions. A matrix structure that causes
trouble at n=10 often causes worse trouble at n=100.


## 6. Verification and Ground Truth

The red agent must provide ground-truth inertia for every generated matrix.
Without verified ground truth, we cannot distinguish "FERAL got the wrong answer"
from "the red agent claimed a wrong ground truth."

### 6.1 Dense Matrices (Stage 1)

For eigenvalue-constructed matrices `A = Q D Q^T`, the ground-truth inertia is known
by construction: it is the sign distribution of D's diagonal. However, floating-point
arithmetic in forming `Q D Q^T` may perturb eigenvalues near zero, so the *assembled*
matrix may have different inertia than the prescribed D. The verification step is:

1. Construct A = Q D Q^T in double precision.
2. Compute eigenvalues of the assembled A using a high-precision eigensolver.
3. The ground-truth inertia is determined from these eigenvalues, not from D.

For moderate n (up to a few hundred), eigenvalue computation is feasible. For larger
dense matrices, Sylvester's law of inertia applied to a high-precision LDL^T
(e.g., using quad-precision arithmetic) provides the reference.

### 6.2 Sparse Matrices (Stage 2)

Use MUMPS/rmumps as the reference oracle, consistent with the Tier 2 sidecar approach
in the spec. MUMPS is mature and well-tested; disagreements between FERAL and MUMPS
on inertia are overwhelmingly likely to be FERAL bugs.

### 6.3 Cross-Validation

For the most critical findings (inertia disagreements), cross-validate with at least
two independent methods. For example, if FERAL and MUMPS disagree on a matrix's
inertia, compute eigenvalues directly (feasible for moderate-sized matrices) to
determine which solver is correct.


## 7. Deliverables

### 7.1 Implementation Artifacts

- `src/bin/redteam.rs`: The adversarial matrix generator binary. Parameterized by
  matrix family, search strategy, and difficulty target. Outputs matrices in MatrixMarket
  format with sidecar metadata, consistent with the benchmark harness format.

- `src/generators/mod.rs`: Matrix generation library with constructors for each family
  (eigenvalue-based, perturbation-based, KKT-structured, arrow, bordered block-diagonal).

- `scripts/adversarial-campaign.sh`: Orchestration script that runs a full red-team
  campaign, verifies ground truth, filters for diversity, and promotes hard instances
  to the benchmark set.

### 7.2 Benchmark Artifacts

- A curated set of adversarially generated matrices, promoted to Tier 3 in the
  benchmark suite. Each matrix includes:
  - The matrix in MatrixMarket format
  - Ground-truth inertia (verified by independent method)
  - RHS vector and reference solution
  - Metadata: generation parameters, difficulty score, which failure mode it targets
  - Provenance: which red-team campaign produced it and why it was selected

### 7.3 Analysis Artifacts

- A report characterizing the difficulty frontier: what matrix properties make
  Bunch-Kaufman pivoting struggle, where element growth approaches theoretical
  bounds, which sparsity patterns defeat AMD ordering.

- Comparison of FERAL's difficulty frontier against MUMPS. Instances that are hard
  for FERAL but easy for MUMPS reveal implementation-specific weaknesses. Instances
  that are hard for both reveal fundamental algorithmic limitations.


## 8. Phasing

### Phase 1 Integration (Dense Solver)

The adversarial framework starts during Stage 1, targeting dense matrices only.
This is the simplest case: eigenvalue-based construction gives full control over
the spectrum, ground truth is cheap to verify, and evaluation is fast for small n.

Concrete first step: after the dense LDL^T with Bunch-Kaufman pivoting is implemented
and passes the hand-calculated tests from the spec, run a red-team campaign with
eigenvalue-based construction at n=4 through n=64. Any inertia misclassification is
a bug to fix before moving to Stage 2.

### Phase 2 Integration (Sparse Solver)

Once the multifrontal engine exists, extend the red agent to generate sparse matrices.
The KKT-structured generator becomes primary, as these are the matrices FERAL must
handle in production. The ordering-adversarial generator tests AMD robustness.

### Ongoing

After initial campaigns, the red-team binary becomes part of the CI pipeline. Not
every commit triggers a full campaign (too expensive), but a reduced "smoke test"
campaign (small matrices, limited rounds) runs on every PR. Full campaigns run
weekly or before releases.


## 9. Relation to Existing Literature

Adversarial testing of numerical software is not new, but systematic adversarial
benchmark generation for sparse direct solvers appears to be unexplored. Related
work includes:

- **Random matrix theory** provides distributions over matrix ensembles (GOE, GUE,
  Wishart) but these are not adversarial; they sample typical-case behavior, not
  worst-case. See citet:anderson2010introduction for background.

- **Condition number estimation** (Higham, 2002) characterizes when a specific matrix
  is hard but does not generate hard matrices. The red agent inverts this: given a
  target difficulty, construct a matrix that achieves it.

- **Fuzzing for numerical code** (e.g., FPGen by Zou et al., 2015) generates
  floating-point inputs that trigger edge cases in math libraries. The adversarial
  approach here is more structured: instead of random bit patterns, the red agent
  searches over mathematically meaningful parameterizations.

- **Generative adversarial networks** (GANs) in ML use a similar two-agent
  structure but optimize a differentiable objective. The discrete nature of inertia
  correctness makes the numerical solver setting more naturally suited to
  gradient-free search.

**Note:** The references above are cited for context. The formal citations for FERAL's
core algorithms are in `dev/references.bib`. If any of these references are added to
the project bibliography, they must be verified via the citation-verifier skill before
inclusion.


## 10. Open Questions

1. **Computational budget.** A full adversarial campaign at n=200 with 10,000 rounds
   takes approximately 10,000 factorizations, each O(n^3) = O(8 * 10^6) flops. At
   1 GFLOP/s effective throughput, this is roughly 80 seconds. At n=1000, it becomes
   roughly 3 hours. What is the right budget/dimension tradeoff for CI integration?

2. **Diversity metric.** How do we ensure the promoted benchmark set covers distinct
   failure modes rather than clustering around one type of hard instance? Possible
   approaches: clustering in parameter space, minimum pairwise distance in a matrix
   feature space, or explicit stratification by failure mode.

3. **Transferability.** Do instances that are hard for FERAL also stress other solvers
   (MA57, PARDISO, MUMPS)? If so, the adversarial benchmarks have broad community
   value. If not, they are still valuable for FERAL development but less interesting
   as a published artifact.

4. **Red agent intelligence.** How sophisticated does the search strategy need to be?
   It is possible that simple random search with filtering finds most of the
   interesting instances, and Bayesian optimization adds marginal value. An empirical
   comparison of search strategies on the dense case would answer this.

5. **Human-in-the-loop.** Should a human review promoted instances before they enter
   the benchmark set? This adds quality control (filtering out instances that are hard
   due to bugs in the generator rather than genuine solver difficulty) but creates a
   bottleneck. A compromise: auto-promote instances that pass cross-validation, flag
   others for review.
