# Research Note: SuiteSparse Mongoose

**Session:** 2026-04-18 (follow-up research, pre-plan)
**Scope:** Evaluation of SuiteSparse Mongoose as a possible fifth
fill-reducing ordering crate in FERAL (`feral-mongoose`), a source of
ideas for existing crates, or a no-op.
**Trigger:** `tasks.org` lines 9–16 — "another data point alongside
`feral-amd`/`feral-metis`/`feral-scotch` for the large-KKT bakeoff."
**Decision required:** proceed with crate, cherry-pick ideas into
existing crates, or skip.

## 1. Origin, authors, citation

Mongoose was developed at Texas A&M University and University of
Florida by Scott P. Kolodziej, Timothy A. Davis, William W. Hager,
and S. Nuri Yeralan. The published reference is:

> T. A. Davis, W. W. Hager, S. P. Kolodziej, and S. N. Yeralan,
> "Algorithm 1003: Mongoose, A Graph Coarsening and Partitioning
> Library," *ACM Transactions on Mathematical Software* 46(1),
> Article 7, March 2020, 18 pp. DOI: 10.1145/3337792.

Open-access copies are at
<https://people.clas.ufl.edu/hager/files/mongoose.pdf> and
<https://people.engr.tamu.edu/davis/suitesparse_files/mongoose-ACMTOMS.pdf>.
ACM landing page: <https://dl.acm.org/doi/10.1145/3337792>.
NSF PAR copy: <https://par.nsf.gov/biblio/10170049>.

The standalone repository (archival, last touched before the code was
folded into SuiteSparse) is <https://github.com/ScottKolo/Mongoose>.
The living codebase is under SuiteSparse at
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/stable/Mongoose>.

## 2. License — BLOCKER

SuiteSparse Mongoose is **GNU General Public License, version 3
(`GPL-3.0-only`)**. Verbatim from the README on the SuiteSparse
`stable` branch
(<https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/stable/Mongoose/README.md>):

> "Mongoose is licensed under the GNU Public License, version 3."
>
> `SPDX-License-Identifier: GPL-3.0-only`

Implications for FERAL:

- FERAL is MIT-licensed and must stay that way (`CLAUDE.md`
  "Constraints"). GPL source code cannot be copied, translated, or
  closely paraphrased into FERAL without relicensing FERAL.
- Davis-authored SuiteSparse components are split across licenses:
  AMD/COLAMD/CCOLAMD are BSD-3-Clause, CHOLMOD is LGPL/GPL by module,
  but Mongoose is uniformly GPL-3.0. No BSD/MIT-licensed
  Davis-authored reference implementation of Mongoose exists.
- The **paper** is freely readable and citable. Clean-room
  implementation from the paper alone is the only acceptable path,
  per FERAL's existing protocol for METIS, SCOTCH, and KaHIP (also
  GPL or restricted but with open papers). The Mongoose paper is
  detailed enough that a clean-room implementation is feasible — see
  §3.

The license does not block citing Mongoose as prior art, reporting
benchmarks against the GPL binary externally (as METIS comparisons
already do), or reimplementing from the paper.

## 3. Algorithm summary (from the paper)

### 3.1 What Mongoose actually solves

**Important:** The published version of Mongoose computes
**edge separators**, not vertex separators, and does **not** perform
nested dissection ordering. The README says:

> "Currently, Mongoose only supports edge partitioning, but in the
> future a vertex separator extension will be added."

The paper's future-work section announces the intent to "extend this
edge partitioning library to compute vertex separators and to
ultimately incorporate this work into a nested dissection framework
for computing fill-reducing orderings" — but as of the 2020
publication and the 2026-04 SuiteSparse `stable` branch, that
extension has not shipped. So the description "ND-outer + AMD-inner
hybrid ordering" in `tasks.org` is at best the project's stated
goal, not what the released code does.

In other words, Mongoose as shipped is **an edge-bisection engine**
comparable to the "bisection kernel" inside
`feral-metis::bisection` / `feral-kahip::bisection`, not a complete
ordering library like AMD or METIS.

### 3.2 Multilevel framework

Mongoose follows the standard Karypis/Kumar three-phase multilevel
template, instantiated with novel choices:

1. **Coarsening** — shrink `G_0 → G_1 → … → G_L` by matching.
2. **Initial partitioning** — compute an edge cut on the coarsest
   graph `G_L`.
3. **Uncoarsening + refinement** — project the cut from `G_k` back to
   `G_{k-1}` and refine.

### 3.3 Coarsening — stall-free HEMSR

Heavy-Edge Matching (HEM) alone can "stall": a high-degree vertex
may have every neighbor already matched, forcing it to remain
unmatched at this level and blocking further size reduction. Mongoose
augments HEM with a second pass of three matching rules applied to
the leftover unmatched vertices. The paper refers to the combined
scheme as **HEMSR — Heavy-Edge Matching with Stall-Reducing /
Stall-Free matches**:

- **Heavy-Edge Matching (HEM):** first pass. Each unmatched vertex
  `v` is matched with the unmatched neighbor `u` sharing the
  maximum-weight edge. Same as METIS/SCOTCH.
- **Brotherly matching:** two unmatched vertices that share a common
  (already-matched) neighbor can be matched to each other, even
  though they are not directly adjacent. Addresses the case where
  HEM leaves both vertices of a "V" structure unmatched.
- **Adoption matching:** an unmatched vertex with a matched neighbor
  `m` joins `m`'s existing match group, growing it from two vertices
  to three. Produces super-nodes of size 3+.
- **Community matching:** more aggressive regrouping of unmatched
  vertices by small-community heuristic. The paper notes this
  "does not appear to offer a significant improvement, but it can be
  mildly helpful in coarsening graphs that are prone to stalling."

**Theoretical guarantee:** using HEM + brotherly + adoption +
community with no degree threshold is **stall-free** (every level
strictly shrinks); using HEM + brotherly + adoption only is
**stall-reducing** (usually shrinks, not guaranteed). The paper
claims this is the first coarsening scheme with a stall-free
guarantee and demonstrates real wins on power-law / social-graph
instances where METIS-style HEM-only stalls badly.

### 3.4 Initial partitioning

The paper tries several "guess functions" on the coarsest graph and
picks the best cut. The options explored include a pseudoperipheral
BFS-style seed (similar to METIS's GGP) and a uniform
`x = 0.5`-everywhere initializer fed into the QP refinement (see
§3.5). The key insight is that because refinement is strong, the
initial partition quality matters less than the downstream QP + FM.

### 3.5 Refinement — the "2-phase" hybrid

This is Mongoose's main algorithmic contribution. Each uncoarsening
level applies a **hybrid 2-phase refinement**:

**Phase A — Boundary FM (Fiduccia–Mattheyses, combinatoric).**
Standard boundary-restricted FM: only vertices currently on the cut
boundary are candidates for swap. Each pass computes per-vertex
gain, picks the best (subject to balance), swaps it, and updates
neighbors. Vertices enter the boundary set when a swap places them
next to the cut; they leave when further swaps pull them away.

**Phase B — Gradient-projection QP (continuous).** The edge-cut
problem is lifted to the continuous quadratic program

> minimise `xᵀ(D − A)x`
> subject to `0 ≤ x ≤ 1`, `1ᵀx = k`

where `A` is the adjacency matrix, `D = diag(A·1)`, and `x_v ∈ [0,1]`
encodes the fractional partition assignment of vertex `v`. Solving
this QP to global optimality is NP-hard, but a **single iteration of
gradient projection** is cheap, and the paper shows it complements FM
well: FM makes discrete swaps that QP could not reach, and QP moves
away from FM's local minima by perturbing many vertices at once. The
continuous `x_v` is discretised at the end of the phase with
threshold `0.5` to recover a valid bipartition.

The two phases are **interleaved** within a level: the paper's
pseudocode alternates FM rounds with QP iterations until the cut no
longer improves (or a cap is hit) before proceeding to uncoarsen to
the next finer graph. On the way back up, the same pair runs again
at each level.

The paper's core claim is that **the hybrid strictly outperforms
either phase alone** and is the main driver of Mongoose's edge-cut
quality vs. METIS.

### 3.6 What is NOT in the paper

- No vertex separator. No min-vertex-cover or flow step to shrink an
  edge cut into a vertex separator. (FERAL's `feral-metis::separator`
  already does this, M6; FERAL's `feral-kahip` plans flow-based
  refinement in K4–K6.)
- No nested-dissection recursion. No AMD-on-leaves. No postorder of
  elimination tree.
- No separator-to-ordering permutation step. The permutation
  `π = [π_left, π_right, π_sep]` that FERAL's pipeline produces is
  entirely outside Mongoose's scope.

So the `tasks.org` description — "ND-style outer separator recursion
with AMD on the leaves" — describes **what Mongoose would become** if
the promised vertex-separator + ND extension ever shipped. Today it
is only the edge-bisection engine.

## 4. Relationship to existing FERAL crates

| FERAL crate     | Overlap with Mongoose                                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `feral-amd`     | None. AMD is the leaf orderer in a *hypothetical* Mongoose-ND; Mongoose itself does not do minimum-degree.                   |
| `feral-metis`   | Heavy overlap. Both are multilevel edge-bisection → vertex-separator → nested-dissection pipelines using HEM + FM.           |
| `feral-scotch`  | Partial overlap (multilevel framework, FM-style refinement). SCOTCH's band-FM and graph compression are distinct techniques. |
| `feral-kahip`   | Partial overlap (multilevel framework; KaHIP uses flow-based refinement instead of QP).                                       |

Where Mongoose differs from everything FERAL already has:

1. **HEMSR coarsening** (brotherly + adoption + community matching).
   `feral-metis::coarsen` uses HEM only and therefore can stall on
   power-law graphs. This is a concrete, portable idea that could
   be added to `feral-metis` without a new crate.
2. **QP refinement** as an alternating partner to FM. Neither METIS
   nor SCOTCH nor KaHIP use gradient-projection QP. KaHIP uses
   max-flow / min-cut; SCOTCH uses diffusion and band-FM; METIS uses
   FM and KL. QP is a genuinely different discrete-continuous
   hybrid.

Where Mongoose is weaker than what FERAL already plans:

1. No vertex separator — FERAL has min-vertex-cover (M6 in
   `feral-metis`) and plans flow-based separators (K4 in
   `feral-kahip`). Mongoose would need the same M6-style step bolted
   on.
2. No ND driver — FERAL has `feral-metis::driver` (M7). A new crate
   would duplicate that driver wholesale.
3. No leaf-AMD integration — FERAL already composes
   `feral-amd` under `feral-metis`. Same wiring would work for any
   new bisection engine, so this is not a differentiator.

Net assessment: Mongoose's genuinely new contributions relative to
FERAL's existing coverage are (a) HEMSR coarsening and (b) QP
refinement. The rest (ND driver, vertex separator, AMD leaves) is
either not in Mongoose or already in FERAL.

## 5. Recommendation

**Primary recommendation: option (b) — adopt specific ideas into
existing crates, do not create `feral-mongoose`.**

Concretely:

1. In `feral-metis::coarsen`, add an optional HEMSR second pass
   (brotherly + adoption; skip community per the paper's own
   lukewarm assessment). This directly addresses the known weakness
   of HEM-only coarsening on power-law graphs and is a small,
   testable change behind a feature flag. Attribute to the Mongoose
   paper in the module doc.
2. In `feral-metis::refine` (and later `feral-kahip`), evaluate
   adding a **QP-projection pass** as an alternative or complement
   to FM. This is a larger experiment and should start as its own
   research note + plan before any code.
3. Do **not** create a `feral-mongoose` crate. Reasons:
   - Mongoose as published is an edge-bisector only. A FERAL-shaped
     `feral-mongoose` crate would be ~70 % duplicated plumbing from
     `feral-metis` (ND driver, vertex separator, AMD wiring,
     postorder, compression) wrapped around one new bisection
     kernel.
   - The algorithmic novelty (HEMSR + QP) can be absorbed into
     existing crates with far less engineering and exposes them to
     the bakeoff directly.
   - Five ordering crates is already a lot of surface to maintain.
     A fifth crate pays for itself only if it produces qualitatively
     different orderings on the KKT bakeoff; given that its top-level
     pipeline would be identical to `feral-metis`, that payoff is
     unlikely.

**Secondary recommendation:** if the HEMSR + QP experiments inside
`feral-metis` produce clearly distinct cut-quality or fill on the
bakeoff (say, >5 % nnz(L) difference on ≥3 matrices), revisit the
`feral-mongoose`-as-separate-crate question. Until then, the
"another data point" in `tasks.org` is better served by a
`feral-metis` configuration flag than by a new crate.

**Tradeoff for option (a) — build `feral-mongoose` anyway.** The
upside is a cleaner separation between coarsening strategies and a
less-coupled comparison. The downsides are the duplicated plumbing,
the maintenance cost of a fifth crate, and the fact that Mongoose
does not in fact supply the ND outer loop or vertex separator — we
would be writing those from scratch while labelling them "Mongoose,"
which is misleading attribution. License compatibility is fine for
clean-room-from-paper (as §2 established); the problem is scope
creep, not licensing.

**Tradeoff for option (c) — skip entirely.** Acceptable. The
HEMSR coarsening idea is genuinely valuable for power-law graphs
and should not be lost, but if KKT bakeoff matrices are mostly mesh
or AC-OPF structure (not power-law), the improvement may not appear.
If the bakeoff corpus does not include any social-graph or scale-free
instance, option (c) is defensible.

## 6. Open questions

1. **Has the Mongoose vertex-separator extension shipped?** The 2020
   paper promised it as future work; the 2026-04 SuiteSparse
   `stable/Mongoose/README.md` still says "in the future." Worth a
   direct check of the SuiteSparse `dev` branch and Mongoose release
   notes before any FERAL work starts — if an extension exists, §3.6
   and the recommendation may change.
2. **Does the FERAL KKT bakeoff corpus include power-law structure?**
   If not, HEMSR's advantage will not show up and the benefit of any
   Mongoose-inspired work is smaller. We should answer this before
   scheduling the `feral-metis` HEMSR experiment.
3. **QP gradient-projection cost profile.** The paper reports QP as
   cheap because only one iteration is run per level, but FERAL has
   no infrastructure for sparse gradient projection. Rough costing
   (added dependency on a QR / projection primitive? a hand-rolled
   boxed projection?) is needed before committing to the QP
   experiment. Relevant to the decision boundary between option (b)
   and option (c).
4. **Confirm GPL-3.0 status on the standalone `ScottKolo/Mongoose`
   repo.** The SuiteSparse copy is GPL-3.0. The archival ScottKolo
   mirror should match but has not been fetched in this note; I did
   not independently verify it. Matters only for completeness of
   §2; does not change the recommendation.
5. **Community matching details.** The paper describes community
   matching only briefly and self-reports it as marginal. We
   deliberately exclude it from the HEMSR proposal in §5, but if a
   FERAL-specific workload shows stalling even with brotherly +
   adoption, revisit the paper for the exact community-match rule.

## Sources

- Paper (OA, Hager lab): <https://people.clas.ufl.edu/hager/files/mongoose.pdf>
- Paper (OA, Davis lab): <https://people.engr.tamu.edu/davis/suitesparse_files/mongoose-ACMTOMS.pdf>
- Paper (ACM DL landing, DOI 10.1145/3337792): <https://dl.acm.org/doi/10.1145/3337792>
- Paper (HTML, ACM): <https://dl.acm.org/doi/fullHtml/10.1145/3337792>
- Paper (NSF PAR): <https://par.nsf.gov/biblio/10170049>
- License (SuiteSparse stable): <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/stable/Mongoose/README.md>
- Archival repo: <https://github.com/ScottKolo/Mongoose>
- SuiteSparse repository root: <https://github.com/DrTimothyAldenDavis/SuiteSparse>
