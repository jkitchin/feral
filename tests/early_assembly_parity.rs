//! Accumulator assembly is bit-identical to classic assembly.
//!
//! `NumericParams::early_assembly` selects the sequential drivers'
//! scheme: `Off` (the default, classic: every contribution block held
//! until its parent is assembled), `WideParents` (opt-in: the symbolic
//! `accumulate_from` plan) and `Always` (every parent's accumulator opens
//! at its first child, so accumulator assembly runs on every front,
//! delayed-pivot fronts included). Assembly performs the same additions
//! in the same order under all three (`numeric::contrib` module docs), so
//! every factor must hash the same — on the parity corpus, the issue-#227
//! synthetic KKTs, a wide star with and without delayed pivots, a delay
//! chain through a wide star's parent (the issue-#221 backstop reads the
//! area an accumulator absorbs), and the Schur-complement API with and
//! without delayed pivots at its root. The
//! parallel driver ignores the scheme; it runs under each one too and
//! must not change. The digest covers each front's row indices,
//! delayed-in count and cascade-break flag as well as its factor. The
//! digests of both drivers under `Off` and `WideParents` are also pinned to
//! the release before this change (on aarch64, where they were computed;
//! ignored elsewhere).

#[path = "common227/mod.rs"]
mod common227;
#[path = "wide_star/mod.rs"]
mod wide_star;

use common227::{factor_digest, synthetic_kkt, Fnv, N_ITER};
use feral::numeric::factorize::{
    factorize_multifrontal_parallel, factorize_multifrontal_supernodal,
    factorize_multifrontal_with_schur, EarlyAssembly, NumericParams, SparseFactors,
};
use feral::symbolic::{
    symbolic_factorize, symbolic_factorize_with_schur, OrderingMethod, SupernodeParams,
};
use feral::{read_mtx, CscMatrix, Inertia, Solver};
use std::path::{Path, PathBuf};
use wide_star::{star_kkt, Star};

const MODES: [EarlyAssembly; 3] = [
    EarlyAssembly::Off,
    EarlyAssembly::WideParents,
    EarlyAssembly::Always,
];

fn params(mode: EarlyAssembly) -> NumericParams {
    NumericParams {
        early_assembly: mode,
        ..NumericParams::default()
    }
}

fn parity_matrices() -> Vec<(String, CscMatrix)> {
    let root = Path::new("tests/data/parity");
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(root) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(e) => panic!("read_dir {root:?}: {e}"),
    };
    dirs.sort();
    let mut out = Vec::new();
    for d in dirs {
        let mut files: Vec<PathBuf> = match std::fs::read_dir(&d) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
                .collect(),
            Err(_) => continue,
        };
        files.sort();
        for p in files {
            match read_mtx(&p).and_then(|m| m.to_csc()) {
                Ok(a) => out.push((p.display().to_string(), a)),
                Err(e) => panic!("{p:?}: {e}"),
            }
        }
    }
    out
}

/// A star of `groups` dense SPD leaves of `local` variables, each
/// coupled to all `shared` root variables: every leaf is one front whose
/// factorization costs `O(local² · (local + shared))` flops, so big
/// leaves are parallel tasks of their own in the parallel driver.
fn dense_star(groups: usize, local: usize, shared: usize) -> CscMatrix {
    let n = shared + groups * local;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..shared {
        r.push(s);
        c.push(s);
        v.push(-(1.0 + 0.01 * (s % 5) as f64));
    }
    for g in 0..groups {
        let base = shared + g * local;
        for a in 0..local {
            let i = base + a;
            r.push(i);
            c.push(i);
            v.push(local as f64 + shared as f64);
            for b in (0..a).map(|b| base + b).chain(0..shared) {
                r.push(i);
                c.push(b);
                v.push(0.1 + 0.001 * ((a * 7 + b + g) % 11) as f64);
            }
        }
    }
    match CscMatrix::from_triplets(n, &r, &c, &v) {
        Ok(m) => m,
        Err(e) => panic!("dense star: {e}"),
    }
}

const CHAIN_DELAYS: usize = 100;
const CHAIN_LEN: usize = 220;
const STAR_AT: usize = 100;
const STAR_LEAVES: usize = 340;
const LEAF: usize = 20;

/// Column of chain variable `x_j` in [`delay_chain_star`].
fn chain_column(j: usize) -> usize {
    CHAIN_DELAYS + STAR_LEAVES * LEAF + j
}

/// Delays riding a chain into a wide star's parent (the issue-#221
/// backstop fixture, `tests/issue_221_delay_chain_backstop.rs`, plus
/// leaves). Constraints `z_i` (`i < CHAIN_DELAYS`, zero diagonal,
/// coupled to the hub `h` and by `1e-6` to chain variable `x_i`) all
/// delay, and the delays ride the tridiagonal chain `x_0 .. x_L` to the
/// hub; `STAR_LEAVES` dense SPD leaves of `LEAF` variables hang off
/// `x_STAR_AT`, so its children are the chain's next link, the leaves
/// and nothing else. Natural order is the elimination order.
///
/// The backstop engages a front once the chain's excess area passes
/// `DELAY_CHAIN_FLOOR` (2^20) and `DELAY_CHAIN_GROWTH` (10) times the
/// front's subtree area. Without the leaves' area it engages at chain
/// index 166. The leaves' area (`STAR_LEAVES · LEAF · (LEAF + 1)` ≈
/// 143K) lifts the growth bound to about 1.43M, deferring it to 191 —
/// only if the accumulator that absorbs the leaves at `x_STAR_AT`
/// carries their area: dropped, it engages at 166; doubled, the bound
/// passes what the chain reaches and it never engages.
fn delay_chain_star() -> CscMatrix {
    let n = chain_column(CHAIN_LEN + 1) + 1;
    let x = chain_column;
    let h = n - 1;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |i: usize, j: usize, val: f64| {
        r.push(i.max(j));
        c.push(i.min(j));
        v.push(val);
    };
    for i in 0..CHAIN_DELAYS {
        push(i, i, 0.0);
        push(x(i), i, 1e-6);
        push(h, i, 1.0);
    }
    for g in 0..STAR_LEAVES {
        let base = CHAIN_DELAYS + g * LEAF;
        for a in 0..LEAF {
            push(base + a, base + a, LEAF as f64 + 4.0);
            for b in 0..a {
                push(
                    base + a,
                    base + b,
                    0.1 + 0.001 * ((a * 7 + b + g) % 11) as f64,
                );
            }
        }
        push(x(STAR_AT), base + LEAF - 1, 0.01);
    }
    for j in 0..=CHAIN_LEN {
        push(x(j), x(j), 4.0);
        if j < CHAIN_LEN {
            push(x(j + 1), x(j), 1.0);
        }
    }
    push(h, h, 10.0);
    match CscMatrix::from_triplets(n, &r, &c, &v) {
        Ok(m) => m,
        Err(e) => panic!("delay chain star: {e}"),
    }
}

/// [`factor_digest`] plus each front's row indices, delayed-in count
/// and cascade-break flag — the structure the issue-#221 chain backstop
/// decides from the subtree areas an accumulator carries.
fn digest(f: &SparseFactors, i: &Inertia) -> u64 {
    let mut h = Fnv::new();
    h.word(factor_digest(f, i));
    for nf in &f.node_factors {
        h.usizes(&nf.row_indices);
        h.word(nf.n_delayed_in as u64);
        h.word(u64::from(nf.cascade_break));
    }
    h.0
}

/// Fronts that received delayed pivots.
fn delayed_fronts(f: &SparseFactors) -> usize {
    f.node_factors
        .iter()
        .filter(|nf| nf.n_delayed_in > 0)
        .count()
}

/// Digest of everything a mode can influence, plus coverage counts.
struct Run {
    digests: Vec<(String, u64)>,
    delayed_fronts: usize,
}

fn run(mode: EarlyAssembly, matrices: &[(String, CscMatrix)]) -> Run {
    let mut digests = Vec::new();
    let mut delayed = 0;
    for (name, a) in matrices {
        // Full `Solver` pipeline (the default ordering race, scaling,
        // dispatch), sequential and parallel.
        for parallel in [false, true] {
            let mut s = Solver::with_params(params(mode), SupernodeParams::default())
                .with_parallel(parallel);
            let status = s.factor(a, None);
            let (Some(f), Some(i)) = (s.factors(), s.inertia()) else {
                panic!("{name} solver parallel={parallel} {mode:?}: no factors ({status:?})");
            };
            delayed += delayed_fronts(f);
            digests.push((format!("{name} solver parallel={parallel}"), digest(f, i)));
        }
        // The multifrontal drivers directly, the parallel one forced past
        // its flop gate.
        let symbolic = match symbolic_factorize(a, &SupernodeParams::default()) {
            Ok(s) => s,
            Err(e) => panic!("{name} symbolic: {e}"),
        };
        let mut p = params(mode);
        let seq = factorize_multifrontal_supernodal(a, &symbolic, &p);
        p.min_parallel_flops = Some(0);
        let par = factorize_multifrontal_parallel(a, &symbolic, &p);
        for (label, res) in [("sequential", seq), ("parallel", par)] {
            match res {
                Ok((f, i)) => {
                    delayed += delayed_fronts(&f);
                    digests.push((format!("{name} {label}"), digest(&f, &i)));
                }
                Err(e) => panic!("{name} {label} {mode:?}: {e}"),
            }
        }
    }
    Run {
        digests,
        delayed_fronts: delayed,
    }
}

/// The Schur-complement API on a star, Schur block = the shared
/// variables (the star's root front). On a delaying star the delayed
/// pivots reach it; otherwise none do, and an accumulator for the root
/// becomes the Schur front as it is.
fn schur_digest(mode: EarlyAssembly, star: Star) -> u64 {
    let a = star_kkt(star);
    let schur: Vec<usize> = (0..star.shared).collect();
    let symbolic = symbolic_factorize_with_schur(&a, &SupernodeParams::default(), &schur)
        .expect("schur symbolic");
    let (f, i, block) =
        factorize_multifrontal_with_schur(&a, &symbolic, &params(mode)).expect("schur factor");
    let root = f.node_factors.last().expect("schur root");
    assert_eq!(
        root.n_delayed_in > 0,
        star.zero_local_diagonal,
        "{mode:?}: delayed pivots must reach the Schur root exactly on a delaying star"
    );
    let mut h = Fnv::new();
    h.word(digest(&f, &i));
    h.f64s(&block.data);
    h.0
}

#[test]
fn accumulator_assembly_is_bit_identical_to_classic() {
    let mut matrices = parity_matrices();
    for k in 0..N_ITER {
        matrices.push((format!("synthetic_kkt({k})"), synthetic_kkt(k)));
    }
    // Delaying star: every leaf's one column has a zero pivot, so all of
    // them are delayed into the shared front, through the accumulator's
    // remainder path.
    let delaying = Star {
        groups: 300,
        local: 1,
        shared: 24,
        zero_local_diagonal: true,
    };
    let strong = Star {
        local: 4,
        zero_local_diagonal: false,
        ..delaying
    };
    for star in [strong, delaying] {
        matrices.push((format!("{star:?}"), star_kkt(star)));
    }

    let results: Vec<_> = MODES
        .iter()
        .map(|&mode| {
            let schur = [schur_digest(mode, delaying), schur_digest(mode, strong)];
            (mode, run(mode, &matrices), schur)
        })
        .collect();

    let (_, classic, classic_schur) = &results[0];
    assert!(
        classic.delayed_fronts > 0,
        "the corpus must exercise delayed pivots into accumulator fronts"
    );
    for (mode, other, schur) in &results[1..] {
        assert_eq!(other.delayed_fronts, classic.delayed_fronts, "{mode:?}");
        for ((name, want), (_, got)) in classic.digests.iter().zip(&other.digests) {
            assert_eq!(got, want, "{mode:?} changed the factor of {name}");
        }
        assert_eq!(schur, classic_schur, "{mode:?} changed a Schur factor");
    }
}

/// The chain backstop reads the subtree area an accumulator carries.
/// On [`delay_chain_star`] every scheme must engage cascade-break at the
/// same front (191), past where the chain alone engages it (166): the
/// leaves' area, absorbed at `x_STAR_AT`, has to reach it.
#[test]
fn chain_backstop_sees_the_area_an_accumulator_absorbed() {
    let a = delay_chain_star();
    let supernodes = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    let mut classic = None;
    for mode in MODES {
        for parallel in [false, true] {
            let numeric = NumericParams {
                min_parallel_flops: Some(0),
                ..params(mode)
            };
            let mut s = Solver::with_params(numeric, supernodes.clone())
                .with_parallel(parallel)
                .with_ordering(OrderingMethod::External((0..a.n).collect()));
            let status = s.factor(&a, None);
            let (Some(f), Some(i)) = (s.factors(), s.inertia()) else {
                panic!("{mode:?} parallel={parallel}: no factors ({status:?})");
            };
            let engaged = f
                .node_factors
                .iter()
                .find(|nf| nf.cascade_break)
                .map(|nf| nf.first_col - chain_column(0));
            let got = (digest(f, i), engaged);
            match &classic {
                None => classic = Some(got),
                Some(want) => assert_eq!(&got, want, "{mode:?} parallel={parallel}"),
            }
        }
    }
    let engaged = classic.and_then(|(_, engaged)| engaged);
    assert!(
        engaged.is_some_and(|at| at > 180),
        "the leaves' area must defer cascade-break past chain index 180 (engaged at {engaged:?})"
    );
}

/// Digests of the wide, delaying and heavy-leaf stars (both drivers) and
/// of [`delay_chain_star`] (`Solver`, natural order, both drivers) under
/// `base`, in a fixed order. With `NumericParams::default()` the same
/// code runs on releases before `EarlyAssembly` existed.
fn golden_digests(base: &NumericParams) -> Vec<(String, u64)> {
    let strong = Star {
        groups: 300,
        local: 4,
        shared: 24,
        zero_local_diagonal: false,
    };
    let delaying = Star {
        local: 1,
        zero_local_diagonal: true,
        ..strong
    };
    let mut out = Vec::new();
    for (name, a) in [
        (format!("{strong:?}"), star_kkt(strong)),
        (format!("{delaying:?}"), star_kkt(delaying)),
        (
            "dense_star(64, 120, 60)".to_string(),
            dense_star(64, 120, 60),
        ),
    ] {
        let symbolic = symbolic_factorize(&a, &SupernodeParams::default()).expect("symbolic");
        let mut p = base.clone();
        let seq = factorize_multifrontal_supernodal(&a, &symbolic, &p).expect("sequential");
        p.min_parallel_flops = Some(0);
        let par = factorize_multifrontal_parallel(&a, &symbolic, &p).expect("parallel");
        out.push((format!("{name} sequential"), digest(&seq.0, &seq.1)));
        out.push((format!("{name} parallel"), digest(&par.0, &par.1)));
    }
    let a = delay_chain_star();
    let supernodes = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    for parallel in [false, true] {
        let numeric = NumericParams {
            min_parallel_flops: Some(0),
            ..base.clone()
        };
        let mut s = Solver::with_params(numeric, supernodes.clone())
            .with_parallel(parallel)
            .with_ordering(OrderingMethod::External((0..a.n).collect()));
        let status = s.factor(&a, None);
        let (Some(f), Some(i)) = (s.factors(), s.inertia()) else {
            panic!("delay_chain_star parallel={parallel}: no factors ({status:?})");
        };
        out.push((
            format!("delay_chain_star solver parallel={parallel}"),
            digest(f, i),
        ));
    }
    out
}

// [`golden_digests`] on `main` @ d3f7eb6, before contribution blocks were
// packed or accumulated, computed by running `golden_digests` there
// (aarch64, macOS). The full default pipeline is not bit-identical across
// architectures (issue #211 class; `issue227_never_golden`; on x86_64 the
// Schur kernel dispatches on the CPU's features at run time), so the
// digests hold on aarch64 only. No x86_64 digests are pinned: they need a
// run of the parent on an x86_64 machine, and the test is ignored there
// rather than passing without a comparison.
const PARENT_GOLDEN: [u64; 8] = [
    0xc392686a5ff99c30,
    0xc392686a5ff99c30,
    0xf2d02bc596baa72d,
    0xf2d02bc596baa72d,
    0x7db536ed1a7288c6,
    0x7db536ed1a7288c6,
    0xb5b5cc4c9953dc6c,
    0xb5b5cc4c9953dc6c,
];

/// The default (`Off`: packed blocks) and `WideParents` reproduce the
/// factors of the release before this change on wide, delaying and
/// heavy-leaf stars and on the delay chain, in the sequential driver and
/// the parallel one (unchanged): bit-identity to the parent, not only
/// between this build's schemes.
/// `FERAL_EARLY_ASSEMBLY_PRINT=1` prints the digests.
#[test]
#[cfg_attr(
    not(target_arch = "aarch64"),
    ignore = "parent digests are pinned for aarch64 only"
)]
fn assembly_matches_the_parent_release() {
    for mode in [EarlyAssembly::Off, EarlyAssembly::WideParents] {
        let got = golden_digests(&params(mode));
        if std::env::var("FERAL_EARLY_ASSEMBLY_PRINT").is_ok() {
            for (name, d) in &got {
                println!("{d:#018x} {mode:?} {name}");
            }
        }
        assert_eq!(got.len(), PARENT_GOLDEN.len());
        for ((name, got), want) in got.iter().zip(PARENT_GOLDEN) {
            assert_eq!(
                *got, want,
                "{mode:?} {name} differs from the parent release"
            );
        }
    }
}
