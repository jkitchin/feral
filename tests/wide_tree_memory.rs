//! Contribution memory of the sequential driver under
//! `EarlyAssembly::WideParents` against the classic scheme (`Off`).
//!
//! A star — thousands of leaf fronts, each passing a dense contribution
//! block to one small shared front — makes the classic scheme hold the
//! *sum* of the children's blocks until the shared front is assembled.
//! `WideParents` opens an accumulator for such a parent
//! (`dev/research/wide-tree-contribution-memory.md`). Its claim, checked
//! here on generated trees: the peak live heap of a factorization is
//! `Off`'s plus bookkeeping (a few words per planned parent, one block's
//! row map, `O(order)` words at a planned front), and the retained
//! factor is no larger. The parallel task-graph driver ignores the
//! scheme and is not measured here.
//!
//! A counting `#[global_allocator]` measures live heap. The tests take
//! one lock while measuring, so nothing else in this binary allocates
//! meanwhile.

#[path = "wide_star/mod.rs"]
mod wide_star;

use feral::numeric::factorize::{factorize_multifrontal_supernodal, EarlyAssembly, NumericParams};
use feral::symbolic::{
    symbolic_factorize, symbolic_factorize_with_method, OrderingMethod, SupernodeParams,
    SymbolicFactorization,
};
use feral::{CscMatrix, FactorStatus, Solver};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use wide_star::{star_kkt, Star};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static MEASURING: Mutex<()> = Mutex::new(());

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

struct Counting;

// SAFETY: every method forwards verbatim to `System` and returns its
// pointer unchanged; the only added work is relaxed atomic bookkeeping of
// the requested sizes. No allocator invariant is altered.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= layout.size() {
            grow(new_size - layout.size());
        } else {
            LIVE.fetch_sub(layout.size() - new_size, Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Live heap of one sequential factorization of `a` under `mode`, above
/// what was live before it: `(peak, retained factor)`.
fn measure(a: &CscMatrix, symbolic: &SymbolicFactorization, mode: EarlyAssembly) -> (usize, usize) {
    let params = NumericParams {
        early_assembly: mode,
        ..NumericParams::default()
    };
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let factors = factorize_multifrontal_supernodal(a, symbolic, &params).expect("factor");
    let peak = PEAK.load(Relaxed) - before;
    let retained = LIVE.load(Relaxed) - before;
    // Contribution blocks are consumed during the factorization; a root's
    // empty block must not pin a recycled buffer for the factor's lifetime.
    assert!(
        factors
            .0
            .node_factors
            .iter()
            .all(|node| node.frontal_factors.contrib.capacity() == 0),
        "{mode:?}: the returned factor owns contribution storage"
    );
    drop(factors);
    (peak, retained)
}

/// `Off` and `WideParents` on one analysis. `WideParents` runs once first: the
/// plan is cached on the analysis (bookkeeping that outlives a
/// factorization), so it must not count as retained factor.
fn off_and_wide_parents(a: &CscMatrix, symbolic: &SymbolicFactorization) -> [(usize, usize); 2] {
    measure(a, symbolic, EarlyAssembly::WideParents);
    [
        measure(a, symbolic, EarlyAssembly::Off),
        measure(a, symbolic, EarlyAssembly::WideParents),
    ]
}

/// `Solver::with_early_assembly` reaches the sequential driver: on the
/// star, a `Solver` factorization (analysis included) peaks far lower
/// with `WideParents` than with `Off`.
#[test]
fn solver_setter_reaches_the_sequential_driver() {
    let _measuring = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let matrix = star_kkt(Star {
        groups: 1500,
        local: 4,
        shared: 96,
        zero_local_diagonal: false,
    });
    let peak = |mode: EarlyAssembly| {
        let mut solver = Solver::new().with_parallel(false).with_early_assembly(mode);
        let before = LIVE.load(Relaxed);
        PEAK.store(before, Relaxed);
        let status = solver.factor(&matrix, None);
        assert!(
            matches!(status, FactorStatus::Success),
            "{mode:?}: {status:?}"
        );
        PEAK.load(Relaxed) - before
    };
    let (off, wide) = (peak(EarlyAssembly::Off), peak(EarlyAssembly::WideParents));
    assert!(
        wide * 2 < off,
        "Solver peak: Off {off} B, WideParents {wide} B"
    );
}

#[test]
fn wide_star_peak_is_far_below_the_classic_scheme() {
    let _measuring = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let star = Star {
        groups: 1500,
        local: 4,
        shared: 96,
        zero_local_diagonal: false,
    };
    let matrix = star_kkt(star);
    let symbolic = symbolic_factorize(&matrix, &SupernodeParams::default()).expect("symbolic");
    let widest = symbolic
        .supernodes
        .iter()
        .map(|sn| sn.children.len())
        .max()
        .unwrap_or(0);
    assert!(
        widest >= star.groups,
        "fixture must be a wide star: widest supernode has {widest} children"
    );
    let [(off, off_factor), (planned, planned_factor)] = off_and_wide_parents(&matrix, &symbolic);
    let mib = |b: usize| b as f64 / 1048576.0;
    eprintln!(
        "peak live heap: Off {:.2} MiB, WideParents {:.2} MiB; factor {:.2} MiB",
        mib(off),
        mib(planned),
        mib(off_factor),
    );
    assert!(
        planned * 4 < off,
        "WideParents peak {planned} B must be far below Off's {off} B: the children's \
         blocks must not all be live at once"
    );
    assert!(planned_factor <= off_factor);
}

/// A tree of fronts for [`tree_matrix`]: `own` columns, coupled to the
/// first row of the parent's front and its last `take - 1` rows.
struct Spec {
    own: usize,
    take: usize,
    zero_diagonal: bool,
    children: Vec<Spec>,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }
}

/// A random tree of depth `depth` below a front of order `parent_order`
/// (0 at the root): from no children to wide fans. With `delays`, some
/// one-column fronts have a zero diagonal, so their pivot is delayed.
fn random_spec(rng: &mut Rng, depth: usize, parent_order: usize, delays: bool) -> Spec {
    let own = if depth == 0 {
        rng.range(1, 3)
    } else {
        rng.range(1, 40)
    };
    let take = if parent_order == 0 {
        0
    } else {
        rng.range(1, parent_order)
    };
    let mut children = Vec::new();
    if depth > 0 {
        let fan = match rng.range(0, 5) {
            0 => 0,
            1 => rng.range(1, 3),
            2 => rng.range(2, 8),
            3 => rng.range(5, 60),
            _ => rng.range(1, 4),
        };
        for _ in 0..fan {
            children.push(random_spec(rng, depth - 1, own + take, delays));
        }
    }
    let zero_diagonal = delays && own == 1 && take > 0 && rng.range(0, 3) == 0;
    Spec {
        own,
        take,
        zero_diagonal,
        children,
    }
}

/// The first column of every front, numbered in postorder (children
/// first) so the identity ordering reproduces the tree.
struct Numbered {
    first: usize,
    children: Vec<Numbered>,
}

fn number(spec: &Spec, next: &mut usize) -> Numbered {
    let children = spec.children.iter().map(|c| number(c, next)).collect();
    let first = *next;
    *next += spec.own;
    Numbered { first, children }
}

/// Lower-triangle entries of `spec`'s front, coupled to `trailing` rows
/// of its parent's front.
fn entries(spec: &Spec, at: &Numbered, trailing: &[usize], out: &mut Vec<(usize, usize, f64)>) {
    let own: Vec<usize> = (at.first..at.first + spec.own).collect();
    for (k, &i) in own.iter().enumerate() {
        let diagonal = if spec.zero_diagonal {
            0.0
        } else {
            1000.0 + (i % 7) as f64
        };
        out.push((i, i, diagonal));
        if k > 0 {
            out.push((i, own[0], 0.01 + 0.0001 * ((i + own[0]) % 13) as f64));
        } else {
            for &t in trailing {
                out.push((t, i, 0.02 + 0.0001 * ((i * 3 + t) % 11) as f64));
            }
        }
    }
    let mut rows = own;
    rows.extend_from_slice(trailing);
    for (child, child_at) in spec.children.iter().zip(&at.children) {
        let take = child.take.clamp(1, rows.len());
        let mut child_trailing = vec![rows[0]];
        child_trailing.extend_from_slice(&rows[rows.len() - (take - 1)..]);
        entries(child, child_at, &child_trailing, out);
    }
}

fn tree_matrix(spec: &Spec) -> CscMatrix {
    let mut n = 0;
    let numbered = number(spec, &mut n);
    let mut triplets = Vec::new();
    entries(spec, &numbered, &[], &mut triplets);
    let rows: Vec<usize> = triplets.iter().map(|t| t.0).collect();
    let cols: Vec<usize> = triplets.iter().map(|t| t.1).collect();
    let vals: Vec<f64> = triplets.iter().map(|t| t.2).collect();
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tree matrix")
}

/// The bookkeeping `WideParents` may add to `Off`'s peak: a plan entry per
/// parent (at most 16 words), one block's row map and `O(order)` words at
/// a planned front (8 words per row of the largest front).
fn bookkeeping_allowance(symbolic: &SymbolicFactorization) -> usize {
    let word = std::mem::size_of::<usize>();
    let parents = symbolic
        .supernodes
        .iter()
        .filter(|sn| !sn.children.is_empty())
        .count();
    let max_order = (0..symbolic.supernodes.len())
        .map(|s| symbolic.static_rows(s).len())
        .max()
        .unwrap_or(0);
    parents * 16 * word + max_order * 9 * word
}

/// A tree on which an exactly sized accumulator, adopted as the frontal
/// pool, made the pool grow differently from the classic scheme: a
/// front of area 10,000, then a planned parent of area 19,044, then
/// later fronts of area up to 19,600.
fn pool_growth_tree() -> Spec {
    let leaf = |own, take| Spec {
        own,
        take,
        zero_diagonal: false,
        children: Vec::new(),
    };
    let fan = |take, leaves: Vec<Spec>| Spec {
        own: 1,
        take,
        zero_diagonal: false,
        children: leaves,
    };
    let mut children = vec![
        leaf(99, 1),
        fan(137, (0..22).map(|_| leaf(5, 60)).collect()),
    ];
    children.extend((0..6).map(|_| fan(90, (0..120).map(|_| leaf(1, 1)).collect())));
    Spec {
        own: 140,
        take: 0,
        zero_diagonal: false,
        children,
    }
}

#[test]
fn wide_parents_never_exceeds_off_on_generated_trees() {
    let _measuring = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let supernodes = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    // Generated trees are numbered in postorder: the identity ordering
    // keeps them. The star is ordered by the analysis.
    let identity = |a: &CscMatrix| OrderingMethod::External((0..a.n).collect());
    let mut cases = vec![(
        "pool growth".to_string(),
        tree_matrix(&pool_growth_tree()),
        true,
    )];
    // Every leaf's pivot is delayed into the shared front, so the
    // accumulator leaves remainders.
    let delaying = Star {
        groups: 300,
        local: 1,
        shared: 24,
        zero_local_diagonal: true,
    };
    cases.push((format!("{delaying:?}"), star_kkt(delaying), false));
    // Sizes keep the debug run within seconds; with these the delaying
    // trees include one whose remainders, pooled, would grow the
    // retained factor.
    for (delays, count) in [(false, 150), (true, 40)] {
        for t in 0..count {
            let label = format!("random tree {t} (delays {delays})");
            cases.push((
                label,
                tree_matrix(&random_spec(&mut rng, 3, 0, delays)),
                true,
            ));
        }
    }
    let (mut planned_below, mut worst_excess) = (0, 0);
    for (label, a, postordered) in &cases {
        let symbolic = if *postordered {
            symbolic_factorize_with_method(a, &supernodes, identity(a))
        } else {
            symbolic_factorize(a, &SupernodeParams::default())
        }
        .expect("symbolic");
        let [(off, off_factor), (planned, planned_factor)] = off_and_wide_parents(a, &symbolic);
        let allowance = bookkeeping_allowance(&symbolic);
        assert!(
            planned <= off + allowance,
            "{label}, n {}: WideParents peak {planned} B above Off's {off} B by more than the \
             bookkeeping allowance {allowance} B",
            a.n
        );
        assert!(
            planned_factor <= off_factor,
            "{label}: WideParents retained {planned_factor} B, Off {off_factor} B"
        );
        planned_below += usize::from(planned < off);
        worst_excess = worst_excess.max(planned.saturating_sub(off));
    }
    eprintln!(
        "{} matrices; WideParents peak below Off's on {planned_below}, at most {worst_excess} B above",
        cases.len()
    );
    assert!(
        planned_below > 0,
        "the generator must produce trees a plan helps"
    );
}
