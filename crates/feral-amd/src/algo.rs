//! AMD elimination-loop primitives: pivot selection and element
//! construction (plus standard absorption).
//!
//! This module lands Commit 4 of the Slice A plan. It ports faer's
//! `amd.rs:220-365` line-by-line:
//!
//! - [`select_pivot`]: linear scan from `mindeg`, LIFO unlink.
//! - [`create_element`]: both the in-place (`elenme == 0`) and
//!   out-of-place (`elenme > 0`) branches, plus standard absorption
//!   fired at the end of each `knt1` iter (faer `amd.rs:355-358`),
//!   and the final bookkeeping write-back to `pe[me]/len[me]/elen[me]`
//!   with the post-step `clear_flag` call.
//!
//! Garbage collection (triggered when `pfree >= iwlen` inside the
//! out-of-place branch) is deferred to Commit 6. Until then,
//! [`create_element`] returns [`AmdError::IndexOverflow`] on that
//! path. Fixtures whose oracle `ncmpa == 0` do not hit it.
//!
//! Pass-1 `w[e]` seeding (`amd.rs:366-385`), Pass-2 approximate
//! degree (`amd.rs:386-465`), mass elimination, supervariable
//! detection, and re-insertion into degree lists are Commit 5+.

#![allow(dead_code)]

use crate::error::AmdError;
use crate::workspace::{clear_flag, flip, AmdWorkspace, NONE};

/// Scan `head` from `ws.mindeg` upward and return the first
/// non-empty degree-list head. Unlink the chosen variable. Returns
/// `None` if no bucket in `[ws.mindeg, ws.n)` is non-empty (i.e.
/// all remaining supervariables have been dense-deferred and the
/// main loop should stop).
///
/// Side effects: `ws.mindeg` advances to the degree of the chosen
/// pivot. `head[deg]` is advanced to the next element. `last[next]`
/// is cleared if a successor exists.
///
/// Reference: faer `amd.rs:220-235`.
pub(crate) fn select_pivot(ws: &mut AmdWorkspace) -> Option<usize> {
    let n = ws.n;
    let mut deg = ws.mindeg;
    let mut me_signed: i32 = NONE;
    while deg < n {
        let h = ws.head[deg];
        if h != NONE {
            me_signed = h;
            break;
        }
        deg += 1;
    }
    if me_signed == NONE {
        return None;
    }
    ws.mindeg = deg;
    let me = me_signed as usize;
    let inext = ws.next[me];
    if inext != NONE {
        ws.last[inext as usize] = NONE;
    }
    ws.head[deg] = inext;
    Some(me)
}

/// Build the new element `me` by merging the (variable) tail of
/// `me`'s list with every element `e` already in `me`'s list.
///
/// On success returns `(pme1, pme2, nvpiv, degme)`:
/// - `pme1..=pme2` is the contiguous region in `ws.iw` holding the
///   new element's **variable** members (supervariables, listed
///   once each).
/// - `nvpiv` is the supervariable count of the pivot.
/// - `degme` is the tentative new element's external degree (sum of
///   `nv[i]` over the assembled variables, before any absorption
///   correction made by Pass-2).
///
/// Post-conditions also persisted on the workspace:
/// - `nv[me] = -nvpiv` (marker — Pass-2 will flip sign back via
///   `-nv[i]`).
/// - `nv[i] = -nv[i]` for every `i` assembled into the new element
///   (ditto — marker for Pass-2's w-seed walk).
/// - `pe[me] = pme1`, `len[me] = pme2 - pme1 + 1`, `elen[me] =
///   flip(nvpiv + degme)` (dead-variable sentinel carrying the
///   pivot-front size for the postorder phase).
/// - `degree[me] = degme` (temporary; Pass-2 overwrites).
/// - For every absorbed element `e != me` in the elenme>0 branch:
///   `pe[e] = flip(me)`, `w[e] = 0`. This is **standard absorption**
///   (faer `amd.rs:355-358`) — it fires unconditionally at each
///   `knt1` iter's end. Aggressive absorption (Pass-2 only) lands
///   in Commit 5.
/// - `ws.wflg` bumped via `clear_flag`.
/// - `ws.nel` incremented by `nvpiv`.
///
/// Reference: faer `amd.rs:236-366`.
///
/// Errors: [`AmdError::IndexOverflow`] if the out-of-place branch
/// would write past `iw` — this is the point where Commit 6 will
/// install the inline GC compaction.
pub(crate) fn create_element(
    ws: &mut AmdWorkspace,
    me: usize,
) -> Result<(usize, usize, i32, usize), AmdError> {
    let elenme = ws.elen[me];
    let nvpiv = ws.nv[me];
    ws.nel += nvpiv as usize;
    ws.nv[me] = -nvpiv;
    let mut degme: usize = 0;
    let pme1: i32;
    let pme2: i32;

    if elenme == 0 {
        // In-place: me has no elements in its list — just variables.
        // Compact them at pe[me]: advance pme2 to the final position,
        // overwriting absorbed entries in-place.
        pme1 = ws.pe[me];
        let list_start = pme1 as usize;
        let list_end = list_start + ws.len[me] as usize;
        let mut pme2_s = pme1 - 1;
        for p in list_start..list_end {
            let i = ws.iw[p] as usize;
            let nvi = ws.nv[i];
            if nvi > 0 {
                degme += nvi as usize;
                ws.nv[i] = -nvi;
                pme2_s += 1;
                ws.iw[pme2_s as usize] = i as i32;
                // Unlink i from its degree list.
                let ilast = ws.last[i];
                let inext = ws.next[i];
                if inext != NONE {
                    ws.last[inext as usize] = ilast;
                }
                if ilast != NONE {
                    ws.next[ilast as usize] = inext;
                } else {
                    ws.head[ws.degree[i] as usize] = inext;
                }
            }
        }
        pme2 = pme2_s;
    } else {
        // Out-of-place: start a new region at pfree. Walk every
        // element e in me's list (first `elenme` entries of me's
        // adjacency), then walk the variable tail (remaining
        // `slenme` entries) with `knt1 = elenme + 1` as the flag.
        let mut p = ws.pe[me] as usize;
        pme1 = ws.pfree as i32;
        let slenme = (ws.len[me] - elenme) as usize;
        let elenme_u = elenme as usize;
        for knt1 in 1..=elenme_u + 1 {
            let e: usize;
            let mut pj: usize;
            let ln: usize;
            if knt1 > elenme_u {
                // Variable tail of me's own list.
                e = me;
                pj = p;
                ln = slenme;
            } else {
                e = ws.iw[p] as usize;
                p += 1;
                pj = ws.pe[e] as usize;
                ln = ws.len[e] as usize;
            }
            for _knt2 in 0..ln {
                let i = ws.iw[pj] as usize;
                pj += 1;
                let nvi = ws.nv[i];
                if nvi > 0 {
                    if ws.pfree >= ws.iwlen {
                        // Commit 6 replaces this with inline GC.
                        return Err(AmdError::IndexOverflow);
                    }
                    degme += nvi as usize;
                    ws.nv[i] = -nvi;
                    ws.iw[ws.pfree] = i as i32;
                    ws.pfree += 1;
                    // Unlink i from its degree list.
                    let ilast = ws.last[i];
                    let inext = ws.next[i];
                    if inext != NONE {
                        ws.last[inext as usize] = ilast;
                    }
                    if ilast != NONE {
                        ws.next[ilast as usize] = inext;
                    } else {
                        ws.head[ws.degree[i] as usize] = inext;
                    }
                }
            }
            // Standard absorption (amd.rs:355-358): every element e
            // that was in me's list is now absorbed by me.
            if e != me {
                ws.pe[e] = flip(me as i32);
                ws.w[e] = 0;
            }
        }
        pme2 = (ws.pfree - 1) as i32;
    }

    ws.degree[me] = degme as i32;
    ws.pe[me] = pme1;
    ws.len[me] = pme2 - pme1 + 1;
    ws.elen[me] = flip(nvpiv + degme as i32);
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    Ok((pme1 as usize, pme2 as usize, nvpiv, degme))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::CscPattern;
    use crate::AmdOptions;

    fn ws_for<'a>(n: usize, cp: &'a [usize], ri: &'a [usize]) -> AmdWorkspace {
        let p = CscPattern::new(n, cp, ri).unwrap();
        AmdWorkspace::new(&p, &AmdOptions::default()).unwrap()
    }

    #[test]
    fn select_pivot_empty() {
        // diag_4: every var pre-eliminated, no degree bucket populated.
        let cp = [0, 1, 2, 3, 4];
        let ri = [0, 1, 2, 3];
        let mut ws = ws_for(4, &cp, &ri);
        assert_eq!(select_pivot(&mut ws), None);
    }

    #[test]
    fn select_pivot_lifo_on_tridiag() {
        // Tridiag 5: head[1] contains 4 -> 0 (LIFO).
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let mut ws = ws_for(5, &cp, &ri);
        assert_eq!(select_pivot(&mut ws), Some(4));
        assert_eq!(ws.mindeg, 1);
        // Head of deg-1 list now points to the remaining spoke (0).
        assert_eq!(ws.head[1], 0);
        assert_eq!(ws.last[0], NONE, "new head has no predecessor");

        assert_eq!(select_pivot(&mut ws), Some(0));
        assert_eq!(ws.head[1], NONE, "deg-1 bucket drained");

        // Next call scans from mindeg=1 upward; only deg-2 non-empty.
        assert_eq!(select_pivot(&mut ws), Some(3));
        assert_eq!(ws.mindeg, 2);
    }

    #[test]
    fn create_element_elenme_zero_on_arrow_5_hub() {
        // Arrow 5: hub has deg 4, but it's dense-deferred? Let's check.
        // For n=5 default, dense = max(16, min(5, 10*sqrt(5))) = 5.
        // deg 4 < 5, so hub is LIVE and sits in head[4].
        // Spokes (deg 1) all share head[1]. The min-degree pivot is a
        // spoke. Let's pick spoke 4 (LIFO head of deg-1).
        let cp = [0, 5, 7, 9, 11, 13];
        let ri = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let mut ws = ws_for(5, &cp, &ri);
        let me = select_pivot(&mut ws).unwrap();
        assert_eq!(me, 4, "first pivot is the LIFO head of deg-1");
        // elen[4] == 0 (no elements yet).
        assert_eq!(ws.elen[4], 0);
        let (pme1, pme2, nvpiv, degme) = create_element(&mut ws, me).unwrap();
        assert_eq!(nvpiv, 1, "singleton supervariable");
        assert_eq!(degme, 1, "only neighbor is the hub (nv=1)");
        // The new element's var list contains {hub} = {0}.
        assert_eq!(pme2 - pme1 + 1, 1);
        assert_eq!(ws.iw[pme1], 0);
        assert_eq!(ws.pe[4], pme1 as i32);
        assert_eq!(ws.len[4], 1);
        assert_eq!(ws.elen[4], flip(1 + 1), "flip(nvpiv + degme)");
        assert_eq!(ws.nv[4], -1, "pivot marker");
        assert_eq!(ws.nv[0], -1, "hub marked");
        // Hub was in head[4] with no siblings; it was removed.
        assert_eq!(ws.head[4], NONE);
        // nel advanced by nvpiv.
        assert_eq!(ws.nel, 1);
    }

    #[test]
    fn create_element_elenme_zero_unlinks_from_degree_list() {
        // Tridiag 5: pivot var 4 (deg 1), neighbor var 3 (deg 2).
        // var 3 should be unlinked from head[2], which currently
        // threads 3 -> 2 -> 1.
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let mut ws = ws_for(5, &cp, &ri);
        let me = select_pivot(&mut ws).unwrap();
        assert_eq!(me, 4);
        let (_, _, _, _) = create_element(&mut ws, me).unwrap();
        // After unlinking 3 (head of deg-2), head[2] -> 2.
        assert_eq!(ws.head[2], 2);
        assert_eq!(ws.last[2], NONE);
        // Unlinked var's last/next are stale but the list is valid.
        assert_eq!(ws.nv[3], -1);
    }

    #[test]
    fn create_element_skips_absorbed_neighbors() {
        // Construct a workspace where one neighbor is already absorbed
        // (nv == 0 or nv < 0). That neighbor must NOT contribute.
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let mut ws = ws_for(5, &cp, &ri);
        // Mark var 0 as already-absorbed (nv <= 0).
        ws.nv[0] = 0;
        let me = select_pivot(&mut ws).unwrap();
        assert_eq!(me, 4);
        let (_, _, nvpiv, degme) = create_element(&mut ws, me).unwrap();
        // Only neighbor 3 counts; not 0 (absorbed) and not the diagonal
        // (skipped by init).
        assert_eq!(nvpiv, 1);
        assert_eq!(degme, 1);
    }

    /// Arrow 200 hub is dense-deferred; first pivot is a spoke with
    /// no elements in its list, so the elenme==0 path is exercised
    /// on a larger graph. Smoke test that the path completes without
    /// indexing out of bounds.
    #[test]
    fn arrow_200_first_pivot_smoke() {
        let n = 200usize;
        let mut cp: Vec<usize> = vec![0];
        let mut ri: Vec<usize> = Vec::new();
        ri.push(0);
        for r in 1..n {
            ri.push(r);
        }
        cp.push(ri.len());
        for j in 1..n {
            ri.push(0);
            ri.push(j);
            cp.push(ri.len());
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = AmdWorkspace::new(&p, &AmdOptions::default()).unwrap();
        // Hub (var 0) is dense-deferred; its nv was set to 0 during
        // init. So when a spoke's list references 0, it's skipped.
        let me = select_pivot(&mut ws).unwrap();
        let (_, _, nvpiv, degme) = create_element(&mut ws, me).unwrap();
        assert_eq!(nvpiv, 1);
        assert_eq!(degme, 0, "spoke's only neighbor (hub) is deferred");
        // Exactly one var (the pivot itself) has been eliminated plus
        // the deferred hub that init already counted.
        assert_eq!(ws.nel, 2, "1 deferred hub + 1 pivot");
    }
}
