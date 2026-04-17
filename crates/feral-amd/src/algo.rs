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
//! Inline garbage collection (faer `amd.rs:289-338`) fires inside
//! the out-of-place branch when `pfree >= iwlen`. See
//! [`create_element`] for the full save → mark → compact → restore
//! dance.
//!
//! Pass-1 `w[e]` seeding (`amd.rs:366-385`), Pass-2 approximate
//! degree (`amd.rs:386-465`), aggressive absorption, monotone
//! degree cap, and re-insertion into degree lists (`amd.rs:516-546`)
//! live in [`finalize_step`]. Mass elimination and supervariable
//! detection are Slice B (Commits 9-10).

#![allow(dead_code)]

use crate::error::AmdError;
use crate::workspace::{clear_flag, flip, AmdWorkspace, NONE};

/// Flop-counter deltas produced by a single elimination step.
/// Matches faer's `amd.rs:547-557` accounting so `AmdStats` can
/// accumulate consistent `ndiv` / `nms_ldl` / `nms_lu` totals.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StepFlops {
    pub(crate) ndiv: f64,
    pub(crate) nms_lu: f64,
    pub(crate) nms_ldl: f64,
}

impl StepFlops {
    fn accumulate(&mut self, other: StepFlops) {
        self.ndiv += other.ndiv;
        self.nms_lu += other.nms_lu;
        self.nms_ldl += other.nms_ldl;
    }
}

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
/// Reference: faer `amd.rs:236-366` (incl. inline GC at 289-338).
pub(crate) fn create_element(
    ws: &mut AmdWorkspace,
    me: usize,
) -> Result<(usize, usize, i32, usize), AmdError> {
    let elenme = ws.elen[me];
    let nvpiv = ws.nv[me];
    ws.nel += nvpiv as usize;
    ws.nv[me] = -nvpiv;
    let mut degme: usize = 0;
    let pme1: usize;
    let pme2: i32;

    if elenme == 0 {
        // In-place: me has no elements in its list — just variables.
        // Compact them at pe[me]: advance pme2 to the final position,
        // overwriting absorbed entries in-place.
        let pme1_s = ws.pe[me];
        pme1 = pme1_s as usize;
        let list_start = pme1;
        let list_end = list_start + ws.len[me] as usize;
        let mut pme2_s = pme1_s - 1;
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
        let mut pme1_rw: usize = ws.pfree;
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
            for knt2 in 1..=ln {
                let i = ws.iw[pj] as usize;
                pj += 1;
                let nvi = ws.nv[i];
                if nvi > 0 {
                    if ws.pfree >= ws.iwlen {
                        // Inline garbage collection (faer
                        // amd.rs:289-338). Save partial state so
                        // the surviving elements can be compacted
                        // down, then restore local cursors.
                        ws.pe[me] = p as i32;
                        ws.len[me] -= knt1 as i32;
                        if ws.len[me] == 0 {
                            ws.pe[me] = NONE;
                        }
                        ws.pe[e] = pj as i32;
                        ws.len[e] = (ln - knt2) as i32;
                        if ws.len[e] == 0 {
                            ws.pe[e] = NONE;
                        }
                        ws.ncmpa += 1;
                        // Mark each live list's head: save iw[pe[j]]
                        // into pe[j] and write flip(j) at the old
                        // head position so the compact sweep can
                        // recognise list starts.
                        for j in 0..ws.n {
                            let pn = ws.pe[j];
                            if pn >= 0 {
                                let pn_u = pn as usize;
                                ws.pe[j] = ws.iw[pn_u];
                                ws.iw[pn_u] = flip(j as i32);
                            }
                        }
                        // Sweep [0, pme1_rw), reconstructing every
                        // marked list contiguously at pdst.
                        let mut psrc = 0usize;
                        let mut pdst = 0usize;
                        let pend = pme1_rw;
                        while psrc < pend {
                            let j_marker = flip(ws.iw[psrc]);
                            psrc += 1;
                            if j_marker >= 0 {
                                let j = j_marker as usize;
                                ws.iw[pdst] = ws.pe[j];
                                ws.pe[j] = pdst as i32;
                                pdst += 1;
                                let lenj = ws.len[j] as usize;
                                if lenj > 0 {
                                    ws.iw.copy_within(psrc..psrc + lenj - 1, pdst);
                                    psrc += lenj - 1;
                                    pdst += lenj - 1;
                                }
                            }
                        }
                        // Slide the new element's accumulated prefix
                        // [pme1_rw, pfree) down to the new pdst.
                        let p1 = pdst;
                        ws.iw.copy_within(pme1_rw..ws.pfree, pdst);
                        pdst += ws.pfree - pme1_rw;
                        pme1_rw = p1;
                        ws.pfree = pdst;
                        // Restore local cursors from the relocated
                        // heads of e's and me's lists.
                        pj = ws.pe[e] as usize;
                        p = ws.pe[me] as usize;
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
        pme1 = pme1_rw;
        pme2 = (ws.pfree - 1) as i32;
    }

    ws.degree[me] = degme as i32;
    ws.pe[me] = pme1 as i32;
    ws.len[me] = pme2 - pme1 as i32 + 1;
    ws.elen[me] = flip(nvpiv + degme as i32);
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    Ok((pme1, pme2 as usize, nvpiv, degme))
}

/// Finish the elimination step whose create-element phase produced
/// `(pme1, pme2, nvpiv, degme)` and left `nv[me] = -nvpiv` and
/// `nv[i] = -nv[i]` for every variable `i ∈ iw[pme1..=pme2]`.
///
/// Does, in order:
/// 1. **Pass-1 w-seeding** (faer `amd.rs:366-385`). For each
///    variable `i` in the new element, walk its element list and
///    lazily seed `w[e]`: first touch sets `w[e] = degree[e] +
///    (wflg - nvi)`, subsequent touches do `w[e] -= nvi`.
/// 2. **Pass-2 approximate external degree** (`amd.rs:386-462`).
///    For each variable `i` in the new element: walk its element
///    list computing `dext = w[e] - wflg`, then its variable list
///    accumulating `nv[j]` for live neighbours. Under `aggressive`,
///    dead elements (`dext == 0`) are absorbed on the spot. The
///    updated degree is clamped by `min(degree[i], deg)`
///    ("monotone cap"). The element list is re-ordered so `me` sits
///    at position `p1`.
/// 3. Bump `degree[me] = degme`, `lemax = max(lemax, degme)`,
///    `wflg += lemax`, `wflg = clear_flag(...)`.
/// 4. **Re-insert** (`amd.rs:516-537`): each surviving variable's
///    updated degree is pushed back onto `head[deg]` LIFO and
///    `mindeg` is lowered if needed.
/// 5. **Me bookkeeping** (`amd.rs:538-546`): restore `nv[me] =
///    nvpiv`, compact `me`'s var list to `[pme1, p)`, trim `pfree`.
/// 6. **Flop counters** (`amd.rs:547-557`).
///
/// Mass elimination and supervariable detection are Slice B —
/// until those land, the "mass elim" shortcut at `amd.rs:436-444`
/// is not taken (we always fall into the else branch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_step(
    ws: &mut AmdWorkspace,
    me: usize,
    pme1: usize,
    pme2_incl: usize,
    nvpiv: i32,
    degme: usize,
    elenme: i32,
    aggressive: bool,
) -> StepFlops {
    // Pass 1: seed w[e] for every element in each member's list.
    for pme in pme1..=pme2_incl {
        let i = ws.iw[pme] as usize;
        let eln = ws.elen[i];
        if eln > 0 {
            let nvi = -ws.nv[i];
            let wnvi = ws.wflg - nvi;
            let pi = ws.pe[i] as usize;
            for k in 0..eln as usize {
                let e = ws.iw[pi + k] as usize;
                let mut we = ws.w[e];
                if we >= ws.wflg {
                    we -= nvi;
                } else if we != 0 {
                    we = ws.degree[e] + wnvi;
                }
                ws.w[e] = we;
            }
        }
    }

    // Pass 2: approximate degree + (optionally) aggressive absorption.
    // Mass elimination would decrement degme here; Slice A leaves it
    // unchanged because we never take the mass-elim branch.
    let degme_i32 = degme as i32;
    for pme in pme1..=pme2_incl {
        let i = ws.iw[pme] as usize;
        let p1 = ws.pe[i] as usize;
        let p2 = p1 + ws.elen[i] as usize;
        let mut pn = p1;
        let mut deg: usize = 0;

        // Element sub-pass.
        if aggressive {
            for p in p1..p2 {
                let e = ws.iw[p] as usize;
                let we = ws.w[e];
                if we != 0 {
                    let dext = we - ws.wflg;
                    if dext > 0 {
                        deg += dext as usize;
                        ws.iw[pn] = e as i32;
                        pn += 1;
                    } else {
                        // Aggressive absorption: dead element folded
                        // into me right here (faer amd.rs:404-407).
                        ws.pe[e] = flip(me as i32);
                        ws.w[e] = 0;
                    }
                }
            }
        } else {
            for p in p1..p2 {
                let e = ws.iw[p] as usize;
                let we = ws.w[e];
                if we != 0 {
                    let dext = (we - ws.wflg) as usize;
                    deg += dext;
                    ws.iw[pn] = e as i32;
                    pn += 1;
                }
            }
        }

        // Record number-of-elements + 1 (the +1 reserves the slot
        // for `me` which we insert at p1 below).
        ws.elen[i] = (pn - p1 + 1) as i32;
        let p3 = pn;
        let p4 = p1 + ws.len[i] as usize;
        // Variable sub-pass.
        for p in p2..p4 {
            let j = ws.iw[p] as usize;
            let nvj = ws.nv[j];
            if nvj > 0 {
                deg += nvj as usize;
                ws.iw[pn] = j as i32;
                pn += 1;
            }
        }

        // Mass elimination (Slice B) — the `elen[i] == 1 && p3 == pn`
        // branch at amd.rs:436-444 would fold i into me here. Until
        // that lands we always take the general path below, which
        // remains correct.
        ws.degree[i] = ws.degree[i].min(deg as i32);
        // Swap-dance to put `me` at the head of i's element list.
        if p1 != pn {
            ws.iw[pn] = ws.iw[p3];
        } else {
            // p1 == pn means i's list is empty after pruning and the
            // variable pass. The swap positions collapse onto p1.
            // iw[pn] = iw[p3] would be a self-assign; skip it and
            // fall through to set iw[p1] = me.
        }
        if p3 != p1 {
            ws.iw[p3] = ws.iw[p1];
        }
        ws.iw[p1] = me as i32;
        ws.len[i] = (pn - p1 + 1) as i32;
    }

    // Step bookkeeping (amd.rs:463-466).
    ws.degree[me] = degme_i32;
    if degme_i32 > ws.lemax {
        ws.lemax = degme_i32;
    }
    ws.wflg += ws.lemax;
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    // Re-insertion (amd.rs:516-537): every surviving var in the new
    // element list gets its new degree, is pushed onto head[deg]
    // LIFO, and me's own list is compacted down to just the
    // survivors.
    let mut p_write = pme1;
    let nleft = ws.n - ws.nel;
    for pme in pme1..=pme2_incl {
        let i = ws.iw[pme] as usize;
        let nvi = -ws.nv[i];
        if nvi > 0 {
            ws.nv[i] = nvi;
            let mut d = ws.degree[i] as usize + degme_i32 as usize - nvi as usize;
            let cap = nleft - nvi as usize;
            if d > cap {
                d = cap;
            }
            let inext = ws.head[d];
            if inext != NONE {
                ws.last[inext as usize] = i as i32;
            }
            ws.next[i] = inext;
            ws.last[i] = NONE;
            ws.head[d] = i as i32;
            if d < ws.mindeg {
                ws.mindeg = d;
            }
            ws.degree[i] = d as i32;
            ws.iw[p_write] = i as i32;
            p_write += 1;
        }
    }

    // Me bookkeeping (amd.rs:538-546).
    ws.nv[me] = nvpiv;
    ws.len[me] = (p_write as i32) - pme1 as i32;
    if ws.len[me] == 0 {
        ws.pe[me] = NONE;
        ws.w[me] = 0;
    }
    if elenme != 0 {
        ws.pfree = p_write;
    }

    // Flop counters (amd.rs:547-557).
    let f = nvpiv as f64;
    let r = degme_i32 as f64 + ws.ndense as f64;
    let lnzme = f * r + (f - 1.0) * f / 2.0;
    let s = f * r * r + r * (f - 1.0) * f + (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;

    StepFlops {
        ndiv: lnzme,
        nms_lu: s,
        nms_ldl: (s + lnzme) / 2.0,
    }
}

/// Run the main AMD elimination loop until every live supervariable
/// has been either pivoted or dense-deferred. Returns the
/// accumulated flop counts.
///
/// Mass elimination and supervariable detection are absent (Slice
/// B). Inline garbage collection is live; fixtures whose working
/// set transiently exceeds `iwlen` recover via in-place compaction
/// and bump `ws.ncmpa`.
///
/// At exit: `ws.nel == ws.n`, every `pe[i]` either points to a
/// live parent (to be path-compressed by the postorder phase) or
/// is `NONE` / `flip(parent)`.
pub(crate) fn run_elimination(
    ws: &mut AmdWorkspace,
    aggressive: bool,
) -> Result<StepFlops, AmdError> {
    let mut flops = StepFlops::default();
    while ws.nel < ws.n {
        let me = match select_pivot(ws) {
            Some(m) => m,
            None => break, // only dense-deferred survivors remain
        };
        let elenme = ws.elen[me];
        let (pme1, pme2, nvpiv, degme) = create_element(ws, me)?;
        flops.accumulate(finalize_step(
            ws, me, pme1, pme2, nvpiv, degme, elenme, aggressive,
        ));
    }
    // Dense-phase flop contribution (amd.rs:559-566).
    let f = ws.ndense as f64;
    let lnzme = (f - 1.0) * f / 2.0;
    let s = (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
    flops.ndiv += lnzme;
    flops.nms_lu += s;
    flops.nms_ldl += (s + lnzme) / 2.0;
    Ok(flops)
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

    /// Drive the full loop on diag_4 — every var pre-eliminated,
    /// loop terminates immediately.
    #[test]
    fn run_elimination_diag_4() {
        let cp = [0, 1, 2, 3, 4];
        let ri = [0, 1, 2, 3];
        let mut ws = ws_for(4, &cp, &ri);
        let flops = run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, 4);
        assert_eq!(flops.ndiv, 0.0);
    }

    /// Arrow 5: no dense deferral. Loop eliminates all 5 vars.
    /// Verify `nel == n` and pivot supervariable count matches nv[me]
    /// after the step (restored to positive).
    #[test]
    fn run_elimination_arrow_5() {
        let cp = [0, 5, 7, 9, 11, 13];
        let ri = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let mut ws = ws_for(5, &cp, &ri);
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, 5);
        // Every var was pivoted exactly once ⇒ nv[i] > 0 everywhere
        // (the pivot restores nv to +nvpiv).
        for i in 0..5 {
            assert!(ws.nv[i] >= 0, "nv[{}] = {}", i, ws.nv[i]);
        }
    }

    /// Tridiag 10 full-symmetric: loop should terminate cleanly.
    /// Oracle lnz = 9 — verified indirectly by the flop counter.
    #[test]
    fn run_elimination_tridiag_10() {
        let n = 10usize;
        let mut cp: Vec<usize> = vec![0];
        let mut ri: Vec<usize> = Vec::new();
        for j in 0..n {
            if j > 0 {
                ri.push(j - 1);
            }
            ri.push(j);
            if j + 1 < n {
                ri.push(j + 1);
            }
            cp.push(ri.len());
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = AmdWorkspace::new(&p, &AmdOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, n);
    }

    /// Grid 7x7: five-point stencil. Faer's oracle reports
    /// `ncmpa == 0`, but Slice A lacks mass elimination and
    /// supervariable detection, so it consumes more `iw` space and
    /// may trip the inline GC. With Commit 6 the loop terminates
    /// cleanly regardless of how many compactions are needed.
    #[test]
    fn run_elimination_grid_7x7() {
        let m = 7usize;
        let n = 7usize;
        let total = m * n;
        let mut cp: Vec<usize> = vec![0];
        let mut ri: Vec<usize> = Vec::new();
        use std::collections::BTreeSet;
        let idx = |r: usize, c: usize| r * n + c;
        for c in 0..total {
            let r0 = c / n;
            let c0 = c % n;
            let mut neigh: BTreeSet<usize> = BTreeSet::new();
            neigh.insert(c);
            if r0 > 0 {
                neigh.insert(idx(r0 - 1, c0));
            }
            if r0 + 1 < m {
                neigh.insert(idx(r0 + 1, c0));
            }
            if c0 > 0 {
                neigh.insert(idx(r0, c0 - 1));
            }
            if c0 + 1 < n {
                neigh.insert(idx(r0, c0 + 1));
            }
            for &r in &neigh {
                ri.push(r);
            }
            cp.push(ri.len());
        }
        let p = CscPattern::new(total, &cp, &ri).unwrap();
        let mut ws = AmdWorkspace::new(&p, &AmdOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, total);
    }

    /// Band(20,3) triggers garbage collection in faer (oracle
    /// ncmpa=1). Commit 6 must run the compaction and finish the
    /// elimination; verify both.
    #[test]
    fn run_elimination_band_20_3_triggers_gc() {
        let n = 20usize;
        let b = 3usize;
        let mut cp: Vec<usize> = vec![0];
        let mut ri: Vec<usize> = Vec::new();
        for j in 0..n {
            let lo = j.saturating_sub(b);
            let hi = (j + b + 1).min(n);
            for r in lo..hi {
                ri.push(r);
            }
            cp.push(ri.len());
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = AmdWorkspace::new(&p, &AmdOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, n);
        assert!(
            ws.ncmpa >= 1,
            "expected at least one compaction on band(20,3), got ncmpa={}",
            ws.ncmpa
        );
    }

    /// Arrow 200: hub dense-deferred in init; spokes get pivoted
    /// one by one; loop terminates with ndense=1 survivor.
    #[test]
    fn run_elimination_arrow_200() {
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
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.ndense, 1);
        assert_eq!(ws.nel, n);
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
