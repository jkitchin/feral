//! PR #247 review: guard the interrupt's single-thread assumption. Every
//! interrupt poll made while `metis_order_interruptible` runs must come from
//! the calling thread (which has the scope). If the ordering ever moves work
//! onto other threads without forwarding the interrupt, those polls are
//! counted as unscoped and this test fails.
//!
//! Kept as the only test in its own binary, so that no unrelated thread
//! polls while the census is armed.

use feral_metis::{metis_order_interruptible, CscPattern, Interrupt, MetisOptions};
use feral_ordering_core::interrupt::census;
use std::time::Duration;

/// 5-point Laplacian pattern of a `k x k` grid, full symmetric, with diagonal.
fn grid(k: usize) -> (usize, Vec<i32>, Vec<i32>) {
    let n = k * k;
    let mut cp = vec![0i32];
    let mut ri = Vec::new();
    for j in 0..n {
        let (r, c) = (j / k, j % k);
        let mut col = vec![j];
        if r > 0 {
            col.push(j - k);
        }
        if c > 0 {
            col.push(j - 1);
        }
        if c + 1 < k {
            col.push(j + 1);
        }
        if r + 1 < k {
            col.push(j + k);
        }
        col.sort_unstable();
        ri.extend(col.iter().map(|&x| x as i32));
        cp.push(ri.len() as i32);
    }
    (n, cp, ri)
}

#[test]
fn every_poll_runs_on_the_scoped_thread() {
    let (n, cp, ri) = grid(120);
    let pat = CscPattern::new(n, &cp, &ri).expect("pattern");
    let far = Interrupt::after(Duration::from_secs(3600));
    census::arm();
    let r = metis_order_interruptible(&pat, &MetisOptions::default(), &far);
    let counts = census::take();
    r.expect("ordering");
    assert_eq!(
        counts.unscoped, 0,
        "polls from a thread with no scope: {counts:?}"
    );
    assert!(counts.scoped > 0, "the ordering never polled: {counts:?}");
}
