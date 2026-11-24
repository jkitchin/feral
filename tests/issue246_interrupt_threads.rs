//! PR #247 review: guard the interrupt's single-thread assumption across
//! feral's whole symbolic pipeline. Every interrupt poll during
//! `symbolic_factorize_with_method` (all ordering methods, both races, and
//! the MC64 matching behind a forced `LdltCompress`) must come from the
//! calling thread. A poll from a thread with no scope means the deadline
//! would silently not apply there.
//!
//! Kept as the only test in its own binary, so that no unrelated thread
//! polls while the census is armed.

use feral::symbolic::supernode::{OrderingPreprocess, SupernodeParams};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod};
use feral::{CscMatrix, Interrupt};
use feral_ordering_core::interrupt::census;
use std::time::Duration;

fn grid(k: usize) -> CscMatrix {
    let n = k * k;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(4.0);
        if j % k + 1 < k {
            r.push(j + 1);
            c.push(j);
            v.push(-1.0);
        }
        if j + k < n {
            r.push(j + k);
            c.push(j);
            v.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).expect("grid")
}

#[test]
fn every_symbolic_poll_runs_on_the_scoped_thread() {
    let m = grid(80);
    let far = Interrupt::after(Duration::from_secs(3600));
    census::arm();
    for pre in [OrderingPreprocess::None, OrderingPreprocess::LdltCompress] {
        for method in [
            OrderingMethod::Amd,
            OrderingMethod::Amf,
            OrderingMethod::MetisND,
            OrderingMethod::ScotchND,
            OrderingMethod::KahipND,
            OrderingMethod::Auto,
            OrderingMethod::AutoRace,
        ] {
            let params = SupernodeParams {
                preprocess: pre,
                interrupt: Some(far.clone()),
                ..Default::default()
            };
            symbolic_factorize_with_method(&m, &params, method.clone())
                .unwrap_or_else(|e| panic!("{method:?}/{pre:?}: {e:?}"));
        }
    }
    let counts = census::take();
    assert_eq!(
        counts.unscoped, 0,
        "polls from a thread with no scope: {counts:?}"
    );
    assert!(counts.scoped > 0, "the pipeline never polled: {counts:?}");
}
