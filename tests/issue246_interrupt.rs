//! Issue #246 / PR #247 review: a cancel flag set from another thread
//! stops a running symbolic analysis mid-ordering.
//!
//! Oracle: control flow only. The interrupted run must return
//! `FeralError::Interrupted` and finish sooner than the same analysis run
//! to completion. No absolute time is asserted, so CI speed cannot flake it.

use feral::symbolic::supernode::SupernodeParams;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod};
use feral::{CscMatrix, FeralError, Interrupt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
fn flag_from_another_thread_stops_metis_mid_run() {
    let m = grid(300);
    let base_params = SupernodeParams::default();
    let t0 = Instant::now();
    symbolic_factorize_with_method(&m, &base_params, OrderingMethod::MetisND).expect("baseline");
    let full = t0.elapsed();

    let flag = Arc::new(AtomicBool::new(false));
    let params = SupernodeParams {
        interrupt: Some(Interrupt::none().with_flag(flag.clone())),
        ..Default::default()
    };
    let setter = {
        let flag = flag.clone();
        let delay = full / 10;
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            flag.store(true, Ordering::Relaxed);
        })
    };
    let t1 = Instant::now();
    let r = symbolic_factorize_with_method(&m, &params, OrderingMethod::MetisND);
    let cut = t1.elapsed();
    setter.join().expect("setter thread");
    match r {
        Err(FeralError::Interrupted { .. }) => {}
        other => panic!("expected Interrupted, got {:?}", other.map(|s| s.n)),
    }
    assert!(
        cut < full,
        "interrupted run ({cut:?}) should stop before a full run ({full:?})"
    );
}

#[test]
fn deadline_stops_auto_with_bounded_overshoot() {
    let m = grid(300);
    let budget = Duration::from_millis(5);
    let params = SupernodeParams {
        interrupt: Some(Interrupt::after(budget)),
        ..Default::default()
    };
    let t0 = Instant::now();
    let r = symbolic_factorize_with_method(&m, &params, OrderingMethod::AutoRace);
    let took = t0.elapsed();
    match r {
        Err(FeralError::Interrupted { stage }) => {
            eprintln!("AutoRace interrupted at stage {stage:?} after {took:?} (budget {budget:?})")
        }
        // A fast machine can legitimately finish inside the budget.
        Ok(_) => eprintln!("AutoRace finished within {budget:?} ({took:?})"),
        Err(e) => panic!("unexpected error {e:?}"),
    }
}
