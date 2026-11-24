//! Issue #246: `amf_order_interruptible` stops on a triggered interrupt and
//! is bit-identical to `amf_order_full` with `Interrupt::none()`.

use feral_amf::{
    amf_order_full, amf_order_interruptible, AmfOptions, CscPattern, Interrupt, OrderingError,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
fn expired_deadline_and_set_flag_interrupt() {
    let (n, cp, ri) = grid(60);
    let pat = CscPattern::new(n, &cp, &ri).expect("pattern");
    let opts = AmfOptions::default();
    let expired = Interrupt::at(Instant::now() - Duration::from_secs(1));
    assert_eq!(
        amf_order_interruptible(&pat, &opts, &expired).err(),
        Some(OrderingError::Interrupted)
    );
    let flagged = Interrupt::none().with_flag(Arc::new(AtomicBool::new(true)));
    assert_eq!(
        amf_order_interruptible(&pat, &opts, &flagged).err(),
        Some(OrderingError::Interrupted)
    );
}

#[test]
fn inert_interrupt_is_bit_identical() {
    let (n, cp, ri) = grid(60);
    let pat = CscPattern::new(n, &cp, &ri).expect("pattern");
    let opts = AmfOptions::default();
    let base = amf_order_full(&pat, &opts).expect("order").0;
    let far = Interrupt::after(Duration::from_secs(3600));
    for i in [Interrupt::none(), far] {
        let got = amf_order_interruptible(&pat, &opts, &i).expect("order").0;
        assert_eq!(got, base);
    }
}
