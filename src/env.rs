//! One parse policy for the numeric `FERAL_*` tuning knobs (issue #176).
//!
//! Every numeric knob used to be read as
//!
//! ```ignore
//! std::env::var(NAME).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(DEFAULT)
//! ```
//!
//! which drops the parse error on the floor: `FERAL_PAR_TASK_MIN_FLOPS=1e18`
//! is not a `u64` literal, so the knob was *silently* replaced by its
//! built-in default. The path the operator believed they had switched off
//! stayed on, and the timing they took from that run looked valid.
//! Issue #176 caught two such measurements.
//!
//! The policy here, applied by every numeric knob in the tree:
//!
//! - **Scientific notation parses.** The defaults are written `1e6` /
//!   `1e8` in the docs and in pounce's option help, so `1e18` must mean
//!   `1_000_000_000_000_000_000` on an integer knob. Plain integers keep
//!   parsing exactly — the integer parse is tried *first*, so
//!   `18446744073709551615` survives as `u64::MAX` instead of rounding
//!   through `f64`.
//! - **A set-but-unusable value warns on stderr** in the same shape the
//!   `FERAL_SCALING` vocabulary check already uses (`capi.rs`), and only
//!   then falls back to the caller's default. The knob's intent is still
//!   lost, but the run no longer *looks* like the knob took effect.
//! - **Warned once per (name, value)**: knobs like
//!   [`crate::dense::factor`]'s intra-front area gate are read once per
//!   front, so an unconditional `eprintln!` would flood stderr.
//! - **An above-range magnitude clamps rather than falls back.**
//!   `FERAL_CB_THRESH=1e30` means "switch this path off"; clamping to
//!   `u64::MAX` honours that, whereas falling back to the default would
//!   reproduce exactly the bug being fixed. The clamp is announced.
//!
//! Boolean and enum-valued knobs (`FERAL_PARALLEL`, `FERAL_PACKED_SIMD`,
//! ...) match a literal vocabulary rather than parsing a number and are
//! not routed through here.
//!
//! Rationale and the full knob inventory:
//! `dev/research/env-knob-parsing-2026-08-19.md`.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// The outcome of parsing one raw value, before any reporting. Split
/// from the env read so the policy is unit-testable without mutating
/// process-global state (the same split `bench.rs` uses for the
/// `FERAL_SCALING` / `FERAL_ORDERING` vocabularies).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Knob<T> {
    /// Usable as given.
    Ok(T),
    /// Usable, but the magnitude exceeded what the knob's type holds and
    /// was clamped to the value carried here.
    Clamped(T),
    /// Unusable. The string says why, for the stderr warning.
    Bad(&'static str),
}

/// Print `warning: NAME="RAW" MSG` to stderr the first time this exact
/// `(name, raw)` pair is rejected. Later reads of the same bad value are
/// silent — the knobs are read per factorization, per solve, or per
/// front.
fn warn_once(name: &str, raw: &str, msg: &str) {
    static SEEN: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let first = match seen.lock() {
        Ok(mut set) => set.insert((name.to_string(), raw.to_string())),
        // Poisoned: a warning that repeats beats a warning that vanishes.
        Err(_) => true,
    };
    if first {
        eprintln!("warning: {name}=\"{raw}\" {msg}");
    }
}

/// Turn a parse outcome into an `Option`, warning on anything the caller
/// would otherwise never hear about.
fn report<T>(name: &str, raw: &str, parsed: Knob<T>) -> Option<T> {
    match parsed {
        Knob::Ok(v) => Some(v),
        Knob::Clamped(v) => {
            warn_once(
                name,
                raw,
                "is larger than this knob can hold; clamped to its maximum",
            );
            Some(v)
        }
        Knob::Bad(why) => {
            warn_once(
                name,
                raw,
                &format!("{why}; falling back to the built-in default"),
            );
            None
        }
    }
}

/// Parse a non-negative integer knob, accepting both plain integer and
/// float/scientific spellings. `max` is the largest value the target
/// type holds.
///
/// Integer-first is load-bearing: `u64::MAX as f64` rounds up to 2^64,
/// so routing `18446744073709551615` through `f64` would report a clamp
/// on a value that is exactly representable.
///
/// Fractional input rounds half-away-from-zero rather than truncating:
/// truncation would turn `FERAL_PAR_MIN_SEEDS=0.9` into 0 ("always
/// parallel"), the opposite of what the value asks for.
fn parse_unsigned(raw: &str, max: u128) -> Knob<u128> {
    let s = raw.trim();
    if s.is_empty() {
        return Knob::Bad("is empty");
    }
    if let Ok(v) = s.parse::<u128>() {
        return if v > max {
            Knob::Clamped(max)
        } else {
            Knob::Ok(v)
        };
    }
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 => {
            let rounded = v.round();
            // `max as f64` rounds *up* for u64/usize, so `>=` (not `>`)
            // keeps a value that lands on 2^64 out of the `as` cast.
            if rounded >= max as f64 {
                Knob::Clamped(max)
            } else {
                Knob::Ok(rounded as u128)
            }
        }
        Ok(v) if v.is_finite() => Knob::Bad("is negative, and this knob counts work"),
        // `+inf` (any magnitude past f64 range, e.g. `1e400`) is the
        // operator asking for "as large as possible". It clamps for the
        // same reason `1e30` does: falling back to the default here
        // would invert the intent rather than merely lose it, which is
        // issue #176's failure mode. Anything else non-finite (`-inf`,
        // `nan`) has no such reading.
        Ok(v) if v == f64::INFINITY => Knob::Clamped(max),
        Ok(_) => Knob::Bad("is not a finite number"),
        Err(_) => Knob::Bad("is not a number"),
    }
}

/// Parse a finite floating-point knob. Scientific notation is native to
/// `f64::from_str`; the added policy is that `nan` and `inf` are refused
/// rather than propagated into a pivot threshold.
fn parse_float(raw: &str) -> Knob<f64> {
    let s = raw.trim();
    if s.is_empty() {
        return Knob::Bad("is empty");
    }
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() => Knob::Ok(v),
        Ok(_) => Knob::Bad("is not a finite number"),
        Err(_) => Knob::Bad("is not a number"),
    }
}

/// Read and parse an unsigned knob, keeping the raw text for any
/// follow-up warning from a caller-side validity check.
fn unsigned_var_raw(name: &str, max: u128) -> Option<(String, u128)> {
    let raw = std::env::var(name).ok()?;
    let v = report(name, &raw, parse_unsigned(&raw, max))?;
    Some((raw, v))
}

/// Read and parse a floating-point knob, keeping the raw text.
fn float_var_raw(name: &str) -> Option<(String, f64)> {
    let raw = std::env::var(name).ok()?;
    let v = report(name, &raw, parse_float(&raw))?;
    Some((raw, v))
}

/// `None` when the knob is unset, or when it is set to something this
/// module refused (after warning) — so the call site's own default
/// expression stays in one place.
pub fn u64_var(name: &str) -> Option<u64> {
    unsigned_var_raw(name, u64::MAX as u128).map(|(_, v)| v as u64)
}

/// [`u64_var`], for the `usize`-typed knobs.
pub fn usize_var(name: &str) -> Option<usize> {
    unsigned_var_raw(name, usize::MAX as u128).map(|(_, v)| v as usize)
}

/// [`usize_var`] with a caller-side validity check. `requirement`
/// completes the sentence "must be ..." in the warning, so an
/// out-of-range value is as loud as an unparseable one.
pub fn usize_var_where(
    name: &str,
    requirement: &str,
    accept: impl Fn(usize) -> bool,
) -> Option<usize> {
    let (raw, v) = unsigned_var_raw(name, usize::MAX as u128)?;
    let v = v as usize;
    if accept(v) {
        return Some(v);
    }
    warn_once(
        name,
        &raw,
        &format!("must be {requirement}; falling back to the built-in default"),
    );
    None
}

/// A comma-separated list knob (the diagnostics sweeps: `arms` lists
/// like `FERAL_NEMIN_LIST=1,4,8`), parsed token-by-token under the same
/// policy as [`usize_var`].
///
/// The failure this exists to prevent is narrower than #176 but worse:
/// a locally-parsed list drops its unusable tokens through `filter_map`
/// and hands the caller a *shorter* list, so a sweep written
/// `0,1e3,1e6` silently runs with one arm and reports "no difference"
/// from an experiment that never had a second arm. Refused tokens warn
/// individually, and a list with nothing usable left returns `None` so
/// the caller's own default list is used rather than an empty sweep.
fn unsigned_list_raw(name: &str, max: u128) -> Option<Vec<u128>> {
    let raw = std::env::var(name).ok()?;
    let mut out = Vec::new();
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some(v) = report(name, token, parse_unsigned(token, max)) {
            out.push(v);
        }
    }
    if out.is_empty() {
        warn_once(
            name,
            &raw,
            "has no usable values; falling back to the built-in default list",
        );
        return None;
    }
    Some(out)
}

/// [`unsigned_list_raw`] for the `u128`-typed sweep knobs.
pub fn u128_list_var(name: &str) -> Option<Vec<u128>> {
    unsigned_list_raw(name, u128::MAX)
}

/// [`unsigned_list_raw`] for the `usize`-typed sweep knobs.
pub fn usize_list_var(name: &str) -> Option<Vec<usize>> {
    unsigned_list_raw(name, usize::MAX as u128).map(|v| v.into_iter().map(|x| x as usize).collect())
}

/// `None` when the knob is unset, or when it is set to something that is
/// not a finite number (after warning).
pub fn f64_var(name: &str) -> Option<f64> {
    float_var_raw(name).map(|(_, v)| v)
}

/// [`f64_var`] with a caller-side validity check; see
/// [`usize_var_where`] for `requirement`.
pub fn f64_var_where(name: &str, requirement: &str, accept: impl Fn(f64) -> bool) -> Option<f64> {
    let (raw, v) = float_var_raw(name)?;
    if accept(v) {
        return Some(v);
    }
    warn_once(
        name,
        &raw,
        &format!("must be {requirement}; falling back to the built-in default"),
    );
    None
}

/// Comma-separated sweep list of floats, e.g.
/// `STATIC_PIVOTS=0,1e-12,1e-10`. The float twin of [`usize_list_var`],
/// and it exists for the same reason: a locally-parsed list silently
/// *drops* the tokens it cannot read, so a sweep reports "no difference"
/// from an experiment that never ran the arms the operator asked for.
/// Each unusable token warns and is skipped; an all-unusable list warns
/// and yields `None` so the caller falls back to its own default list.
pub fn f64_list_var(name: &str) -> Option<Vec<f64>> {
    let raw = std::env::var(name).ok()?;
    let mut out = Vec::new();
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some(v) = report(name, token, parse_float(token)) {
            out.push(v);
        }
    }
    if out.is_empty() {
        warn_once(
            name,
            &raw,
            "has no usable values; falling back to the built-in default list",
        );
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const U64: u128 = u64::MAX as u128;

    /// The reporter's case, verbatim from issue #176: this is the value
    /// that used to vanish into the default.
    #[test]
    fn scientific_notation_parses_on_an_integer_knob() {
        assert_eq!(
            parse_unsigned("1e18", U64),
            Knob::Ok(1_000_000_000_000_000_000)
        );
    }

    /// The two spellings of the documented default must agree — the docs
    /// write `1e6`, the code writes `1_000_000`.
    #[test]
    fn integer_and_scientific_spellings_agree() {
        assert_eq!(parse_unsigned("1e6", U64), Knob::Ok(1_000_000));
        assert_eq!(parse_unsigned("1000000", U64), Knob::Ok(1_000_000));
        assert_eq!(parse_unsigned("1.0e6", U64), Knob::Ok(1_000_000));
    }

    /// `u64::MAX as f64` is 2^64, so a float round-trip would report a
    /// clamp here. The integer parse has to be tried first.
    #[test]
    fn u64_max_survives_exactly() {
        assert_eq!(parse_unsigned("18446744073709551615", U64), Knob::Ok(U64));
    }

    #[test]
    fn above_range_clamps_rather_than_falling_back() {
        assert_eq!(parse_unsigned("1e30", U64), Knob::Clamped(U64));
        assert_eq!(
            parse_unsigned("18446744073709551616", U64),
            Knob::Clamped(U64)
        );
        // Exactly 2^64: representable as f64, one past the type.
        assert_eq!(
            parse_unsigned("1.8446744073709552e19", U64),
            Knob::Clamped(U64)
        );
    }

    /// The hole the first version of this module left: `1e30` clamped
    /// but `1e400` did not, because it parses to `+inf` and fell into
    /// the non-finite arm. An operator escalating past `1e30` to be
    /// *extra* sure of "never" would have landed back on the default —
    /// more parallelism, not less. Intent inverted, which is exactly
    /// what the clamp policy exists to prevent.
    #[test]
    fn past_f64_range_clamps_like_any_other_magnitude() {
        assert_eq!(parse_unsigned("1e400", U64), Knob::Clamped(U64));
        assert_eq!(parse_unsigned("inf", U64), Knob::Clamped(U64));
        assert_eq!(parse_unsigned("infinity", U64), Knob::Clamped(U64));
        assert_eq!(parse_unsigned("1e400", 255), Knob::Clamped(255));
        // The sign still matters: `-inf` is not a count of work.
        assert!(matches!(parse_unsigned("-inf", U64), Knob::Bad(_)));
    }

    #[test]
    fn unusable_values_are_refused_not_defaulted_silently() {
        assert!(matches!(parse_unsigned("", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("   ", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("abc", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("1e", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("1e18x", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("-1", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("-1e6", U64), Knob::Bad(_)));
        assert!(matches!(parse_unsigned("nan", U64), Knob::Bad(_)));
        // `-inf` has no "as large as possible" reading; `+inf` does and
        // clamps instead (see `past_f64_range_clamps_like_any_other_magnitude`).
        assert!(matches!(parse_unsigned("-inf", U64), Knob::Bad(_)));
    }

    #[test]
    fn surrounding_whitespace_is_ignored() {
        assert_eq!(parse_unsigned("  42  ", U64), Knob::Ok(42));
        assert_eq!(parse_float(" 1e-8 "), Knob::Ok(1e-8));
    }

    /// Half-away-from-zero, so a fractional seed count cannot round down
    /// into "no gate at all".
    #[test]
    fn fractional_input_rounds_rather_than_truncating() {
        assert_eq!(parse_unsigned("0.9", U64), Knob::Ok(1));
        assert_eq!(parse_unsigned("2.5", U64), Knob::Ok(3));
        assert_eq!(parse_unsigned("2.4", U64), Knob::Ok(2));
        assert_eq!(parse_unsigned("0.4", U64), Knob::Ok(0));
    }

    #[test]
    fn small_type_max_clamps_at_its_own_bound() {
        assert_eq!(parse_unsigned("300", 255), Knob::Clamped(255));
        assert_eq!(parse_unsigned("255", 255), Knob::Ok(255));
    }

    /// A list knob's tokens go through the same policy as a scalar
    /// one — the point of routing them here rather than through a local
    /// `filter_map(|t| t.parse().ok())`, which drops `1e3` and silently
    /// shortens the sweep.
    #[test]
    fn list_tokens_use_the_same_policy_as_scalar_knobs() {
        for token in ["0", "1e3", " 1e6 ", "2.5"] {
            assert!(
                matches!(parse_unsigned(token.trim(), U64), Knob::Ok(_)),
                "list token {token:?} must be usable"
            );
        }
        assert_eq!(parse_unsigned("1e3", U64), Knob::Ok(1000));
    }

    #[test]
    fn float_knobs_take_the_usual_pivot_threshold_spellings() {
        assert_eq!(parse_float("1e-8"), Knob::Ok(1e-8));
        assert_eq!(parse_float("0.001"), Knob::Ok(0.001));
        assert_eq!(parse_float("-0.5"), Knob::Ok(-0.5));
        assert!(matches!(parse_float("nan"), Knob::Bad(_)));
        // Unlike the counting knobs, a float knob is a threshold or a
        // ratio: `inf` has no usable reading and stays refused.
        assert!(matches!(parse_float("inf"), Knob::Bad(_)));
        assert!(matches!(parse_float("1e-"), Knob::Bad(_)));
        assert!(matches!(parse_float(""), Knob::Bad(_)));
    }
}
