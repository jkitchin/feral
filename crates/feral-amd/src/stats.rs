//! Diagnostic counters exposed alongside the permutation.

/// Diagnostic counters collected during AMD ordering.
///
/// In release builds only `ncmpa` has non-zero cost; other fields
/// are populated without adding branches to the hot loop. In debug
/// builds every field is populated.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AmdStats {
    /// Number of garbage-collection compactions fired.
    pub ncmpa: u32,
    /// Number of mark-array generation-counter resets.
    pub n_clear_flag: u32,
    /// Number of variables absorbed by mass elimination
    /// (Slice B).
    pub n_mass_elim: u32,
    /// Number of supervariable merges detected (Slice B).
    pub n_supervar_merge: u32,
    /// Number of variables placed into the dense-deferred bucket
    /// at initialization.
    pub n_dense_deferred: u32,
    /// Flop counter: divisions (faer amd.rs:547-566).
    pub ndiv: u64,
    /// Flop counter: LU multiply-subtracts.
    pub nms_lu: u64,
    /// Flop counter: LDLᵀ multiply-subtracts.
    pub nms_ldl: u64,
}
