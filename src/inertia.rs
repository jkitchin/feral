/// Inertia of a symmetric matrix: counts of positive, negative, zero eigenvalues.
/// Invariant: positive + negative + zero == n.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

impl Inertia {
    /// Create a new Inertia with explicit counts.
    pub fn new(positive: usize, negative: usize, zero: usize) -> Self {
        Self {
            positive,
            negative,
            zero,
        }
    }

    /// Total dimension: positive + negative + zero.
    pub fn total(&self) -> usize {
        self.positive + self.negative + self.zero
    }
}

impl std::fmt::Display for Inertia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.positive, self.negative, self.zero)
    }
}
