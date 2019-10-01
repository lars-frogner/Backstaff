//! Utilities related to numbers.

use ieee754;
use num;
use std::{cmp, fmt};

/// Floating point marker trait for easier control over trait bounds.
pub trait BFloat:
    Sync + Send + num::Float + num::cast::FromPrimitive + ieee754::Ieee754 + fmt::Debug
{
}

impl BFloat for f32 {}
impl BFloat for f64 {}

/// Integer-float pair that can be ordered based on the float.
pub struct OrderableIndexValuePair<I: num::Integer, F: BFloat>(pub I, pub F);

impl<I: num::Integer, F: BFloat> PartialEq for OrderableIndexValuePair<I, F> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<I: num::Integer, F: BFloat> PartialOrd for OrderableIndexValuePair<I, F> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl<I: num::Integer, F: BFloat> Eq for OrderableIndexValuePair<I, F> {}

impl<I: num::Integer, F: BFloat> Ord for OrderableIndexValuePair<I, F> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(&other)
            .expect("NaN in floating point comparison.")
    }
}
