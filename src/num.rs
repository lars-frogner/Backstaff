//! Utilities related to numbers.

use ieee754;
use num;
use std::{cmp, fmt};

/// Floating point marker trait for easier control over trait bounds.
pub trait BFloat:
    Sync + Send + num::Float + num::cast::FromPrimitive + ieee754::Ieee754 + fmt::Debug + fmt::Display
{
}

impl BFloat for f32 {}
impl BFloat for f64 {}

/// Key-value pair that can be ordered based on the value.
pub struct KeyValueOrderableByValue<K, V>(pub K, pub V);

impl<K, V: PartialEq> PartialEq for KeyValueOrderableByValue<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<K, V: PartialOrd> PartialOrd for KeyValueOrderableByValue<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl<K, V: PartialEq> Eq for KeyValueOrderableByValue<K, V> {}

impl<K, V: PartialOrd> Ord for KeyValueOrderableByValue<K, V> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).expect("NaN in value comparison")
    }
}
