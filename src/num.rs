//! Utilities related to numbers.

use ieee754;
use num;
use std::{cmp, fmt};

#[cfg(feature = "comparison")]
use approx::{AbsDiffEq, RelativeEq};

/// Floating point marker trait for easier control over trait bounds.
pub trait BFloat:
    Sync
    + Send
    + num::Float
    + num::cast::FromPrimitive
    + ieee754::Ieee754
    + fmt::Debug
    + fmt::Display
    + Into<f64>
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

/// Wrapper for the slice type that implements comparison traits.
#[cfg(feature = "comparison")]
#[derive(Debug, Clone, PartialEq)]
pub struct ComparableSlice<'a, T>(pub &'a [T]);

#[cfg(feature = "comparison")]
impl<'a, T> AbsDiffEq for ComparableSlice<'a, T>
where
    T: AbsDiffEq,
    T::Epsilon: Copy,
{
    type Epsilon = <T as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| T::abs_diff_eq(a, b, epsilon))
    }
}

#[cfg(feature = "comparison")]
impl<'a, T> RelativeEq for ComparableSlice<'a, T>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| T::relative_eq(a, b, epsilon, max_relative))
    }
}
