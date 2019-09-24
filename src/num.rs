//! Utilities related to numbers.

use ieee754;
use num;
use std::fmt;

/// Floating point marker trait for easier control over trait bounds.
pub trait BFloat:
    Sync + Send + num::Float + num::cast::FromPrimitive + ieee754::Ieee754 + fmt::Debug
{
}

impl BFloat for f32 {}
impl BFloat for f64 {}
