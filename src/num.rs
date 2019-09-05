//! Utilities related to numbers.

use std::fmt;
use num;
use ieee754;

/// Floating point marker trait for easier control over trait bounds.
pub trait BFloat: num::Float + num::cast::FromPrimitive + ieee754::Ieee754 + fmt::Debug {}

impl BFloat for f32 {}
impl BFloat for f64 {}