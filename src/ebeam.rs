//! Non-thermal electron beam physics in Bifrost simulations.

pub mod acceleration;
pub mod distribution;

/// Floating-point precision to use for electron beam physics.
#[allow(non_camel_case_types)]
pub type feb = f64;
