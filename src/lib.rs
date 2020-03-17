//! Useful tools for working with Bifrost in Rust.

#![allow(incomplete_features)]
#![feature(const_generics, maybe_uninit_extra)]

pub mod constants;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod interpolation;
pub mod io;
pub mod math;
pub mod num;
pub mod plasma;
pub mod random;
pub mod units;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "tracing")]
pub mod tracing;

#[cfg(feature = "ebeam")]
pub mod ebeam;
