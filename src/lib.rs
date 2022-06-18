//! Useful tools for working with Bifrost in Rust.

#![allow(clippy::too_many_arguments)]

pub mod constants;
pub mod error;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod interpolation;
pub mod io;
pub mod num;
pub mod units;

#[cfg(feature = "seeding")]
pub mod random;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "seeding")]
pub mod seeding;

#[cfg(feature = "tracing")]
pub mod tracing;

#[cfg(feature = "corks")]
pub mod corks;

#[cfg(feature = "ebeam")]
pub mod ebeam;
#[cfg(feature = "ebeam")]
pub mod math;
#[cfg(feature = "ebeam")]
pub mod plasma;

// mod test {
//     use crate::{
//         field::ScalarField3,
//         geometry::Point3,
//         grid::{fgr, GridPointQuery3},
//         interpolation::{fip, InterpGridVerifier3},
//         num::BFloat,
//     };

//     pub trait Interpolator3<F: BFloat>: InterpGridVerifier3 + Copy + Sync + Send {
//         fn interp_scalar_field(
//             &self,
//             field: &ScalarField3<F>,
//             interp_point: &Point3<fgr>,
//         ) -> GridPointQuery3<fgr, fip>;
//     }

//     fn test<F>(a: Box<dyn Interpolator3<F>>) {}
// }
