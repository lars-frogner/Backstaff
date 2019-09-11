//! Generation of seed points for field line tracing.

pub mod slice;

use crate::geometry::Point3;
use super::ftr;

/// Defines the properties of a 3D seed point generator.
pub trait Seeder3: IntoIterator<Item = Point3<ftr>> {}

// Let a vector of points work as a seeder.
impl Seeder3 for Vec<Point3<ftr>> {}
