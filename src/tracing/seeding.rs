//! Generation of seed points for field line tracing.

pub mod slice;
pub mod criterion;

use rayon::prelude::*;
use crate::geometry::Point3;
use super::ftr;

/// Defines the properties of a 3D seed point generator.
pub trait Seeder3: IntoIterator<Item = Point3<ftr>> + IntoParallelIterator<Item = Point3<ftr>> {
    /// Returns the number of seed points that will be produced by the seeder.
    fn number_of_points(&self) -> usize;

    /// Filters the seed points using the given predicate.
    fn retain<P>(&mut self, predicate: P)
    where P: FnMut(&Point3<ftr>) -> bool;
}

// Let a vector of points work as a seeder.
impl Seeder3 for Vec<Point3<ftr>> {
    fn number_of_points(&self) -> usize { self.len() }

    fn retain<P>(&mut self, predicate: P)
    where P: FnMut(&Point3<ftr>) -> bool
    {
        self.retain(predicate);
    }
}
