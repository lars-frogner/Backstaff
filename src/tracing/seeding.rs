//! Generation of seed points for field line tracing.

pub mod criterion;
pub mod slice;

use super::ftr;
use crate::geometry::{Idx3, Point3};
use rayon::prelude::*;

/// Defines the properties of a 3D seed point generator.
pub trait Seeder3:
    IntoIterator<Item = Point3<ftr>> + IntoParallelIterator<Item = Point3<ftr>>
{
    /// Returns the number of seed points that will be produced by the seeder.
    fn number_of_points(&self) -> usize;

    /// Filters the seed points using the given predicate.
    fn retain_points<P>(&mut self, predicate: P)
    where
        P: FnMut(&Point3<ftr>) -> bool;
}

/// Defines the properties of a 3D seed index generator.
pub trait IndexSeeder3:
    IntoIterator<Item = Idx3<usize>> + IntoParallelIterator<Item = Idx3<usize>>
{
    /// Returns the number of seed indices that will be produced by the seeder.
    fn number_of_indices(&self) -> usize;

    /// Filters the seed indices using the given predicate.
    fn retain_indices<P>(&mut self, predicate: P)
    where
        P: FnMut(&Idx3<usize>) -> bool;
}

// Let a vector of points work as a seeder.
impl Seeder3 for Vec<Point3<ftr>> {
    fn number_of_points(&self) -> usize {
        self.len()
    }

    fn retain_points<P>(&mut self, predicate: P)
    where
        P: FnMut(&Point3<ftr>) -> bool,
    {
        self.retain(predicate);
    }
}

// Let a vector of indices work as a seeder.
impl IndexSeeder3 for Vec<Idx3<usize>> {
    fn number_of_indices(&self) -> usize {
        self.len()
    }

    fn retain_indices<P>(&mut self, predicate: P)
    where
        P: FnMut(&Idx3<usize>) -> bool,
    {
        self.retain(predicate);
    }
}
