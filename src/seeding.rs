//! Generation of seed points for field line tracing.

pub mod criterion;
pub mod manual;
pub mod slice;
pub mod volume;

use crate::{
    geometry::{Idx3, Point3},
    grid::{fgr, Grid3},
    num::BFloat,
};
use rayon::prelude::*;

/// Defines the properties of a 3D seed point generator.
pub trait Seeder3:
    IntoIterator<Item = Point3<fgr>> + IntoParallelIterator<Item = Point3<fgr>>
{
    /// Returns the number of seed points that will be produced by the seeder.
    fn number_of_points(&self) -> usize;

    /// Filters the seed points using the given predicate.
    fn retain_points<P>(&mut self, predicate: P)
    where
        P: FnMut(&Point3<fgr>) -> bool;

    /// Creates a list of seed indices from the seed points by looking up the grid cells
    /// of the given grid containing the seed points.
    fn to_index_seeder<F, G>(&self, grid: &G) -> Vec<Idx3<usize>>
    where
        F: BFloat,
        G: Grid3<F>;
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

    /// Creates a list of seed points from the seed indices by indexing the center coordinates
    /// of the given grid.
    fn to_point_seeder<F, G>(&self, grid: &G) -> Vec<Point3<fgr>>
    where
        F: BFloat,
        G: Grid3<F>;
}

// Let a vector of points work as a seeder.
impl Seeder3 for Vec<Point3<fgr>> {
    fn number_of_points(&self) -> usize {
        self.len()
    }

    fn retain_points<P>(&mut self, predicate: P)
    where
        P: FnMut(&Point3<fgr>) -> bool,
    {
        self.retain(predicate);
    }

    fn to_index_seeder<F, G>(&self, grid: &G) -> Vec<Idx3<usize>>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        self.iter()
            .map(|point| {
                grid.find_closest_grid_cell(&Point3::from(point))
                    .expect_inside_or_moved()
            })
            .collect()
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

    fn to_point_seeder<F, G>(&self, grid: &G) -> Vec<Point3<fgr>>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        self.par_iter()
            .map(|indices| Point3::from(&grid.centers().point(indices)))
            .collect()
    }
}
