//! Generation of seed points for field line tracing.

pub mod criterion;
pub mod manual;
pub mod slice;
pub mod volume;

use crate::{
    field::FieldGrid3,
    geometry::{Idx3, Point3},
    grid::{fgr, Grid3},
};
use rayon::prelude::*;

pub type DynSeeder3 = Box<dyn Seeder3>;
pub type DynIndexSeeder3 = Box<dyn IndexSeeder3>;

/// Defines the properties of a 3D seed point generator.
pub trait Seeder3 {
    /// Returns the number of seed points that will be produced by the seeder.
    fn number_of_points(&self) -> usize;

    /// Returns a slice with all seed points.
    fn points(&self) -> &[Point3<fgr>];

    /// Creates a list of seed indices from the seed points by looking up the grid cells
    /// of the given grid containing the seed points.
    fn to_index_seeder(&self, grid: &FieldGrid3) -> Vec<Idx3<usize>>;
}

/// Defines the properties of a 3D seed index generator.
pub trait IndexSeeder3 {
    /// Returns the number of seed indices that will be produced by the seeder.
    fn number_of_indices(&self) -> usize;

    /// Returns a slice with all seed indices.
    fn indices(&self) -> &[Idx3<usize>];

    /// Creates a list of seed points from the seed indices by indexing the center coordinates
    /// of the given grid.
    fn to_point_seeder(&self, grid: &FieldGrid3) -> Vec<Point3<fgr>>;
}

// Let a vector of points work as a seeder.
impl Seeder3 for Vec<Point3<fgr>> {
    fn number_of_points(&self) -> usize {
        self.len()
    }

    fn points(&self) -> &[Point3<fgr>] {
        self
    }

    fn to_index_seeder(&self, grid: &FieldGrid3) -> Vec<Idx3<usize>> {
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

    fn indices(&self) -> &[Idx3<usize>] {
        self
    }

    fn to_point_seeder(&self, grid: &FieldGrid3) -> Vec<Point3<fgr>> {
        self.par_iter()
            .map(|indices| Point3::from(&grid.centers().point(indices)))
            .collect()
    }
}
