//! Generation of seed points in a slice through a field.

use std::vec;
use std::collections::HashSet;
use std::iter::FromIterator;
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::SampleUniform;
use rayon;
use rayon::prelude::*;
use crate::num::BFloat;
use crate::geometry::{Dim3, Dim2, In2D, Vec3, Point3, Point2};
use crate::grid::{Grid3, Grid2, CoordLocation};
use crate::field::{ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use crate::random;
use super::Seeder3;
use super::super::ftr;
use Dim3::{X, Y, Z};

/// Generator for seed points in a slice of a 3D field.
#[derive(Clone)]
pub struct SliceSeeder3 {
    seed_points: Vec<Point3<ftr>>
}

impl SliceSeeder3 {
    /// Creates a new seeder producing regularly spaced seed points in a 2D slice of a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to slice through.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `shape`: Number of seed points to generate in each direction.
    ///
    /// # Returns
    ///
    /// An new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    pub fn regular<F, G>(grid: &G, axis: Dim3, coord: ftr, shape: In2D<usize>) -> Self
    where F: BFloat,
          G: Grid3<F>
    {
        let slice_grid = grid.regular_slice_across_axis(axis).reshaped(shape);
        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        SliceSeeder3{ seed_points: Self::construct_seed_points_from_slice_points(slice_centers, axis, coord) }
    }

    /// Creates a new seeder producing randomly spaced seed points in a 2D slice of a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to slice through.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `n_seeds`: Number of seed points to generate.
    ///
    /// # Returns
    ///
    /// An new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    pub fn random<F, G>(grid: &G, axis: Dim3, coord: ftr, n_seeds: usize) -> Self
    where F: BFloat + SampleUniform,
          G: Grid3<F>
    {
        Self::stratified(grid, axis, coord, In2D::same(1), n_seeds, 1.0)
    }

    /// Creates a new seeder producing stratified seed points in a 2D slice of a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to slice through.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `shape`: Shape of the stratification grid.
    /// - `n_seeds_per_cell`: Number of seed points to generate in each cell of the stratification grid.
    /// - `randomness`: How far from the cell centers the seed points can be generated, going from 0 (cell center) to 1 (cell edge).
    ///
    /// # Returns
    ///
    /// An new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    pub fn stratified<F, G>(grid: &G, axis: Dim3, coord: ftr, shape: In2D<usize>, n_seeds_per_cell: usize, randomness: ftr) -> Self
    where F: BFloat + SampleUniform,
          G: Grid3<F>
    {
        assert_ne!(n_seeds_per_cell, 0, "Number of seeds per cell must be larger than zero.");
        assert!(randomness >= 0.0 && randomness <= 1.0, "Randomness must be in the range [0, 1].");

        if randomness == 0.0 {
            return Self::regular(grid, axis, coord, shape)
        }

        let slice_grid = grid.regular_slice_across_axis(axis).reshaped(shape);
        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        let slice_cell_extents = slice_grid.cell_extents();

        let offset_limit = F::from(0.5*randomness).expect("Conversion failed.");
        let rng = rand::thread_rng();
        let mut uniform_offset_samples = Uniform::new(-offset_limit, offset_limit).sample_iter(rng);

        let mut stratified_points = Vec::with_capacity(slice_centers.len()*n_seeds_per_cell);
        for center in slice_centers {
            for _ in 0..n_seeds_per_cell {
                stratified_points.push(Point2::new(
                    center[Dim2::X] + uniform_offset_samples.next().unwrap()*slice_cell_extents[Dim2::X],
                    center[Dim2::Y] + uniform_offset_samples.next().unwrap()*slice_cell_extents[Dim2::Y]
                ));
            }
        }
        SliceSeeder3{ seed_points: Self::construct_seed_points_from_slice_points(stratified_points, axis, coord) }
    }

    /// Creates a new seeder producing seed points in a 2D slice of a 3D scalar field,
    /// with positions following a probability density function evaluated on the local
    /// field values.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to slice through.
    /// - `interpolator`: Interpolator to use for sampling field values.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `compute_pdf_value`: Closure computing a positive, un-normalized probability density from a field value.
    /// - `n_seeds`: Number of seed points to generate (note that duplicate seed points will be discarded).
    ///
    /// # Returns
    ///
    /// An new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking and returning a floating point value.
    pub fn scalar_field_pdf<F, G, I, C>(field: &ScalarField3<F, G>, interpolator: &I, axis: Dim3, coord: ftr, compute_pdf_value: &C, n_seeds: usize) -> Self
    where F: BFloat + SampleUniform + Sync + Send,
          G: Grid3<F> + Sync + Send,
          I: Interpolator3 + Sync,
          C: Fn(F) -> F
    {
        assert_ne!(n_seeds, 0, "Number of seeds must be larger than zero.");

        let slice_field = field.regular_slice_across_axis(interpolator, axis, F::from(coord).expect("Conversion failed."), CoordLocation::Center);
        let slice_values = slice_field.values();
        let slice_grid = slice_field.grid();
        let slice_shape = slice_grid.shape();

        let mut pdf = Vec::with_capacity(slice_shape[Dim2::X]*slice_shape[Dim2::Y]);
        for j in 0..slice_shape[Dim2::Y] {
            for i in 0..slice_shape[Dim2::X] {
                pdf.push(compute_pdf_value(slice_values[[i, j]]));
            }
        }
        let indices = HashSet::<usize>::from_iter(random::draw_from_distribution(&pdf, n_seeds).into_iter());

        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        let slice_seed_points = indices.into_iter().map(|index| slice_centers[index].clone()).collect();
        SliceSeeder3{ seed_points: Self::construct_seed_points_from_slice_points(slice_seed_points, axis, coord) }
    }

    /// Creates a new seeder producing seed points in a 2D slice of a 3D vector field,
    /// with positions following a probability density function evaluated on the local
    /// field vectors.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to slice through.
    /// - `interpolator`: Interpolator to use for sampling field values.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `compute_pdf_value`: Closure computing a positive, un-normalized probability density from a field vector.
    /// - `n_seeds`: Number of seed points to generate (note that duplicate seed points will be discarded).
    ///
    /// # Returns
    ///
    /// An new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking a reference to a vector and returning a floating point value.
    pub fn vector_field_pdf<F, G, I, C>(field: &VectorField3<F, G>, interpolator: &I, axis: Dim3, coord: ftr, compute_pdf_value: &C, n_seeds: usize) -> Self
    where F: BFloat + SampleUniform + Sync + Send,
          G: Grid3<F> + Sync + Send,
          I: Interpolator3 + Sync,
          C: Fn(&Vec3<F>) -> F
    {
        assert_ne!(n_seeds, 0, "Number of seeds must be larger than zero.");

        let slice_field = field.regular_slice_across_axis(interpolator, axis, F::from(coord).expect("Conversion failed."), CoordLocation::Center);
        let slice_values = slice_field.all_values();
        let slice_grid = slice_field.grid();
        let slice_shape = slice_grid.shape();

        let mut pdf = Vec::with_capacity(slice_shape[Dim2::X]*slice_shape[Dim2::Y]);
        for j in 0..slice_shape[Dim2::Y] {
            for i in 0..slice_shape[Dim2::X] {
                pdf.push(compute_pdf_value(&Vec3::new(
                    slice_values[X][[i, j]],
                    slice_values[Y][[i, j]],
                    slice_values[Z][[i, j]]
                )));
            }
        }
        let indices = HashSet::<usize>::from_iter(random::draw_from_distribution(&pdf, n_seeds).into_iter());

        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        let slice_seed_points = indices.into_iter().map(|index| slice_centers[index].clone()).collect();
        SliceSeeder3{ seed_points: Self::construct_seed_points_from_slice_points(slice_seed_points, axis, coord) }
    }

    fn construct_seed_points_from_slice_points<F>(slice_points: Vec<Point2<F>>, axis: Dim3, coord: ftr) -> Vec<Point3<ftr>>
    where F: BFloat
    {
        match axis {
            X => slice_points.into_iter().map(|point| Point3::from_components(coord, point[Dim2::X], point[Dim2::Y])).collect(),
            Y => slice_points.into_iter().map(|point| Point3::from_components(point[Dim2::X], coord, point[Dim2::Y])).collect(),
            Z => slice_points.into_iter().map(|point| Point3::from_components(point[Dim2::X], point[Dim2::Y], coord)).collect()
        }
    }
}

impl IntoIterator for SliceSeeder3 {
    type Item = Point3<ftr>;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.seed_points.into_iter()
    }
}

impl IntoParallelIterator for SliceSeeder3 {
    type Item = Point3<ftr>;
    type Iter = rayon::vec::IntoIter<Self::Item>;
    fn into_par_iter(self) -> Self::Iter {
        self.seed_points.into_par_iter()
    }
}

impl Seeder3 for SliceSeeder3 {}