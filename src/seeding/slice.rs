//! Generation of seed points in a slice through a field.

use super::Seeder3;
use crate::{
    field::{FieldGrid3, ScalarField3, VectorField3},
    geometry::{
        Dim2,
        Dim3::{self, X, Y, Z},
        Idx3, In2D, Point2, Point3, Vec3,
    },
    grid::{fgr, CoordLocation, Grid2, Grid3},
    interpolation::Interpolator3,
    num::BFloat,
    random,
};
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rayon::{self, prelude::*};
use std::{collections::HashSet, iter::FromIterator, vec};

/// Generator for seed points in a slice of a 3D field.
#[derive(Clone, Debug)]
pub struct SliceSeeder3 {
    seed_points: Vec<Point3<fgr>>,
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
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `S`: Function type taking a reference to a 2D point and returning a boolean value.
    pub fn regular<S>(
        grid: &FieldGrid3,
        axis: Dim3,
        coord: fgr,
        shape: In2D<usize>,
        satisfies_constraints: &S,
    ) -> Self
    where
        S: Fn(&Point2<fgr>) -> bool + Sync,
    {
        let slice_grid = grid.regular_slice_across_axis(axis).reshaped(shape);
        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        SliceSeeder3 {
            seed_points: Self::construct_seed_points_from_slice_points(
                slice_centers,
                axis,
                coord,
                satisfies_constraints,
            ),
        }
    }

    /// Creates a new seeder producing randomly spaced seed points in a 2D slice of a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to slice through.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `n_seeds`: Number of seed points to generate.
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `S`: Function type taking a reference to a 2D point and returning a boolean value.
    pub fn random<S>(
        grid: &FieldGrid3,
        axis: Dim3,
        coord: fgr,
        n_seeds: usize,
        satisfies_constraints: &S,
    ) -> Self
    where
        S: Fn(&Point2<fgr>) -> bool + Sync,
    {
        Self::stratified(
            grid,
            axis,
            coord,
            In2D::same(1),
            n_seeds,
            1.0,
            satisfies_constraints,
        )
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
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `C`: Function type taking a reference to a 2D point and returning a boolean value.
    pub fn stratified<S>(
        grid: &FieldGrid3,
        axis: Dim3,
        coord: fgr,
        shape: In2D<usize>,
        n_seeds_per_cell: usize,
        randomness: fgr,
        satisfies_constraints: &S,
    ) -> Self
    where
        S: Fn(&Point2<fgr>) -> bool + Sync,
    {
        assert_ne!(
            n_seeds_per_cell, 0,
            "Number of seeds per cell must be larger than zero."
        );
        assert!(
            (0.0..=1.0).contains(&randomness),
            "Randomness must be in the range [0, 1]."
        );

        if randomness == 0.0 {
            return Self::regular(grid, axis, coord, shape, satisfies_constraints);
        }

        let slice_grid = grid.regular_slice_across_axis(axis).reshaped(shape);
        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        let slice_cell_extents = slice_grid.cell_extents();

        let offset_limit = 0.5 * randomness;
        let rng = rand::thread_rng();
        let mut uniform_offset_samples = Uniform::new(-offset_limit, offset_limit).sample_iter(rng);

        let mut stratified_points = Vec::with_capacity(slice_centers.len() * n_seeds_per_cell);
        for center in slice_centers {
            for _ in 0..n_seeds_per_cell {
                stratified_points.push(Point2::new(
                    center[Dim2::X]
                        + uniform_offset_samples.next().unwrap() * slice_cell_extents[Dim2::X],
                    center[Dim2::Y]
                        + uniform_offset_samples.next().unwrap() * slice_cell_extents[Dim2::Y],
                ));
            }
        }
        SliceSeeder3 {
            seed_points: Self::construct_seed_points_from_slice_points(
                stratified_points,
                axis,
                coord,
                satisfies_constraints,
            ),
        }
    }

    /// Creates a new seeder producing seed points in a 2D slice of a 3D scalar field,
    /// with positions following a probability density function evaluated on the local
    /// field values.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to slice through.
    /// - `interpolator`: Interpolator to use for sampling field values.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `compute_pdf_value`: Closure computing a positive, un-normalized probability density from a field value.
    /// - `n_seeds`: Number of seed points to generate (note that duplicate seed points will be discarded).
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking and returning a floating point value.
    /// - `S`: Function type taking a reference to a 2D point and returning a boolean value.
    pub fn scalar_field_pdf<F, I, C, S>(
        field: &ScalarField3<F>,
        interpolator: &I,
        axis: Dim3,
        coord: fgr,
        compute_pdf_value: &C,
        n_seeds: usize,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat + SampleUniform,
        I: Interpolator3,
        C: Fn(F) -> F,
        S: Fn(&Point2<fgr>) -> bool + Sync,
    {
        assert_ne!(n_seeds, 0, "Number of seeds must be larger than zero.");

        let slice_field =
            field.regular_slice_across_axis(interpolator, axis, coord, CoordLocation::Center);
        let slice_values = slice_field.values();
        let slice_grid = slice_field.grid();
        let slice_shape = slice_grid.shape();

        let mut pdf = Vec::with_capacity(slice_shape[Dim2::X] * slice_shape[Dim2::Y]);
        for j in 0..slice_shape[Dim2::Y] {
            for i in 0..slice_shape[Dim2::X] {
                pdf.push(compute_pdf_value(slice_values[[i, j]]));
            }
        }
        let indices =
            HashSet::<usize>::from_iter(random::draw_from_distribution(&pdf, n_seeds).into_iter());

        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        let slice_seed_points = indices
            .into_iter()
            .map(|index| slice_centers[index].clone())
            .collect();
        SliceSeeder3 {
            seed_points: Self::construct_seed_points_from_slice_points(
                slice_seed_points,
                axis,
                coord,
                satisfies_constraints,
            ),
        }
    }

    /// Creates a new seeder producing seed points in a 2D slice of a 3D vector field,
    /// with positions following a probability density function evaluated on the local
    /// field vectors.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to slice through.
    /// - `interpolator`: Interpolator to use for sampling field values.
    /// - `axis`: Axis to slice across.
    /// - `coord`: Coordinate of the slice along `axis`.
    /// - `compute_pdf_value`: Closure computing a positive, un-normalized probability density from a field vector.
    /// - `n_seeds`: Number of seed points to generate (note that duplicate seed points will be discarded).
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `SliceSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking a reference to a vector and returning a floating point value.
    /// - `S`: Function type taking a reference to a 2D point and returning a boolean value.
    pub fn vector_field_pdf<F, I, C, S>(
        field: &VectorField3<F>,
        interpolator: &I,
        axis: Dim3,
        coord: fgr,
        compute_pdf_value: &C,
        n_seeds: usize,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat + SampleUniform,
        I: Interpolator3,
        C: Fn(&Vec3<F>) -> F,
        S: Fn(&Point2<fgr>) -> bool + Sync,
    {
        assert_ne!(n_seeds, 0, "Number of seeds must be larger than zero.");

        let slice_field =
            field.regular_slice_across_axis(interpolator, axis, coord, CoordLocation::Center);
        let slice_values = slice_field.all_values();
        let slice_grid = slice_field.grid();
        let slice_shape = slice_grid.shape();

        let mut pdf = Vec::with_capacity(slice_shape[Dim2::X] * slice_shape[Dim2::Y]);
        for j in 0..slice_shape[Dim2::Y] {
            for i in 0..slice_shape[Dim2::X] {
                pdf.push(compute_pdf_value(&Vec3::with_each_component(|dim| {
                    slice_values[dim][[i, j]]
                })));
            }
        }
        let indices =
            HashSet::<usize>::from_iter(random::draw_from_distribution(&pdf, n_seeds).into_iter());

        let slice_centers = slice_grid.create_point_list(CoordLocation::Center);
        let slice_seed_points = indices
            .into_iter()
            .map(|index| slice_centers[index].clone())
            .collect();
        SliceSeeder3 {
            seed_points: Self::construct_seed_points_from_slice_points(
                slice_seed_points,
                axis,
                coord,
                satisfies_constraints,
            ),
        }
    }

    fn construct_seed_points_from_slice_points<S>(
        slice_points: Vec<Point2<fgr>>,
        axis: Dim3,
        coord: fgr,
        satisfies_constraints: &S,
    ) -> Vec<Point3<fgr>>
    where
        S: Fn(&Point2<fgr>) -> bool + Sync,
    {
        match axis {
            X => slice_points
                .into_par_iter()
                .filter_map(|point| {
                    if satisfies_constraints(&point) {
                        Some(Point3::new(coord, point[Dim2::X], point[Dim2::Y]))
                    } else {
                        None
                    }
                })
                .collect(),
            Y => slice_points
                .into_par_iter()
                .filter_map(|point| {
                    if satisfies_constraints(&point) {
                        Some(Point3::new(point[Dim2::X], coord, point[Dim2::Y]))
                    } else {
                        None
                    }
                })
                .collect(),
            Z => slice_points
                .into_par_iter()
                .filter_map(|point| {
                    if satisfies_constraints(&point) {
                        Some(Point3::new(point[Dim2::X], point[Dim2::Y], coord))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }
}

impl IntoIterator for SliceSeeder3 {
    type Item = Point3<fgr>;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.seed_points.into_iter()
    }
}

impl IntoParallelIterator for SliceSeeder3 {
    type Item = Point3<fgr>;
    type Iter = rayon::vec::IntoIter<Self::Item>;
    fn into_par_iter(self) -> Self::Iter {
        self.seed_points.into_par_iter()
    }
}

impl Seeder3 for SliceSeeder3 {
    fn number_of_points(&self) -> usize {
        self.seed_points.len()
    }

    fn retain_points<P>(&mut self, predicate: P)
    where
        P: FnMut(&Point3<fgr>) -> bool,
    {
        self.seed_points.retain(predicate);
    }

    fn to_index_seeder(&self, grid: &FieldGrid3) -> Vec<Idx3<usize>> {
        self.seed_points.to_index_seeder(grid)
    }
}
