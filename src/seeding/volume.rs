//! Generation of seed points in a volume of a 3D field.

use super::Seeder3;
use crate::{
    field::{FieldGrid3, ScalarField3, VectorField3},
    geometry::{
        Dim3::{X, Y, Z},
        Idx3, In3D, Point3, Vec3,
    },
    grid::{fgr, regular::RegularGrid3, CoordLocation, Grid3},
    interpolation::Interpolator3,
    num::BFloat,
    random,
};
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rayon::{self, prelude::*};
use std::{collections::HashSet, iter::FromIterator};

/// Generator for seed points in a volume of a 3D field.
#[derive(Clone, Debug)]
pub struct VolumeSeeder3 {
    seed_points: Vec<Point3<fgr>>,
}

impl VolumeSeeder3 {
    /// Creates a new seeder producing regularly spaced seed points in a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `grid`: Regular grid defining the location of the seed points.
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `VolumeSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn regular<S>(grid: &RegularGrid3<fgr>, satisfies_constraints: &S) -> Self
    where
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        let centers = grid.create_point_list(CoordLocation::Center);
        Self {
            seed_points: Self::apply_constraints(centers, satisfies_constraints),
        }
    }

    /// Creates a new seeder producing randomly spaced seed points in a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `lower_bounds`: Vector specifying the lower bounds of the volume to place seed points in.
    /// - `upper_bounds`: Vector specifying the upper bounds of the volume to place seed points in.
    /// - `n_seeds`: Number of seed points to generate.
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `VolumeSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn random<S>(
        lower_bounds: &Vec3<fgr>,
        upper_bounds: &Vec3<fgr>,
        n_seeds: usize,
        satisfies_constraints: &S,
    ) -> Self
    where
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        let grid = RegularGrid3::from_bounds(
            In3D::same(1),
            lower_bounds.clone(),
            upper_bounds.clone(),
            In3D::same(false),
        );
        Self::stratified(&grid, n_seeds, 1.0, satisfies_constraints)
    }

    /// Creates a new seeder producing stratified seed points in a 3D grid.
    ///
    /// # Parameters
    ///
    /// - `grid`: Regular grid defining the cells in which to generate seed points.
    /// - `n_seeds_per_cell`: Number of seed points to generate in each cell of the stratification grid.
    /// - `randomness`: How far from the cell centers the seed points can be generated, going from 0 (cell center) to 1 (cell edge).
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `VolumeSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `C`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn stratified<S>(
        grid: &RegularGrid3<fgr>,
        n_seeds_per_cell: usize,
        randomness: fgr,
        satisfies_constraints: &S,
    ) -> Self
    where
        S: Fn(&Point3<fgr>) -> bool + Sync,
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
            return Self::regular(grid, satisfies_constraints);
        }

        let centers = grid.create_point_list(CoordLocation::Center);
        let cell_extents = grid.cell_extents();

        let offset_limit = 0.5 * randomness;
        let rng = rand::thread_rng();
        let mut uniform_offset_samples = Uniform::new(-offset_limit, offset_limit).sample_iter(rng);

        let mut stratified_points = Vec::with_capacity(centers.len() * n_seeds_per_cell);
        for center in centers {
            for _ in 0..n_seeds_per_cell {
                stratified_points.push(Point3::new(
                    center[X] + uniform_offset_samples.next().unwrap() * cell_extents[X],
                    center[Y] + uniform_offset_samples.next().unwrap() * cell_extents[Y],
                    center[Z] + uniform_offset_samples.next().unwrap() * cell_extents[Z],
                ));
            }
        }
        Self {
            seed_points: Self::apply_constraints(stratified_points, satisfies_constraints),
        }
    }

    /// Creates a new seeder producing seed points in a volume of a 3D scalar field,
    /// with positions following a probability density function evaluated on the local
    /// field values.
    ///
    /// # Parameters
    ///
    /// - `grid`: Regular grid defining the positions for which to evaluate the PDF.
    /// - `field`: Scalar field to use.
    /// - `interpolator`: Interpolator to use for sampling field values.
    /// - `compute_pdf_value`: Closure computing a positive, un-normalized probability density from a field value.
    /// - `n_seeds`: Number of seed points to generate (note that duplicate seed points will be discarded).
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `VolumeSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `C`: Function type taking and returning a floating point value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn scalar_field_pdf<F, C, S>(
        grid: &RegularGrid3<fgr>,
        field: &ScalarField3<F>,
        interpolator: &dyn Interpolator3<F>,
        compute_pdf_value: &C,
        n_seeds: usize,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat + SampleUniform,
        C: Fn(F) -> F + Sync,
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        assert_ne!(n_seeds, 0, "Number of seeds must be larger than zero.");

        let centers = grid.create_point_list(CoordLocation::Center);

        let pdf: Vec<F> = centers
            .par_iter()
            .map(|point| {
                compute_pdf_value(
                    F::from(
                        interpolator
                            .interp_extrap_scalar_field(field, point)
                            .expect_inside_or_moved(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let indices =
            HashSet::<usize>::from_iter(random::draw_from_distribution(&pdf, n_seeds).into_iter());

        let seed_points = indices
            .into_iter()
            .map(|index| centers[index].clone())
            .collect();
        Self {
            seed_points: Self::apply_constraints(seed_points, satisfies_constraints),
        }
    }

    /// Creates a new seeder producing seed points in a volume of a 3D vector field,
    /// with positions following a probability density function evaluated on the local
    /// field vectors.
    ///
    /// # Parameters
    ///
    /// - `grid`: Regular grid defining the positions for which to evaluate the PDF.
    /// - `field`: Vector field to use.
    /// - `interpolator`: Interpolator to use for sampling field values.
    /// - `compute_pdf_value`: Closure computing a positive, un-normalized probability density from a field vector.
    /// - `n_seeds`: Number of seed points to generate (note that duplicate seed points will be discarded).
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `VolumeSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `C`: Function type taking a reference to a vector and returning a floating point value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn vector_field_pdf<F, C, S>(
        grid: &RegularGrid3<fgr>,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        compute_pdf_value: &C,
        n_seeds: usize,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat + SampleUniform,
        C: Fn(&Vec3<F>) -> F + Sync,
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        assert_ne!(n_seeds, 0, "Number of seeds must be larger than zero.");

        let centers = grid.create_point_list(CoordLocation::Center);

        let pdf: Vec<F> = centers
            .par_iter()
            .map(|point| {
                compute_pdf_value(
                    &interpolator
                        .interp_extrap_vector_field(field, point)
                        .expect_inside_or_moved()
                        .cast(),
                )
            })
            .collect();
        let indices =
            HashSet::<usize>::from_iter(random::draw_from_distribution(&pdf, n_seeds).into_iter());

        let seed_points = indices
            .into_iter()
            .map(|index| centers[index].clone())
            .collect();
        Self {
            seed_points: Self::apply_constraints(seed_points, satisfies_constraints),
        }
    }

    fn apply_constraints<S>(points: Vec<Point3<fgr>>, satisfies_constraints: &S) -> Vec<Point3<fgr>>
    where
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        points
            .into_par_iter()
            .filter_map(|point| {
                if satisfies_constraints(&point) {
                    Some(Point3::from(&point))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Seeder3 for VolumeSeeder3 {
    fn number_of_points(&self) -> usize {
        self.seed_points.len()
    }

    fn points(&self) -> &[Point3<fgr>] {
        &self.seed_points
    }

    fn to_index_seeder(&self, grid: &FieldGrid3) -> Vec<Idx3<usize>> {
        self.seed_points.to_index_seeder(grid)
    }
}
