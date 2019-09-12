//! Generation of seed points by evaluating a criterion on field values.

use std::vec;
use rayon::prelude::*;
use crate::num::BFloat;
use crate::geometry::{Dim3, Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{self, ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use super::Seeder3;
use super::super::ftr;
use Dim3::{X, Y, Z};

/// Generator for seed points found by evaluating a criterion on values of a 3D field.
#[derive(Clone, Debug)]
pub struct CriterionSeeder3 {
    seed_points: Vec<Point3<ftr>>
}

impl CriterionSeeder3 {
    /// Creates a new seeder producing seed points at positions where
    /// a given criterion on the local scalar field value is satisfied.
    ///
    /// Uses the original locations of the field values.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field whose values to evaluate.
    /// - `evaluate_criterion`: Closure returning true for a field value whose position should be a seed point.
    ///
    /// # Returns
    ///
    /// An new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `C`: Function type taking a floating point value and returning a boolean value.
    pub fn on_scalar_field_values<F, G, C>(field: &ScalarField3<F, G>, evaluate_criterion: &C) -> Self
    where F: BFloat,
          G: Grid3<F>,
          C: Fn(F) -> bool + Sync
    {
        let shape = field.shape();
        let coords = field.coords();
        let values_slice = field.values().as_slice_memory_order().expect("Values array not contiguous.");

        let seed_points = values_slice.par_iter().enumerate().filter_map(
            |(idx, &value)| {
                if evaluate_criterion(value) {
                    let indices = field::compute_3d_array_indices_from_flat_idx(&shape, idx);
                    Some(Point3::from(&coords.point(&indices)))
                } else {
                    None
                }
            }
        ).collect();

        CriterionSeeder3{ seed_points }
    }

    /// Creates a new seeder producing seed points at positions where
    /// a given criterion on the local scalar field value is satisfied.
    ///
    /// Interpolates the field values to the grid cell centers before evaluating the criterion.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field whose values to evaluate.
    /// - `interpolator`: Interpolator to use.
    /// - `evaluate_criterion`: Closure returning true for a field value whose position should be a seed point.
    ///
    /// # Returns
    ///
    /// An new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking a floating point value and returning a boolean value.
    pub fn on_centered_scalar_field_values<F, G, I, C>(field: &ScalarField3<F, G>, interpolator: &I, evaluate_criterion: &C) -> Self
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3,
          C: Fn(F) -> bool + Sync
    {
        let shape = field.shape();
        let center_coords = field.grid().centers();

        let seed_points = (0..shape[X]*shape[Y]*shape[Z]).into_par_iter().filter_map(
            |idx| {
                let indices = field::compute_3d_array_indices_from_flat_idx(&shape, idx);
                let point = center_coords.point(&indices);
                let value = interpolator.interp_scalar_field(field, &point).expect_inside();
                if evaluate_criterion(value) {
                    Some(Point3::from(&point))
                } else {
                    None
                }
            }
        ).collect();

        CriterionSeeder3{ seed_points }
    }

    /// Creates a new seeder producing seed points at positions where
    /// a given criterion on the local vector field value is satisfied.
    ///
    /// Interpolates the field vectors to the grid cell centers before evaluating the criterion.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field whose values to evaluate.
    /// - `interpolator`: Interpolator to use.
    /// - `evaluate_criterion`: Closure returning true for a field vector whose position should be a seed point.
    ///
    /// # Returns
    ///
    /// An new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking a reference to a vector and returning a boolean value.
    pub fn on_centered_vector_field_values<F, G, I, C>(field: &VectorField3<F, G>, interpolator: &I, evaluate_criterion: &C) -> Self
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3,
          C: Fn(&Vec3<F>) -> bool + Sync
    {
        let shape = field.shape();
        let center_coords = field.grid().centers();

        let seed_points = (0..shape[X]*shape[Y]*shape[Z]).into_par_iter().filter_map(
            |idx| {
                let indices = field::compute_3d_array_indices_from_flat_idx(&shape, idx);
                let point = center_coords.point(&indices);
                let vector = interpolator.interp_vector_field(field, &point).expect_inside();
                if evaluate_criterion(&vector) {
                    Some(Point3::from(&point))
                } else {
                    None
                }
            }
        ).collect();

        CriterionSeeder3{ seed_points }
    }
}

impl IntoIterator for CriterionSeeder3 {
    type Item = Point3<ftr>;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.seed_points.into_iter()
    }
}

impl IntoParallelIterator for CriterionSeeder3 {
    type Item = Point3<ftr>;
    type Iter = rayon::vec::IntoIter<Self::Item>;
    fn into_par_iter(self) -> Self::Iter {
        self.seed_points.into_par_iter()
    }
}

impl Seeder3 for CriterionSeeder3 {}