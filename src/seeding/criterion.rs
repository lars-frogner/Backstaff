//! Generation of seed indices by evaluating a criterion on field values.

use super::IndexSeeder3;
use crate::{
    field::{self, FieldGrid3, ScalarField3, VectorField3},
    geometry::{
        Dim3::{X, Y, Z},
        Idx3, Point3, Vec3,
    },
    grid::{fgr, Grid3},
    interpolation::Interpolator3,
    num::BFloat,
};
use rayon::prelude::*;

/// Generator for seed indices found by evaluating a criterion on values of a 3D field.
#[derive(Clone, Debug)]
pub struct CriterionSeeder3 {
    seed_indices: Vec<Idx3<usize>>,
}

impl CriterionSeeder3 {
    /// Creates a new seeder producing indices corresponding to positions where
    /// a given criterion on the local scalar field value is satisfied.
    ///
    /// Uses the original locations of the field values.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field whose values to evaluate.
    /// - `evaluate_criterion`: Closure returning true for a field value whose indices should be a seed.
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `C`: Function type taking a floating point value and returning a boolean value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn on_scalar_field_values<F, C, S>(
        field: &ScalarField3<F>,
        evaluate_criterion: &C,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat,
        C: Fn(F) -> bool + Sync,
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        let shape = field.shape();
        let values_slice = field
            .values()
            .as_slice_memory_order()
            .expect("Values array not contiguous");

        let seed_indices = values_slice
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                let indices = field::compute_3d_array_indices_from_flat_idx(shape, idx);
                let point = field.coords().point(&indices);
                if evaluate_criterion(value) && satisfies_constraints(&point) {
                    Some(indices)
                } else {
                    None
                }
            })
            .collect();

        CriterionSeeder3 { seed_indices }
    }

    /// Creates a new seeder producing indices corresponding to positions where
    /// a given criterion on the local scalar field value is satisfied.
    ///
    /// Interpolates the field values to the grid cell centers before evaluating the criterion.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field whose values to evaluate.
    /// - `interpolator`: Interpolator to use.
    /// - `evaluate_criterion`: Closure returning true for a field value whose indices should be a seed.
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `C`: Function type taking a floating point value and returning a boolean value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn on_centered_scalar_field_values<F, C, S>(
        field: &ScalarField3<F>,
        interpolator: &dyn Interpolator3<F>,
        evaluate_criterion: &C,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat,
        C: Fn(F) -> bool + Sync,
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        let shape = field.shape();
        let center_coords = field.grid().centers();

        let seed_indices = (0..shape[X] * shape[Y] * shape[Z])
            .into_par_iter()
            .filter_map(|idx| {
                let indices = field::compute_3d_array_indices_from_flat_idx(shape, idx);
                let point = center_coords.point(&indices);
                if !satisfies_constraints(&point) {
                    return None;
                }
                let value = interpolator
                    .interp_scalar_field(field, &point)
                    .expect_inside();
                if evaluate_criterion(F::from(value).unwrap()) {
                    Some(indices)
                } else {
                    None
                }
            })
            .collect();

        CriterionSeeder3 { seed_indices }
    }

    /// Creates a new seeder producing indices corresponding to positions where
    /// a given criterion on the local vector field value is satisfied.
    ///
    /// Interpolates the field vectors to the grid cell centers before evaluating the criterion.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field whose values to evaluate.
    /// - `interpolator`: Interpolator to use.
    /// - `evaluate_criterion`: Closure returning true for a field vector whose indices should be a seed.
    /// - `satisfies_constraints`: Closure taking a potential seed point and returning whether the point is accepted.
    ///
    /// # Returns
    ///
    /// A new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `C`: Function type taking a reference to a vector and returning a boolean value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn on_centered_vector_field_values<F, C, S>(
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        evaluate_criterion: &C,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat,
        C: Fn(&Vec3<F>) -> bool + Sync,
        S: Fn(&Point3<fgr>) -> bool + Sync,
    {
        let shape = field.shape();
        let center_coords = field.grid().centers();

        let seed_indices = (0..shape[X] * shape[Y] * shape[Z])
            .into_par_iter()
            .filter_map(|idx| {
                let indices = field::compute_3d_array_indices_from_flat_idx(shape, idx);
                let point = center_coords.point(&indices);
                if !satisfies_constraints(&point) {
                    return None;
                }
                let vector = interpolator
                    .interp_vector_field(field, &point)
                    .expect_inside();
                if evaluate_criterion(&vector.cast()) {
                    Some(indices)
                } else {
                    None
                }
            })
            .collect();

        CriterionSeeder3 { seed_indices }
    }
}

impl IndexSeeder3 for CriterionSeeder3 {
    fn number_of_indices(&self) -> usize {
        self.seed_indices.len()
    }

    fn indices(&self) -> &[Idx3<usize>] {
        &self.seed_indices
    }

    fn to_point_seeder(&self, grid: &FieldGrid3) -> Vec<Point3<fgr>> {
        self.seed_indices
            .par_iter()
            .map(|indices| Point3::from(&grid.centers().point(indices)))
            .collect()
    }
}
