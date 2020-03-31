//! Generation of seed indices by evaluating a criterion on field values.

use super::{super::ftr, IndexSeeder3};
use crate::{
    field::{self, ScalarField3, VectorField3},
    geometry::{
        Dim3::{X, Y, Z},
        Idx3, Point3, Vec3,
    },
    grid::Grid3,
    interpolation::Interpolator3,
    num::BFloat,
};
use rayon::prelude::*;
use std::vec;

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
    /// An new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `C`: Function type taking a floating point value and returning a boolean value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn on_scalar_field_values<F, G, C, S>(
        field: &ScalarField3<F, G>,
        evaluate_criterion: &C,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat,
        G: Grid3<F>,
        C: Fn(F) -> bool + Sync,
        S: Fn(&Point3<F>) -> bool + Sync,
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
                let indices = field::compute_3d_array_indices_from_flat_idx(&shape, idx);
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
    /// An new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking a floating point value and returning a boolean value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn on_centered_scalar_field_values<F, G, I, C, S>(
        field: &ScalarField3<F, G>,
        interpolator: &I,
        evaluate_criterion: &C,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        C: Fn(F) -> bool + Sync,
        S: Fn(&Point3<F>) -> bool + Sync,
    {
        let shape = field.shape();
        let center_coords = field.grid().centers();

        let seed_indices = (0..shape[X] * shape[Y] * shape[Z])
            .into_par_iter()
            .filter_map(|idx| {
                let indices = field::compute_3d_array_indices_from_flat_idx(&shape, idx);
                let point = center_coords.point(&indices);
                if !satisfies_constraints(&point) {
                    return None;
                }
                let value = interpolator
                    .interp_scalar_field(field, &point)
                    .expect_inside();
                if evaluate_criterion(value) {
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
    /// An new `CriterionSeeder3`.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `C`: Function type taking a reference to a vector and returning a boolean value.
    /// - `S`: Function type taking a reference to a 3D point and returning a boolean value.
    pub fn on_centered_vector_field_values<F, G, I, C, S>(
        field: &VectorField3<F, G>,
        interpolator: &I,
        evaluate_criterion: &C,
        satisfies_constraints: &S,
    ) -> Self
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        C: Fn(&Vec3<F>) -> bool + Sync,
        S: Fn(&Point3<F>) -> bool + Sync,
    {
        let shape = field.shape();
        let center_coords = field.grid().centers();

        let seed_indices = (0..shape[X] * shape[Y] * shape[Z])
            .into_par_iter()
            .filter_map(|idx| {
                let indices = field::compute_3d_array_indices_from_flat_idx(&shape, idx);
                let point = center_coords.point(&indices);
                if !satisfies_constraints(&point) {
                    return None;
                }
                let vector = interpolator
                    .interp_vector_field(field, &point)
                    .expect_inside();
                if evaluate_criterion(&vector) {
                    Some(indices)
                } else {
                    None
                }
            })
            .collect();

        CriterionSeeder3 { seed_indices }
    }
}

impl IntoIterator for CriterionSeeder3 {
    type Item = Idx3<usize>;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.seed_indices.into_iter()
    }
}

impl IntoParallelIterator for CriterionSeeder3 {
    type Item = Idx3<usize>;
    type Iter = rayon::vec::IntoIter<Self::Item>;
    fn into_par_iter(self) -> Self::Iter {
        self.seed_indices.into_par_iter()
    }
}

impl IndexSeeder3 for CriterionSeeder3 {
    fn number_of_indices(&self) -> usize {
        self.seed_indices.len()
    }

    fn retain_indices<P>(&mut self, predicate: P)
    where
        P: FnMut(&Idx3<usize>) -> bool,
    {
        self.seed_indices.retain(predicate);
    }

    fn to_point_seeder<F, G>(&self, grid: &G) -> Vec<Point3<ftr>>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        self.seed_indices
            .par_iter()
            .map(|indices| Point3::from(&grid.centers().point(indices)))
            .collect()
    }
}
