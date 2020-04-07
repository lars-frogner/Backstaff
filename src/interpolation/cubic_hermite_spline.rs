//! Interpolation with a cubic Hermite spline.

use super::Interpolator1;
use crate::{
    field::ScalarField1,
    grid::{CoordLocation, Grid1, GridPointQuery1},
    num::BFloat,
};

/// Scheme for computing tangents at interpolation points.
#[derive(Clone, Copy, Debug)]
pub enum TangentScheme {
    FiniteDifference,
    ProximityWeightedFiniteDifference,
}

/// How tangents should be chosen at the boundaries.
#[derive(Clone, Copy, Debug)]
pub enum BoundaryTangents {
    Computed,
    Zero,
}

/// Configuration parameters for cubic Hermite spline interpolators.
#[derive(Clone, Debug)]
pub struct CubicHermiteSplineInterpolatorConfig {
    /// Scheme for computing tangents at interpolation points.
    pub tangent_scheme: TangentScheme,
    /// How tangents should be chosen at the boundaries.
    pub boundary_tangents: BoundaryTangents,
}

/// A 1D interpolator using a cubic Hermite spline to estimate the interpolated value.
#[derive(Clone, Debug)]
pub struct CubicHermiteSplineInterpolator {
    config: CubicHermiteSplineInterpolatorConfig,
}

impl CubicHermiteSplineInterpolator {
    /// Creates a new cubic Hermite spline interpolator.
    pub fn new(config: CubicHermiteSplineInterpolatorConfig) -> Self {
        Self { config }
    }

    fn hermite_basis_00<F: BFloat>(t: F) -> F {
        (F::one() + F::from_f32(2.0).unwrap() * t) * F::powi(F::one() - t, 2)
    }

    fn hermite_basis_10<F: BFloat>(t: F) -> F {
        t * F::powi(F::one() - t, 2)
    }

    fn hermite_basis_01<F: BFloat>(t: F) -> F {
        t * t * (F::from_f32(3.0).unwrap() - F::from_f32(2.0).unwrap() * t)
    }

    fn hermite_basis_11<F: BFloat>(t: F) -> F {
        t * t * (t - F::one())
    }

    fn compute_tangent<F, G>(&self, field: &ScalarField1<F, G>, index: usize) -> F
    where
        F: BFloat,
        G: Grid1<F>,
    {
        let size = field.size();
        let coords = field.coords();
        let values = field.values();

        if index == 0 {
            match self.config.boundary_tangents {
                BoundaryTangents::Computed => (values[1] - values[0]) / (coords[1] - coords[0]),
                BoundaryTangents::Zero => F::zero(),
            }
        } else if index == size - 1 {
            match self.config.boundary_tangents {
                BoundaryTangents::Computed => {
                    (values[size - 1] - values[size - 2]) / (coords[size - 1] - coords[size - 2])
                }
                BoundaryTangents::Zero => F::zero(),
            }
        } else {
            let lower_distance = coords[index] - coords[index - 1];
            let upper_distance = coords[index + 1] - coords[index];
            let (lower_weight, upper_weight) = match self.config.tangent_scheme {
                TangentScheme::FiniteDifference => {
                    (F::from_f32(0.5).unwrap(), F::from_f32(0.5).unwrap())
                }
                TangentScheme::ProximityWeightedFiniteDifference => {
                    (F::one() / lower_distance, F::one() / upper_distance)
                }
            };
            (lower_weight * (values[index] - values[index - 1]) / lower_distance
                + upper_weight * (values[index + 1] - values[index]) / upper_distance)
                / (lower_weight + upper_weight)
        }
    }

    fn interp<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
        mut interp_index: usize,
    ) -> F
    where
        F: BFloat,
        G: Grid1<F>,
    {
        assert!(field.size() >= 2);
        assert!(interp_index < field.size());

        if field.location() == CoordLocation::Center
            && interp_coord < field.grid().centers()[interp_index]
        {
            // If values are located at cell centers and interpolation coordinate
            // is in lower half of the cell:
            // Shift start offset one cell down.
            interp_index -= 1
        }

        if interp_index == field.size() - 1 {
            // We have to extrapolate within the last grid cell
            interp_index -= 1;
        }

        let coords = field.coords();
        let start_coord = coords[interp_index];
        let end_coord = coords[interp_index + 1];
        let span = end_coord - start_coord;

        let values = field.values();
        let start_value = values[interp_index];
        let end_value = values[interp_index + 1];

        let start_tangent = self.compute_tangent(field, interp_index);
        let end_tangent = self.compute_tangent(field, interp_index + 1);

        let t = (interp_coord - start_coord) / span;

        Self::hermite_basis_00(t) * start_value
            + Self::hermite_basis_01(t) * end_value
            + Self::hermite_basis_10(t) * span * start_tangent
            + Self::hermite_basis_11(t) * span * end_tangent
    }
}

impl Interpolator1 for CubicHermiteSplineInterpolator {
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
    ) -> GridPointQuery1<F, F>
    where
        F: BFloat,
        G: Grid1<F>,
    {
        let grid_point_query = field.grid().find_grid_cell(interp_coord);
        match grid_point_query {
            GridPointQuery1::Inside(interp_index) => {
                GridPointQuery1::Inside(self.interp(field, interp_coord, interp_index))
            }
            GridPointQuery1::MovedInside((interp_index, moved_coord)) => {
                GridPointQuery1::MovedInside((
                    self.interp(field, interp_coord, interp_index),
                    moved_coord,
                ))
            }
            GridPointQuery1::Outside => GridPointQuery1::Outside,
        }
    }

    fn interp_scalar_field_known_cell<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
        interp_index: usize,
    ) -> F
    where
        F: BFloat,
        G: Grid1<F>,
    {
        self.interp(field, interp_coord, interp_index)
    }

    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
    ) -> GridPointQuery1<F, F>
    where
        F: BFloat,
        G: Grid1<F>,
    {
        let grid_point_query = field.grid().find_closest_grid_cell(interp_coord);
        match grid_point_query {
            GridPointQuery1::Inside(interp_index) => {
                GridPointQuery1::Inside(self.interp(field, interp_coord, interp_index))
            }
            GridPointQuery1::MovedInside((interp_index, moved_coord)) => {
                GridPointQuery1::MovedInside((
                    self.interp(field, interp_coord, interp_index),
                    moved_coord,
                ))
            }
            GridPointQuery1::Outside => GridPointQuery1::Outside,
        }
    }
}

impl CubicHermiteSplineInterpolatorConfig {
    pub const DEFAULT_TANGENT_SCHEME: TangentScheme = TangentScheme::FiniteDifference;
    pub const DEFAULT_BOUNDARY_TANGENTS: BoundaryTangents = BoundaryTangents::Computed;
}

impl Default for CubicHermiteSplineInterpolatorConfig {
    fn default() -> Self {
        CubicHermiteSplineInterpolatorConfig {
            tangent_scheme: Self::DEFAULT_TANGENT_SCHEME,
            boundary_tangents: Self::DEFAULT_BOUNDARY_TANGENTS,
        }
    }
}
