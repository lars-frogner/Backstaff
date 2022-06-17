//! Interpolation with a cubic Hermite spline.

use super::{fip, Interpolator1};
use crate::{
    field::ScalarField1,
    grid::{fgr, CoordLocation, Grid1, GridPointQuery1},
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

    fn hermite_basis_00(t: fgr) -> fgr {
        (1.0 + 2.0 * t) * fgr::powi(1.0 - t, 2)
    }

    fn hermite_basis_10(t: fgr) -> fgr {
        t * fgr::powi(1.0 - t, 2)
    }

    fn hermite_basis_01(t: fgr) -> fgr {
        t * t * (3.0 - 2.0 * t)
    }

    fn hermite_basis_11(t: fgr) -> fgr {
        t * t * (t - 1.0)
    }

    fn compute_tangent<F>(&self, field: &ScalarField1<F>, index: usize) -> fgr
    where
        F: BFloat,
    {
        let size = field.size();
        let coords = field.coords();
        let values = field.values();

        if index == 0 {
            match self.config.boundary_tangents {
                BoundaryTangents::Computed => {
                    (values[1] - values[0]).into() / (coords[1] - coords[0])
                }
                BoundaryTangents::Zero => 0.0,
            }
        } else if index == size - 1 {
            match self.config.boundary_tangents {
                BoundaryTangents::Computed => {
                    (values[size - 1] - values[size - 2]).into()
                        / (coords[size - 1] - coords[size - 2])
                }
                BoundaryTangents::Zero => 0.0,
            }
        } else {
            let lower_distance = coords[index] - coords[index - 1];
            let upper_distance = coords[index + 1] - coords[index];
            let (lower_weight, upper_weight) = match self.config.tangent_scheme {
                TangentScheme::FiniteDifference => (0.5, 0.5),
                TangentScheme::ProximityWeightedFiniteDifference => {
                    (1.0 / lower_distance, 1.0 / upper_distance)
                }
            };
            (lower_weight * (values[index] - values[index - 1]).into() / lower_distance
                + upper_weight * (values[index + 1] - values[index]).into() / upper_distance)
                / (lower_weight + upper_weight)
        }
    }

    fn interp<F>(&self, field: &ScalarField1<F>, interp_coord: fgr, mut interp_index: usize) -> fip
    where
        F: BFloat,
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
        let start_value = values[interp_index].into();
        let end_value = values[interp_index + 1].into();

        let start_tangent = self.compute_tangent(field, interp_index);
        let end_tangent = self.compute_tangent(field, interp_index + 1);

        let t = (interp_coord - start_coord) / span;

        (Self::hermite_basis_00(t) * start_value
            + Self::hermite_basis_01(t) * end_value
            + Self::hermite_basis_10(t) * span * start_tangent
            + Self::hermite_basis_11(t) * span * end_tangent) as fip
    }
}

impl Interpolator1 for CubicHermiteSplineInterpolator {
    fn interp_scalar_field<F>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, fip>
    where
        F: BFloat,
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

    fn interp_scalar_field_known_cell<F>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
        interp_index: usize,
    ) -> fip
    where
        F: BFloat,
    {
        self.interp(field, interp_coord, interp_index)
    }

    fn interp_extrap_scalar_field<F>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, fip>
    where
        F: BFloat,
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
