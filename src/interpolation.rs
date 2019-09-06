//! Interpolation of scalar and vector fields.

pub mod poly_fit;

use crate::num::BFloat;
use crate::geometry::{Point3, Vec3};
use crate::grid::Grid3;
use crate::field::{ScalarField3, VectorField3};

/// Defines the properties of a 3D interpolator.
pub trait Interpolator3 {
    /// Computes the interpolated value of a scalar field at the given coordinate.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated value should be computed.
    ///
    /// # Returns
    ///
    /// An `Option<F>` which is either:
    ///
    /// - `Some`: Contains the interpolated field value.
    /// - `None`: The interpolation point was outside the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_scalar_field<F, G>(&self, field: &ScalarField3<F, G>, interp_point: &Point3<F>) -> Option<F>
    where F: BFloat,
          G: Grid3<F>;

    /// Computes the interpolated vector of a vector field at the given coordinate.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated vector should be computed.
    ///
    /// # Returns
    ///
    /// An `Option<F>` which is either:
    ///
    /// - `Some`: Contains the interpolated field vector.
    /// - `None`: The interpolation point was outside the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_vector_field<F, G>(&self, field: &VectorField3<F, G>, interp_point: &Point3<F>) -> Option<Vec3<F>>
    where F: BFloat,
          G: Grid3<F>;
}
