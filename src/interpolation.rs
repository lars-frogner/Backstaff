//! Interpolation of Bifrost fields.

pub mod poly_fit;

use crate::geometry::{In3D, Point3, Vec3};
use crate::grid::{BoundsCrossing, Grid3};
use crate::field::{ScalarField3, VectorField3};

/// An interpolated value or a bounds crossing for each dimension.
pub enum InterpResult3<T> {
    Ok(T),
    OutOfBounds(In3D<BoundsCrossing>)
}

impl<T> InterpResult3<T> {
    pub fn unwrap(self) -> T {
        match self {
            InterpResult3::Ok(value) => value,
            InterpResult3::OutOfBounds(_) => panic!("called `InterpResult3::unwrap()` on an `OutOfBounds` value")
        }
    }
}

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
    /// An `InterpResult3<F>` which is either:
    ///
    /// - `Ok`: Contains the interpolated field value.
    /// - `OutOfBounds`: Contains a `BoundsCrossing` for each dimension.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_scalar_field<F, G>(&self, field: &ScalarField3<F, G>, interp_point: &Point3<F>) -> InterpResult3<F>
    where F: num::Float + std::fmt::Display,
          G: Grid3<F> + Clone;

    /// Computes the interpolated vector of a vector field at the given coordinate.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated vector should be computed.
    ///
    /// # Returns
    ///
    /// An `InterpResult3<F>` which is either:
    ///
    /// - `Ok`: Contains the interpolated field vector.
    /// - `OutOfBounds`: Contains a `BoundsCrossing` for each dimension.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_vector_field<F, G>(&self, field: &VectorField3<F, G>, interp_point: &Point3<F>) -> InterpResult3<Vec3<F>>
    where F: num::Float + std::fmt::Display,
          G: Grid3<F> + Clone;
}
