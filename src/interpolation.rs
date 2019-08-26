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
pub trait Interpolator3<T, G>
where T: num::Float,
      G: Grid3<T> + Clone
{
    /// Interpolates the given scalar field at the given coordinate.
    fn interp_scalar_field(field: &ScalarField3<T, G>, interp_point: &Point3<T>) -> InterpResult3<T>;

    /// Interpolates the given vector field at the given coordinate.
    fn interp_vector_field(field: &VectorField3<T, G>, interp_point: &Point3<T>) -> InterpResult3<Vec3<T>>;
}
