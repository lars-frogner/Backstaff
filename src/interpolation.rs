//! Interpolation of scalar and vector fields.

pub mod poly_fit;

use crate::field::{ScalarField3, VectorField3};
use crate::geometry::{Idx3, Point3, Vec3};
use crate::grid::{Grid3, GridPointQuery3};
use crate::num::BFloat;

/// Defines the properties of a 3D interpolator.
pub trait Interpolator3: Clone + Sync + Send {
    /// Computes the interpolated value of a scalar field at the given coordinate,
    /// wrapping around any periodic boundaries.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated value should be computed.
    ///
    /// # Returns
    ///
    /// A `GridPointQuery3<F, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field value.
    /// - `MovedInside`: Contains the interpolated field value and a wrapped version of the
    /// interpolation point located on the inside of the grid.
    /// - `Outside`: The interpolation point was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, F>
    where
        F: BFloat,
        G: Grid3<F>;

    /// Computes the interpolated value of a scalar field at the given coordinate
    /// known to lie inside the grid cell with the given indices.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated value should be computed.
    /// - `interp_indices`: Indices of the grid cell containing the interpolation coordinate.
    ///
    /// # Returns
    ///
    /// The interpolated field value.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    ///
    /// # Panics
    ///
    /// If the interpolation coordinate does not lie within the specified grid cell.
    fn interp_scalar_field_known_cell<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<F>,
        interp_indices: &Idx3<usize>,
    ) -> F
    where
        F: BFloat,
        G: Grid3<F>;

    /// Computes the interpolated or extrapolated value of a scalar field at the given coordinate,
    /// wrapping around any periodic boundaries.
    ///
    /// Extrapolation is performed if the coordinate is outside a non-periodic boundary.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate/extrapolate.
    /// - `interp_point`: Coordinate where the interpolated/extrapolated value should be computed.
    ///
    /// # Returns
    ///
    /// A `GridPointQuery3<F, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field value.
    /// - `MovedInside`: Contains the interpolated/extrapolated field value and a wrapped/truncated
    /// version of the interpolation point located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, F>
    where
        F: BFloat,
        G: Grid3<F>;

    /// Computes the interpolated vector of a vector field at the given coordinate,
    /// wrapping around any periodic boundaries.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated vector should be computed.
    ///
    /// # Returns
    ///
    /// A `GridPointQuery3<F, Vec3<F>>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field vector.
    /// - `MovedInside`: Contains the interpolated field vector and a wrapped version of the
    /// interpolation point located on the inside of the grid.
    /// - `Outside`: The interpolation point was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_vector_field<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, Vec3<F>>
    where
        F: BFloat,
        G: Grid3<F>;

    /// Computes the interpolated vector of a vector field at the given coordinate
    /// known to lie inside the grid cell with the given indices.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to interpolate.
    /// - `interp_point`: Coordinate where the interpolated vector should be computed.
    /// - `interp_indices`: Indices of the grid cell containing the interpolation coordinate.
    ///
    /// # Returns
    ///
    /// The interpolated field vector.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    ///
    /// # Panics
    ///
    /// If the interpolation coordinate does not lie within the specified grid cell.
    fn interp_vector_field_known_cell<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<F>,
        interp_indices: &Idx3<usize>,
    ) -> Vec3<F>
    where
        F: BFloat,
        G: Grid3<F>;

    /// Computes the interpolated or extrapolated value of a vector field at the given coordinate,
    /// wrapping around any periodic boundaries.
    ///
    /// Extrapolation is performed if the coordinate is outside a non-periodic boundary.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to interpolate/extrapolate.
    /// - `interp_point`: Coordinate where the interpolated/extrapolated vector should be computed.
    ///
    /// # Returns
    ///
    /// A `GridPointQuery3<F, Vec3<F>>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field vector.
    /// - `MovedInside`: Contains the interpolated/extrapolated field vector and a wrapped/truncated
    /// version of the interpolation point located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    fn interp_extrap_vector_field<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, Vec3<F>>
    where
        F: BFloat,
        G: Grid3<F>;
}
