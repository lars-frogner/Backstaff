//! Interpolation of scalar and vector fields.

pub mod cubic_hermite_spline;
pub mod poly_fit;

use crate::{
    field::{FieldGrid3, ScalarField1, ScalarField2, ScalarField3, VectorField2, VectorField3},
    geometry::{Idx2, Idx3, Point2, Point3, Vec2, Vec3},
    grid::{fgr, GridPointQuery1, GridPointQuery2, GridPointQuery3},
    num::BFloat,
};

/// Default floating-point precision to use for interpolated values.
#[allow(non_camel_case_types)]
pub type fip = f64;

pub trait InterpGridVerifier3 {
    /// Verifies that a field with the given grid safely can be interpolated
    /// with this interpolator.
    ///
    /// # Parameters
    ///
    /// - `grid`: Grid to verify.
    ///
    /// # Returns
    ///
    /// A `Result<(), String>` which is either
    /// - `Ok`: The grid is valid.
    /// - `Err`: Contains a message describing why the grid is invalid.
    fn verify_grid(&self, grid: &FieldGrid3) -> Result<(), String>;
}

/// Defines the properties of a 3D interpolator.
pub trait Interpolator3<F: BFloat>: InterpGridVerifier3 + Clone + Sync + Send {
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
    /// A `GridPointQuery3<fgr, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field value.
    /// - `MovedInside`: Contains the interpolated field value and a wrapped version of the
    /// interpolation point located on the inside of the grid.
    /// - `Outside`: The interpolation point was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_scalar_field(
        &self,
        field: &ScalarField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, fip>;

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
    fn interp_scalar_field_known_cell(
        &self,
        field: &ScalarField3<F>,
        interp_point: &Point3<fgr>,
        interp_indices: &Idx3<usize>,
    ) -> fip;

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
    /// A `GridPointQuery3<fgr, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field value.
    /// - `MovedInside`: Contains the interpolated/extrapolated field value and a wrapped/truncated
    /// version of the interpolation point located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_extrap_scalar_field(
        &self,
        field: &ScalarField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, fip>;

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
    /// A `GridPointQuery3<fgr, Vec3<F>>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field vector.
    /// - `MovedInside`: Contains the interpolated field vector and a wrapped version of the
    /// interpolation point located on the inside of the grid.
    /// - `Outside`: The interpolation point was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_vector_field(
        &self,
        field: &VectorField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, Vec3<fip>>;

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
    fn interp_vector_field_known_cell(
        &self,
        field: &VectorField3<F>,
        interp_point: &Point3<fgr>,
        interp_indices: &Idx3<usize>,
    ) -> Vec3<fip>;

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
    /// A `GridPointQuery3<fgr, Vec3<F>>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field vector.
    /// - `MovedInside`: Contains the interpolated/extrapolated field vector and a wrapped/truncated
    /// version of the interpolation point located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_extrap_vector_field(
        &self,
        field: &VectorField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, Vec3<fip>>;
}

/// Defines the properties of a 2D interpolator.
pub trait Interpolator2: Clone + Sync + Send {
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
    /// A `GridPointQuery2<fgr, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field value.
    /// - `MovedInside`: Contains the interpolated field value and a wrapped version of the
    /// interpolation point located on the inside of the grid.
    /// - `Outside`: The interpolation point was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_scalar_field<F: BFloat>(
        &self,
        field: &ScalarField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, fip>;

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
    fn interp_scalar_field_known_cell<F: BFloat>(
        &self,
        field: &ScalarField2<F>,
        interp_point: &Point2<fgr>,
        interp_indices: &Idx2<usize>,
    ) -> fip;

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
    /// A `GridPointQuery3<fgr, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field value.
    /// - `MovedInside`: Contains the interpolated/extrapolated field value and a wrapped/truncated
    /// version of the interpolation point located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_extrap_scalar_field<F: BFloat>(
        &self,
        field: &ScalarField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, fip>;

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
    /// A `GridPointQuery2<fgr, Vec2<F>>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field vector.
    /// - `MovedInside`: Contains the interpolated field vector and a wrapped version of the
    /// interpolation point located on the inside of the grid.
    /// - `Outside`: The interpolation point was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_vector_field<F: BFloat>(
        &self,
        field: &VectorField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, Vec2<fip>>;

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
    fn interp_vector_field_known_cell<F: BFloat>(
        &self,
        field: &VectorField2<F>,
        interp_point: &Point2<fgr>,
        interp_indices: &Idx2<usize>,
    ) -> Vec2<fip>;

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
    /// A `GridPointQuery2<fgr, Vec2<F>>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field vector.
    /// - `MovedInside`: Contains the interpolated/extrapolated field vector and a wrapped/truncated
    /// version of the interpolation point located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_extrap_vector_field<F: BFloat>(
        &self,
        field: &VectorField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, Vec2<fip>>;
}

/// Defines the properties of a 1D interpolator.
pub trait Interpolator1: Clone + Sync + Send {
    /// Computes the interpolated value of a scalar field at the given coordinate,
    /// wrapping around any periodic boundaries.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate.
    /// - `interp_coord`: Coordinate where the interpolated value should be computed.
    ///
    /// # Returns
    ///
    /// A `GridPointQuery1<fgr, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated field value.
    /// - `MovedInside`: Contains the interpolated field value and a wrapped version of the
    /// interpolation coordinate located on the inside of the grid.
    /// - `Outside`: The interpolation coordinate was outside a non-periodic boundary.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_scalar_field<F: BFloat>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, fip>;

    /// Computes the interpolated value of a scalar field at the given coordinate
    /// known to lie inside the grid cell with the given indices.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate.
    /// - `interp_coord`: Coordinate where the interpolated value should be computed.
    /// - `interp_index`: Index of the grid cell containing the interpolation coordinate.
    ///
    /// # Returns
    ///
    /// The interpolated field value.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_scalar_field_known_cell<F: BFloat>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
        interp_index: usize,
    ) -> fip;

    /// Computes the interpolated or extrapolated value of a scalar field at the given coordinate,
    /// wrapping around any periodic boundaries.
    ///
    /// Extrapolation is performed if the coordinate is outside a non-periodic boundary.
    ///
    /// # Parameters
    ///
    /// - `field`: Scalar field to interpolate/extrapolate.
    /// - `interp_coord`: Coordinate where the interpolated/extrapolated value should be computed.
    ///
    /// # Returns
    ///
    /// A `GridPointQuery1<fgr, F>` which is either:
    ///
    /// - `Inside`: Contains the interpolated/extrapolated field value.
    /// - `MovedInside`: Contains the interpolated/extrapolated field value and a wrapped/truncated
    /// version of the interpolation coordinate located on the inside of the grid.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn interp_extrap_scalar_field<F: BFloat>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, fip>;
}
