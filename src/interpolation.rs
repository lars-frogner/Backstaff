//! Interpolation of scalar and vector fields.

pub mod cubic_hermite_spline;
pub mod poly_fit;

use crate::{
    field::{ScalarField1, ScalarField2, ScalarField3, VectorField2, VectorField3},
    geometry::{Idx2, Idx3, Point2, Point3, Vec2, Vec3},
    grid::{fgr, Grid1, Grid2, Grid3, GridPointQuery1, GridPointQuery2, GridPointQuery3},
    num::BFloat,
};

/// Default floating-point precision to use for interpolated values.
#[allow(non_camel_case_types)]
pub type fin = f64;

/// Defines the properties of a 3D interpolator.
pub trait Interpolator3: Clone + Sync + Send {
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
    ///
    /// # Type parameters
    ///
    /// - `G`: Type of grid.
    fn verify_grid<G>(&self, grid: &G) -> Result<(), String>
    where
        G: Grid3<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, F>
    where
        F: BFloat,
        G: Grid3<fgr>;

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
        interp_point: &Point3<fgr>,
        interp_indices: &Idx3<usize>,
    ) -> F
    where
        F: BFloat,
        G: Grid3<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, F>
    where
        F: BFloat,
        G: Grid3<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_vector_field<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, Vec3<F>>
    where
        F: BFloat,
        G: Grid3<fgr>;

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
        interp_point: &Point3<fgr>,
        interp_indices: &Idx3<usize>,
    ) -> Vec3<F>
    where
        F: BFloat,
        G: Grid3<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_extrap_vector_field<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, Vec3<F>>
    where
        F: BFloat,
        G: Grid3<fgr>;
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
    /// - `G`: Type of grid.
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField2<F, G>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, F>
    where
        F: BFloat,
        G: Grid2<fgr>;

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
        field: &ScalarField2<F, G>,
        interp_point: &Point2<fgr>,
        interp_indices: &Idx2<usize>,
    ) -> F
    where
        F: BFloat,
        G: Grid2<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField2<F, G>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, F>
    where
        F: BFloat,
        G: Grid2<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_vector_field<F, G>(
        &self,
        field: &VectorField2<F, G>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, Vec2<F>>
    where
        F: BFloat,
        G: Grid2<fgr>;

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
        field: &VectorField2<F, G>,
        interp_point: &Point2<fgr>,
        interp_indices: &Idx2<usize>,
    ) -> Vec2<F>
    where
        F: BFloat,
        G: Grid2<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_extrap_vector_field<F, G>(
        &self,
        field: &VectorField2<F, G>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, Vec2<F>>
    where
        F: BFloat,
        G: Grid2<fgr>;
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
    /// - `G`: Type of grid.
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, F>
    where
        F: BFloat,
        G: Grid1<fgr>;

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
    /// - `G`: Type of grid.
    ///
    /// # Panics
    ///
    /// If the interpolation coordinate does not lie within the specified grid cell.
    fn interp_scalar_field_known_cell<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: fgr,
        interp_index: usize,
    ) -> F
    where
        F: BFloat,
        G: Grid1<fgr>;

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
    /// - `G`: Type of grid.
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, F>
    where
        F: BFloat,
        G: Grid1<fgr>;
}
