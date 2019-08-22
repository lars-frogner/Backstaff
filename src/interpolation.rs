//! Structures for interpolating Bifrost fields.

use ndarray::prelude::*;
use ndarray::s;
use crate::geometry::{Dim, In3D, Coord3, Idx3, Vec3, CoordRefs3};
use crate::grid::{BoundsCrossing, FoundIdx3, Grid3};
use crate::field::{ScalarField3, VectorField3};
use Dim::{X, Y, Z};

/// An interpolated value or a bounds crossing for each dimension.
pub enum InterpResult<T> {
    Ok(T),
    OutOfBounds(In3D<BoundsCrossing>)
}

/// Defines the properties of a 3D interpolator.
pub trait Interpolator3<T, G>
where T: num::Float,
      G: Grid3<T> + Clone
{
    /// Interpolates the given scalar field at the given coordinate.
    fn interp_scalar_field(field: &ScalarField3<T, G>, interp_coord: &Coord3<T>) -> InterpResult<T>;

    /// Interpolates the given vector field at the given coordinate.
    fn interp_vector_field(field: &VectorField3<T, G>, interp_coord: &Coord3<T>) -> InterpResult<Vec3<T>>;
}

/// A 3D interpolator using polynomial fitting to estimate the interpolated value.
pub struct PolyInterpolator3;

impl PolyInterpolator3 {
    /// Order of interpolation between grid points.
    pub const ORDER:    usize = 3;

    /// How many cells to shift the interval of interpolation points towards higher indices.
    pub const BIAS:     isize = 0;

    /// Number of interpolation points.
    const POINTS:       usize = Self::ORDER + 1;

    /// Offset from a point to the lower edge of the interpolation range.
    const START_OFFSET: isize = Self::BIAS - (Self::POINTS as isize)/2;

    fn compute_start_idx(grid_shape: &In3D<usize>, interp_idx: &Idx3<usize>) -> Idx3<usize> {

        let mut idx = interp_idx.clone();

        for dim in Dim::xyz_slice().iter() {
            let i = (idx[*dim] as isize) + Self::START_OFFSET;
            if i < 0 {
                idx[*dim] = 0;
            } else if (i as usize) + Self::POINTS > grid_shape[*dim] {
                idx[*dim] = grid_shape[*dim] - 1;
            } else {
                idx[*dim] = i as usize;
            }
        }

        idx
    }

    fn interp_array<T>(coords: &CoordRefs3<T>, values: &Array3<T>, interp_coord: &Coord3<T>, start_idx: &Idx3<usize>) -> T
    where T: num::Float + std::ops::AddAssign
    {
        let x_coords = coords[X];
        let y_coords = coords[Y];
        let z_coords = coords[Z];

        let values_slice = values.slice(s![start_idx[X]..(start_idx[X] + Self::POINTS),
                                           start_idx[Y]..(start_idx[Y] + Self::POINTS),
                                           start_idx[Z]..(start_idx[Z] + Self::POINTS)]);
        let values_slice = values_slice.as_slice().unwrap();

        let mut vals_c = [T::zero(); Self::POINTS];
        let mut vals_d = [T::zero(); Self::POINTS];
        let mut poly_x = [T::zero(); Self::POINTS*Self::POINTS];
        let mut poly_xy = [T::zero(); Self::POINTS];
        let mut poly_xyz;
        let mut accum;
        let mut correction;

        for k in 0..Self::POINTS {
            for j in 0..Self::POINTS {

                vals_c.copy_from_slice(&values_slice[(k*Self::POINTS + j)*Self::POINTS..(k*Self::POINTS + j + 1)*Self::POINTS]);
                vals_d.copy_from_slice(&vals_c);

                accum = vals_c[0];

                for n in 1..Self::POINTS {

                    for i in 0..(Self::POINTS - n) {
                        correction = (vals_c[i + 1] - vals_d[i])/(x_coords[start_idx[X] + i + n] - x_coords[start_idx[X] + i]);
                        vals_c[i] = (interp_coord[X] - x_coords[start_idx[X] + i])*correction;
                        vals_d[i] = (interp_coord[X] - x_coords[start_idx[X] + i + n])*correction;
                    }

                    accum += vals_c[0];
                }

                poly_x[k*Self::POINTS + j] = accum;
            }
        }

        for k in 0..Self::POINTS {

            vals_c.copy_from_slice(&poly_x[k*Self::POINTS..(k + 1)*Self::POINTS]);
            vals_d.copy_from_slice(&vals_c);

            accum = vals_c[0];

            for n in 1..Self::POINTS {

                for j in 0..(Self::POINTS - n) {
                    correction = (vals_c[j + 1] - vals_d[j])/(y_coords[start_idx[Y] + j + n] - y_coords[start_idx[Y] + j]);
                    vals_c[j] = (interp_coord[Y] - y_coords[start_idx[Y] + j])*correction;
                    vals_d[j] = (interp_coord[Y] - y_coords[start_idx[Y] + j + n])*correction;
                }

                accum += vals_c[0];
            }

            poly_xy[k] = accum;
        }

        vals_c.copy_from_slice(&poly_xy);
        vals_d.copy_from_slice(&vals_c);

        poly_xyz = vals_c[0];

        for n in 1..Self::POINTS {

            for k in 0..(Self::POINTS - n) {
                correction = (vals_c[k + 1] - vals_d[k])/(z_coords[start_idx[Z] + k + n] - z_coords[start_idx[Z] + k]);
                vals_c[k] = (interp_coord[Z] - z_coords[start_idx[Z] + k])*correction;
                vals_d[k] = (interp_coord[Z] - z_coords[start_idx[Z] + k + n])*correction;
            }

            poly_xyz += vals_c[0];
        }

        poly_xyz
    }
}

impl<T, G> Interpolator3<T, G> for PolyInterpolator3
where T: num::Float + std::ops::AddAssign,
      G: Grid3<T> + Clone
{
    fn interp_scalar_field(field: &ScalarField3<T, G>, interp_coord: &Coord3<T>) -> InterpResult<T> {
        let grid = field.grid();
        let coords = field.coords();
        let values = field.values();

        match grid.find_grid_cell(interp_coord) {
            FoundIdx3::Inside(interp_idx) => {
                let start_idx = Self::compute_start_idx(grid.shape(), &interp_idx);
                InterpResult::Ok(Self::interp_array(&coords, values, interp_coord, &start_idx))
            },
            FoundIdx3::Outside(crossings) => InterpResult::OutOfBounds(crossings),
        }
    }

    fn interp_vector_field(field: &VectorField3<T, G>, interp_coord: &Coord3<T>) -> InterpResult<Vec3<T>> {
        let grid = field.grid();
        let coords = field.coords();
        let values = field.values();

        match grid.find_grid_cell(interp_coord) {
            FoundIdx3::Inside(interp_idx) => {
                let start_idx = Self::compute_start_idx(grid.shape(), &interp_idx);
                InterpResult::Ok(Vec3::new(
                    Self::interp_array(&coords[X], &values[X], interp_coord, &start_idx),
                    Self::interp_array(&coords[Y], &values[Y], interp_coord, &start_idx),
                    Self::interp_array(&coords[Z], &values[Z], interp_coord, &start_idx)
                ))
            },
            FoundIdx3::Outside(crossings) => InterpResult::OutOfBounds(crossings),
        }
    }
}
