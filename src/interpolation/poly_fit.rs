//! Structures for interpolating Bifrost fields.

use ndarray::prelude::*;
use super::{InterpResult3, Interpolator3};
use crate::geometry::{Dim3, In3D, Point3, Idx3, Vec3, CoordRefs3};
use crate::grid::{FoundIdx3, Grid3};
use crate::field::{ScalarField3, VectorField3};
use Dim3::{X, Y, Z};

/// A 3D interpolator using polynomial fitting to estimate the interpolated value.
pub struct PolyFitInterpolator3;

impl PolyFitInterpolator3 {
    /// Order of interpolation between grid points.
    pub const ORDER:    usize = 3;

    /// How many cells to shift the interval of interpolation points towards higher indices.
    pub const BIAS:     isize = 0;

    /// Number of interpolation points.
    const POINTS:       usize = Self::ORDER + 1;

    /// Offset from a point to the lower edge of the interpolation range.
    const START_OFFSET: isize = Self::BIAS - (Self::POINTS as isize)/2;

    fn interp<T, G>(grid: &G, coords: &CoordRefs3<T>, values: &Array3<T>, interp_point: &Point3<T>, interp_idx: &Idx3<usize>) -> T
    where T: num::Float + std::ops::AddAssign,
          G: Grid3<T> + Clone
    {
        let grid_shape = grid.shape();

        let mut start_idx = Idx3::new((interp_idx[X] as isize) + Self::START_OFFSET,
                                      (interp_idx[Y] as isize) + Self::START_OFFSET,
                                      (interp_idx[Z] as isize) + Self::START_OFFSET);

        let mut crosses_periodic_bound = In3D::new(false, false, false);
        let mut any_crosses_periodic_bound = false;

        for dim in Dim3::slice().iter() {
            // Check if start index is outside lower bound
            if start_idx[*dim] < 0 {
                if grid.is_periodic(*dim) {
                    // If the dimension is periodic, make a note of the crossing
                    crosses_periodic_bound[*dim] = true;
                    any_crosses_periodic_bound = true;
                } else {
                    // If the dimension is not periodic, shift the interpolation interval to lie on the inside
                    start_idx[*dim] = 0;
                }
            // Check the upper bound accordingly
            } else if (start_idx[*dim] as usize) + Self::POINTS > grid_shape[*dim] {
                if grid.is_periodic(*dim) {
                    crosses_periodic_bound[*dim] = true;
                    any_crosses_periodic_bound = true;
                } else {
                    start_idx[*dim] = grid_shape[*dim] as isize - 1;
                }
            }
        }

        // Create appropriate subarrays according to the detected crossings

        let x_coord_subarray = if crosses_periodic_bound[X] {
            Self::create_coordinate_subarray_for_periodic(coords[X], start_idx[X])
        } else {
            Self::create_coordinate_subarray_for_interior(coords[X], start_idx[X])
        };

        let y_coord_subarray = if crosses_periodic_bound[Y] {
            Self::create_coordinate_subarray_for_periodic(coords[Y], start_idx[Y])
        } else {
            Self::create_coordinate_subarray_for_interior(coords[Y], start_idx[Y])
        };

        let z_coord_subarray = if crosses_periodic_bound[Z] {
            Self::create_coordinate_subarray_for_periodic(coords[Z], start_idx[Z])
        } else {
            Self::create_coordinate_subarray_for_interior(coords[Z], start_idx[Z])
        };

        let value_subarray = if any_crosses_periodic_bound {
            Self::create_value_subarray_for_periodic(values, &start_idx)
        } else {
            Self::create_value_subarray_for_interior(values, &start_idx)
        };

        Self::interp_subarrays(In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray), &value_subarray, interp_point)
    }

    fn create_coordinate_subarray_for_interior<T>(coords: &Array1<T>, start_idx: isize) -> [T; Self::POINTS]
    where T: num::Float
    {
        let mut subarray = [T::zero(); Self::POINTS];
        let offset = start_idx as usize;
        for idx in 0..Self::POINTS {
            subarray[idx] = coords[offset + idx];
        }
        subarray
    }

    fn create_coordinate_subarray_for_periodic<T>(coords: &Array1<T>, start_idx: isize) -> [T; Self::POINTS]
    where T: num::Float
    {
        let mut subarray = [T::zero(); Self::POINTS];
        let offset = (start_idx + (Self::POINTS as isize)) as usize;
        for idx in 0..Self::POINTS {
            subarray[idx] = coords[(offset + idx) % Self::POINTS];
        }
        subarray
    }

    fn create_value_subarray_for_interior<T>(values: &Array3<T>, start_idx: &Idx3<isize>) -> [T; Self::POINTS*Self::POINTS*Self::POINTS]
    where T: num::Float
    {
        let mut subarray = [T::zero(); Self::POINTS*Self::POINTS*Self::POINTS];
        let offsets = In3D::new(start_idx[X] as usize,
                                start_idx[Y] as usize,
                                start_idx[Z] as usize);
        let mut idx = 0;

        for k in offsets[Z]..(offsets[Z] + Self::POINTS) {
            for j in offsets[Y]..(offsets[Y] + Self::POINTS) {
                for i in offsets[X]..(offsets[X] + Self::POINTS) {
                    subarray[idx] = values[[i, j, k]];
                    idx += 1;
                }
            }
        }

        subarray
    }

    fn create_value_subarray_for_periodic<T>(values: &Array3<T>, start_idx: &Idx3<isize>) -> [T; Self::POINTS*Self::POINTS*Self::POINTS]
    where T: num::Float
    {
        let mut subarray = [T::zero(); Self::POINTS*Self::POINTS*Self::POINTS];
        let offsets = In3D::new((start_idx[X] + (Self::POINTS as isize)) as usize,
                                (start_idx[Y] + (Self::POINTS as isize)) as usize,
                                (start_idx[Z] + (Self::POINTS as isize)) as usize);
        let mut idx = 0;

        for k in 0..Self::POINTS {
            for j in 0..Self::POINTS {
                for i in 0..Self::POINTS {
                    subarray[idx] = values[[(offsets[X] + i) % Self::POINTS,
                                            (offsets[Y] + j) % Self::POINTS,
                                            (offsets[Z] + k) % Self::POINTS]];
                    idx += 1;
                }
            }
        }

        subarray
    }

    fn interp_subarrays<T>(coords: In3D<&[T; Self::POINTS]>, values: &[T; Self::POINTS*Self::POINTS*Self::POINTS], interp_point: &Point3<T>) -> T
    where T: num::Float + std::ops::AddAssign
    {
        let x_coords = coords[X];
        let y_coords = coords[Y];
        let z_coords = coords[Z];

        let mut vals_c = [T::zero(); Self::POINTS];
        let mut vals_d = [T::zero(); Self::POINTS];
        let mut poly_x = [T::zero(); Self::POINTS*Self::POINTS];
        let mut poly_xy = [T::zero(); Self::POINTS];
        let mut poly_xyz;
        let mut accum;
        let mut correction;

        for k in 0..Self::POINTS {
            for j in 0..Self::POINTS {

                vals_c.copy_from_slice(&values[(k*Self::POINTS + j)*Self::POINTS..(k*Self::POINTS + j + 1)*Self::POINTS]);
                vals_d.copy_from_slice(&vals_c);

                accum = vals_c[0];

                for n in 1..Self::POINTS {

                    for i in 0..(Self::POINTS - n) {
                        correction = (vals_c[i + 1] - vals_d[i])/(x_coords[i + n] - x_coords[i]);
                        vals_c[i] = (interp_point[X] - x_coords[i])*correction;
                        vals_d[i] = (interp_point[X] - x_coords[i + n])*correction;
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
                    correction = (vals_c[j + 1] - vals_d[j])/(y_coords[j + n] - y_coords[j]);
                    vals_c[j] = (interp_point[Y] - y_coords[j])*correction;
                    vals_d[j] = (interp_point[Y] - y_coords[j + n])*correction;
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
                correction = (vals_c[k + 1] - vals_d[k])/(z_coords[k + n] - z_coords[k]);
                vals_c[k] = (interp_point[Z] - z_coords[k])*correction;
                vals_d[k] = (interp_point[Z] - z_coords[k + n])*correction;
            }

            poly_xyz += vals_c[0];
        }

        poly_xyz
    }
}

impl<T, G> Interpolator3<T, G> for PolyFitInterpolator3
where T: num::Float + std::ops::AddAssign,
      G: Grid3<T> + Clone
{
    fn interp_scalar_field(field: &ScalarField3<T, G>, interp_point: &Point3<T>) -> InterpResult3<T> {
        let grid = field.grid();
        let coords = field.coords();
        let values = field.values();

        match grid.find_grid_cell(interp_point) {
            FoundIdx3::Inside(interp_idx) => InterpResult3::Ok(Self::interp(grid, &coords, values, interp_point, &interp_idx)),
            FoundIdx3::Outside(crossings) => InterpResult3::OutOfBounds(crossings),
        }
    }

    fn interp_vector_field(field: &VectorField3<T, G>, interp_point: &Point3<T>) -> InterpResult3<Vec3<T>> {
        let grid = field.grid();
        let coords = field.coords();
        let values = field.values();

        match grid.find_grid_cell(interp_point) {
            FoundIdx3::Inside(interp_idx) => InterpResult3::Ok(Vec3::new(
                                                 Self::interp(grid, &coords[X], &values[X], interp_point, &interp_idx),
                                                 Self::interp(grid, &coords[Y], &values[Y], interp_point, &interp_idx),
                                                 Self::interp(grid, &coords[Z], &values[Z], interp_point, &interp_idx)
                                             )),
            FoundIdx3::Outside(crossings) => InterpResult3::OutOfBounds(crossings),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::path;
    use crate::grid::Grid3VaryingZ;
    use crate::reading::{SnapshotReader, Endianness};

    #[test]
    fn regular_grid_index_search_works() {
        let params_path = path::PathBuf::from("data/en024031_emer3.0sml_ebeam_631.idl");
        let reader: SnapshotReader<Grid3VaryingZ<f32>> = SnapshotReader::new(&params_path, Endianness::Little).unwrap();
        let field = reader.read_3d_scalar_field("r").unwrap();

    }
}
