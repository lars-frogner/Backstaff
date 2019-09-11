//! Interpolation by polynomial fitting.

use ndarray::prelude::*;
use crate::num::BFloat;
use crate::geometry::{Dim3, In3D, Point3, Idx3, Vec3, CoordRefs3};
use crate::grid::{Grid3, GridPointQuery3, CoordLocation};
use crate::field::{ScalarField3, VectorField3};
use super::Interpolator3;
use Dim3::{X, Y, Z};

/// A 3D interpolator using polynomial fitting to estimate the interpolated value.
pub struct PolyFitInterpolator3;

impl PolyFitInterpolator3 {
    /// Order of interpolation between grid points.
    pub const ORDER:    usize = 3;

    /// Number of interpolation points.
    const POINTS:       usize = Self::ORDER + 1;

    /// Offset from a point to the lower edge of the interpolation range.
    const DEFAULT_START_OFFSET: isize = 1 - ((Self::POINTS + 1) as isize)/2;

    fn interp<F, G>(grid: &G, coords: &CoordRefs3<F>, locations: &In3D<CoordLocation>, values: &Array3<F>, interp_point: &Point3<F>, interp_idx: &Idx3<usize>) -> F
    where F: BFloat,
          G: Grid3<F>
    {
        let grid_shape = grid.shape();

        let center_coords = grid.centers();
        let start_offset_x = Self::compute_start_offset(&center_coords[X], locations[X], interp_point[X], interp_idx[X]);
        let start_offset_y = Self::compute_start_offset(&center_coords[Y], locations[Y], interp_point[Y], interp_idx[Y]);
        let start_offset_z = Self::compute_start_offset(&center_coords[Z], locations[Z], interp_point[Z], interp_idx[Z]);

        let mut start_idx = Idx3::new((interp_idx[X] as isize) + start_offset_x,
                                      (interp_idx[Y] as isize) + start_offset_y,
                                      (interp_idx[Z] as isize) + start_offset_z);

        let mut crosses_periodic_bound = In3D::new(false, false, false);
        let mut any_crosses_periodic_bound = false;

        for &dim in Dim3::slice().iter() {
            // Check if start index is outside lower bound
            if start_idx[dim] < 0 {
                if grid.is_periodic(dim) {
                    // If the dimension is periodic, make a note of the crossing
                    crosses_periodic_bound[dim] = true;
                    any_crosses_periodic_bound = true;
                } else {
                    // If the dimension is not periodic, shift the interpolation interval to lie on the inside
                    start_idx[dim] = 0;
                }
            // Check the upper bound accordingly
            } else if (start_idx[dim] as usize) + Self::POINTS > grid_shape[dim] {
                if grid.is_periodic(dim) {
                    crosses_periodic_bound[dim] = true;
                    any_crosses_periodic_bound = true;
                } else {
                    start_idx[dim] = (grid_shape[dim] - Self::POINTS) as isize;
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

    fn compute_start_offset<F: BFloat>(center_coords: &[F], locations: CoordLocation, interp_coord: F, interp_idx: usize) -> isize {
        match locations {
            CoordLocation::Center => if Self::ORDER % 2 != 0 && interp_coord < center_coords[interp_idx] {
                // If coordinates are located at cell centers, interpolation order is odd and interpolation coordinate is in lower half of the cell:
                // Shift start offset one cell down.
                Self::DEFAULT_START_OFFSET - 1
            } else {
                Self::DEFAULT_START_OFFSET
            },
            CoordLocation::LowerEdge => if Self::ORDER % 2 == 0 && interp_coord > center_coords[interp_idx] {
                // If coordinates are located at lower cell edges, interpolation order is event and interpolation coordinate is in upper half of the cell:
                // Shift start offset one cell up.
                Self::DEFAULT_START_OFFSET + 1
            } else {
                Self::DEFAULT_START_OFFSET
            }
        }
    }

    fn create_coordinate_subarray_for_interior<F: BFloat>(coords: &[F], start_idx: isize) -> [F; Self::POINTS] {
        let mut subarray = [F::zero(); Self::POINTS];
        let offset = start_idx as usize;
        subarray[..Self::POINTS].clone_from_slice(&coords[offset..(Self::POINTS + offset)]);
        subarray
    }

    fn create_coordinate_subarray_for_periodic<F: BFloat>(coords: &[F], start_idx: isize) -> [F; Self::POINTS] {
        let len = coords.len();
        let offset = (start_idx + (len as isize)) as usize;
        let mut subarray = [F::zero(); Self::POINTS];
        for idx in 0..Self::POINTS {
            subarray[idx] = coords[(offset + idx) % len];
        }
        subarray
    }

    fn create_value_subarray_for_interior<F: BFloat>(values: &Array3<F>, start_idx: &Idx3<isize>) -> [F; Self::POINTS*Self::POINTS*Self::POINTS] {
        let mut subarray = [F::zero(); Self::POINTS*Self::POINTS*Self::POINTS];
        let offsets = Idx3::from(&start_idx);
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

    fn create_value_subarray_for_periodic<F: BFloat>(values: &Array3<F>, start_idx: &Idx3<isize>) -> [F; Self::POINTS*Self::POINTS*Self::POINTS] {
        let grid_shape = values.shape();
        let offsets = In3D::new((start_idx[X] + (grid_shape[0] as isize)) as usize,
                                (start_idx[Y] + (grid_shape[1] as isize)) as usize,
                                (start_idx[Z] + (grid_shape[2] as isize)) as usize);
        let mut subarray = [F::zero(); Self::POINTS*Self::POINTS*Self::POINTS];
        let mut idx = 0;

        for k in 0..Self::POINTS {
            for j in 0..Self::POINTS {
                for i in 0..Self::POINTS {
                    subarray[idx] = values[[(offsets[X] + i) % grid_shape[0],
                                            (offsets[Y] + j) % grid_shape[1],
                                            (offsets[Z] + k) % grid_shape[2]]];
                    idx += 1;
                }
            }
        }
        subarray
    }

    fn interp_subarrays<F: BFloat>(coords: In3D<&[F; Self::POINTS]>, values: &[F; Self::POINTS*Self::POINTS*Self::POINTS], interp_point: &Point3<F>) -> F {
        let x_coords = coords[X];
        let y_coords = coords[Y];
        let z_coords = coords[Z];

        let mut vals_c = [F::zero(); Self::POINTS];
        let mut vals_d = [F::zero(); Self::POINTS];
        let mut poly_x = [F::zero(); Self::POINTS*Self::POINTS];
        let mut poly_xy = [F::zero(); Self::POINTS];
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

                    accum = accum + vals_c[0];
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

                accum = accum + vals_c[0];
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

            poly_xyz = poly_xyz + vals_c[0];
        }

        poly_xyz
    }
}

impl Interpolator3 for PolyFitInterpolator3 {
    fn interp_scalar_field<F, G>(&self, field: &ScalarField3<F, G>, interp_point: &Point3<F>) -> GridPointQuery3<F, F>
    where F: BFloat,
          G: Grid3<F>
    {
        let grid = field.grid();
        match grid.find_grid_cell(interp_point) {
            GridPointQuery3::Inside(interp_idx) => {
                GridPointQuery3::Inside(Self::interp(grid, &field.coords(), field.locations(), field.values(), interp_point, &interp_idx))
            },
            GridPointQuery3::WrappedInside((interp_idx, wrapped_point)) => {
                GridPointQuery3::WrappedInside((Self::interp(grid, &field.coords(), field.locations(), field.values(), interp_point, &interp_idx), wrapped_point))
            },
            GridPointQuery3::Outside => GridPointQuery3::Outside
        }
    }

    fn interp_extrap_scalar_field<F, G>(&self, field: &ScalarField3<F, G>, interp_point: &Point3<F>) -> F
    where F: BFloat,
          G: Grid3<F>
    {
        let grid = field.grid();
        let interp_idx = grid.find_closest_grid_cell(interp_point);
        Self::interp(grid, &field.coords(), field.locations(), field.values(), interp_point, &interp_idx)
    }

    fn interp_vector_field<F, G>(&self, field: &VectorField3<F, G>, interp_point: &Point3<F>) -> GridPointQuery3<F, Vec3<F>>
    where F: BFloat,
          G: Grid3<F>
    {
        let grid = field.grid();
        match grid.find_grid_cell(interp_point) {
            GridPointQuery3::Inside(interp_idx) => {
                GridPointQuery3::Inside(Vec3::new(
                    Self::interp(grid, &field.coords(X), field.locations(X), &field.values(X), interp_point, &interp_idx),
                    Self::interp(grid, &field.coords(Y), field.locations(Y), &field.values(Y), interp_point, &interp_idx),
                    Self::interp(grid, &field.coords(Z), field.locations(Z), &field.values(Z), interp_point, &interp_idx)
                ))
            },
            GridPointQuery3::WrappedInside((interp_idx, wrapped_point)) => {
                GridPointQuery3::WrappedInside((Vec3::new(
                    Self::interp(grid, &field.coords(X), field.locations(X), &field.values(X), interp_point, &interp_idx),
                    Self::interp(grid, &field.coords(Y), field.locations(Y), &field.values(Y), interp_point, &interp_idx),
                    Self::interp(grid, &field.coords(Z), field.locations(Z), &field.values(Z), interp_point, &interp_idx)
                ),
                wrapped_point))
            },
            GridPointQuery3::Outside => GridPointQuery3::Outside
        }
    }

    fn interp_extrap_vector_field<F, G>(&self, field: &VectorField3<F, G>, interp_point: &Point3<F>) -> Vec3<F>
    where F: BFloat,
          G: Grid3<F>
    {
        let grid = field.grid();
        let interp_idx = grid.find_closest_grid_cell(interp_point);
        Vec3::new(
            Self::interp(grid, &field.coords(X), field.locations(X), &field.values(X), interp_point, &interp_idx),
            Self::interp(grid, &field.coords(X), field.locations(Y), &field.values(X), interp_point, &interp_idx),
            Self::interp(grid, &field.coords(X), field.locations(Z), &field.values(X), interp_point, &interp_idx)
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray_stats::QuantileExt;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::field::ResampledCoordLocations;
    use crate::io::Endianness;
    use crate::io::snapshot::{fdt, SnapshotReader3};

    #[test]
    fn interpolation_at_original_data_points_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new("data/en024031_emer3.0sml_ebeam_631.idl", Endianness::Little).unwrap();
        let field = reader.read_scalar_field("r").unwrap();

        let coords = field.coords();
        let idx = 300;
        let slice_values_idx = field.slice_across_axis_at_idx(Y, idx);
        let interpolator = PolyFitInterpolator3;
        let slice_field_coord = field.slice_across_y(&interpolator, coords[Y][idx], ResampledCoordLocations::Original);
        let slice_values_coord = slice_field_coord.values();

        let rel_diffs = (slice_values_idx.to_owned() - slice_values_coord).mapv(fdt::abs)/slice_values_idx;
        assert!(*rel_diffs.max().unwrap() < 1e-6);
    }
}
