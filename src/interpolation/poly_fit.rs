//! Interpolation by polynomial fitting.

#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(unused_variables)]

use super::{Interpolator1, Interpolator2, Interpolator3};
use crate::{
    field::{ScalarField1, ScalarField2, ScalarField3, VectorField2, VectorField3},
    geometry::{
        Dim2::{self, X as X2, Y as Y2},
        Dim3::{self, X, Y, Z},
        Idx2, Idx3, In2D, In3D, Point2, Point3, Vec2, Vec3,
    },
    grid::{CoordLocation, Grid1, Grid2, Grid3, GridPointQuery1, GridPointQuery2, GridPointQuery3},
    num::BFloat,
};

macro_rules! compute_start_offset {
    ($center_coords:expr, $location:expr, $interp_coord:expr, $interp_idx:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;
        const DEFAULT_START_OFFSET: isize = 1 - ((N_POINTS + 1) as isize) / 2;

        match $location {
            CoordLocation::Center => {
                if $order % 2 != 0 && $interp_coord < $center_coords[$interp_idx] {
                    // If values are located at cell centers, interpolation order is odd
                    // and interpolation coordinate is in lower half of the cell:
                    // Shift start offset one cell down.
                    DEFAULT_START_OFFSET - 1
                } else {
                    DEFAULT_START_OFFSET
                }
            }
            CoordLocation::LowerEdge => {
                if $order % 2 == 0 && $interp_coord > $center_coords[$interp_idx] {
                    // If values are located at lower cell edges, interpolation order is even
                    // and interpolation coordinate is in upper half of the cell:
                    // Shift start offset one cell up.
                    DEFAULT_START_OFFSET + 1
                } else {
                    DEFAULT_START_OFFSET
                }
            }
        }
    }};
}

macro_rules! compute_start_indices_3d {
    ($grid:expr, $locations:expr, $interp_point:expr, $interp_indices:expr, $order:expr) => {{
        let center_coords = $grid.centers();
        let start_offset_x = compute_start_offset!(
            &center_coords[X],
            $locations[X],
            $interp_point[X],
            $interp_indices[X],
            $order
        );
        let start_offset_y = compute_start_offset!(
            &center_coords[Y],
            $locations[Y],
            $interp_point[Y],
            $interp_indices[Y],
            $order
        );
        let start_offset_z = compute_start_offset!(
            &center_coords[Z],
            $locations[Z],
            $interp_point[Z],
            $interp_indices[Z],
            $order
        );
        Idx3::new(
            ($interp_indices[X] as isize) + start_offset_x,
            ($interp_indices[Y] as isize) + start_offset_y,
            ($interp_indices[Z] as isize) + start_offset_z,
        )
    }};
}

macro_rules! compute_start_indices_2d {
    ($grid:expr, $locations:expr, $interp_point:expr, $interp_indices:expr, $order:expr) => {{
        let center_coords = $grid.centers();
        let start_offset_x = compute_start_offset!(
            &center_coords[X2],
            $locations[X2],
            $interp_point[X2],
            $interp_indices[X2],
            $order
        );
        let start_offset_y = compute_start_offset!(
            &center_coords[Y2],
            $locations[Y2],
            $interp_point[Y2],
            $interp_indices[Y2],
            $order
        );
        Idx2::new(
            ($interp_indices[X2] as isize) + start_offset_x,
            ($interp_indices[Y2] as isize) + start_offset_y,
        )
    }};
}

macro_rules! compute_start_index_1d {
    ($grid:expr, $location:expr, $interp_coord:expr, $interp_index:expr, $order:expr) => {{
        let center_coords = $grid.centers();
        let start_offset = compute_start_offset!(
            center_coords,
            $location,
            $interp_coord,
            $interp_index,
            $order
        );
        ($interp_index as isize) + start_offset
    }};
}

macro_rules! find_start_indices_and_crossings_3d {
    ($grid:expr, $locations:expr, $interp_point:expr, $interp_indices:expr, $order:expr) => {
        {
            const N_POINTS: usize = $order + 1;

            let mut start_indices = compute_start_indices_3d!(
                $grid,
                $locations,
                $interp_point,
                $interp_indices,
                $order
            );

            let grid_shape = $grid.shape();
            let mut crosses_periodic_bound = In3D::new(false, false, false);
            let mut any_crosses_periodic_bound = false;

            for &dim in &Dim3::slice() {
                // Check if start index is outside lower bound
                if start_indices[dim] < 0 {
                    if $grid.is_periodic(dim) {
                        // If the dimension is periodic, make a note of the crossing
                        crosses_periodic_bound[dim] = true;
                        any_crosses_periodic_bound = true;
                    } else {
                        // If the dimension is not periodic, shift the interpolation interval to lie on the inside
                        start_indices[dim] = 0;
                    }
                // Check the upper bound accordingly
                } else if (start_indices[dim] as usize) + N_POINTS > grid_shape[dim] {
                    if $grid.is_periodic(dim) {
                        crosses_periodic_bound[dim] = true;
                        any_crosses_periodic_bound = true;
                    } else {
                        start_indices[dim] = (grid_shape[dim] - N_POINTS) as isize;
                    }
                }
            }

            (start_indices, any_crosses_periodic_bound, crosses_periodic_bound)
        }
    };
}

macro_rules! find_start_indices_and_crossings_2d {
    ($grid:expr, $locations:expr, $interp_point:expr, $interp_indices:expr, $order:expr) => {
        {
            const N_POINTS: usize = $order + 1;

            let mut start_indices = compute_start_indices_2d!(
                $grid,
                $locations,
                $interp_point,
                $interp_indices,
                $order
            );

            let grid_shape = $grid.shape();
            let mut crosses_periodic_bound = In2D::new(false, false);
            let mut any_crosses_periodic_bound = false;

            for &dim in &Dim2::slice() {
                // Check if start index is outside lower bound
                if start_indices[dim] < 0 {
                    if $grid.is_periodic(dim) {
                        // If the dimension is periodic, make a note of the crossing
                        crosses_periodic_bound[dim] = true;
                        any_crosses_periodic_bound = true;
                    } else {
                        // If the dimension is not periodic, shift the interpolation interval to lie on the inside
                        start_indices[dim] = 0;
                    }
                // Check the upper bound accordingly
                } else if (start_indices[dim] as usize) + N_POINTS > grid_shape[dim] {
                    if $grid.is_periodic(dim) {
                        crosses_periodic_bound[dim] = true;
                        any_crosses_periodic_bound = true;
                    } else {
                        start_indices[dim] = (grid_shape[dim] - N_POINTS) as isize;
                    }
                }
            }

            (start_indices, any_crosses_periodic_bound, crosses_periodic_bound)
        }
    };
}

macro_rules! find_start_index_and_crossing_1d {
    ($grid:expr, $location:expr, $interp_coord:expr, $interp_index:expr, $order:expr) => {
        {
            const N_POINTS: usize = $order + 1;

            let mut start_index = compute_start_index_1d!(
                $grid,
                $location,
                $interp_coord,
                $interp_index,
                $order
            );

            let grid_size = $grid.size();
            let mut crosses_periodic_bound = false;

            // Check if start index is outside lower bound
            if start_index < 0 {
                if $grid.is_periodic() {
                    // If the grid is periodic, make a note of the crossing
                    crosses_periodic_bound = true;
                } else {
                    // If the grid is not periodic, shift the interpolation interval to lie on the inside
                    start_index = 0;
                }
            // Check the upper bound accordingly
            } else if (start_index as usize) + N_POINTS > grid_size {
                if $grid.is_periodic() {
                    crosses_periodic_bound = true;
                } else {
                    start_index = (grid_size - N_POINTS) as isize;
                }
            }

            (start_index, crosses_periodic_bound)
        }
    };
}

macro_rules! create_value_subarray_for_interior_3d {
    ($values:expr, $start_indices:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let mut subarray = [F::zero(); N_POINTS * N_POINTS * N_POINTS];
        let offsets = Idx3::from(&$start_indices);
        let mut idx = 0;

        let mut sum = F::zero();
        let mut sum_of_squares = F::zero();

        for k in offsets[Z]..(offsets[Z] + N_POINTS) {
            for j in offsets[Y]..(offsets[Y] + N_POINTS) {
                for i in offsets[X]..(offsets[X] + N_POINTS) {
                    let value = $values[[i, j, k]];
                    subarray[idx] = value;
                    idx += 1;

                    sum = sum + value;
                    sum_of_squares = sum_of_squares + value * value;
                }
            }
        }

        let variation = F::one()
            - (sum * sum) / (sum_of_squares * F::from(N_POINTS * N_POINTS * N_POINTS).unwrap());

        (subarray, variation)
    }};
}

macro_rules! create_value_subarray_for_interior_2d {
    ($values:expr, $start_indices:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let mut subarray = [F::zero(); N_POINTS * N_POINTS];
        let offsets = Idx2::from(&$start_indices);
        let mut idx = 0;

        let mut sum = F::zero();
        let mut sum_of_squares = F::zero();

        for j in offsets[Y2]..(offsets[Y2] + N_POINTS) {
            for i in offsets[X2]..(offsets[X2] + N_POINTS) {
                let value = $values[[i, j]];
                subarray[idx] = value;
                idx += 1;

                sum = sum + value;
                sum_of_squares = sum_of_squares + value * value;
            }
        }

        let variation =
            F::one() - (sum * sum) / (sum_of_squares * F::from(N_POINTS * N_POINTS).unwrap());

        (subarray, variation)
    }};
}

macro_rules! create_value_subarray_for_interior_1d {
    ($values:expr, $start_index:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let mut subarray = [F::zero(); N_POINTS];
        let offset = $start_index as usize;
        let mut idx = 0;

        let mut sum = F::zero();
        let mut sum_of_squares = F::zero();

        for i in offset..(offset + N_POINTS) {
            let value = $values[i];
            subarray[idx] = value;
            idx += 1;

            sum = sum + value;
            sum_of_squares = sum_of_squares + value * value;
        }

        let variation = F::one() - (sum * sum) / (sum_of_squares * F::from(N_POINTS).unwrap());

        (subarray, variation)
    }};
}

macro_rules! create_value_subarray_for_periodic_3d {
    ($values:expr, $start_indices:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let grid_shape = $values.shape();
        let offsets = In3D::new(
            ($start_indices[X] + (grid_shape[0] as isize)) as usize,
            ($start_indices[Y] + (grid_shape[1] as isize)) as usize,
            ($start_indices[Z] + (grid_shape[2] as isize)) as usize,
        );
        let mut subarray = [F::zero(); N_POINTS * N_POINTS * N_POINTS];
        let mut idx = 0;

        let mut sum = F::zero();
        let mut sum_of_squares = F::zero();

        for k in 0..N_POINTS {
            for j in 0..N_POINTS {
                for i in 0..N_POINTS {
                    let value = $values[[
                        (offsets[X] + i) % grid_shape[0],
                        (offsets[Y] + j) % grid_shape[1],
                        (offsets[Z] + k) % grid_shape[2],
                    ]];
                    subarray[idx] = value;
                    idx += 1;

                    sum = sum + value;
                    sum_of_squares = sum_of_squares + value * value;
                }
            }
        }

        let variation = F::one()
            - (sum * sum) / (sum_of_squares * F::from(N_POINTS * N_POINTS * N_POINTS).unwrap());

        (subarray, variation)
    }};
}

macro_rules! create_value_subarray_for_periodic_2d {
    ($values:expr, $start_indices:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let grid_shape = $values.shape();
        let offsets = In2D::new(
            ($start_indices[X2] + (grid_shape[0] as isize)) as usize,
            ($start_indices[Y2] + (grid_shape[1] as isize)) as usize,
        );
        let mut subarray = [F::zero(); N_POINTS * N_POINTS];
        let mut idx = 0;

        let mut sum = F::zero();
        let mut sum_of_squares = F::zero();

        for j in 0..N_POINTS {
            for i in 0..N_POINTS {
                let value = $values[[
                    (offsets[X2] + i) % grid_shape[0],
                    (offsets[Y2] + j) % grid_shape[1],
                ]];
                subarray[idx] = value;
                idx += 1;

                sum = sum + value;
                sum_of_squares = sum_of_squares + value * value;
            }
        }

        let variation =
            F::one() - (sum * sum) / (sum_of_squares * F::from(N_POINTS * N_POINTS).unwrap());

        (subarray, variation)
    }};
}

macro_rules! create_value_subarray_for_periodic_1d {
    ($values:expr, $start_index:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let grid_size = $values.len();
        let offset = ($start_index + (grid_size as isize)) as usize;
        let mut subarray = [F::zero(); N_POINTS];
        let mut idx = 0;

        let mut sum = F::zero();
        let mut sum_of_squares = F::zero();

        for i in 0..N_POINTS {
            let value = $values[(offset + i) % grid_size];
            subarray[idx] = value;
            idx += 1;

            sum = sum + value;
            sum_of_squares = sum_of_squares + value * value;
        }

        let variation = F::one() - (sum * sum) / (sum_of_squares * F::from(N_POINTS).unwrap());

        (subarray, variation)
    }};
}

macro_rules! create_value_subarray_3d {
    ($any_crosses_periodic_bound:expr, $values:expr, $start_indices:expr, $order:expr) => {
        if $any_crosses_periodic_bound {
            create_value_subarray_for_periodic_3d!($values, &$start_indices, $order)
        } else {
            create_value_subarray_for_interior_3d!($values, &$start_indices, $order)
        };
    };
}

macro_rules! create_value_subarray_2d {
    ($any_crosses_periodic_bound:expr, $values:expr, $start_indices:expr, $order:expr) => {
        if $any_crosses_periodic_bound {
            create_value_subarray_for_periodic_2d!($values, &$start_indices, $order)
        } else {
            create_value_subarray_for_interior_2d!($values, &$start_indices, $order)
        };
    };
}

macro_rules! create_value_subarray_1d {
    ($crosses_periodic_bound:expr, $values:expr, $start_index:expr, $order:expr) => {
        if $crosses_periodic_bound {
            create_value_subarray_for_periodic_1d!($values, $start_index, $order)
        } else {
            create_value_subarray_for_interior_1d!($values, $start_index, $order)
        };
    };
}

macro_rules! create_coordinate_subarray_for_interior {
    ($coords:expr, $start_idx:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let mut subarray = [F::zero(); N_POINTS];
        let offset = $start_idx as usize;
        subarray[..N_POINTS].clone_from_slice(&$coords[offset..(N_POINTS + offset)]);
        subarray
    }};
}

macro_rules! create_coordinate_subarray_for_periodic {
    ($coords:expr, $extent:expr, $start_idx:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let len = $coords.len();
        let mut subarray = [F::zero(); N_POINTS];
        if $start_idx < 0 {
            let n_points_below = (-$start_idx) as usize;
            let offset = len - n_points_below;
            for idx in 0..n_points_below {
                subarray[idx] = $coords[offset + idx] - $extent;
            }
            for idx in n_points_below..N_POINTS {
                subarray[idx] = $coords[idx - n_points_below];
            }
        } else {
            // start_idx + N_POINTS >= len
            let offset = $start_idx as usize;
            let n_points_below = len - offset;
            for idx in 0..n_points_below {
                subarray[idx] = $coords[offset + idx];
            }
            for idx in n_points_below..N_POINTS {
                subarray[idx] = $coords[idx - n_points_below] + $extent;
            }
        }
        subarray
    }};
}

macro_rules! create_coordinate_subarrays_3d {
    ($crosses_periodic_bound:expr, $coords:expr, $extents:expr, $start_indices:expr, $order:expr) => {{
        let x_coord_subarray = if $crosses_periodic_bound[X] {
            create_coordinate_subarray_for_periodic!(
                $coords[X],
                $extents[X],
                $start_indices[X],
                $order
            )
        } else {
            create_coordinate_subarray_for_interior!($coords[X], $start_indices[X], $order)
        };
        let y_coord_subarray = if $crosses_periodic_bound[Y] {
            create_coordinate_subarray_for_periodic!(
                $coords[Y],
                $extents[Y],
                $start_indices[Y],
                $order
            )
        } else {
            create_coordinate_subarray_for_interior!($coords[Y], $start_indices[Y], $order)
        };
        let z_coord_subarray = if $crosses_periodic_bound[Z] {
            create_coordinate_subarray_for_periodic!(
                $coords[Z],
                $extents[Z],
                $start_indices[Z],
                $order
            )
        } else {
            create_coordinate_subarray_for_interior!($coords[Z], $start_indices[Z], $order)
        };
        (x_coord_subarray, y_coord_subarray, z_coord_subarray)
    }};
}

macro_rules! create_coordinate_subarrays_2d {
    ($crosses_periodic_bound:expr, $coords:expr, $extents:expr, $start_indices:expr, $order:expr) => {{
        let x_coord_subarray = if $crosses_periodic_bound[X2] {
            create_coordinate_subarray_for_periodic!(
                $coords[X2],
                $extents[X2],
                $start_indices[X2],
                $order
            )
        } else {
            create_coordinate_subarray_for_interior!($coords[X2], $start_indices[X2], $order)
        };
        let y_coord_subarray = if $crosses_periodic_bound[Y2] {
            create_coordinate_subarray_for_periodic!(
                $coords[Y2],
                $extents[Y2],
                $start_indices[Y2],
                $order
            )
        } else {
            create_coordinate_subarray_for_interior!($coords[Y2], $start_indices[Y2], $order)
        };
        (x_coord_subarray, y_coord_subarray)
    }};
}

macro_rules! create_coordinate_subarray_1d {
    ($crosses_periodic_bound:expr, $coords:expr, $extent:expr, $start_index:expr, $order:expr) => {{
        if $crosses_periodic_bound {
            create_coordinate_subarray_for_periodic!($coords, $extent, $start_index, $order)
        } else {
            create_coordinate_subarray_for_interior!($coords, $start_index, $order)
        }
    }};
}

macro_rules! interp_subarrays_3d {
    ($coords:expr, $values:expr, $interp_point:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let x_coords = $coords[X];
        let y_coords = $coords[Y];
        let z_coords = $coords[Z];

        let mut vals_c = [F::zero(); N_POINTS];
        let mut vals_d = [F::zero(); N_POINTS];
        let mut poly_x = [F::zero(); N_POINTS * N_POINTS];
        let mut poly_xy = [F::zero(); N_POINTS];
        let mut poly_xyz;
        let mut accum;
        let mut correction;

        for k in 0..N_POINTS {
            for j in 0..N_POINTS {
                vals_c.copy_from_slice(
                    &$values[(k * N_POINTS + j) * N_POINTS..(k * N_POINTS + j + 1) * N_POINTS],
                );
                vals_d.copy_from_slice(&vals_c);

                accum = vals_c[0];

                for n in 1..N_POINTS {
                    for i in 0..(N_POINTS - n) {
                        correction = (vals_c[i + 1] - vals_d[i]) / (x_coords[i + n] - x_coords[i]);
                        vals_c[i] = ($interp_point[X] - x_coords[i]) * correction;
                        vals_d[i] = ($interp_point[X] - x_coords[i + n]) * correction;
                    }

                    accum = accum + vals_c[0];
                }

                poly_x[k * N_POINTS + j] = accum;
            }
        }

        for k in 0..N_POINTS {
            vals_c.copy_from_slice(&poly_x[k * N_POINTS..(k + 1) * N_POINTS]);
            vals_d.copy_from_slice(&vals_c);

            accum = vals_c[0];

            for n in 1..N_POINTS {
                for j in 0..(N_POINTS - n) {
                    correction = (vals_c[j + 1] - vals_d[j]) / (y_coords[j + n] - y_coords[j]);
                    vals_c[j] = ($interp_point[Y] - y_coords[j]) * correction;
                    vals_d[j] = ($interp_point[Y] - y_coords[j + n]) * correction;
                }

                accum = accum + vals_c[0];
            }

            poly_xy[k] = accum;
        }

        vals_c.copy_from_slice(&poly_xy);
        vals_d.copy_from_slice(&vals_c);

        poly_xyz = vals_c[0];

        for n in 1..N_POINTS {
            for k in 0..(N_POINTS - n) {
                correction = (vals_c[k + 1] - vals_d[k]) / (z_coords[k + n] - z_coords[k]);
                vals_c[k] = ($interp_point[Z] - z_coords[k]) * correction;
                vals_d[k] = ($interp_point[Z] - z_coords[k + n]) * correction;
            }

            poly_xyz = poly_xyz + vals_c[0];
        }

        poly_xyz
    }};
}

macro_rules! interp_subarrays_2d {
    ($coords:expr, $values:expr, $interp_point:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let x_coords = $coords[X2];
        let y_coords = $coords[Y2];

        let mut vals_c = [F::zero(); N_POINTS];
        let mut vals_d = [F::zero(); N_POINTS];
        let mut poly_x = [F::zero(); N_POINTS];
        let mut poly_xy;
        let mut accum;
        let mut correction;

        for j in 0..N_POINTS {
            vals_c.copy_from_slice(&$values[j * N_POINTS..(j + 1) * N_POINTS]);
            vals_d.copy_from_slice(&vals_c);

            accum = vals_c[0];

            for n in 1..N_POINTS {
                for i in 0..(N_POINTS - n) {
                    correction = (vals_c[i + 1] - vals_d[i]) / (x_coords[i + n] - x_coords[i]);
                    vals_c[i] = ($interp_point[X2] - x_coords[i]) * correction;
                    vals_d[i] = ($interp_point[X2] - x_coords[i + n]) * correction;
                }

                accum = accum + vals_c[0];
            }

            poly_x[j] = accum;
        }

        vals_c.copy_from_slice(&poly_x);
        vals_d.copy_from_slice(&vals_c);

        poly_xy = vals_c[0];

        for n in 1..N_POINTS {
            for j in 0..(N_POINTS - n) {
                correction = (vals_c[j + 1] - vals_d[j]) / (y_coords[j + n] - y_coords[j]);
                vals_c[j] = ($interp_point[Y2] - y_coords[j]) * correction;
                vals_d[j] = ($interp_point[Y2] - y_coords[j + n]) * correction;
            }

            poly_xy = poly_xy + vals_c[0];
        }
        poly_xy
    }};
}

macro_rules! interp_subarray_1d {
    ($coords:expr, $values:expr, $interp_coord:expr, $order:expr) => {{
        const N_POINTS: usize = $order + 1;

        let mut vals_c = [F::zero(); N_POINTS];
        let mut vals_d = [F::zero(); N_POINTS];
        let mut poly;
        let mut correction;

        vals_c.copy_from_slice($values);
        vals_d.copy_from_slice(&vals_c);

        poly = vals_c[0];

        for n in 1..N_POINTS {
            for i in 0..(N_POINTS - n) {
                correction = (vals_c[i + 1] - vals_d[i]) / ($coords[i + n] - $coords[i]);
                vals_c[i] = ($interp_coord - $coords[i]) * correction;
                vals_d[i] = ($interp_coord - $coords[i + n]) * correction;
            }

            poly = poly + vals_c[0];
        }
        poly
    }};
}

macro_rules! interp_3d {
    (
        $grid:expr,
        $coords:expr,
        $locations:expr,
        $values:expr,
        $interp_point:expr,
        $interp_indices:expr,
        $variation_threshold_for_linear:expr,
        $order:expr
     ) => {
        {
            let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                find_start_indices_and_crossings_3d!(
                    $grid,
                    $locations,
                    $interp_point,
                    $interp_indices,
                    $order
                );

            let (value_subarray, variation) = create_value_subarray_3d!(
                any_crosses_periodic_bound,
                $values,
                &start_indices,
                $order
            );

            // If the variation exceeds the given threshold, use linear interpolation
            // in order to avoid overshoot.
            if variation > $variation_threshold_for_linear && $order > 1 {

                let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                    find_start_indices_and_crossings_3d!(
                        $grid,
                        $locations,
                        $interp_point,
                        $interp_indices,
                        1
                    );

                let (value_subarray, _) = create_value_subarray_3d!(
                    any_crosses_periodic_bound,
                    $values,
                    &start_indices,
                    1
                );

                let (x_coord_subarray, y_coord_subarray, z_coord_subarray) =
                    create_coordinate_subarrays_3d!(
                        crosses_periodic_bound,
                        $coords,
                        $grid.extents(),
                        start_indices,
                        1
                    );

                interp_subarrays_3d!(
                    In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray),
                    &value_subarray,
                    $interp_point,
                    1
                )

            } else {

                let (x_coord_subarray, y_coord_subarray, z_coord_subarray) =
                    create_coordinate_subarrays_3d!(
                        crosses_periodic_bound,
                        $coords,
                        $grid.extents(),
                        start_indices,
                        $order
                    );

                interp_subarrays_3d!(
                    In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray),
                    &value_subarray,
                    $interp_point,
                    $order
                )
            }
        }
    };
}

macro_rules! interp_2d {
    (
        $grid:expr,
        $coords:expr,
        $locations:expr,
        $values:expr,
        $interp_point:expr,
        $interp_indices:expr,
        $variation_threshold_for_linear:expr,
        $order:expr
     ) => {
        {
            let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                find_start_indices_and_crossings_2d!(
                    $grid,
                    $locations,
                    $interp_point,
                    $interp_indices,
                    $order
                );

            let (value_subarray, variation) = create_value_subarray_2d!(
                any_crosses_periodic_bound,
                $values,
                &start_indices,
                $order
            );

            // If the variation exceeds the given threshold, use linear interpolation
            // in order to avoid overshoot.
            if variation > $variation_threshold_for_linear && $order > 1 {

                let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                    find_start_indices_and_crossings_2d!(
                        $grid,
                        $locations,
                        $interp_point,
                        $interp_indices,
                        1
                    );

                let (value_subarray, _) = create_value_subarray_2d!(
                    any_crosses_periodic_bound,
                    $values,
                    &start_indices,
                    1
                );

                let (x_coord_subarray, y_coord_subarray) =
                    create_coordinate_subarrays_2d!(
                        crosses_periodic_bound,
                        $coords,
                        $grid.extents(),
                        start_indices,
                        1
                    );

                interp_subarrays_2d!(
                    In2D::new(&x_coord_subarray, &y_coord_subarray),
                    &value_subarray,
                    $interp_point,
                    1
                )

            } else {

                let (x_coord_subarray, y_coord_subarray) =
                    create_coordinate_subarrays_2d!(
                        crosses_periodic_bound,
                        $coords,
                        $grid.extents(),
                        start_indices,
                        $order
                    );

                interp_subarrays_2d!(
                    In2D::new(&x_coord_subarray, &y_coord_subarray),
                    &value_subarray,
                    $interp_point,
                    $order
                )
            }
        }
    };
}

macro_rules! interp_1d {
    (
        $grid:expr,
        $coords:expr,
        $location:expr,
        $values:expr,
        $interp_coord:expr,
        $interp_index:expr,
        $variation_threshold_for_linear:expr,
        $order:expr
     ) => {{
        let (start_index, crosses_periodic_bound) = find_start_index_and_crossing_1d!(
            $grid,
            $location,
            $interp_coord,
            $interp_index,
            $order
        );

        let (value_subarray, variation) =
            create_value_subarray_1d!(crosses_periodic_bound, $values, start_index, $order);

        // If the variation exceeds the given threshold, use linear interpolation
        // in order to avoid overshoot.
        if variation > $variation_threshold_for_linear && $order > 1 {
            let (start_index, crosses_periodic_bound) = find_start_index_and_crossing_1d!(
                $grid,
                $location,
                $interp_coord,
                $interp_index,
                1
            );

            let (value_subarray, _) =
                create_value_subarray_1d!(crosses_periodic_bound, $values, start_index, 1);

            let coord_subarray = create_coordinate_subarray_1d!(
                crosses_periodic_bound,
                $coords,
                $grid.extent(),
                start_index,
                1
            );

            interp_subarray_1d!(&coord_subarray, &value_subarray, $interp_coord, 1)
        } else {
            let coord_subarray = create_coordinate_subarray_1d!(
                crosses_periodic_bound,
                $coords,
                $grid.extent(),
                start_index,
                $order
            );

            interp_subarray_1d!(&coord_subarray, &value_subarray, $interp_coord, $order)
        }
    }};
}

macro_rules! interp_scalar_field_from_grid_point_query_3d {
    ($field:expr, $grid_point_query:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        match $grid_point_query {
            GridPointQuery3::Inside(interp_indices) => {
                GridPointQuery3::Inside(interp_scalar_field_in_known_grid_cell_3d!(
                    $field,
                    $interp_point,
                    &interp_indices,
                    $variation_threshold_for_linear,
                    $order
                ))
            }
            GridPointQuery3::MovedInside((interp_indices, moved_point)) => {
                GridPointQuery3::MovedInside((
                    interp_scalar_field_in_known_grid_cell_3d!(
                        $field,
                        &moved_point,
                        &interp_indices,
                        $variation_threshold_for_linear,
                        $order
                    ),
                    moved_point,
                ))
            }
            GridPointQuery3::Outside => GridPointQuery3::Outside,
        }
    }};
}

macro_rules! interp_scalar_field_from_grid_point_query_2d {
    ($field:expr, $grid_point_query:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        match $grid_point_query {
            GridPointQuery2::Inside(interp_indices) => {
                GridPointQuery2::Inside(interp_scalar_field_in_known_grid_cell_2d!(
                    $field,
                    $interp_point,
                    &interp_indices,
                    $variation_threshold_for_linear,
                    $order
                ))
            }
            GridPointQuery2::MovedInside((interp_indices, moved_point)) => {
                GridPointQuery2::MovedInside((
                    interp_scalar_field_in_known_grid_cell_2d!(
                        $field,
                        &moved_point,
                        &interp_indices,
                        $variation_threshold_for_linear,
                        $order
                    ),
                    moved_point,
                ))
            }
            GridPointQuery2::Outside => GridPointQuery2::Outside,
        }
    }};
}

macro_rules! interp_scalar_field_from_grid_point_query_1d {
    ($field:expr, $grid_point_query:expr, $interp_coord:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        match $grid_point_query {
            GridPointQuery1::Inside(interp_index) => {
                GridPointQuery1::Inside(interp_scalar_field_in_known_grid_cell_1d!(
                    $field,
                    $interp_coord,
                    interp_index,
                    $variation_threshold_for_linear,
                    $order
                ))
            }
            GridPointQuery1::MovedInside((interp_index, moved_coord)) => {
                GridPointQuery1::MovedInside((
                    interp_scalar_field_in_known_grid_cell_1d!(
                        $field,
                        moved_coord,
                        interp_index,
                        $variation_threshold_for_linear,
                        $order
                    ),
                    moved_coord,
                ))
            }
            GridPointQuery1::Outside => GridPointQuery1::Outside,
        }
    }};
}

macro_rules! interp_scalar_field_in_known_grid_cell_3d {
    ($field:expr, $interp_point:expr, $interp_indices:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        interp_3d!(
            $field.grid(),
            &$field.coords(),
            $field.locations(),
            $field.values(),
            $interp_point,
            $interp_indices,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_scalar_field_in_known_grid_cell_2d {
    ($field:expr, $interp_point:expr, $interp_indices:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        interp_2d!(
            $field.grid(),
            &$field.coords(),
            $field.locations(),
            $field.values(),
            $interp_point,
            $interp_indices,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_scalar_field_in_known_grid_cell_1d {
    ($field:expr, $interp_coord:expr, $interp_index:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        interp_1d!(
            $field.grid(),
            $field.coords(),
            $field.location(),
            $field.values(),
            $interp_coord,
            $interp_index,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_vector_field_from_grid_point_query_3d {
    ($field:expr, $grid_point_query:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        match $grid_point_query {
            GridPointQuery3::Inside(interp_indices) => {
                GridPointQuery3::Inside(interp_vector_field_in_known_grid_cell_3d!(
                    $field,
                    $interp_point,
                    &interp_indices,
                    $variation_threshold_for_linear,
                    $order
                ))
            }
            GridPointQuery3::MovedInside((interp_indices, moved_point)) => {
                GridPointQuery3::MovedInside((
                    interp_vector_field_in_known_grid_cell_3d!(
                        $field,
                        &moved_point,
                        &interp_indices,
                        $variation_threshold_for_linear,
                        $order
                    ),
                    moved_point,
                ))
            }
            GridPointQuery3::Outside => GridPointQuery3::Outside,
        }
    }};
}

macro_rules! interp_vector_field_from_grid_point_query_2d {
    ($field:expr, $grid_point_query:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        match $grid_point_query {
            GridPointQuery2::Inside(interp_indices) => {
                GridPointQuery2::Inside(interp_vector_field_in_known_grid_cell_2d!(
                    $field,
                    $interp_point,
                    &interp_indices,
                    $variation_threshold_for_linear,
                    $order
                ))
            }
            GridPointQuery2::MovedInside((interp_indices, moved_point)) => {
                GridPointQuery2::MovedInside((
                    interp_vector_field_in_known_grid_cell_2d!(
                        $field,
                        &moved_point,
                        &interp_indices,
                        $variation_threshold_for_linear,
                        $order
                    ),
                    moved_point,
                ))
            }
            GridPointQuery2::Outside => GridPointQuery2::Outside,
        }
    }};
}

macro_rules! interp_vector_field_in_known_grid_cell_3d {
    ($field:expr, $interp_point:expr, $interp_indices:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid = $field.grid();
        Vec3::new(
            interp_3d!(
                grid,
                &$field.coords(X),
                $field.locations(X),
                &$field.values(X),
                $interp_point,
                $interp_indices,
                $variation_threshold_for_linear,
                $order
            ),
            interp_3d!(
                grid,
                &$field.coords(Y),
                $field.locations(Y),
                &$field.values(Y),
                $interp_point,
                $interp_indices,
                $variation_threshold_for_linear,
                $order
            ),
            interp_3d!(
                grid,
                &$field.coords(Z),
                $field.locations(Z),
                &$field.values(Z),
                $interp_point,
                $interp_indices,
                $variation_threshold_for_linear,
                $order
            ),
        )
    }};
}

macro_rules! interp_vector_field_in_known_grid_cell_2d {
    ($field:expr, $interp_point:expr, $interp_indices:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid = $field.grid();
        Vec2::new(
            interp_2d!(
                grid,
                &$field.coords(X2),
                $field.locations(X2),
                &$field.values(X2),
                $interp_point,
                $interp_indices,
                $variation_threshold_for_linear,
                $order
            ),
            interp_2d!(
                grid,
                &$field.coords(Y2),
                $field.locations(Y2),
                &$field.values(Y2),
                $interp_point,
                $interp_indices,
                $variation_threshold_for_linear,
                $order
            ),
        )
    }};
}

macro_rules! interp_scalar_field_3d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_grid_cell($interp_point);
        interp_scalar_field_from_grid_point_query_3d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_scalar_field_2d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_grid_cell($interp_point);
        interp_scalar_field_from_grid_point_query_2d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_scalar_field_1d {
    ($field:expr, $interp_coord:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_grid_cell($interp_coord);
        interp_scalar_field_from_grid_point_query_1d!(
            $field,
            grid_point_query,
            $interp_coord,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_extrap_scalar_field_3d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_closest_grid_cell($interp_point);
        interp_scalar_field_from_grid_point_query_3d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_extrap_scalar_field_2d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_closest_grid_cell($interp_point);
        interp_scalar_field_from_grid_point_query_2d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_extrap_scalar_field_1d {
    ($field:expr, $interp_coord:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_closest_grid_cell($interp_coord);
        interp_scalar_field_from_grid_point_query_1d!(
            $field,
            grid_point_query,
            $interp_coord,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_vector_field_3d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_grid_cell($interp_point);
        interp_vector_field_from_grid_point_query_3d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_vector_field_2d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_grid_cell($interp_point);
        interp_vector_field_from_grid_point_query_2d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_extrap_vector_field_3d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_closest_grid_cell($interp_point);
        interp_vector_field_from_grid_point_query_3d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

macro_rules! interp_extrap_vector_field_2d {
    ($field:expr, $interp_point:expr, $variation_threshold_for_linear:expr, $order:expr) => {{
        let grid_point_query = $field.grid().find_closest_grid_cell($interp_point);
        interp_vector_field_from_grid_point_query_2d!(
            $field,
            grid_point_query,
            $interp_point,
            $variation_threshold_for_linear,
            $order
        )
    }};
}

/// Configuration parameters for polynomial fitting interpolators.
#[derive(Clone, Debug)]
pub struct PolyFitInterpolatorConfig {
    /// Order of the polynomials to fit.
    pub order: usize,
    /// Linear interpolation will be used when a normalized variance of
    /// the values surrounding the interpolation point exceeds this value.
    ///
    /// This can help to avoid overshoot in regions with strong gradients.
    pub variation_threshold_for_linear: f64,
}

/// A 3D interpolator using polynomial fitting to estimate the interpolated value.
#[derive(Clone, Debug)]
pub struct PolyFitInterpolator3 {
    config: PolyFitInterpolatorConfig,
}

impl PolyFitInterpolator3 {
    /// Creates a new polynomial fitting interpolator.
    pub fn new(config: PolyFitInterpolatorConfig) -> Self {
        config.validate();
        Self { config }
    }
}

impl Interpolator3 for PolyFitInterpolator3 {
    #[allow(clippy::cognitive_complexity)]
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, F>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_3d!(field, interp_point, variation_threshold_for_linear, 1),
            2 => interp_scalar_field_3d!(field, interp_point, variation_threshold_for_linear, 2),
            3 => interp_scalar_field_3d!(field, interp_point, variation_threshold_for_linear, 3),
            4 => interp_scalar_field_3d!(field, interp_point, variation_threshold_for_linear, 4),
            5 => interp_scalar_field_3d!(field, interp_point, variation_threshold_for_linear, 5),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_scalar_field_known_cell<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<F>,
        interp_indices: &Idx3<usize>,
    ) -> F
    where
        F: BFloat,
        G: Grid3<F>,
    {
        assert!(field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices));

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_in_known_grid_cell_3d!(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
                1
            ),
            2 => interp_scalar_field_in_known_grid_cell_3d!(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
                2
            ),
            3 => interp_scalar_field_in_known_grid_cell_3d!(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
                3
            ),
            4 => interp_scalar_field_in_known_grid_cell_3d!(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
                4
            ),
            5 => interp_scalar_field_in_known_grid_cell_3d!(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
                5
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, F>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_extrap_scalar_field_3d!(
                field,
                interp_point,
                variation_threshold_for_linear,
                1
            ),
            2 => interp_extrap_scalar_field_3d!(
                field,
                interp_point,
                variation_threshold_for_linear,
                2
            ),
            3 => interp_extrap_scalar_field_3d!(
                field,
                interp_point,
                variation_threshold_for_linear,
                3
            ),
            4 => interp_extrap_scalar_field_3d!(
                field,
                interp_point,
                variation_threshold_for_linear,
                4
            ),
            5 => interp_extrap_scalar_field_3d!(
                field,
                interp_point,
                variation_threshold_for_linear,
                5
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_vector_field<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, Vec3<F>>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_vector_field_3d!(field, interp_point, variation_threshold_for_linear, 1),
            2 => interp_vector_field_3d!(field, interp_point, variation_threshold_for_linear, 2),
            3 => interp_vector_field_3d!(field, interp_point, variation_threshold_for_linear, 3),
            4 => interp_vector_field_3d!(field, interp_point, variation_threshold_for_linear, 4),
            5 => interp_vector_field_3d!(field, interp_point, variation_threshold_for_linear, 5),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_vector_field_known_cell<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<F>,
        interp_indices: &Idx3<usize>,
    ) -> Vec3<F>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        assert!(field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices));

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_vector_field_in_known_grid_cell_3d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_vector_field_in_known_grid_cell_3d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_vector_field_in_known_grid_cell_3d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_vector_field_in_known_grid_cell_3d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_vector_field_in_known_grid_cell_3d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_extrap_vector_field<F, G>(
        &self,
        field: &VectorField3<F, G>,
        interp_point: &Point3<F>,
    ) -> GridPointQuery3<F, Vec3<F>>
    where
        F: BFloat,
        G: Grid3<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_extrap_vector_field_3d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_extrap_vector_field_3d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_extrap_vector_field_3d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_extrap_vector_field_3d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_extrap_vector_field_3d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }
}

/// A 2D interpolator using polynomial fitting to estimate the interpolated value.
#[derive(Clone, Debug)]
pub struct PolyFitInterpolator2 {
    config: PolyFitInterpolatorConfig,
}

impl PolyFitInterpolator2 {
    /// Creates a new polynomial fitting interpolator.
    pub fn new(config: PolyFitInterpolatorConfig) -> Self {
        config.validate();
        Self { config }
    }
}

impl Interpolator2 for PolyFitInterpolator2 {
    #[allow(clippy::cognitive_complexity)]
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField2<F, G>,
        interp_point: &Point2<F>,
    ) -> GridPointQuery2<F, F>
    where
        F: BFloat,
        G: Grid2<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_scalar_field_2d!(field, interp_point, variation_threshold_for_linear, 1),
            // 2 => interp_scalar_field_2d!(field, interp_point, variation_threshold_for_linear, 2),
            // 3 => interp_scalar_field_2d!(field, interp_point, variation_threshold_for_linear, 3),
            // 4 => interp_scalar_field_2d!(field, interp_point, variation_threshold_for_linear, 4),
            // 5 => interp_scalar_field_2d!(field, interp_point, variation_threshold_for_linear, 5),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_scalar_field_known_cell<F, G>(
        &self,
        field: &ScalarField2<F, G>,
        interp_point: &Point2<F>,
        interp_indices: &Idx2<usize>,
    ) -> F
    where
        F: BFloat,
        G: Grid2<F>,
    {
        assert!(field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices));

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_scalar_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_scalar_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_scalar_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_scalar_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_scalar_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField2<F, G>,
        interp_point: &Point2<F>,
    ) -> GridPointQuery2<F, F>
    where
        F: BFloat,
        G: Grid2<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_extrap_scalar_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_extrap_scalar_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_extrap_scalar_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_extrap_scalar_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_extrap_scalar_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_vector_field<F, G>(
        &self,
        field: &VectorField2<F, G>,
        interp_point: &Point2<F>,
    ) -> GridPointQuery2<F, Vec2<F>>
    where
        F: BFloat,
        G: Grid2<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_vector_field_2d!(field, interp_point, variation_threshold_for_linear, 1),
            // 2 => interp_vector_field_2d!(field, interp_point, variation_threshold_for_linear, 2),
            // 3 => interp_vector_field_2d!(field, interp_point, variation_threshold_for_linear, 3),
            // 4 => interp_vector_field_2d!(field, interp_point, variation_threshold_for_linear, 4),
            // 5 => interp_vector_field_2d!(field, interp_point, variation_threshold_for_linear, 5),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_vector_field_known_cell<F, G>(
        &self,
        field: &VectorField2<F, G>,
        interp_point: &Point2<F>,
        interp_indices: &Idx2<usize>,
    ) -> Vec2<F>
    where
        F: BFloat,
        G: Grid2<F>,
    {
        assert!(field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices));

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_vector_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_vector_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_vector_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_vector_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_vector_field_in_known_grid_cell_2d!(
            //     field,
            //     interp_point,
            //     interp_indices,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_extrap_vector_field<F, G>(
        &self,
        field: &VectorField2<F, G>,
        interp_point: &Point2<F>,
    ) -> GridPointQuery2<F, Vec2<F>>
    where
        F: BFloat,
        G: Grid2<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_extrap_vector_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_extrap_vector_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_extrap_vector_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_extrap_vector_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_extrap_vector_field_2d!(
            //     field,
            //     interp_point,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }
}

/// A 1D interpolator using polynomial fitting to estimate the interpolated value.
#[derive(Clone, Debug)]
pub struct PolyFitInterpolator1 {
    config: PolyFitInterpolatorConfig,
}

impl PolyFitInterpolator1 {
    /// Creates a new polynomial fitting interpolator.
    pub fn new(config: PolyFitInterpolatorConfig) -> Self {
        config.validate();
        Self { config }
    }
}

impl Interpolator1 for PolyFitInterpolator1 {
    #[allow(clippy::cognitive_complexity)]
    fn interp_scalar_field<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
    ) -> GridPointQuery1<F, F>
    where
        F: BFloat,
        G: Grid1<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_scalar_field_1d!(field, interp_coord, variation_threshold_for_linear, 1),
            // 2 => interp_scalar_field_1d!(field, interp_coord, variation_threshold_for_linear, 2),
            // 3 => interp_scalar_field_1d!(field, interp_coord, variation_threshold_for_linear, 3),
            // 4 => interp_scalar_field_1d!(field, interp_coord, variation_threshold_for_linear, 4),
            // 5 => interp_scalar_field_1d!(field, interp_coord, variation_threshold_for_linear, 5),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_scalar_field_known_cell<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
        interp_index: usize,
    ) -> F
    where
        F: BFloat,
        G: Grid1<F>,
    {
        assert!(field
            .grid()
            .coord_is_inside_cell(interp_coord, interp_index));

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_scalar_field_in_known_grid_cell_1d!(
            //     field,
            //     interp_coord,
            //     interp_index,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_scalar_field_in_known_grid_cell_1d!(
            //     field,
            //     interp_coord,
            //     interp_index,
            //     variation_threshold_for_linear,
            //     2
            // ),
            // 3 => interp_scalar_field_in_known_grid_cell_1d!(
            //     field,
            //     interp_coord,
            //     interp_index,
            //     variation_threshold_for_linear,
            //     3
            // ),
            // 4 => interp_scalar_field_in_known_grid_cell_1d!(
            //     field,
            //     interp_coord,
            //     interp_index,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_scalar_field_in_known_grid_cell_1d!(
            //     field,
            //     interp_coord,
            //     interp_index,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn interp_extrap_scalar_field<F, G>(
        &self,
        field: &ScalarField1<F, G>,
        interp_coord: F,
    ) -> GridPointQuery1<F, F>
    where
        F: BFloat,
        G: Grid1<F>,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            // 1 => interp_extrap_scalar_field_1d!(
            //     field,
            //     interp_coord,
            //     variation_threshold_for_linear,
            //     1
            // ),
            // 2 => interp_extrap_scalar_field_1d!(
            //     field,
            //     interp_coord,
            //     variation_threshold_for_linear,
            //     2
            // ),
            3 => interp_extrap_scalar_field_1d!(
                field,
                interp_coord,
                variation_threshold_for_linear,
                3
            ),
            // 4 => interp_extrap_scalar_field_1d!(
            //     field,
            //     interp_coord,
            //     variation_threshold_for_linear,
            //     4
            // ),
            // 5 => interp_extrap_scalar_field_1d!(
            //     field,
            //     interp_coord,
            //     variation_threshold_for_linear,
            //     5
            // ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }
}

impl PolyFitInterpolatorConfig {
    pub const DEFAULT_ORDER: usize = 3;
    pub const DEFAULT_VARIATION_THRESHOLD_FOR_LINEAR: f64 = 0.3;

    /// Panics if any of the configuration parameter values are invalid.
    fn validate(&self) {
        assert!(
            self.order >= 1 && self.order <= 5,
            "Order must be in the range [1, 5]."
        );
        assert!(
            self.variation_threshold_for_linear >= 0.0,
            "Variation threshold for linear interpolation must be larger than or equal to zero."
        );
    }
}

impl Default for PolyFitInterpolatorConfig {
    fn default() -> Self {
        PolyFitInterpolatorConfig {
            order: Self::DEFAULT_ORDER,
            variation_threshold_for_linear: Self::DEFAULT_VARIATION_THRESHOLD_FOR_LINEAR,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::field::ResampledCoordLocation;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::io::snapshot::{
        fdt,
        native::{NativeSnapshotReader3, NativeSnapshotReaderConfig},
        SnapshotReader3,
    };
    use crate::io::{Endianness, Verbose};
    use ndarray_stats::QuantileExt;

    #[test]
    fn interpolation_at_original_data_points_works() {
        let reader =
            NativeSnapshotReader3::<HorRegularGrid3<_>>::new(NativeSnapshotReaderConfig::new(
                "data/en024031_emer3.0sml_ebeam_631.idl",
                Endianness::Little,
                Verbose::No,
            ))
            .unwrap();
        let field = reader.read_scalar_field("r").unwrap();

        let coords = field.coords();
        let idx = 300;
        let slice_values_idx = field.slice_across_axis_at_idx(Y, idx);
        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());
        let slice_field_coord = field.slice_across_y(
            &interpolator,
            coords[Y][idx],
            ResampledCoordLocation::Original,
        );
        let slice_values_coord = slice_field_coord.values();

        let rel_diffs =
            (slice_values_idx.to_owned() - slice_values_coord).mapv(fdt::abs) / slice_values_idx;
        assert!(*rel_diffs.max().unwrap() < 1e-6);
    }
}
