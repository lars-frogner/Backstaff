//! Interpolation by polynomial fitting.

use super::{fip, InterpGridVerifier3, Interpolator1, Interpolator2, Interpolator3};
use crate::{
    field::{
        FieldGrid1, FieldGrid2, FieldGrid3, ScalarField1, ScalarField2, ScalarField3, VectorField2,
        VectorField3,
    },
    geometry::{
        CoordRefs2, CoordRefs3,
        Dim2::{self, X as X2, Y as Y2},
        Dim3::{self, X, Y, Z},
        Idx2, Idx3, In2D, In3D, Point2, Point3, Vec2, Vec3,
    },
    grid::{
        fgr, CoordLocation, Grid1, Grid2, Grid3, GridPointQuery1, GridPointQuery2, GridPointQuery3,
    },
    num::BFloat,
};
use ndarray::prelude::*;
use std::mem::MaybeUninit;

fn compute_start_offset<const N_POINTS: usize>(
    center_coords: &[fgr],
    location: CoordLocation,
    interp_coord: fgr,
    interp_idx: usize,
) -> isize {
    let default_start_offset = 1 - ((N_POINTS + 1) as isize) / 2;

    match location {
        CoordLocation::Center => {
            if (N_POINTS - 1) % 2 != 0 && interp_coord < center_coords[interp_idx] {
                // If coordinates are located at cell centers, interpolation order is odd
                // and interpolation coordinate is in lower half of the cell:
                // Shift start offset one cell down.
                default_start_offset - 1
            } else {
                default_start_offset
            }
        }
        CoordLocation::LowerEdge => {
            if (N_POINTS - 1) % 2 == 0 && interp_coord > center_coords[interp_idx] {
                // If coordinates are located at lower cell edges, interpolation order is even
                // and interpolation coordinate is in upper half of the cell:
                // Shift start offset one cell up.
                default_start_offset + 1
            } else {
                default_start_offset
            }
        }
    }
}

fn compute_start_indices_3d<const N_POINTS: usize>(
    grid: &FieldGrid3,
    locations: &In3D<CoordLocation>,
    interp_point: &Point3<fgr>,
    interp_indices: &Idx3<usize>,
) -> Idx3<isize> {
    let center_coords = grid.centers();
    let start_offset_x = compute_start_offset::<N_POINTS>(
        &center_coords[X],
        locations[X],
        interp_point[X],
        interp_indices[X],
    );
    let start_offset_y = compute_start_offset::<N_POINTS>(
        &center_coords[Y],
        locations[Y],
        interp_point[Y],
        interp_indices[Y],
    );
    let start_offset_z = compute_start_offset::<N_POINTS>(
        &center_coords[Z],
        locations[Z],
        interp_point[Z],
        interp_indices[Z],
    );
    Idx3::new(
        (interp_indices[X] as isize) + start_offset_x,
        (interp_indices[Y] as isize) + start_offset_y,
        (interp_indices[Z] as isize) + start_offset_z,
    )
}

fn compute_start_indices_2d<const N_POINTS: usize>(
    grid: &FieldGrid2,
    locations: &In2D<CoordLocation>,
    interp_point: &Point2<fgr>,
    interp_indices: &Idx2<usize>,
) -> Idx2<isize> {
    let center_coords = grid.centers();
    let start_offset_x = compute_start_offset::<N_POINTS>(
        &center_coords[X2],
        locations[X2],
        interp_point[X2],
        interp_indices[X2],
    );
    let start_offset_y = compute_start_offset::<N_POINTS>(
        &center_coords[Y2],
        locations[Y2],
        interp_point[Y2],
        interp_indices[Y2],
    );
    Idx2::new(
        (interp_indices[X2] as isize) + start_offset_x,
        (interp_indices[Y2] as isize) + start_offset_y,
    )
}

fn compute_start_idx_1d<const N_POINTS: usize>(
    grid: &FieldGrid1,
    location: CoordLocation,
    interp_coord: fgr,
    interp_idx: usize,
) -> isize {
    let center_coords = grid.centers();
    let start_offset =
        compute_start_offset::<N_POINTS>(center_coords, location, interp_coord, interp_idx);
    (interp_idx as isize) + start_offset
}

fn find_start_indices_and_crossings_3d<const N_POINTS: usize>(
    grid: &FieldGrid3,
    locations: &In3D<CoordLocation>,
    interp_point: &Point3<fgr>,
    interp_indices: &Idx3<usize>,
) -> (Idx3<isize>, bool, In3D<bool>) {
    let mut start_indices =
        compute_start_indices_3d::<N_POINTS>(grid, locations, interp_point, interp_indices);

    let grid_shape = grid.shape();
    let mut crosses_periodic_bound = In3D::new(false, false, false);
    let mut any_crosses_periodic_bound = false;

    for &dim in &Dim3::slice() {
        // Check if start index is outside lower bound
        if start_indices[dim] < 0 {
            if grid.is_periodic(dim) {
                // If the dimension is periodic, make a note of the crossing
                crosses_periodic_bound[dim] = true;
                any_crosses_periodic_bound = true;
            } else {
                // If the dimension is not periodic, shift the interpolation interval to lie on the inside
                start_indices[dim] = 0;
            }
        // Check the upper bound accordingly
        } else if (start_indices[dim] as usize) + N_POINTS > grid_shape[dim] {
            if grid.is_periodic(dim) {
                crosses_periodic_bound[dim] = true;
                any_crosses_periodic_bound = true;
            } else {
                start_indices[dim] = (grid_shape[dim] - N_POINTS) as isize;
            }
        }
    }

    (
        start_indices,
        any_crosses_periodic_bound,
        crosses_periodic_bound,
    )
}

fn find_start_indices_and_crossings_2d<const N_POINTS: usize>(
    grid: &FieldGrid2,
    locations: &In2D<CoordLocation>,
    interp_point: &Point2<fgr>,
    interp_indices: &Idx2<usize>,
) -> (Idx2<isize>, bool, In2D<bool>) {
    let mut start_indices =
        compute_start_indices_2d::<N_POINTS>(grid, locations, interp_point, interp_indices);

    let grid_shape = grid.shape();
    let mut crosses_periodic_bound = In2D::new(false, false);
    let mut any_crosses_periodic_bound = false;

    for &dim in &Dim2::slice() {
        // Check if start index is outside lower bound
        if start_indices[dim] < 0 {
            if grid.is_periodic(dim) {
                // If the dimension is periodic, make a note of the crossing
                crosses_periodic_bound[dim] = true;
                any_crosses_periodic_bound = true;
            } else {
                // If the dimension is not periodic, shift the interpolation interval to lie on the inside
                start_indices[dim] = 0;
            }
        // Check the upper bound accordingly
        } else if (start_indices[dim] as usize) + N_POINTS > grid_shape[dim] {
            if grid.is_periodic(dim) {
                crosses_periodic_bound[dim] = true;
                any_crosses_periodic_bound = true;
            } else {
                start_indices[dim] = (grid_shape[dim] - N_POINTS) as isize;
            }
        }
    }

    (
        start_indices,
        any_crosses_periodic_bound,
        crosses_periodic_bound,
    )
}

fn find_start_idx_and_crossing_1d<const N_POINTS: usize>(
    grid: &FieldGrid1,
    location: CoordLocation,
    interp_coord: fgr,
    interp_idx: usize,
) -> (isize, bool) {
    let mut start_idx = compute_start_idx_1d::<N_POINTS>(grid, location, interp_coord, interp_idx);

    let grid_size = grid.size();
    let mut crosses_periodic_bound = false;

    // Check if start index is outside lower bound
    if start_idx < 0 {
        if grid.is_periodic() {
            // If the grid is periodic, make a note of the crossing
            crosses_periodic_bound = true;
        } else {
            // If the grid is not periodic, shift the interpolation interval to lie on the inside
            start_idx = 0;
        }
    // Check the upper bound accordingly
    } else if (start_idx as usize) + N_POINTS > grid_size {
        if grid.is_periodic() {
            crosses_periodic_bound = true;
        } else {
            start_idx = (grid_size - N_POINTS) as isize;
        }
    }

    (start_idx, crosses_periodic_bound)
}

fn create_value_subarray_for_interior_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_CUBED: usize,
>(
    values: &Array3<F>,
    start_indices: &Idx3<isize>,
) -> ([fgr; N_POINTS_CUBED], F) {
    let mut subarray: [MaybeUninit<fgr>; N_POINTS_CUBED] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let offsets = Idx3::from(start_indices);
    let mut idx = 0;

    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;

    for k in offsets[Z]..(offsets[Z] + N_POINTS) {
        for j in offsets[Y]..(offsets[Y] + N_POINTS) {
            for i in offsets[X]..(offsets[X] + N_POINTS) {
                let value = values[[i, j, k]].into();
                subarray[idx].write(value);
                idx += 1;

                if N_POINTS > 2 {
                    sum += value;
                    sum_of_squares += value * value;
                }
            }
        }
    }

    debug_assert_eq!(idx, N_POINTS_CUBED);

    let variation = if N_POINTS > 2 {
        1.0 - (sum * sum) / (sum_of_squares * (N_POINTS_CUBED as fgr))
    } else {
        0.0
    };

    (
        unsafe { (&subarray as *const _ as *const [fgr; N_POINTS_CUBED]).read() },
        F::from(variation).unwrap(),
    )
}

fn create_value_subarray_for_interior_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    values: &Array2<F>,
    start_indices: &Idx2<isize>,
) -> ([fgr; N_POINTS_SQUARED], F) {
    let mut subarray: [MaybeUninit<fgr>; N_POINTS_SQUARED] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let offsets = Idx2::from(start_indices);
    let mut idx = 0;

    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;

    for j in offsets[Y2]..(offsets[Y2] + N_POINTS) {
        for i in offsets[X2]..(offsets[X2] + N_POINTS) {
            let value = values[[i, j]].into();
            subarray[idx].write(value);
            idx += 1;

            if N_POINTS > 2 {
                sum += value;
                sum_of_squares += value * value;
            }
        }
    }

    debug_assert_eq!(idx, N_POINTS_SQUARED);

    let variation = if N_POINTS > 2 {
        1.0 - (sum * sum) / (sum_of_squares * (N_POINTS_SQUARED as fgr))
    } else {
        0.0
    };

    (
        unsafe { (&subarray as *const _ as *const [fgr; N_POINTS_SQUARED]).read() },
        F::from(variation).unwrap(),
    )
}

fn create_value_subarray_for_interior_1d<F: BFloat, const N_POINTS: usize>(
    values: &Array1<F>,
    start_idx: isize,
) -> ([fgr; N_POINTS], F) {
    let mut subarray: [MaybeUninit<fgr>; N_POINTS] = unsafe { MaybeUninit::uninit().assume_init() };
    let offset = start_idx as usize;

    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;

    for (idx, i) in (offset..(offset + N_POINTS)).enumerate() {
        let value = values[i].into();
        subarray[idx].write(value);

        if N_POINTS > 2 {
            sum += value;
            sum_of_squares += value * value;
        }
    }

    let variation = if N_POINTS > 2 {
        1.0 - (sum * sum) / (sum_of_squares * (N_POINTS as fgr))
    } else {
        0.0
    };

    (
        unsafe { (&subarray as *const _ as *const [fgr; N_POINTS]).read() },
        F::from(variation).unwrap(),
    )
}

fn create_value_subarray_for_periodic_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_CUBED: usize,
>(
    values: &Array3<F>,
    start_indices: &Idx3<isize>,
) -> ([fgr; N_POINTS_CUBED], F) {
    let grid_shape = values.shape();
    let offsets = In3D::new(
        (start_indices[X] + (grid_shape[0] as isize)) as usize,
        (start_indices[Y] + (grid_shape[1] as isize)) as usize,
        (start_indices[Z] + (grid_shape[2] as isize)) as usize,
    );
    let mut subarray: [MaybeUninit<fgr>; N_POINTS_CUBED] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut idx = 0;

    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;

    for k in 0..N_POINTS {
        for j in 0..N_POINTS {
            for i in 0..N_POINTS {
                let value = values[[
                    (offsets[X] + i) % grid_shape[0],
                    (offsets[Y] + j) % grid_shape[1],
                    (offsets[Z] + k) % grid_shape[2],
                ]]
                .into();
                subarray[idx].write(value);
                idx += 1;

                if N_POINTS > 2 {
                    sum += value;
                    sum_of_squares += value * value;
                }
            }
        }
    }

    debug_assert_eq!(idx, N_POINTS_CUBED);

    let variation = if N_POINTS > 2 {
        1.0 - (sum * sum) / (sum_of_squares * (N_POINTS_CUBED as fgr))
    } else {
        0.0
    };

    (
        unsafe { (&subarray as *const _ as *const [fgr; N_POINTS_CUBED]).read() },
        F::from(variation).unwrap(),
    )
}

fn create_value_subarray_for_periodic_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    values: &Array2<F>,
    start_indices: &Idx2<isize>,
) -> ([fgr; N_POINTS_SQUARED], F) {
    let grid_shape = values.shape();
    let offsets = In2D::new(
        (start_indices[X2] + (grid_shape[0] as isize)) as usize,
        (start_indices[Y2] + (grid_shape[1] as isize)) as usize,
    );
    let mut subarray: [MaybeUninit<fgr>; N_POINTS_SQUARED] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut idx = 0;

    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;

    for j in 0..N_POINTS {
        for i in 0..N_POINTS {
            let value = values[[
                (offsets[X2] + i) % grid_shape[0],
                (offsets[Y2] + j) % grid_shape[1],
            ]]
            .into();
            subarray[idx].write(value);
            idx += 1;

            if N_POINTS > 2 {
                sum += value;
                sum_of_squares += value * value;
            }
        }
    }

    debug_assert_eq!(idx, N_POINTS_SQUARED);

    let variation = if N_POINTS > 2 {
        1.0 - (sum * sum) / (sum_of_squares * (N_POINTS_SQUARED as fgr))
    } else {
        0.0
    };

    (
        unsafe { (&subarray as *const _ as *const [fgr; N_POINTS_SQUARED]).read() },
        F::from(variation).unwrap(),
    )
}

fn create_value_subarray_for_periodic_1d<F: BFloat, const N_POINTS: usize>(
    values: &Array1<F>,
    start_idx: isize,
) -> ([fgr; N_POINTS], F) {
    let grid_size = values.len();
    let offset = (start_idx + (grid_size as isize)) as usize;
    let mut subarray: [MaybeUninit<fgr>; N_POINTS] = unsafe { MaybeUninit::uninit().assume_init() };

    let mut sum = 0.0;
    let mut sum_of_squares = 0.0;

    for i in 0..N_POINTS {
        let value = values[(offset + i) % grid_size].into();
        subarray[i].write(value);

        if N_POINTS > 2 {
            sum += value;
            sum_of_squares += value * value;
        }
    }

    let variation = if N_POINTS > 2 {
        1.0 - (sum * sum) / (sum_of_squares * (N_POINTS as fgr))
    } else {
        0.0
    };

    (
        unsafe { (&subarray as *const _ as *const [fgr; N_POINTS]).read() },
        F::from(variation).unwrap(),
    )
}

fn create_value_subarray_3d<F: BFloat, const N_POINTS: usize, const N_POINTS_CUBED: usize>(
    any_crosses_periodic_bound: bool,
    values: &Array3<F>,
    start_indices: &Idx3<isize>,
) -> ([fgr; N_POINTS_CUBED], F) {
    if any_crosses_periodic_bound {
        create_value_subarray_for_periodic_3d::<_, N_POINTS, N_POINTS_CUBED>(values, start_indices)
    } else {
        create_value_subarray_for_interior_3d::<_, N_POINTS, N_POINTS_CUBED>(values, start_indices)
    }
}

fn create_value_subarray_2d<F: BFloat, const N_POINTS: usize, const N_POINTS_SQUARED: usize>(
    any_crosses_periodic_bound: bool,
    values: &Array2<F>,
    start_indices: &Idx2<isize>,
) -> ([fgr; N_POINTS_SQUARED], F) {
    if any_crosses_periodic_bound {
        create_value_subarray_for_periodic_2d::<_, N_POINTS, N_POINTS_SQUARED>(
            values,
            start_indices,
        )
    } else {
        create_value_subarray_for_interior_2d::<_, N_POINTS, N_POINTS_SQUARED>(
            values,
            start_indices,
        )
    }
}

fn create_value_subarray_1d<F: BFloat, const N_POINTS: usize>(
    crosses_periodic_bound: bool,
    values: &Array1<F>,
    start_idx: isize,
) -> ([fgr; N_POINTS], F) {
    if crosses_periodic_bound {
        create_value_subarray_for_periodic_1d::<_, N_POINTS>(values, start_idx)
    } else {
        create_value_subarray_for_interior_1d::<_, N_POINTS>(values, start_idx)
    }
}

fn create_coordinate_subarray_for_interior<const N_POINTS: usize>(
    coords: &[fgr],
    start_idx: isize,
) -> [fgr; N_POINTS] {
    let offset = start_idx as usize;
    let mut subarray: [MaybeUninit<fgr>; N_POINTS] = unsafe { MaybeUninit::uninit().assume_init() };
    for idx in 0..N_POINTS {
        subarray[idx].write(coords[offset + idx]);
    }
    unsafe { (&subarray as *const _ as *const [fgr; N_POINTS]).read() }
}

fn create_coordinate_subarray_for_periodic<const N_POINTS: usize>(
    coords: &[fgr],
    extent: fgr,
    start_idx: isize,
) -> [fgr; N_POINTS] {
    let len = coords.len();
    let mut subarray: [MaybeUninit<fgr>; N_POINTS] = unsafe { MaybeUninit::uninit().assume_init() };
    if start_idx < 0 {
        let n_points_below = (-start_idx) as usize;
        let offset = len - n_points_below;
        for idx in 0..n_points_below {
            subarray[idx].write(coords[offset + idx] - extent);
        }
        for idx in n_points_below..N_POINTS {
            subarray[idx].write(coords[idx - n_points_below]);
        }
    } else {
        // start_idx + N_POINTS >= len
        let offset = start_idx as usize;
        let n_points_below = len - offset;
        for idx in 0..n_points_below {
            subarray[idx].write(coords[offset + idx]);
        }
        for idx in n_points_below..N_POINTS {
            subarray[idx].write(coords[idx - n_points_below] + extent);
        }
    }
    unsafe { (&subarray as *const _ as *const [fgr; N_POINTS]).read() }
}

fn create_coordinate_subarrays_3d<const N_POINTS: usize>(
    crosses_periodic_bound: &In3D<bool>,
    coords: &CoordRefs3<fgr>,
    extents: &Vec3<fgr>,
    start_indices: &Idx3<isize>,
) -> ([fgr; N_POINTS], [fgr; N_POINTS], [fgr; N_POINTS]) {
    let x_coord_subarray = if crosses_periodic_bound[X] {
        create_coordinate_subarray_for_periodic::<N_POINTS>(coords[X], extents[X], start_indices[X])
    } else {
        create_coordinate_subarray_for_interior::<N_POINTS>(coords[X], start_indices[X])
    };
    let y_coord_subarray = if crosses_periodic_bound[Y] {
        create_coordinate_subarray_for_periodic::<N_POINTS>(coords[Y], extents[Y], start_indices[Y])
    } else {
        create_coordinate_subarray_for_interior::<N_POINTS>(coords[Y], start_indices[Y])
    };
    let z_coord_subarray = if crosses_periodic_bound[Z] {
        create_coordinate_subarray_for_periodic::<N_POINTS>(coords[Z], extents[Z], start_indices[Z])
    } else {
        create_coordinate_subarray_for_interior::<N_POINTS>(coords[Z], start_indices[Z])
    };
    (x_coord_subarray, y_coord_subarray, z_coord_subarray)
}

fn create_coordinate_subarrays_2d<const N_POINTS: usize>(
    crosses_periodic_bound: &In2D<bool>,
    coords: &CoordRefs2<fgr>,
    extents: &Vec2<fgr>,
    start_indices: &Idx2<isize>,
) -> ([fgr; N_POINTS], [fgr; N_POINTS]) {
    let x_coord_subarray = if crosses_periodic_bound[X2] {
        create_coordinate_subarray_for_periodic::<N_POINTS>(
            coords[X2],
            extents[X2],
            start_indices[X2],
        )
    } else {
        create_coordinate_subarray_for_interior::<N_POINTS>(coords[X2], start_indices[X2])
    };
    let y_coord_subarray = if crosses_periodic_bound[Y2] {
        create_coordinate_subarray_for_periodic::<N_POINTS>(
            coords[Y2],
            extents[Y2],
            start_indices[Y2],
        )
    } else {
        create_coordinate_subarray_for_interior::<N_POINTS>(coords[Y2], start_indices[Y2])
    };
    (x_coord_subarray, y_coord_subarray)
}

fn create_coordinate_subarray_1d<const N_POINTS: usize>(
    crosses_periodic_bound: bool,
    coords: &[fgr],
    extent: fgr,
    start_idx: isize,
) -> [fgr; N_POINTS] {
    if crosses_periodic_bound {
        create_coordinate_subarray_for_periodic::<N_POINTS>(coords, extent, start_idx)
    } else {
        create_coordinate_subarray_for_interior::<N_POINTS>(coords, start_idx)
    }
}

macro_rules! init_array {
    ($T:ty, $LEN:expr, $val:expr) => {{
        let mut data: [MaybeUninit<$T>; $LEN] = unsafe { MaybeUninit::uninit().assume_init() };
        for elem in &mut data[..] {
            elem.write($val);
        }
        unsafe { (&data as *const _ as *const [$T; $LEN]).read() }
    }};
}

fn interp_subarrays_3d<
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    coords: In3D<&[fgr; N_POINTS]>,
    values: &[fgr; N_POINTS_CUBED],
    interp_point: &Point3<fgr>,
) -> fip {
    let x_coords = coords[X];
    let y_coords = coords[Y];
    let z_coords = coords[Z];

    let mut vals_c = init_array!(fgr, N_POINTS, 0.0);
    let mut vals_d = init_array!(fgr, N_POINTS, 0.0);
    let mut poly_x = init_array!(fgr, N_POINTS_SQUARED, 0.0);
    let mut poly_xy = init_array!(fgr, N_POINTS, 0.0);
    let mut accum;
    let mut correction;

    debug_assert_eq!(N_POINTS * N_POINTS, N_POINTS_SQUARED);
    debug_assert_eq!(N_POINTS_SQUARED * N_POINTS, N_POINTS_CUBED);

    for k in 0..N_POINTS {
        for j in 0..N_POINTS {
            vals_c.copy_from_slice(
                &values[(k * N_POINTS + j) * N_POINTS..(k * N_POINTS + j + 1) * N_POINTS],
            );
            vals_d.copy_from_slice(&vals_c);

            accum = vals_c[0];

            for n in 1..N_POINTS {
                for i in 0..(N_POINTS - n) {
                    correction = (vals_c[i + 1] - vals_d[i]) / (x_coords[i + n] - x_coords[i]);
                    vals_c[i] = (interp_point[X] - x_coords[i]) * correction;
                    vals_d[i] = (interp_point[X] - x_coords[i + n]) * correction;
                }

                accum += vals_c[0];
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
                vals_c[j] = (interp_point[Y] - y_coords[j]) * correction;
                vals_d[j] = (interp_point[Y] - y_coords[j + n]) * correction;
            }

            accum += vals_c[0];
        }

        poly_xy[k] = accum;
    }

    vals_c.copy_from_slice(&poly_xy);
    vals_d.copy_from_slice(&vals_c);

    let mut poly_xyz = vals_c[0];

    for n in 1..N_POINTS {
        for k in 0..(N_POINTS - n) {
            correction = (vals_c[k + 1] - vals_d[k]) / (z_coords[k + n] - z_coords[k]);
            vals_c[k] = (interp_point[Z] - z_coords[k]) * correction;
            vals_d[k] = (interp_point[Z] - z_coords[k + n]) * correction;
        }

        poly_xyz += vals_c[0];
    }

    poly_xyz as fip
}

fn interp_subarrays_2d<const N_POINTS: usize, const N_POINTS_SQUARED: usize>(
    coords: In2D<&[fgr; N_POINTS]>,
    values: &[fgr; N_POINTS_SQUARED],
    interp_point: &Point2<fgr>,
) -> fip {
    let x_coords = coords[X2];
    let y_coords = coords[Y2];

    let mut vals_c = init_array!(fgr, N_POINTS, 0.0);
    let mut vals_d = init_array!(fgr, N_POINTS, 0.0);
    let mut poly_x = init_array!(fgr, N_POINTS, 0.0);
    let mut accum;
    let mut correction;

    debug_assert_eq!(N_POINTS * N_POINTS, N_POINTS_SQUARED);

    for j in 0..N_POINTS {
        vals_c.copy_from_slice(&values[j * N_POINTS..(j + 1) * N_POINTS]);
        vals_d.copy_from_slice(&vals_c);

        accum = vals_c[0];

        for n in 1..N_POINTS {
            for i in 0..(N_POINTS - n) {
                correction = (vals_c[i + 1] - vals_d[i]) / (x_coords[i + n] - x_coords[i]);
                vals_c[i] = (interp_point[X2] - x_coords[i]) * correction;
                vals_d[i] = (interp_point[X2] - x_coords[i + n]) * correction;
            }

            accum += vals_c[0];
        }

        poly_x[j] = accum;
    }

    vals_c.copy_from_slice(&poly_x);
    vals_d.copy_from_slice(&vals_c);

    let mut poly_xy = vals_c[0];

    for n in 1..N_POINTS {
        for j in 0..(N_POINTS - n) {
            correction = (vals_c[j + 1] - vals_d[j]) / (y_coords[j + n] - y_coords[j]);
            vals_c[j] = (interp_point[Y2] - y_coords[j]) * correction;
            vals_d[j] = (interp_point[Y2] - y_coords[j + n]) * correction;
        }

        poly_xy += vals_c[0];
    }

    poly_xy as fip
}

fn interp_subarray_1d<const N_POINTS: usize>(
    coords: &[fgr; N_POINTS],
    values: &[fgr; N_POINTS],
    interp_coord: fgr,
) -> fip {
    let mut vals_c = init_array!(fgr, N_POINTS, 0.0);
    let mut vals_d = init_array!(fgr, N_POINTS, 0.0);
    let mut correction;

    vals_c.copy_from_slice(values);
    vals_d.copy_from_slice(&vals_c);

    let mut poly = vals_c[0];

    for n in 1..N_POINTS {
        for i in 0..(N_POINTS - n) {
            correction = (vals_c[i + 1] - vals_d[i]) / (coords[i + n] - coords[i]);
            vals_c[i] = (interp_coord - coords[i]) * correction;
            vals_d[i] = (interp_coord - coords[i + n]) * correction;
        }

        poly += vals_c[0];
    }

    poly as fip
}

fn interp_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    grid: &FieldGrid3,
    coords: &CoordRefs3<fgr>,
    locations: &In3D<CoordLocation>,
    values: &Array3<F>,
    interp_point: &Point3<fgr>,
    interp_indices: &Idx3<usize>,
    variation_threshold_for_linear: F,
) -> fip {
    {
        let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
            find_start_indices_and_crossings_3d::<N_POINTS>(
                grid,
                locations,
                interp_point,
                interp_indices,
            );

        let (value_subarray, variation) = create_value_subarray_3d::<_, N_POINTS, N_POINTS_CUBED>(
            any_crosses_periodic_bound,
            values,
            &start_indices,
        );

        // If the variation exceeds the given threshold, use linear interpolation
        // in order to avoid overshoot.
        if variation > variation_threshold_for_linear && N_POINTS > 2 {
            let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                find_start_indices_and_crossings_3d::<2>(
                    grid,
                    locations,
                    interp_point,
                    interp_indices,
                );

            let (value_subarray, _) = create_value_subarray_3d::<_, 2, 8>(
                any_crosses_periodic_bound,
                values,
                &start_indices,
            );

            let (x_coord_subarray, y_coord_subarray, z_coord_subarray) =
                create_coordinate_subarrays_3d::<2>(
                    &crosses_periodic_bound,
                    coords,
                    grid.extents(),
                    &start_indices,
                );

            interp_subarrays_3d::<2, 4, 8>(
                In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray),
                &value_subarray,
                interp_point,
            )
        } else {
            let (x_coord_subarray, y_coord_subarray, z_coord_subarray) =
                create_coordinate_subarrays_3d::<N_POINTS>(
                    &crosses_periodic_bound,
                    coords,
                    grid.extents(),
                    &start_indices,
                );

            interp_subarrays_3d::<N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
                In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray),
                &value_subarray,
                interp_point,
            )
        }
    }
}

fn interp_2d<F: BFloat, const N_POINTS: usize, const N_POINTS_SQUARED: usize>(
    grid: &FieldGrid2,
    coords: &CoordRefs2<fgr>,
    locations: &In2D<CoordLocation>,
    values: &Array2<F>,
    interp_point: &Point2<fgr>,
    interp_indices: &Idx2<usize>,
    variation_threshold_for_linear: F,
) -> fip {
    {
        let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
            find_start_indices_and_crossings_2d::<N_POINTS>(
                grid,
                locations,
                interp_point,
                interp_indices,
            );

        let (value_subarray, variation) = create_value_subarray_2d::<_, N_POINTS, N_POINTS_SQUARED>(
            any_crosses_periodic_bound,
            values,
            &start_indices,
        );

        // If the variation exceeds the given threshold, use linear interpolation
        // in order to avoid overshoot.
        if variation > variation_threshold_for_linear && N_POINTS > 2 {
            let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                find_start_indices_and_crossings_2d::<2>(
                    grid,
                    locations,
                    interp_point,
                    interp_indices,
                );

            let (value_subarray, _) = create_value_subarray_2d::<_, 2, 4>(
                any_crosses_periodic_bound,
                values,
                &start_indices,
            );

            let (x_coord_subarray, y_coord_subarray) = create_coordinate_subarrays_2d::<2>(
                &crosses_periodic_bound,
                coords,
                grid.extents(),
                &start_indices,
            );

            interp_subarrays_2d::<2, 4>(
                In2D::new(&x_coord_subarray, &y_coord_subarray),
                &value_subarray,
                interp_point,
            )
        } else {
            let (x_coord_subarray, y_coord_subarray) = create_coordinate_subarrays_2d::<N_POINTS>(
                &crosses_periodic_bound,
                coords,
                grid.extents(),
                &start_indices,
            );

            interp_subarrays_2d::<N_POINTS, N_POINTS_SQUARED>(
                In2D::new(&x_coord_subarray, &y_coord_subarray),
                &value_subarray,
                interp_point,
            )
        }
    }
}

fn interp_1d<F: BFloat, const N_POINTS: usize>(
    grid: &FieldGrid1,
    coords: &[fgr],
    location: CoordLocation,
    values: &Array1<F>,
    interp_coord: fgr,
    interp_idx: usize,
    variation_threshold_for_linear: F,
) -> fip {
    {
        let (start_idx, crosses_periodic_bound) =
            find_start_idx_and_crossing_1d::<N_POINTS>(grid, location, interp_coord, interp_idx);

        let (value_subarray, variation) =
            create_value_subarray_1d::<_, N_POINTS>(crosses_periodic_bound, values, start_idx);

        // If the variation exceeds the given threshold, use linear interpolation
        // in order to avoid overshoot.
        if variation > variation_threshold_for_linear && N_POINTS > 2 {
            let (start_idx, crosses_periodic_bound) =
                find_start_idx_and_crossing_1d::<2>(grid, location, interp_coord, interp_idx);

            let (value_subarray, _) =
                create_value_subarray_1d::<_, 2>(crosses_periodic_bound, values, start_idx);

            let coord_subarray = create_coordinate_subarray_1d::<2>(
                crosses_periodic_bound,
                coords,
                grid.extent(),
                start_idx,
            );

            interp_subarray_1d::<2>(&coord_subarray, &value_subarray, interp_coord)
        } else {
            let coord_subarray = create_coordinate_subarray_1d::<N_POINTS>(
                crosses_periodic_bound,
                coords,
                grid.extent(),
                start_idx,
            );

            interp_subarray_1d::<N_POINTS>(&coord_subarray, &value_subarray, interp_coord)
        }
    }
}

fn interp_scalar_field_in_known_grid_cell_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F>,
    interp_point: &Point3<fgr>,
    interp_indices: &Idx3<usize>,
    variation_threshold_for_linear: F,
) -> fip {
    interp_3d::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field.grid(),
        &field.coords(),
        field.locations(),
        field.values(),
        interp_point,
        interp_indices,
        variation_threshold_for_linear,
    )
}

fn interp_scalar_field_in_known_grid_cell_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    field: &ScalarField2<F>,
    interp_point: &Point2<fgr>,
    interp_indices: &Idx2<usize>,
    variation_threshold_for_linear: F,
) -> fip {
    interp_2d::<_, N_POINTS, N_POINTS_SQUARED>(
        field.grid(),
        &field.coords(),
        field.locations(),
        field.values(),
        interp_point,
        interp_indices,
        variation_threshold_for_linear,
    )
}

fn interp_scalar_field_in_known_grid_cell_1d<F: BFloat, const N_POINTS: usize>(
    field: &ScalarField1<F>,
    interp_coord: fgr,
    interp_idx: usize,
    variation_threshold_for_linear: F,
) -> fip {
    interp_1d::<_, N_POINTS>(
        field.grid(),
        field.coords(),
        field.location(),
        field.values(),
        interp_coord,
        interp_idx,
        variation_threshold_for_linear,
    )
}

fn interp_scalar_field_from_grid_point_query_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F>,
    grid_point_query: GridPointQuery3<fgr, Idx3<usize>>,
    interp_point: &Point3<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<fgr, fip> {
    match grid_point_query {
        GridPointQuery3::Inside(interp_indices) => {
            GridPointQuery3::Inside(interp_scalar_field_in_known_grid_cell_3d::<
                _,
                N_POINTS,
                N_POINTS_SQUARED,
                N_POINTS_CUBED,
            >(
                field,
                interp_point,
                &interp_indices,
                variation_threshold_for_linear,
            ))
        }
        GridPointQuery3::MovedInside((interp_indices, moved_point)) => {
            GridPointQuery3::MovedInside((
                interp_scalar_field_in_known_grid_cell_3d::<
                    _,
                    N_POINTS,
                    N_POINTS_SQUARED,
                    N_POINTS_CUBED,
                >(
                    field,
                    &moved_point,
                    &interp_indices,
                    variation_threshold_for_linear,
                ),
                moved_point,
            ))
        }
        GridPointQuery3::Outside => GridPointQuery3::Outside,
    }
}

fn interp_scalar_field_from_grid_point_query_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    field: &ScalarField2<F>,
    grid_point_query: GridPointQuery2<fgr, Idx2<usize>>,
    interp_point: &Point2<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery2<fgr, fip> {
    match grid_point_query {
        GridPointQuery2::Inside(interp_indices) => {
            GridPointQuery2::Inside(interp_scalar_field_in_known_grid_cell_2d::<
                _,
                N_POINTS,
                N_POINTS_SQUARED,
            >(
                field,
                interp_point,
                &interp_indices,
                variation_threshold_for_linear,
            ))
        }
        GridPointQuery2::MovedInside((interp_indices, moved_point)) => {
            GridPointQuery2::MovedInside((
                interp_scalar_field_in_known_grid_cell_2d::<_, N_POINTS, N_POINTS_SQUARED>(
                    field,
                    &moved_point,
                    &interp_indices,
                    variation_threshold_for_linear,
                ),
                moved_point,
            ))
        }
        GridPointQuery2::Outside => GridPointQuery2::Outside,
    }
}

fn interp_scalar_field_from_grid_point_query_1d<F: BFloat, const N_POINTS: usize>(
    field: &ScalarField1<F>,
    grid_point_query: GridPointQuery1<fgr, usize>,
    interp_coord: fgr,
    variation_threshold_for_linear: F,
) -> GridPointQuery1<fgr, fip> {
    match grid_point_query {
        GridPointQuery1::Inside(interp_idx) => {
            GridPointQuery1::Inside(interp_scalar_field_in_known_grid_cell_1d::<_, N_POINTS>(
                field,
                interp_coord,
                interp_idx,
                variation_threshold_for_linear,
            ))
        }
        GridPointQuery1::MovedInside((interp_idx, moved_coord)) => GridPointQuery1::MovedInside((
            interp_scalar_field_in_known_grid_cell_1d::<_, N_POINTS>(
                field,
                moved_coord,
                interp_idx,
                variation_threshold_for_linear,
            ),
            moved_coord,
        )),
        GridPointQuery1::Outside => GridPointQuery1::Outside,
    }
}

fn interp_vector_field_in_known_grid_cell_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F>,
    interp_point: &Point3<fgr>,
    interp_indices: &Idx3<usize>,
    variation_threshold_for_linear: F,
) -> Vec3<fip> {
    let grid = field.grid();
    Vec3::with_each_component(|dim| {
        interp_3d::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
            grid,
            &field.coords(dim),
            field.locations(dim),
            field.values(dim),
            interp_point,
            interp_indices,
            variation_threshold_for_linear,
        )
    })
}

fn interp_vector_field_in_known_grid_cell_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    field: &VectorField2<F>,
    interp_point: &Point2<fgr>,
    interp_indices: &Idx2<usize>,
    variation_threshold_for_linear: F,
) -> Vec2<fip> {
    let grid = field.grid();
    Vec2::with_each_component(|dim| {
        interp_2d::<_, N_POINTS, N_POINTS_SQUARED>(
            grid,
            &field.coords(dim),
            field.locations(dim),
            field.values(dim),
            interp_point,
            interp_indices,
            variation_threshold_for_linear,
        )
    })
}

fn interp_vector_field_from_grid_point_query_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F>,
    grid_point_query: GridPointQuery3<fgr, Idx3<usize>>,
    interp_point: &Point3<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<fgr, Vec3<fip>> {
    match grid_point_query {
        GridPointQuery3::Inside(interp_indices) => {
            GridPointQuery3::Inside(interp_vector_field_in_known_grid_cell_3d::<
                _,
                N_POINTS,
                N_POINTS_SQUARED,
                N_POINTS_CUBED,
            >(
                field,
                interp_point,
                &interp_indices,
                variation_threshold_for_linear,
            ))
        }
        GridPointQuery3::MovedInside((interp_indices, moved_point)) => {
            GridPointQuery3::MovedInside((
                interp_vector_field_in_known_grid_cell_3d::<
                    _,
                    N_POINTS,
                    N_POINTS_SQUARED,
                    N_POINTS_CUBED,
                >(
                    field,
                    &moved_point,
                    &interp_indices,
                    variation_threshold_for_linear,
                ),
                moved_point,
            ))
        }
        GridPointQuery3::Outside => GridPointQuery3::Outside,
    }
}

fn interp_vector_field_from_grid_point_query_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    field: &VectorField2<F>,
    grid_point_query: GridPointQuery2<fgr, Idx2<usize>>,
    interp_point: &Point2<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery2<fgr, Vec2<fip>> {
    match grid_point_query {
        GridPointQuery2::Inside(interp_indices) => {
            GridPointQuery2::Inside(interp_vector_field_in_known_grid_cell_2d::<
                _,
                N_POINTS,
                N_POINTS_SQUARED,
            >(
                field,
                interp_point,
                &interp_indices,
                variation_threshold_for_linear,
            ))
        }
        GridPointQuery2::MovedInside((interp_indices, moved_point)) => {
            GridPointQuery2::MovedInside((
                interp_vector_field_in_known_grid_cell_2d::<_, N_POINTS, N_POINTS_SQUARED>(
                    field,
                    &moved_point,
                    &interp_indices,
                    variation_threshold_for_linear,
                ),
                moved_point,
            ))
        }
        GridPointQuery2::Outside => GridPointQuery2::Outside,
    }
}

fn interp_scalar_field_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F>,
    interp_point: &Point3<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<fgr, fip> {
    let grid_point_query = field.grid().find_grid_cell(interp_point);
    interp_scalar_field_from_grid_point_query_3d::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_scalar_field_2d<F: BFloat, const N_POINTS: usize, const N_POINTS_SQUARED: usize>(
    field: &ScalarField2<F>,
    interp_point: &Point2<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery2<fgr, fip> {
    let grid_point_query = field.grid().find_grid_cell(interp_point);
    interp_scalar_field_from_grid_point_query_2d::<_, N_POINTS, N_POINTS_SQUARED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_scalar_field_1d<F: BFloat, const N_POINTS: usize>(
    field: &ScalarField1<F>,
    interp_coord: fgr,
    variation_threshold_for_linear: F,
) -> GridPointQuery1<fgr, fip> {
    let grid_point_query = field.grid().find_grid_cell(interp_coord);
    interp_scalar_field_from_grid_point_query_1d::<_, N_POINTS>(
        field,
        grid_point_query,
        interp_coord,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_scalar_field_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F>,
    interp_point: &Point3<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<fgr, fip> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_point);
    interp_scalar_field_from_grid_point_query_3d::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_scalar_field_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    field: &ScalarField2<F>,
    interp_point: &Point2<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery2<fgr, fip> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_point);
    interp_scalar_field_from_grid_point_query_2d::<_, N_POINTS, N_POINTS_SQUARED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_scalar_field_1d<F: BFloat, const N_POINTS: usize>(
    field: &ScalarField1<F>,
    interp_coord: fgr,
    variation_threshold_for_linear: F,
) -> GridPointQuery1<fgr, fip> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_coord);
    interp_scalar_field_from_grid_point_query_1d::<_, N_POINTS>(
        field,
        grid_point_query,
        interp_coord,
        variation_threshold_for_linear,
    )
}

fn interp_vector_field_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F>,
    interp_point: &Point3<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<fgr, Vec3<fip>> {
    let grid_point_query = field.grid().find_grid_cell(interp_point);
    interp_vector_field_from_grid_point_query_3d::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_vector_field_2d<F: BFloat, const N_POINTS: usize, const N_POINTS_SQUARED: usize>(
    field: &VectorField2<F>,
    interp_point: &Point2<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery2<fgr, Vec2<fip>> {
    let grid_point_query = field.grid().find_grid_cell(interp_point);
    interp_vector_field_from_grid_point_query_2d::<_, N_POINTS, N_POINTS_SQUARED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_vector_field_3d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F>,
    interp_point: &Point3<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<fgr, Vec3<fip>> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_point);
    interp_vector_field_from_grid_point_query_3d::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_vector_field_2d<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
>(
    field: &VectorField2<F>,
    interp_point: &Point2<fgr>,
    variation_threshold_for_linear: F,
) -> GridPointQuery2<fgr, Vec2<fip>> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_point);
    interp_vector_field_from_grid_point_query_2d::<_, N_POINTS, N_POINTS_SQUARED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
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
    /// Creates a new quadratic interpolator.
    pub fn new(config: PolyFitInterpolatorConfig) -> Self {
        config.validate();
        PolyFitInterpolator3 { config }
    }
}

impl InterpGridVerifier3 for PolyFitInterpolator3 {
    fn verify_grid(&self, grid: &FieldGrid3) -> Result<(), String> {
        if grid.shape().into_iter().any(|&n| n <= self.config.order) {
            Err(format!(
                "Grid with shape {} is too small for polynomial fitting interpolator with order {}",
                grid.shape(),
                self.config.order
            ))
        } else {
            Ok(())
        }
    }
}

impl<F: BFloat> Interpolator3<F> for PolyFitInterpolator3 {
    fn interp_scalar_field(
        &self,
        field: &ScalarField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, fip>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_3d::<_, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field_3d::<_, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field_3d::<_, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field_3d::<_, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field_3d::<_, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_scalar_field_known_cell(
        &self,
        field: &ScalarField3<F>,
        interp_point: &Point3<fgr>,
        interp_indices: &Idx3<usize>,
    ) -> fip
    where
        F: BFloat,
    {
        let is_inside = field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices);

        if !is_inside {
            return self
                .interp_scalar_field(field, interp_point)
                .expect_inside_or_moved();
        }

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_in_known_grid_cell_3d::<_, 2, 4, 8>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field_in_known_grid_cell_3d::<_, 3, 9, 27>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field_in_known_grid_cell_3d::<_, 4, 16, 64>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field_in_known_grid_cell_3d::<_, 5, 25, 125>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field_in_known_grid_cell_3d::<_, 6, 36, 216>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_extrap_scalar_field(
        &self,
        field: &ScalarField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, fip>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_extrap_scalar_field_3d::<_, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_scalar_field_3d::<_, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_scalar_field_3d::<_, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_scalar_field_3d::<_, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_scalar_field_3d::<_, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_vector_field(
        &self,
        field: &VectorField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, Vec3<fip>>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_vector_field_3d::<_, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_vector_field_3d::<_, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_vector_field_3d::<_, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_vector_field_3d::<_, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_vector_field_3d::<_, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_vector_field_known_cell(
        &self,
        field: &VectorField3<F>,
        interp_point: &Point3<fgr>,
        interp_indices: &Idx3<usize>,
    ) -> Vec3<fip>
    where
        F: BFloat,
    {
        let is_inside = field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices);

        if !is_inside {
            return self
                .interp_vector_field(field, interp_point)
                .expect_inside_or_moved();
        }

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_vector_field_in_known_grid_cell_3d::<_, 2, 4, 8>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            2 => interp_vector_field_in_known_grid_cell_3d::<_, 3, 9, 27>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            3 => interp_vector_field_in_known_grid_cell_3d::<_, 4, 16, 64>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            4 => interp_vector_field_in_known_grid_cell_3d::<_, 5, 25, 125>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            5 => interp_vector_field_in_known_grid_cell_3d::<_, 6, 36, 216>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_extrap_vector_field(
        &self,
        field: &VectorField3<F>,
        interp_point: &Point3<fgr>,
    ) -> GridPointQuery3<fgr, Vec3<fip>>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_extrap_vector_field_3d::<_, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_vector_field_3d::<_, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_vector_field_3d::<_, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_vector_field_3d::<_, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_vector_field_3d::<_, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
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
    /// Creates a new quadratic interpolator.
    pub fn new(config: PolyFitInterpolatorConfig) -> Self {
        config.validate();
        PolyFitInterpolator2 { config }
    }
}

impl Interpolator2 for PolyFitInterpolator2 {
    fn interp_scalar_field<F>(
        &self,
        field: &ScalarField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, fip>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_2d::<_, 2, 4>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field_2d::<_, 3, 9>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field_2d::<_, 4, 16>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field_2d::<_, 5, 25>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field_2d::<_, 6, 36>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_scalar_field_known_cell<F>(
        &self,
        field: &ScalarField2<F>,
        interp_point: &Point2<fgr>,
        interp_indices: &Idx2<usize>,
    ) -> fip
    where
        F: BFloat,
    {
        let is_inside = field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices);

        if !is_inside {
            return self
                .interp_scalar_field(field, interp_point)
                .expect_inside_or_moved();
        }

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_in_known_grid_cell_2d::<_, 2, 4>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field_in_known_grid_cell_2d::<_, 3, 9>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field_in_known_grid_cell_2d::<_, 4, 16>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field_in_known_grid_cell_2d::<_, 5, 25>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field_in_known_grid_cell_2d::<_, 6, 36>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_extrap_scalar_field<F>(
        &self,
        field: &ScalarField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, fip>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_extrap_scalar_field_2d::<_, 2, 4>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_scalar_field_2d::<_, 3, 9>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_scalar_field_2d::<_, 4, 16>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_scalar_field_2d::<_, 5, 25>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_scalar_field_2d::<_, 6, 36>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_vector_field<F>(
        &self,
        field: &VectorField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, Vec2<fip>>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_vector_field_2d::<_, 2, 4>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_vector_field_2d::<_, 3, 9>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_vector_field_2d::<_, 4, 16>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_vector_field_2d::<_, 5, 25>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_vector_field_2d::<_, 6, 36>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_vector_field_known_cell<F>(
        &self,
        field: &VectorField2<F>,
        interp_point: &Point2<fgr>,
        interp_indices: &Idx2<usize>,
    ) -> Vec2<fip>
    where
        F: BFloat,
    {
        let is_inside = field
            .grid()
            .point_is_inside_cell(interp_point, interp_indices);

        if !is_inside {
            return self
                .interp_vector_field(field, interp_point)
                .expect_inside_or_moved();
        }

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_vector_field_in_known_grid_cell_2d::<_, 2, 4>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            2 => interp_vector_field_in_known_grid_cell_2d::<_, 3, 9>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            3 => interp_vector_field_in_known_grid_cell_2d::<_, 4, 16>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            4 => interp_vector_field_in_known_grid_cell_2d::<_, 5, 25>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            5 => interp_vector_field_in_known_grid_cell_2d::<_, 6, 36>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_extrap_vector_field<F>(
        &self,
        field: &VectorField2<F>,
        interp_point: &Point2<fgr>,
    ) -> GridPointQuery2<fgr, Vec2<fip>>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_extrap_vector_field_2d::<_, 2, 4>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_vector_field_2d::<_, 3, 9>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_vector_field_2d::<_, 4, 16>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_vector_field_2d::<_, 5, 25>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_vector_field_2d::<_, 6, 36>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
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
    /// Creates a new quadratic interpolator.
    pub fn new(config: PolyFitInterpolatorConfig) -> Self {
        config.validate();
        PolyFitInterpolator1 { config }
    }
}

impl Interpolator1 for PolyFitInterpolator1 {
    fn interp_scalar_field<F>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, fip>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => {
                interp_scalar_field_1d::<_, 2>(field, interp_coord, variation_threshold_for_linear)
            }
            2 => {
                interp_scalar_field_1d::<_, 3>(field, interp_coord, variation_threshold_for_linear)
            }
            3 => {
                interp_scalar_field_1d::<_, 4>(field, interp_coord, variation_threshold_for_linear)
            }
            4 => {
                interp_scalar_field_1d::<_, 5>(field, interp_coord, variation_threshold_for_linear)
            }
            5 => {
                interp_scalar_field_1d::<_, 6>(field, interp_coord, variation_threshold_for_linear)
            }
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_scalar_field_known_cell<F>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
        interp_idx: usize,
    ) -> fip
    where
        F: BFloat,
    {
        let is_inside = field.grid().coord_is_inside_cell(interp_coord, interp_idx);

        if !is_inside {
            return self
                .interp_scalar_field(field, interp_coord)
                .expect_inside_or_moved();
        }

        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_scalar_field_in_known_grid_cell_1d::<_, 2>(
                field,
                interp_coord,
                interp_idx,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field_in_known_grid_cell_1d::<_, 3>(
                field,
                interp_coord,
                interp_idx,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field_in_known_grid_cell_1d::<_, 4>(
                field,
                interp_coord,
                interp_idx,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field_in_known_grid_cell_1d::<_, 5>(
                field,
                interp_coord,
                interp_idx,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field_in_known_grid_cell_1d::<_, 6>(
                field,
                interp_coord,
                interp_idx,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

    fn interp_extrap_scalar_field<F>(
        &self,
        field: &ScalarField1<F>,
        interp_coord: fgr,
    ) -> GridPointQuery1<fgr, fip>
    where
        F: BFloat,
    {
        let variation_threshold_for_linear =
            F::from(self.config.variation_threshold_for_linear).unwrap();
        match self.config.order {
            1 => interp_extrap_scalar_field_1d::<_, 2>(
                field,
                interp_coord,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_scalar_field_1d::<_, 3>(
                field,
                interp_coord,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_scalar_field_1d::<_, 4>(
                field,
                interp_coord,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_scalar_field_1d::<_, 5>(
                field,
                interp_coord,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_scalar_field_1d::<_, 6>(
                field,
                interp_coord,
                variation_threshold_for_linear,
            ),
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
