//! Interpolation by polynomial fitting.

use super::Interpolator3;
use crate::field::{ScalarField3, VectorField3};
use crate::geometry::{CoordRefs3, Dim3, Idx3, In3D, Point3, Vec3};
use crate::grid::{CoordLocation, Grid3, GridPointQuery3};
use crate::num::BFloat;
use ndarray::prelude::*;
use std::mem::MaybeUninit;
use Dim3::{X, Y, Z};

fn compute_start_offset<F: BFloat, const N_POINTS: usize>(
    center_coords: &[F],
    location: CoordLocation,
    interp_coord: F,
    interp_idx: usize,
) -> isize {
    let default_start_offset = 1 - ((N_POINTS + 1) as isize) / 2;

    match location {
        CoordLocation::Center => {
            if N_POINTS - 1 % 2 != 0 && interp_coord < center_coords[interp_idx] {
                // If coordinates are located at cell centers, interpolation order is odd
                // and interpolation coordinate is in lower half of the cell:
                // Shift start offset one cell down.
                default_start_offset - 1
            } else {
                default_start_offset
            }
        }
        CoordLocation::LowerEdge => {
            if N_POINTS - 1 % 2 == 0 && interp_coord > center_coords[interp_idx] {
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

fn compute_start_indices<F: BFloat, G: Grid3<F>, const N_POINTS: usize>(
    grid: &G,
    locations: &In3D<CoordLocation>,
    interp_point: &Point3<F>,
    interp_indices: &Idx3<usize>,
) -> Idx3<isize> {
    let center_coords = grid.centers();
    let start_offset_x = compute_start_offset::<_, N_POINTS>(
        &center_coords[X],
        locations[X],
        interp_point[X],
        interp_indices[X],
    );
    let start_offset_y = compute_start_offset::<_, N_POINTS>(
        &center_coords[Y],
        locations[Y],
        interp_point[Y],
        interp_indices[Y],
    );
    let start_offset_z = compute_start_offset::<_, N_POINTS>(
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

fn find_start_indices_and_crossings<F: BFloat, G: Grid3<F>, const N_POINTS: usize>(
    grid: &G,
    locations: &In3D<CoordLocation>,
    interp_point: &Point3<F>,
    interp_indices: &Idx3<usize>,
) -> (Idx3<isize>, bool, In3D<bool>) {
    let mut start_indices =
        compute_start_indices::<_, _, N_POINTS>(grid, locations, interp_point, interp_indices);

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

fn create_value_subarray_for_interior<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_CUBED: usize,
>(
    values: &Array3<F>,
    start_indices: &Idx3<isize>,
) -> ([F; N_POINTS_CUBED], F) {
    //let mut subarray = MaybeUninit::uninit_array::<N_POINTS_CUBED>();
    let mut subarray: [MaybeUninit<F>; N_POINTS_CUBED] =
        unsafe { MaybeUninit::uninit().assume_init() };
    //unsafe { MaybeUninit::uninit().assume_init() };
    let offsets = Idx3::from(&start_indices);
    let mut idx = 0;

    let mut sum = F::zero();
    let mut sum_of_squares = F::zero();

    for k in offsets[Z]..(offsets[Z] + N_POINTS) {
        for j in offsets[Y]..(offsets[Y] + N_POINTS) {
            for i in offsets[X]..(offsets[X] + N_POINTS) {
                let value = values[[i, j, k]];
                subarray[idx].write(value);
                idx += 1;

                sum = sum + value;
                sum_of_squares = sum_of_squares + value * value;
            }
        }
    }

    assert_eq!(idx, N_POINTS_CUBED);

    let variation =
        F::one() - (sum * sum) / (sum_of_squares * F::from_usize(N_POINTS_CUBED).unwrap());

    (
        unsafe { (&subarray as *const _ as *const [F; N_POINTS_CUBED]).read() },
        variation,
    )
}

fn create_value_subarray_for_periodic<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_CUBED: usize,
>(
    values: &Array3<F>,
    start_indices: &Idx3<isize>,
) -> ([F; N_POINTS_CUBED], F) {
    let grid_shape = values.shape();
    let offsets = In3D::new(
        (start_indices[X] + (grid_shape[0] as isize)) as usize,
        (start_indices[Y] + (grid_shape[1] as isize)) as usize,
        (start_indices[Z] + (grid_shape[2] as isize)) as usize,
    );
    let mut subarray: [MaybeUninit<F>; N_POINTS_CUBED] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut idx = 0;

    let mut sum = F::zero();
    let mut sum_of_squares = F::zero();

    for k in 0..N_POINTS {
        for j in 0..N_POINTS {
            for i in 0..N_POINTS {
                let value = values[[
                    (offsets[X] + i) % grid_shape[0],
                    (offsets[Y] + j) % grid_shape[1],
                    (offsets[Z] + k) % grid_shape[2],
                ]];
                subarray[idx].write(value);
                idx += 1;

                sum = sum + value;
                sum_of_squares = sum_of_squares + value * value;
            }
        }
    }

    let variation = F::one() - (sum * sum) / (sum_of_squares * F::from(N_POINTS_CUBED).unwrap());

    (
        unsafe { (&subarray as *const _ as *const [F; N_POINTS_CUBED]).read() },
        variation,
    )
}

fn create_value_subarray<F: BFloat, const N_POINTS: usize, const N_POINTS_CUBED: usize>(
    any_crosses_periodic_bound: bool,
    values: &Array3<F>,
    start_indices: &Idx3<isize>,
) -> ([F; N_POINTS_CUBED], F) {
    if any_crosses_periodic_bound {
        create_value_subarray_for_periodic::<_, N_POINTS, N_POINTS_CUBED>(values, &start_indices)
    } else {
        create_value_subarray_for_interior::<_, N_POINTS, N_POINTS_CUBED>(values, &start_indices)
    }
}

fn create_coordinate_subarray_for_interior<F: BFloat, const N_POINTS: usize>(
    coords: &[F],
    start_idx: isize,
) -> [F; N_POINTS] {
    let offset = start_idx as usize;
    let mut subarray: [MaybeUninit<F>; N_POINTS] = unsafe { MaybeUninit::uninit().assume_init() };
    for idx in 0..N_POINTS {
        subarray[idx].write(coords[offset + idx]);
    }
    unsafe { (&subarray as *const _ as *const [F; N_POINTS]).read() }
}

fn create_coordinate_subarray_for_periodic<F: BFloat, const N_POINTS: usize>(
    coords: &[F],
    extent: F,
    start_idx: isize,
) -> [F; N_POINTS] {
    let len = coords.len();
    let mut subarray: [MaybeUninit<F>; N_POINTS] = unsafe { MaybeUninit::uninit().assume_init() };
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
    unsafe { (&subarray as *const _ as *const [F; N_POINTS]).read() }
}

fn create_coordinate_subarrays<F: BFloat, const N_POINTS: usize>(
    crosses_periodic_bound: &In3D<bool>,
    coords: &CoordRefs3<F>,
    extents: &Vec3<F>,
    start_indices: &Idx3<isize>,
) -> ([F; N_POINTS], [F; N_POINTS], [F; N_POINTS]) {
    let x_coord_subarray = if crosses_periodic_bound[X] {
        create_coordinate_subarray_for_periodic(coords[X], extents[X], start_indices[X])
    } else {
        create_coordinate_subarray_for_interior(coords[X], start_indices[X])
    };
    let y_coord_subarray = if crosses_periodic_bound[Y] {
        create_coordinate_subarray_for_periodic(coords[Y], extents[Y], start_indices[Y])
    } else {
        create_coordinate_subarray_for_interior(coords[Y], start_indices[Y])
    };
    let z_coord_subarray = if crosses_periodic_bound[Z] {
        create_coordinate_subarray_for_periodic(coords[Z], extents[Z], start_indices[Z])
    } else {
        create_coordinate_subarray_for_interior(coords[Z], start_indices[Z])
    };
    (x_coord_subarray, y_coord_subarray, z_coord_subarray)
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

fn interp_subarrays<
    F: BFloat,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    coords: In3D<&[F; N_POINTS]>,
    values: &[F; N_POINTS_CUBED],
    interp_point: &Point3<F>,
) -> F {
    let x_coords = coords[X];
    let y_coords = coords[Y];
    let z_coords = coords[Z];

    let mut vals_c = init_array!(F, N_POINTS, F::zero());
    let mut vals_d = init_array!(F, N_POINTS, F::zero());
    let mut poly_x = init_array!(F, N_POINTS_SQUARED, F::zero());
    let mut poly_xy = init_array!(F, N_POINTS, F::zero());
    let mut poly_xyz;
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
                vals_c[j] = (interp_point[Y] - y_coords[j]) * correction;
                vals_d[j] = (interp_point[Y] - y_coords[j + n]) * correction;
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
            vals_c[k] = (interp_point[Z] - z_coords[k]) * correction;
            vals_d[k] = (interp_point[Z] - z_coords[k + n]) * correction;
        }

        poly_xyz = poly_xyz + vals_c[0];
    }

    poly_xyz
}

fn interp<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    grid: &G,
    coords: &CoordRefs3<F>,
    locations: &In3D<CoordLocation>,
    values: &Array3<F>,
    interp_point: &Point3<F>,
    interp_indices: &Idx3<usize>,
    variation_threshold_for_linear: F,
) -> F {
    {
        let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
            find_start_indices_and_crossings::<_, _, N_POINTS>(
                grid,
                locations,
                interp_point,
                interp_indices,
            );

        let (value_subarray, variation) = create_value_subarray::<_, N_POINTS, N_POINTS_CUBED>(
            any_crosses_periodic_bound,
            values,
            &start_indices,
        );

        // If the variation exceeds the given threshold, use linear interpolation
        // in order to avoid overshoot.
        if variation > variation_threshold_for_linear && N_POINTS > 2 {
            let (start_indices, any_crosses_periodic_bound, crosses_periodic_bound) =
                find_start_indices_and_crossings::<_, _, 1>(
                    grid,
                    locations,
                    interp_point,
                    interp_indices,
                );

            let (value_subarray, _) = create_value_subarray::<_, 1, 1>(
                any_crosses_periodic_bound,
                values,
                &start_indices,
            );

            let (x_coord_subarray, y_coord_subarray, z_coord_subarray) =
                create_coordinate_subarrays::<_, 1>(
                    &crosses_periodic_bound,
                    coords,
                    grid.extents(),
                    &start_indices,
                );

            interp_subarrays::<_, 1, 1, 1>(
                In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray),
                &value_subarray,
                interp_point,
            )
        } else {
            let (x_coord_subarray, y_coord_subarray, z_coord_subarray) =
                create_coordinate_subarrays::<_, N_POINTS>(
                    &crosses_periodic_bound,
                    coords,
                    grid.extents(),
                    &start_indices,
                );

            interp_subarrays::<_, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
                In3D::new(&x_coord_subarray, &y_coord_subarray, &z_coord_subarray),
                &value_subarray,
                interp_point,
            )
        }
    }
}

fn interp_scalar_field_in_known_grid_cell<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F, G>,
    interp_point: &Point3<F>,
    interp_indices: &Idx3<usize>,
    variation_threshold_for_linear: F,
) -> F {
    interp::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field.grid(),
        &field.coords(),
        field.locations(),
        field.values(),
        interp_point,
        interp_indices,
        variation_threshold_for_linear,
    )
}

fn interp_scalar_field_from_grid_point_query<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F, G>,
    grid_point_query: GridPointQuery3<F, Idx3<usize>>,
    interp_point: &Point3<F>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<F, F> {
    match grid_point_query {
        GridPointQuery3::Inside(interp_indices) => {
            GridPointQuery3::Inside(interp_scalar_field_in_known_grid_cell::<
                _,
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
                interp_scalar_field_in_known_grid_cell::<
                    _,
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

fn interp_vector_field_in_known_grid_cell<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F, G>,
    interp_point: &Point3<F>,
    interp_indices: &Idx3<usize>,
    variation_threshold_for_linear: F,
) -> Vec3<F> {
    let grid = field.grid();
    Vec3::new(
        interp::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
            grid,
            &field.coords(X),
            field.locations(X),
            &field.values(X),
            interp_point,
            interp_indices,
            variation_threshold_for_linear,
        ),
        interp::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
            grid,
            &field.coords(Y),
            field.locations(Y),
            &field.values(Y),
            interp_point,
            interp_indices,
            variation_threshold_for_linear,
        ),
        interp::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
            grid,
            &field.coords(Z),
            field.locations(Z),
            &field.values(Z),
            interp_point,
            interp_indices,
            variation_threshold_for_linear,
        ),
    )
}

fn interp_vector_field_from_grid_point_query<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F, G>,
    grid_point_query: GridPointQuery3<F, Idx3<usize>>,
    interp_point: &Point3<F>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<F, Vec3<F>> {
    match grid_point_query {
        GridPointQuery3::Inside(interp_indices) => {
            GridPointQuery3::Inside(interp_vector_field_in_known_grid_cell::<
                _,
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
                interp_vector_field_in_known_grid_cell::<
                    _,
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

fn interp_scalar_field<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F, G>,
    interp_point: &Point3<F>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<F, F> {
    let grid_point_query = field.grid().find_grid_cell(interp_point);
    interp_scalar_field_from_grid_point_query::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_scalar_field<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &ScalarField3<F, G>,
    interp_point: &Point3<F>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<F, F> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_point);
    interp_scalar_field_from_grid_point_query::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_vector_field<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F, G>,
    interp_point: &Point3<F>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<F, Vec3<F>> {
    let grid_point_query = field.grid().find_grid_cell(interp_point);
    interp_vector_field_from_grid_point_query::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
        field,
        grid_point_query,
        interp_point,
        variation_threshold_for_linear,
    )
}

fn interp_extrap_vector_field<
    F: BFloat,
    G: Grid3<F>,
    const N_POINTS: usize,
    const N_POINTS_SQUARED: usize,
    const N_POINTS_CUBED: usize,
>(
    field: &VectorField3<F, G>,
    interp_point: &Point3<F>,
    variation_threshold_for_linear: F,
) -> GridPointQuery3<F, Vec3<F>> {
    let grid_point_query = field.grid().find_closest_grid_cell(interp_point);
    interp_vector_field_from_grid_point_query::<_, _, N_POINTS, N_POINTS_SQUARED, N_POINTS_CUBED>(
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

impl Interpolator3 for PolyFitInterpolator3 {
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
            1 => interp_scalar_field::<_, _, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field::<_, _, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field::<_, _, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field::<_, _, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field::<_, _, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

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
            1 => interp_scalar_field_in_known_grid_cell::<_, _, 2, 4, 8>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            2 => interp_scalar_field_in_known_grid_cell::<_, _, 3, 9, 27>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            3 => interp_scalar_field_in_known_grid_cell::<_, _, 4, 16, 64>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            4 => interp_scalar_field_in_known_grid_cell::<_, _, 5, 25, 125>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            5 => interp_scalar_field_in_known_grid_cell::<_, _, 6, 36, 216>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

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
            1 => interp_extrap_scalar_field::<_, _, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_scalar_field::<_, _, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_scalar_field::<_, _, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_scalar_field::<_, _, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_scalar_field::<_, _, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

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
            1 => interp_vector_field::<_, _, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_vector_field::<_, _, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_vector_field::<_, _, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_vector_field::<_, _, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_vector_field::<_, _, 6, 36, 216>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

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
            1 => interp_vector_field_in_known_grid_cell::<_, _, 2, 4, 8>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            2 => interp_vector_field_in_known_grid_cell::<_, _, 3, 9, 27>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            3 => interp_vector_field_in_known_grid_cell::<_, _, 4, 16, 64>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            4 => interp_vector_field_in_known_grid_cell::<_, _, 5, 25, 125>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            5 => interp_vector_field_in_known_grid_cell::<_, _, 6, 36, 216>(
                field,
                interp_point,
                interp_indices,
                variation_threshold_for_linear,
            ),
            order => panic!("Invalid interpolation order: {}", order),
        }
    }

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
            1 => interp_extrap_vector_field::<_, _, 2, 4, 8>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            2 => interp_extrap_vector_field::<_, _, 3, 9, 27>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            3 => interp_extrap_vector_field::<_, _, 4, 16, 64>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            4 => interp_extrap_vector_field::<_, _, 5, 25, 125>(
                field,
                interp_point,
                variation_threshold_for_linear,
            ),
            5 => interp_extrap_vector_field::<_, _, 6, 36, 216>(
                field,
                interp_point,
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::field::ResampledCoordLocation;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::io::snapshot::{fdt, SnapshotReader3, SnapshotReaderConfig};
    use crate::io::{Endianness, Verbose};
    use ndarray_stats::QuantileExt;

    #[test]
    fn interpolation_at_original_data_points_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new(SnapshotReaderConfig::new(
            "data/cb24ni_ebeam_offline/cb24ni_ebeam_offline_462.idl",
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
