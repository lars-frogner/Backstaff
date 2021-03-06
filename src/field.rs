//! Scalar and vector fields.

pub mod quantities;

use crate::{
    geometry::{
        CoordRefs2, CoordRefs3, Coords2, Dim2,
        Dim3::{self, X, Y, Z},
        Idx2, Idx3, In2D, In3D, Point3, Vec2, Vec3,
    },
    grid::{regular::RegularGrid2, CoordLocation, Grid1, Grid2, Grid3},
    interpolation::Interpolator3,
    io::utils::save_data_as_pickle,
    num::{BFloat, OrderableIndexValuePair},
};
use itertools::Itertools;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use std::{io, iter, path::Path, sync::Arc};

/// Location in the grid cell for resampled field values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResampledCoordLocation {
    Original,
    Specific(CoordLocation),
}

impl ResampledCoordLocation {
    pub fn center() -> Self {
        Self::Specific(CoordLocation::Center)
    }
    pub fn lower_edge() -> Self {
        Self::Specific(CoordLocation::LowerEdge)
    }
    pub fn into_location(self, original: CoordLocation) -> CoordLocation {
        match self {
            Self::Original => original,
            Self::Specific(location) => location,
        }
    }
    pub fn convert_to_locations_3d(
        resampled: In3D<Self>,
        original: &In3D<CoordLocation>,
    ) -> In3D<CoordLocation> {
        In3D::new(
            resampled[X].into_location(original[X]),
            resampled[Y].into_location(original[Y]),
            resampled[Z].into_location(original[Z]),
        )
    }
}

/// Method for resampling a field.
#[derive(Clone, Copy, Debug)]
pub enum ResamplingMethod {
    WeightedSampleAveraging,
    WeightedCellAveraging,
    DirectSampling,
}

/// A 3D scalar field.
///
/// Holds the grid and values of a 3D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct ScalarField3<F, G>
where
    F: BFloat,
    G: Grid3<F>,
{
    name: String,
    grid: Arc<G>,
    locations: In3D<CoordLocation>,
    values: Array3<F>,
}

impl<F, G> ScalarField3<F, G>
where
    F: BFloat,
    G: Grid3<F>,
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate locations specifying where in the grid cell the values are defined.
    pub fn new(
        name: String,
        grid: Arc<G>,
        locations: In3D<CoordLocation>,
        values: Array3<F>,
    ) -> Self {
        let grid_shape = grid.shape();
        let values_shape = values.shape();
        assert!(
            grid_shape[X] == values_shape[0]
                && grid_shape[Y] == values_shape[1]
                && grid_shape[Z] == values_shape[2],
            "Shape of grid does not match shape of array of values."
        );
        Self {
            name,
            grid,
            locations,
            values,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs3<F> {
        CoordRefs3::new(
            &self.grid.coords_by_type(self.locations[X])[X],
            &self.grid.coords_by_type(self.locations[Y])[Y],
            &self.grid.coords_by_type(self.locations[Z])[Z],
        )
    }

    /// Returns a reference to the 3D array of field values.
    pub fn values(&self) -> &Array3<F> {
        &self.values
    }

    /// Returns a mutable reference to the 3D array of field values.
    pub fn values_mut(&mut self) -> &mut Array3<F> {
        &mut self.values
    }

    /// Returns the field value at the given 3D index.
    pub fn value(&self, indices: &Idx3<usize>) -> F {
        self.values[[indices[X], indices[Y], indices[Z]]]
    }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> {
        self.grid.shape()
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values are defined.
    pub fn locations(&self) -> &In3D<CoordLocation> {
        &self.locations
    }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array3<F> {
        self.values
    }

    /// Computes the 3D indices and value of the minimum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_minimum(&self) -> Option<(Idx3<usize>, F)> {
        self.values
            .as_slice_memory_order()
            .unwrap()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value.is_nan() {
                    None
                } else {
                    Some(OrderableIndexValuePair(idx, value))
                }
            })
            .min()
            .map(|OrderableIndexValuePair(idx_of_min_value, min_value)| {
                (
                    compute_3d_array_indices_from_flat_idx(self.shape(), idx_of_min_value),
                    min_value,
                )
            })
    }

    /// Computes the 3D indices and value of the maximum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_maximum(&self) -> Option<(Idx3<usize>, F)> {
        self.values
            .as_slice_memory_order()
            .unwrap()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value.is_nan() {
                    None
                } else {
                    Some(OrderableIndexValuePair(idx, value))
                }
            })
            .max()
            .map(|OrderableIndexValuePair(idx_of_max_value, max_value)| {
                (
                    compute_3d_array_indices_from_flat_idx(self.shape(), idx_of_max_value),
                    max_value,
                )
            })
    }

    /// Resamples the scalar field onto the given grid using the given method and
    /// returns the resampled field.
    pub fn resampled_to_grid<H, I>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        method: ResamplingMethod,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<F>,
        I: Interpolator3,
    {
        match method {
            ResamplingMethod::WeightedSampleAveraging => self
                .resampled_to_grid_with_weighted_sample_averaging(
                    grid,
                    resampled_locations,
                    interpolator,
                ),
            ResamplingMethod::WeightedCellAveraging => {
                self.resampled_to_grid_with_weighted_cell_averaging(grid, resampled_locations)
            }
            ResamplingMethod::DirectSampling => {
                self.resampled_to_grid_with_direct_sampling(grid, resampled_locations, interpolator)
            }
        }
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, values are interpolated from all overlapped original
    /// grid cells and averaged with weights according to the intersected volumes.
    /// If the new grid cell is contained within an original grid cell, this reduces
    /// to a single interpolation.
    ///
    /// This method gives robust results for arbitrary resampling grids, but is slower
    /// than direct sampling or weighted cell averaging.
    pub fn resampled_to_grid_with_weighted_sample_averaging<H, I>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<F>,
        I: Interpolator3,
    {
        let underlying_locations = self.locations();
        let underlying_lower_edges = self.grid().lower_edges();
        let underlying_extents = self.grid().extents();

        let overlying_grid = grid;
        let overlying_locations = ResampledCoordLocation::convert_to_locations_3d(
            resampled_locations,
            underlying_locations,
        );
        let mut overlying_values =
            unsafe { Array3::uninitialized(overlying_grid.shape().to_tuple().f()) };
        let overlying_values_buffer = overlying_values.as_slice_memory_order_mut().unwrap();

        overlying_values_buffer.par_iter_mut().enumerate().for_each(
            |(overlying_idx, overlying_value)| {
                let (mut lower_overlying_corner, mut upper_overlying_corner) =
                    Self::compute_overlying_grid_cell_corners_for_resampling(
                        overlying_grid.as_ref(),
                        overlying_idx,
                    );
                Self::shift_overlying_grid_cell_corners_for_weighted_sample_averaging(
                    &overlying_locations,
                    &mut lower_overlying_corner,
                    &mut upper_overlying_corner,
                );

                let idx_range_lists = self.compute_underlying_grid_cell_idx_ranges_for_resampling(
                    &lower_overlying_corner,
                    &upper_overlying_corner,
                );

                let compute_overlap_centers_and_lengths_along_dim = |dim| {
                    iter::once(&lower_overlying_corner[dim]) // First edge is lower edge of overlying cell
                        .chain(
                            // Next are the lower edges of the underlying cells completely inside the overlying cell
                            idx_range_lists[dim]
                                .iter()
                                .skip(1) // Skip first underlying lower edge since it is outside lower edge of overlying cell
                                .map(|&idx| &underlying_lower_edges[dim][idx]),
                        )
                        .chain(iter::once(&upper_overlying_corner[dim])) // Last edge is the upper edge of the overlying cell
                        .tuple_windows() // Create sliding window iterator over edge pairs
                        .map(|(&lower_coord, &upper_coord)| {
                            let overlap_length = ((upper_coord - lower_coord)
                                + underlying_extents[dim])
                                % underlying_extents[dim]; // Make sure coordinate difference is correct also for wrapped coordinates
                            let overlap_center =
                                lower_coord + overlap_length * F::from_f32(0.5).unwrap();
                            (overlap_center, overlap_length)
                        })
                        .collect::<Vec<_>>()
                };

                // Compute the center points and extents of the "sub grid cells" found
                // by intersecting the underlying grid with the overlying grid.
                let overlap_centers_and_lengths = In3D::new(
                    compute_overlap_centers_and_lengths_along_dim(X),
                    compute_overlap_centers_and_lengths_along_dim(Y),
                    compute_overlap_centers_and_lengths_along_dim(Z),
                );

                let mut accum_value = F::zero();
                let mut accum_weight = F::zero();

                // Accumulate the interpolated value from each sub grid cell center,
                // weighted with the relative volume of the sub grid cell.
                for &(overlap_center_z, overlap_length_z) in &overlap_centers_and_lengths[Z] {
                    for &(overlap_center_y, overlap_length_y) in &overlap_centers_and_lengths[Y] {
                        for &(overlap_center_x, overlap_length_x) in &overlap_centers_and_lengths[X]
                        {
                            let weight = overlap_length_x * overlap_length_y * overlap_length_z;

                            accum_value = accum_value
                                + interpolator
                                    .interp_extrap_scalar_field(
                                        self,
                                        &Point3::new(
                                            overlap_center_x,
                                            overlap_center_y,
                                            overlap_center_z,
                                        ),
                                    )
                                    .expect_inside_or_moved()
                                    * weight;

                            accum_weight = accum_weight + weight;
                        }
                    }
                }
                *overlying_value = accum_value / accum_weight;
            },
        );
        ScalarField3::new(
            self.name.clone(),
            overlying_grid,
            overlying_locations,
            overlying_values,
        )
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, the values of all overlapped original grid cells are
    /// averaged with weights according to the intersected volumes.
    ///
    /// This method is suited for downsampling. It is faster than weighted sample
    /// averaging, but slightly less accurate.
    pub fn resampled_to_grid_with_weighted_cell_averaging<H: Grid3<F>>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
    ) -> ScalarField3<F, H> {
        let underlying_locations = self.locations();
        let underlying_lower_edges = self.grid().lower_edges();
        let underlying_extents = self.grid().extents();
        let average_underlying_cell_extents = self.grid().average_grid_cell_extents();

        let overlying_grid = grid;
        let overlying_locations = ResampledCoordLocation::convert_to_locations_3d(
            resampled_locations,
            underlying_locations,
        );
        let mut overlying_values =
            unsafe { Array3::uninitialized(overlying_grid.shape().to_tuple().f()) };
        let overlying_values_buffer = overlying_values.as_slice_memory_order_mut().unwrap();

        overlying_values_buffer.par_iter_mut().enumerate().for_each(
            |(overlying_idx, overlying_value)| {
                let (mut lower_overlying_corner, mut upper_overlying_corner) =
                    Self::compute_overlying_grid_cell_corners_for_resampling(
                        overlying_grid.as_ref(),
                        overlying_idx,
                    );
                Self::shift_overlying_grid_cell_corners_for_weighted_cell_averaging(
                    underlying_locations,
                    &overlying_locations,
                    &average_underlying_cell_extents,
                    &mut lower_overlying_corner,
                    &mut upper_overlying_corner,
                );

                let idx_range_lists = self.compute_underlying_grid_cell_idx_ranges_for_resampling(
                    &lower_overlying_corner,
                    &upper_overlying_corner,
                );

                let compute_overlap_lengths_and_indices_along_dim = |dim| {
                    iter::once(&lower_overlying_corner[dim]) // First edge is lower edge of overlying cell
                        .chain(
                            // Next are the lower edges of the underlying cells completely inside the overlying cell
                            idx_range_lists[dim]
                                .iter()
                                .skip(1) // Skip first underlying lower edge since it is outside lower edge of overlying cell
                                .map(|&idx| &underlying_lower_edges[dim][idx]),
                        )
                        .chain(iter::once(&upper_overlying_corner[dim])) // Last edge is the upper edge of the overlying cell
                        .tuple_windows() // Create sliding window iterator over edge pairs
                        .map(|(&lower_coord, &upper_coord)| {
                            ((upper_coord - lower_coord) + underlying_extents[dim])
                                % underlying_extents[dim] // Make sure coordinate difference is correct also for wrapped coordinates
                        })
                        .zip(idx_range_lists[dim].iter())
                        .collect::<Vec<_>>()
                };

                // Compute the extents of the "sub grid cells" found by intersecting
                // the underlying grid with the overlying grid, as well as the indices
                // of the underlying grid cells.
                let overlap_lengths_and_indices = In3D::new(
                    compute_overlap_lengths_and_indices_along_dim(X),
                    compute_overlap_lengths_and_indices_along_dim(Y),
                    compute_overlap_lengths_and_indices_along_dim(Z),
                );

                let mut accum_value = F::zero();
                let mut accum_weight = F::zero();

                // Accumulate the value from each sub grid cell, weighted with the
                // relative volume of the sub grid cell.
                for &(overlap_length_z, &k) in &overlap_lengths_and_indices[Z] {
                    for &(overlap_length_y, &j) in &overlap_lengths_and_indices[Y] {
                        for &(overlap_length_x, &i) in &overlap_lengths_and_indices[X] {
                            let weight = overlap_length_x * overlap_length_y * overlap_length_z;
                            accum_value = accum_value + self.value(&Idx3::new(i, j, k)) * weight;
                            accum_weight = accum_weight + weight;
                        }
                    }
                }
                *overlying_value = accum_value / accum_weight;
            },
        );
        ScalarField3::new(
            self.name.clone(),
            overlying_grid,
            overlying_locations,
            overlying_values,
        )
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    ///
    /// Each value on the new grid is found by interpolation of the values on the old grid
    /// at the new coordinate location.
    ///
    /// This is the preferred method for upsampling. For heavy downsampling it yields a
    /// more noisy result than weighted averaging.
    pub fn resampled_to_grid_with_direct_sampling<H, I>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<F>,
        I: Interpolator3,
    {
        let locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let new_coords = Self::coords_from_grid(grid.as_ref(), &locations);

        let grid_shape = grid.shape();
        let mut new_values = unsafe { Array3::uninitialized(grid_shape.to_tuple().f()) };
        let values_buffer = new_values.as_slice_memory_order_mut().unwrap();

        values_buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let indices = compute_3d_array_indices_from_flat_idx(&grid_shape, idx);
                let point = new_coords.point(&indices);
                *value = interpolator
                    .interp_extrap_scalar_field(self, &point)
                    .expect_inside_or_moved();
            });
        ScalarField3::new(self.name.clone(), grid, locations, new_values)
    }

    /// Returns a view of the 2D slice of the field located at the given index along the given axis.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> ArrayView2<F> {
        self.values.index_axis(Axis(axis as usize), idx)
    }

    /// Returns a 2D scalar field corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(
        &self,
        interpolator: &I,
        x_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::XSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.slice_across_x();
        self.create_slice_across_x(
            Arc::new(slice_grid),
            interpolator,
            x_coord,
            resampled_location,
        )
    }

    /// Returns a 2D scalar field corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(
        &self,
        interpolator: &I,
        y_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::YSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.slice_across_y();
        self.create_slice_across_y(
            Arc::new(slice_grid),
            interpolator,
            y_coord,
            resampled_location,
        )
    }

    /// Returns a 2D scalar field corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(
        &self,
        interpolator: &I,
        z_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::ZSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.slice_across_z();
        self.create_slice_across_z(
            Arc::new(slice_grid),
            interpolator,
            z_coord,
            resampled_location,
        )
    }

    /// Returns a 2D scalar field corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(
        &self,
        interpolator: &I,
        axis: Dim3,
        coord: F,
        location: CoordLocation,
    ) -> ScalarField2<F, RegularGrid2<F>>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.regular_slice_across_axis(axis);
        self.create_regular_slice_across_axis(
            Arc::new(slice_grid),
            interpolator,
            axis,
            coord,
            location,
        )
    }

    fn compute_overlying_grid_cell_corners_for_resampling<H: Grid3<F>>(
        overlying_grid: &H,
        overlying_grid_cell_idx: usize,
    ) -> (Point3<F>, Point3<F>) {
        let overlying_indices =
            compute_3d_array_indices_from_flat_idx(overlying_grid.shape(), overlying_grid_cell_idx);
        overlying_grid.grid_cell_extremal_corners(&overlying_indices)
    }

    fn shift_overlying_grid_cell_corners_for_weighted_sample_averaging(
        overlying_locations: &In3D<CoordLocation>,
        lower_overlying_corner: &mut Point3<F>,
        upper_overlying_corner: &mut Point3<F>,
    ) {
        for &dim in &Dim3::slice() {
            if let CoordLocation::LowerEdge = overlying_locations[dim] {
                // Shift the overlying grid cell half a cell down to be centered around the
                // location of the value to estimate
                let shift = -(upper_overlying_corner[dim] - lower_overlying_corner[dim])
                    * F::from_f32(0.5).unwrap();
                lower_overlying_corner[dim] = lower_overlying_corner[dim] + shift;
                upper_overlying_corner[dim] = upper_overlying_corner[dim] + shift;
            }
        }
    }

    fn shift_overlying_grid_cell_corners_for_weighted_cell_averaging(
        underlying_locations: &In3D<CoordLocation>,
        overlying_locations: &In3D<CoordLocation>,
        average_underlying_cell_extents: &Vec3<F>,
        lower_overlying_corner: &mut Point3<F>,
        upper_overlying_corner: &mut Point3<F>,
    ) {
        for &dim in &Dim3::slice() {
            let mut shift = F::zero();
            if let CoordLocation::LowerEdge = underlying_locations[dim] {
                // Shift overlying grid cell half an underlying grid cell up to compensate
                // for downward bias due the underlying values being located on lower edges
                shift = shift + average_underlying_cell_extents[dim];
            }
            if let CoordLocation::LowerEdge = overlying_locations[dim] {
                // Shift the overlying grid cell half a cell down to be centered around the
                // location of the value to estimate
                shift = shift - (upper_overlying_corner[dim] - lower_overlying_corner[dim]);
            }
            shift = shift * F::from_f32(0.5).unwrap();
            lower_overlying_corner[dim] = lower_overlying_corner[dim] + shift;
            upper_overlying_corner[dim] = upper_overlying_corner[dim] + shift;
        }
    }

    fn compute_underlying_grid_cell_idx_ranges_for_resampling(
        &self,
        lower_overlying_corner: &Point3<F>,
        upper_overlying_corner: &Point3<F>,
    ) -> In3D<Vec<usize>> {
        let lower_underlying_indices = self
            .grid
            .find_closest_grid_cell(lower_overlying_corner)
            .expect_inside_or_moved();
        let upper_underlying_indices = self
            .grid
            .find_closest_grid_cell(upper_overlying_corner)
            .expect_inside_or_moved();

        In3D::new(
            self.grid.create_idx_range_list_wrapped(
                X,
                lower_underlying_indices[X],
                upper_underlying_indices[X],
            ),
            self.grid.create_idx_range_list_wrapped(
                Y,
                lower_underlying_indices[Y],
                upper_underlying_indices[Y],
            ),
            self.grid.create_idx_range_list_wrapped(
                Z,
                lower_underlying_indices[Z],
                upper_underlying_indices[Z],
            ),
        )
    }

    fn coords_from_grid<'a, 'b, H: Grid3<F>>(
        grid: &'a H,
        locations: &'b In3D<CoordLocation>,
    ) -> CoordRefs3<'a, F> {
        CoordRefs3::new(
            &grid.coords_by_type(locations[X])[X],
            &grid.coords_by_type(locations[Y])[Y],
            &grid.coords_by_type(locations[Z])[Z],
        )
    }

    fn set_grid(&mut self, new_grid: Arc<G>) {
        let grid_shape = new_grid.shape();
        let values_shape = self.values.shape();
        assert!(
            grid_shape[X] == values_shape[0]
                && grid_shape[Y] == values_shape[1]
                && grid_shape[Z] == values_shape[2],
            "Shape of new grid does not match shape of array of values."
        );
        self.grid = new_grid;
    }

    fn create_slice_across_x<I>(
        &self,
        slice_grid: Arc<G::XSliceGrid>,
        interpolator: &I,
        x_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::XSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_locations = self.select_slice_locations([Y, Z], resampled_location);
        let slice_values =
            self.compute_slice_values(interpolator, X, x_coord, resampled_location, false);
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn create_slice_across_y<I>(
        &self,
        slice_grid: Arc<G::YSliceGrid>,
        interpolator: &I,
        y_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::YSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_locations = self.select_slice_locations([X, Z], resampled_location);
        let slice_values =
            self.compute_slice_values(interpolator, Y, y_coord, resampled_location, false);
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn create_slice_across_z<I>(
        &self,
        slice_grid: Arc<G::ZSliceGrid>,
        interpolator: &I,
        z_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::ZSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_locations = self.select_slice_locations([X, Y], resampled_location);
        let slice_values =
            self.compute_slice_values(interpolator, Z, z_coord, resampled_location, false);
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn create_regular_slice_across_axis<I>(
        &self,
        slice_grid: Arc<RegularGrid2<F>>,
        interpolator: &I,
        axis: Dim3,
        coord: F,
        location: CoordLocation,
    ) -> ScalarField2<F, RegularGrid2<F>>
    where
        I: Interpolator3,
    {
        let slice_locations = In2D::same(location);
        let slice_values = self.compute_slice_values(
            interpolator,
            axis,
            coord,
            ResampledCoordLocation::Specific(location),
            true,
        );
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn compute_slice_indices_from_flat_idx(&self, axes: [Dim3; 2], idx: usize) -> [usize; 2] {
        let shape = self.shape();
        let indices =
            compute_2d_array_indices_from_flat_idx(&In2D::new(shape[axes[0]], shape[axes[1]]), idx);
        [indices[Dim2::X], indices[Dim2::Y]]
    }

    fn select_slice_locations(
        &self,
        axes: [Dim3; 2],
        resampled_location: ResampledCoordLocation,
    ) -> In2D<CoordLocation> {
        match resampled_location {
            ResampledCoordLocation::Original => {
                In2D::new(self.locations[axes[0]], self.locations[axes[1]])
            }
            ResampledCoordLocation::Specific(location) => In2D::same(location),
        }
    }

    fn select_slice_coords(
        &self,
        axes: [Dim3; 2],
        resampled_location: ResampledCoordLocation,
    ) -> [&[F]; 2] {
        match resampled_location {
            ResampledCoordLocation::Original => {
                let coords = self.coords();
                [coords[axes[0]], coords[axes[1]]]
            }
            ResampledCoordLocation::Specific(CoordLocation::Center) => {
                let centers = self.grid.centers();
                [&centers[axes[0]], &centers[axes[1]]]
            }
            ResampledCoordLocation::Specific(CoordLocation::LowerEdge) => {
                let lower_edges = self.grid.lower_edges();
                [&lower_edges[axes[0]], &lower_edges[axes[1]]]
            }
        }
    }

    fn select_regular_slice_coords(&self, axes: [Dim3; 2], location: CoordLocation) -> [&[F]; 2] {
        match location {
            CoordLocation::Center => {
                let regular_centers = self.grid.regular_centers();
                [&regular_centers[axes[0]], &regular_centers[axes[1]]]
            }
            CoordLocation::LowerEdge => {
                let regular_lower_edges = self.grid.regular_lower_edges();
                [&regular_lower_edges[axes[0]], &regular_lower_edges[axes[1]]]
            }
        }
    }

    fn compute_slice_values<I>(
        &self,
        interpolator: &I,
        axis: Dim3,
        coord: F,
        resampled_location: ResampledCoordLocation,
        regular: bool,
    ) -> Array2<F>
    where
        I: Interpolator3,
    {
        let lower_bound = self.grid.lower_bounds()[axis];
        let upper_bound = self.grid.upper_bounds()[axis];
        if coord < lower_bound || coord >= upper_bound {
            panic!("Slicing coordinate is outside the grid bounds.");
        }

        let axes = Dim3::slice_except(axis);

        let coords = if regular {
            if let ResampledCoordLocation::Specific(location) = resampled_location {
                self.select_regular_slice_coords(axes, location)
            } else {
                panic!("Original coord locations not supported for regular slice.")
            }
        } else {
            self.select_slice_coords(axes, resampled_location)
        };

        self.interpolate_slice_values(interpolator, axes, &coords, coord)
    }

    fn interpolate_slice_values<I>(
        &self,
        interpolator: &I,
        axes: [Dim3; 2],
        coords: &[&[F]; 2],
        slice_axis_coord: F,
    ) -> Array2<F>
    where
        I: Interpolator3,
    {
        let slice_shape = (coords[0].len(), coords[1].len());
        let mut slice_values = unsafe { Array2::uninitialized(slice_shape.f()) };
        let values_buffer = slice_values.as_slice_memory_order_mut().unwrap();

        values_buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let [idx_0, idx_1] = self.compute_slice_indices_from_flat_idx(axes, idx);
                let mut point_in_slice = Point3::equal_components(slice_axis_coord);
                point_in_slice[axes[0]] = coords[0][idx_0];
                point_in_slice[axes[1]] = coords[1][idx_1];
                *value = interpolator
                    .interp_scalar_field(self, &point_in_slice)
                    .expect_inside();
            });
        slice_values
    }
}

/// A 3D vector field.
///
/// Holds the grid and values of the three components of a 3D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct VectorField3<F, G>
where
    F: BFloat,
    G: Grid3<F>,
{
    name: String,
    grid: Arc<G>,
    components: In3D<ScalarField3<F, G>>,
}

impl<F, G> VectorField3<F, G>
where
    F: BFloat,
    G: Grid3<F>,
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In3D<ScalarField3<F, G>>) -> Self {
        components[X].set_grid(Arc::clone(&grid));
        components[Y].set_grid(Arc::clone(&grid));
        components[Z].set_grid(Arc::clone(&grid));
        Self {
            name,
            grid,
            components,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim3) -> &ScalarField3<F, G> {
        &self.components[dim]
    }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim3) -> CoordRefs3<F> {
        self.components[dim].coords()
    }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In3D<CoordRefs3<F>> {
        In3D::new(self.coords(X), self.coords(Y), self.coords(Z))
    }

    /// Returns a reference to the 3D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim3) -> &Array3<F> {
        self.components[dim].values()
    }

    /// Returns a mutable reference to the 3D array of field values
    /// for the specified component.
    pub fn values_mut(&mut self, dim: Dim3) -> &mut Array3<F> {
        self.components[dim].values_mut()
    }

    /// Returns a reference to the 3D array of field values for each component.
    pub fn all_values(&self) -> In3D<&Array3<F>> {
        In3D::new(self.values(X), self.values(Y), self.values(Z))
    }

    /// Returns the field vector at the given 3D index.
    pub fn vector(&self, indices: &Idx3<usize>) -> Vec3<F> {
        Vec3::new(
            self.values(X)[[indices[X], indices[Y], indices[Z]]],
            self.values(Y)[[indices[X], indices[Y], indices[Z]]],
            self.values(Z)[[indices[X], indices[Y], indices[Z]]],
        )
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim3) -> &In3D<CoordLocation> {
        self.components[dim].locations()
    }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> {
        self.grid.shape()
    }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, values are interpolated from all overlapped original
    /// grid cells and averaged with weights according to the intersected volumes.
    /// If the new grid cell is contained within an original grid cell, this reduces
    /// to a single interpolation.
    ///
    /// This method gives robust results for arbitrary resampling grids, but is slower
    /// than direct sampling or weighted cell averaging.
    pub fn resampled_to_grid_with_weighted_sample_averaging<H, I>(
        &self,
        grid: Arc<H>,
        interpolator: &I,
    ) -> VectorField3<F, H>
    where
        H: Grid3<F>,
        I: Interpolator3,
    {
        let components = In3D::new(
            self.components[X].resampled_to_grid_with_weighted_sample_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
            ),
            self.components[Y].resampled_to_grid_with_weighted_sample_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
            ),
            self.components[Z].resampled_to_grid_with_weighted_sample_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
            ),
        );
        VectorField3::new(self.name.clone(), grid, components)
    }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, the values of all overlapped original grid cells are
    /// averaged with weights according to the intersected volumes.
    ///
    /// This method is suited for downsampling. It is faster than weighted sample
    /// averaging, but slightly less accurate.
    pub fn resampled_to_grid_with_weighted_cell_averaging<H: Grid3<F>>(
        &self,
        grid: Arc<H>,
    ) -> VectorField3<F, H> {
        let components = In3D::new(
            self.components[X].resampled_to_grid_with_weighted_cell_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
            ),
            self.components[Y].resampled_to_grid_with_weighted_cell_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
            ),
            self.components[Z].resampled_to_grid_with_weighted_cell_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
            ),
        );
        VectorField3::new(self.name.clone(), grid, components)
    }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    ///
    /// Each value on the new grid is found by interpolation of the values on the old grid
    /// at the new coordinate location.
    ///
    /// This is the preferred method for upsampling. For heavy downsampling it yields a
    /// more noisy result than weighted averaging.
    pub fn resampled_to_grid_with_direct_sampling<H, I>(
        &self,
        grid: Arc<H>,
        interpolator: &I,
    ) -> VectorField3<F, H>
    where
        H: Grid3<F>,
        I: Interpolator3,
    {
        let components = In3D::new(
            self.components[X].resampled_to_grid_with_direct_sampling(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
            ),
            self.components[Y].resampled_to_grid_with_direct_sampling(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
            ),
            self.components[Z].resampled_to_grid_with_direct_sampling(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
            ),
        );
        VectorField3::new(self.name.clone(), grid, components)
    }

    /// Returns a view of the 2D slice located at the given index along the given dimension,
    /// for each component of the field.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> In3D<ArrayView2<F>> {
        In3D::new(
            self.components[X].slice_across_axis_at_idx(axis, idx),
            self.components[Y].slice_across_axis_at_idx(axis, idx),
            self.components[Z].slice_across_axis_at_idx(axis, idx),
        )
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(
        &self,
        interpolator: &I,
        x_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> PlaneVectorField3<F, G::XSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.slice_across_x());
        let slice_field_components = In3D::new(
            self.components[X].create_slice_across_x(
                Arc::clone(&slice_grid),
                interpolator,
                x_coord,
                resampled_location,
            ),
            self.components[Y].create_slice_across_x(
                Arc::clone(&slice_grid),
                interpolator,
                x_coord,
                resampled_location,
            ),
            self.components[Z].create_slice_across_x(
                Arc::clone(&slice_grid),
                interpolator,
                x_coord,
                resampled_location,
            ),
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(
        &self,
        interpolator: &I,
        y_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> PlaneVectorField3<F, G::YSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.slice_across_y());
        let slice_field_components = In3D::new(
            self.components[X].create_slice_across_y(
                Arc::clone(&slice_grid),
                interpolator,
                y_coord,
                resampled_location,
            ),
            self.components[Y].create_slice_across_y(
                Arc::clone(&slice_grid),
                interpolator,
                y_coord,
                resampled_location,
            ),
            self.components[Z].create_slice_across_y(
                Arc::clone(&slice_grid),
                interpolator,
                y_coord,
                resampled_location,
            ),
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(
        &self,
        interpolator: &I,
        z_coord: F,
        resampled_location: ResampledCoordLocation,
    ) -> PlaneVectorField3<F, G::ZSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.slice_across_z());
        let slice_field_components = In3D::new(
            self.components[X].create_slice_across_z(
                Arc::clone(&slice_grid),
                interpolator,
                z_coord,
                resampled_location,
            ),
            self.components[Y].create_slice_across_z(
                Arc::clone(&slice_grid),
                interpolator,
                z_coord,
                resampled_location,
            ),
            self.components[Z].create_slice_across_z(
                Arc::clone(&slice_grid),
                interpolator,
                z_coord,
                resampled_location,
            ),
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(
        &self,
        interpolator: &I,
        axis: Dim3,
        coord: F,
        location: CoordLocation,
    ) -> PlaneVectorField3<F, RegularGrid2<F>>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.regular_slice_across_axis(axis));
        let slice_field_components = In3D::new(
            self.components[X].create_regular_slice_across_axis(
                Arc::clone(&slice_grid),
                interpolator,
                axis,
                coord,
                location,
            ),
            self.components[Y].create_regular_slice_across_axis(
                Arc::clone(&slice_grid),
                interpolator,
                axis,
                coord,
                location,
            ),
            self.components[Z].create_regular_slice_across_axis(
                Arc::clone(&slice_grid),
                interpolator,
                axis,
                coord,
                location,
            ),
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }
}

/// A 2D scalar field.
///
/// Holds the grid and values of a 2D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct ScalarField2<F, G>
where
    F: BFloat,
    G: Grid2<F>,
{
    name: String,
    grid: Arc<G>,
    locations: In2D<CoordLocation>,
    values: Array2<F>,
}

#[derive(Serialize)]
struct ScalarFieldSerializeData2<F: BFloat> {
    coords: Coords2<F>,
    values: Array2<F>,
}

impl<F, G> ScalarField2<F, G>
where
    F: BFloat,
    G: Grid2<F>,
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate locations specifying where in the grid cell the values are defined.
    pub fn new(
        name: String,
        grid: Arc<G>,
        locations: In2D<CoordLocation>,
        values: Array2<F>,
    ) -> Self {
        Self {
            name,
            grid,
            locations,
            values,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs2<F> {
        CoordRefs2::new(
            &self.grid.coords_by_type(self.locations[Dim2::X])[Dim2::X],
            &self.grid.coords_by_type(self.locations[Dim2::Y])[Dim2::Y],
        )
    }

    /// Returns a reference to the 2D array of field values.
    pub fn values(&self) -> &Array2<F> {
        &self.values
    }

    /// Returns a mutable reference to the 2D array of field values.
    pub fn values_mut(&mut self) -> &mut Array2<F> {
        &mut self.values
    }

    /// Returns the field value at the given 2D index.
    pub fn value(&self, indices: &Idx2<usize>) -> F {
        self.values[[indices[Dim2::X], indices[Dim2::Y]]]
    }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> {
        self.grid.shape()
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values are defined.
    pub fn locations(&self) -> &In2D<CoordLocation> {
        &self.locations
    }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array2<F> {
        self.values
    }

    /// Computes the 2D indices and value of the minimum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_minimum(&self) -> Option<(Idx2<usize>, F)> {
        self.values
            .as_slice_memory_order()
            .unwrap()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value.is_nan() {
                    None
                } else {
                    Some(OrderableIndexValuePair(idx, value))
                }
            })
            .min()
            .map(|OrderableIndexValuePair(idx_of_min_value, min_value)| {
                (
                    compute_2d_array_indices_from_flat_idx(self.shape(), idx_of_min_value),
                    min_value,
                )
            })
    }

    /// Computes the 2D indices and value of the maximum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_maximum(&self) -> Option<(Idx2<usize>, F)> {
        self.values
            .as_slice_memory_order()
            .unwrap()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value.is_nan() {
                    None
                } else {
                    Some(OrderableIndexValuePair(idx, value))
                }
            })
            .max()
            .map(|OrderableIndexValuePair(idx_of_max_value, max_value)| {
                (
                    compute_2d_array_indices_from_flat_idx(self.shape(), idx_of_max_value),
                    max_value,
                )
            })
    }

    /// Serializes the field data into pickle format and save at the given path.
    pub fn save_as_pickle<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()>
    where
        F: Serialize,
    {
        let data = ScalarFieldSerializeData2 {
            coords: self.coords().into_owned(),
            values: self.values().clone(),
        };
        save_data_as_pickle(output_file_path, &data)
    }

    fn set_grid(&mut self, new_grid: Arc<G>) {
        let grid_shape = new_grid.shape();
        let values_shape = self.values.shape();
        assert!(
            grid_shape[Dim2::X] == values_shape[0] && grid_shape[Dim2::Y] == values_shape[1],
            "Shape of new grid does not match shape of array of values."
        );
        self.grid = new_grid;
    }
}

/// A 2D vector field.
///
/// Holds the grid and values of the two components of a 2D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct VectorField2<F, G>
where
    F: BFloat,
    G: Grid2<F>,
{
    name: String,
    grid: Arc<G>,
    components: In2D<ScalarField2<F, G>>,
}

impl<F, G> VectorField2<F, G>
where
    F: BFloat,
    G: Grid2<F>,
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In2D<ScalarField2<F, G>>) -> Self {
        components[Dim2::X].set_grid(Arc::clone(&grid));
        components[Dim2::Y].set_grid(Arc::clone(&grid));
        Self {
            name,
            grid,
            components,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim2) -> &ScalarField2<F, G> {
        &self.components[dim]
    }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim2) -> CoordRefs2<F> {
        self.components[dim].coords()
    }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In2D<CoordRefs2<F>> {
        In2D::new(self.coords(Dim2::X), self.coords(Dim2::Y))
    }

    /// Returns a reference to the 2D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim2) -> &Array2<F> {
        self.components[dim].values()
    }

    /// Returns a mutable reference to the 2D array of field values.
    pub fn values_mut(&mut self, dim: Dim2) -> &mut Array2<F> {
        self.components[dim].values_mut()
    }

    /// Returns a reference to the 2D array of field values for each component.
    pub fn all_values(&self) -> In2D<&Array2<F>> {
        In2D::new(self.values(Dim2::X), self.values(Dim2::Y))
    }

    /// Returns the field vector at the given 2D index.
    pub fn vector(&self, indices: &Idx2<usize>) -> Vec2<F> {
        Vec2::new(
            self.values(Dim2::X)[[indices[Dim2::X], indices[Dim2::Y]]],
            self.values(Dim2::Y)[[indices[Dim2::X], indices[Dim2::Y]]],
        )
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim2) -> &In2D<CoordLocation> {
        self.components[dim].locations()
    }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> {
        self.grid.shape()
    }
}

/// A field of 3D vectors in a 2D plane.
///
/// Holds the grid and values of the three components of a vector field in a 2D plane,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct PlaneVectorField3<F, G>
where
    F: BFloat,
    G: Grid2<F>,
{
    name: String,
    grid: Arc<G>,
    components: In3D<ScalarField2<F, G>>,
}

impl<F, G> PlaneVectorField3<F, G>
where
    F: BFloat,
    G: Grid2<F>,
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In3D<ScalarField2<F, G>>) -> Self {
        components[X].set_grid(Arc::clone(&grid));
        components[Y].set_grid(Arc::clone(&grid));
        components[Z].set_grid(Arc::clone(&grid));
        Self {
            name,
            grid,
            components,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim3) -> &ScalarField2<F, G> {
        &self.components[dim]
    }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim3) -> CoordRefs2<F> {
        self.components[dim].coords()
    }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In3D<CoordRefs2<F>> {
        In3D::new(self.coords(X), self.coords(Y), self.coords(Z))
    }

    /// Returns a reference to the 3D array of field values for each component.
    pub fn all_values(&self) -> In3D<&Array2<F>> {
        In3D::new(self.values(X), self.values(Y), self.values(Z))
    }

    /// Returns the field vector at the given 3D index.
    pub fn vector(&self, indices: &Idx2<usize>) -> Vec3<F> {
        Vec3::new(
            self.values(X)[[indices[Dim2::X], indices[Dim2::Y]]],
            self.values(Y)[[indices[Dim2::X], indices[Dim2::Y]]],
            self.values(Y)[[indices[Dim2::X], indices[Dim2::Y]]],
        )
    }

    /// Returns a reference to the 2D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim3) -> &Array2<F> {
        self.components[dim].values()
    }

    /// Returns a mutable reference to the 2D array of field values for the
    /// specified component.
    pub fn values_mut(&mut self, dim: Dim3) -> &mut Array2<F> {
        self.components[dim].values_mut()
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim3) -> &In2D<CoordLocation> {
        self.components[dim].locations()
    }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> {
        self.grid.shape()
    }
}

/// A 1D scalar field.
///
/// Holds the grid and values of a 1D scalar field, as well as the
/// specific coordinates where the values are defined.
#[derive(Clone, Debug)]
pub struct ScalarField1<F, G>
where
    F: BFloat,
    G: Grid1<F>,
{
    name: String,
    grid: Arc<G>,
    location: CoordLocation,
    values: Array1<F>,
}

#[derive(Serialize)]
struct ScalarFieldSerializeData1<F: BFloat> {
    coords: Vec<F>,
    values: Array1<F>,
}

impl<F, G> ScalarField1<F, G>
where
    F: BFloat,
    G: Grid1<F>,
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate location specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: Arc<G>, location: CoordLocation, values: Array1<F>) -> Self {
        Self {
            name,
            grid,
            location,
            values,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> &[F] {
        self.grid.coords_by_type(self.location)
    }

    /// Returns a reference to the 1D array of field values.
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    /// Returns a mutable reference to the 1D array of field values.
    pub fn values_mut(&mut self) -> &mut Array1<F> {
        &mut self.values
    }

    /// Returns the field value at the given index.
    pub fn value(&self, index: usize) -> F {
        self.values[index]
    }

    /// Returns the size of the grid.
    pub fn size(&self) -> usize {
        self.grid.size()
    }

    /// Returns the coordinate location specifying where in the grid cell
    /// the values are defined.
    pub fn location(&self) -> CoordLocation {
        self.location
    }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array1<F> {
        self.values
    }

    /// Computes the index and value of the minimum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_minimum(&self) -> Option<(usize, F)> {
        self.values
            .as_slice_memory_order()
            .unwrap()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value.is_nan() {
                    None
                } else {
                    Some(OrderableIndexValuePair(idx, value))
                }
            })
            .min()
            .map(|OrderableIndexValuePair(idx_of_min_value, min_value)| {
                (idx_of_min_value, min_value)
            })
    }

    /// Computes the index and value of the maximum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_maximum(&self) -> Option<(usize, F)> {
        self.values
            .as_slice_memory_order()
            .unwrap()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value.is_nan() {
                    None
                } else {
                    Some(OrderableIndexValuePair(idx, value))
                }
            })
            .max()
            .map(|OrderableIndexValuePair(idx_of_max_value, max_value)| {
                (idx_of_max_value, max_value)
            })
    }

    /// Serializes the field data into pickle format and save at the given path.
    pub fn save_as_pickle<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()>
    where
        F: Serialize,
    {
        let data = ScalarFieldSerializeData1 {
            coords: self.coords().to_vec(),
            values: self.values().clone(),
        };
        save_data_as_pickle(output_file_path, &data)
    }
}

/// Computes the 3D array indices corresponding to a given index into the flattened version of the array,
/// assuming the array is laid out in column-major order.
pub fn compute_3d_array_indices_from_flat_idx(shape: &In3D<usize>, idx: usize) -> Idx3<usize> {
    let i = idx % shape[X];
    let j = idx / shape[X] % shape[Y];
    let k = idx / (shape[X] * shape[Y]);
    Idx3::new(i, j, k)
}

/// Computes the 2D array indices corresponding to a given index into the flattened version of the array,
/// assuming the array is laid out in column-major order.
pub fn compute_2d_array_indices_from_flat_idx(shape: &In2D<usize>, idx: usize) -> Idx2<usize> {
    let i = idx % shape[Dim2::X];
    let j = idx / shape[Dim2::X];
    Idx2::new(i, j)
}
