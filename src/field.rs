//! Scalar and vector fields.

use std::{io, path};
use std::sync::Arc;
use ndarray::prelude::*;
use serde::Serialize;
use rayon::prelude::*;
use crate::num::BFloat;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Point3, Idx3, Idx2, Coords2, CoordRefs3, CoordRefs2};
use crate::grid::{CoordLocation, Grid3, Grid2};
use crate::grid::regular::RegularGrid2;
use crate::interpolation::Interpolator3;
use crate::io::utils::save_data_as_pickle;
use Dim3::{X, Y, Z};

/// Locations in the grid cell for resampled field values.
#[derive(Clone, Debug)]
pub enum ResampledCoordLocations {
    Original,
    Equal(CoordLocation)
}

impl ResampledCoordLocations {
    pub fn centers() -> Self { ResampledCoordLocations::Equal(CoordLocation::Center) }
    pub fn lower_edges() -> Self { ResampledCoordLocations::Equal(CoordLocation::LowerEdge) }
}

/// A 3D scalar field.
///
/// Holds the grid and values of a 3D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct ScalarField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    name: String,
    grid: Arc<G>,
    locations: In3D<CoordLocation>,
    values: Array3<F>
}

impl<F, G> ScalarField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate locations specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: Arc<G>, locations: In3D<CoordLocation>, values: Array3<F>) -> Self {
        let grid_shape = grid.shape();
        let values_shape = values.shape();
        assert!(grid_shape[X] == values_shape[0] &&
                grid_shape[Y] == values_shape[1] &&
                grid_shape[Z] == values_shape[2],
                "Shape of grid does not match shape of array of values.");
        ScalarField3{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { self.grid.as_ref() }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> { Arc::clone(&self.grid) }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs3<F> {
        CoordRefs3::new(
            &self.grid.coords_by_type(self.locations[X])[X],
            &self.grid.coords_by_type(self.locations[Y])[Y],
            &self.grid.coords_by_type(self.locations[Z])[Z]
        )
    }

    /// Returns a reference to the 3D array of field values.
    pub fn values(&self) -> &Array3<F> { &self.values }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> { self.grid.shape() }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values are defined.
    pub fn locations(&self) -> &In3D<CoordLocation> { &self.locations }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array3<F> { self.values }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    pub fn resampled_to_grid<H, I>(&self, grid: Arc<H>, interpolator: &I) -> ScalarField3<F, H>
    where H: Grid3<F>,
          I: Interpolator3
    {
        let new_coords = self.coords_from_grid(grid.as_ref());

        let grid_shape = grid.shape();
        let mut new_values = unsafe { Array3::uninitialized(grid_shape.to_tuple().f()) };
        let values_buffer = new_values.as_slice_memory_order_mut().unwrap();

        values_buffer.par_iter_mut().enumerate().for_each(
            |(idx, value)| {
                let indices = compute_3d_array_indices_from_flat_idx(&grid_shape, idx);
                let point = new_coords.point(&indices);
                *value = interpolator.interp_extrap_scalar_field(self, &point);
            }
        );
        ScalarField3::new(self.name.clone(), grid, self.locations.clone(), new_values)
    }

    /// Returns a view of the 2D slice of the field located at the given index along the given axis.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> ArrayView2<F> {
        self.values.index_axis(Axis(axis as usize), idx)
    }

    /// Returns a 2D scalar field corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(&self, interpolator: &I, x_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::XSliceGrid>
    where I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_x();
        self.create_slice_across_x(Arc::new(slice_grid), interpolator, x_coord, resampled_locations)
    }

    /// Returns a 2D scalar field corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(&self, interpolator: &I, y_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::YSliceGrid>
    where I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_y();
        self.create_slice_across_y(Arc::new(slice_grid), interpolator, y_coord, resampled_locations)
    }

    /// Returns a 2D scalar field corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(&self, interpolator: &I, z_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::ZSliceGrid>
    where I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_z();
        self.create_slice_across_z(Arc::new(slice_grid), interpolator, z_coord, resampled_locations)
    }

    /// Returns a 2D scalar field corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(&self, interpolator: &I, axis: Dim3, coord: F, location: CoordLocation) -> ScalarField2<F, RegularGrid2<F>>
    where I: Interpolator3
    {
        let slice_grid = self.grid.regular_slice_across_axis(axis);
        self.create_regular_slice_across_axis(Arc::new(slice_grid), interpolator, axis, coord, location)
    }

    fn coords_from_grid<'a, 'b, H: Grid3<F>>(&'a self, grid: &'b H) -> CoordRefs3<'b, F> {
        CoordRefs3::new(
            &grid.coords_by_type(self.locations[X])[X],
            &grid.coords_by_type(self.locations[Y])[Y],
            &grid.coords_by_type(self.locations[Z])[Z]
        )
    }

    fn set_grid(&mut self, new_grid: Arc<G>) {
        let grid_shape = new_grid.shape();
        let values_shape = self.values.shape();
        assert!(grid_shape[X] == values_shape[0] &&
                grid_shape[Y] == values_shape[1] &&
                grid_shape[Z] == values_shape[2],
                "Shape of new grid does not match shape of array of values.");
        self.grid = new_grid;
    }

    fn create_slice_across_x<I>(&self, slice_grid: Arc<G::XSliceGrid>, interpolator: &I, x_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::XSliceGrid>
    where I: Interpolator3
    {
        let slice_locations = self.select_slice_locations([Y, Z], &resampled_locations);
        let slice_values = self.compute_slice_values(interpolator, X, x_coord, resampled_locations, false);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    fn create_slice_across_y<I>(&self, slice_grid: Arc<G::YSliceGrid>, interpolator: &I, y_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::YSliceGrid>
    where I: Interpolator3
    {
        let slice_locations = self.select_slice_locations([X, Z], &resampled_locations);
        let slice_values = self.compute_slice_values(interpolator, Y, y_coord, resampled_locations, false);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    fn create_slice_across_z<I>(&self, slice_grid: Arc<G::ZSliceGrid>, interpolator: &I, z_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::ZSliceGrid>
    where I: Interpolator3
    {
        let slice_locations = self.select_slice_locations([X, Y], &resampled_locations);
        let slice_values = self.compute_slice_values(interpolator, Z, z_coord, resampled_locations, false);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    fn create_regular_slice_across_axis<I>(&self, slice_grid: Arc<RegularGrid2<F>>, interpolator: &I, axis: Dim3, coord: F, location: CoordLocation) -> ScalarField2<F, RegularGrid2<F>>
    where I: Interpolator3
    {
        let slice_locations = In2D::same(location);
        let slice_values = self.compute_slice_values(interpolator, axis, coord, ResampledCoordLocations::Equal(location), true);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    fn compute_slice_indices_from_flat_idx(&self, axes: [Dim3; 2], idx: usize) -> [usize; 2] {
        let shape = self.shape();
        let indices = compute_2d_array_indices_from_flat_idx(&In2D::new(shape[axes[0]], shape[axes[1]]), idx);
        [indices[Dim2::X], indices[Dim2::Y]]
    }

    fn select_slice_locations(&self, axes: [Dim3; 2], resampled_locations: &ResampledCoordLocations) -> In2D<CoordLocation> {
        match *resampled_locations {
            ResampledCoordLocations::Original => In2D::new(self.locations[axes[0]], self.locations[axes[1]]),
            ResampledCoordLocations::Equal(location) => In2D::same(location)
        }
    }

    fn select_slice_coords(&self, axes: [Dim3; 2], resampled_locations: &ResampledCoordLocations) -> [&[F]; 2] {
        match *resampled_locations {
            ResampledCoordLocations::Original => {
                let coords = self.coords();
                [coords[axes[0]], coords[axes[1]]]
            },
            ResampledCoordLocations::Equal(CoordLocation::Center) => {
                let centers = self.grid.centers();
                [&centers[axes[0]], &centers[axes[1]]]
            },
            ResampledCoordLocations::Equal(CoordLocation::LowerEdge) => {
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
            },
            CoordLocation::LowerEdge => {
                let regular_lower_edges = self.grid.regular_lower_edges();
                [&regular_lower_edges[axes[0]], &regular_lower_edges[axes[1]]]
            }
        }
    }

    fn compute_slice_values<I>(&self, interpolator: &I, axis: Dim3, coord: F, resampled_locations: ResampledCoordLocations, regular: bool) -> Array2<F>
    where I: Interpolator3
    {
        let lower_bound = self.grid.lower_bounds()[axis];
        let upper_bound = self.grid.upper_bounds()[axis];
        if coord < lower_bound || coord >= upper_bound{
            panic!("Slicing coordinate is outside the grid bounds.");
        }

        let axes = Dim3::slice_except(axis);

        let coords = if regular {
            if let ResampledCoordLocations::Equal(location) = resampled_locations {
                self.select_regular_slice_coords(axes, location)
            } else {
                panic!("Original coord locations not supported for regular slice.")
            }
        } else {
            self.select_slice_coords(axes, &resampled_locations)
        };

        self.interpolate_slice_values(interpolator, axes, &coords, coord)
    }

    fn interpolate_slice_values<I>(&self, interpolator: &I, axes: [Dim3; 2], coords: &[&[F]; 2], slice_axis_coord: F) -> Array2<F>
    where I: Interpolator3
    {
        let slice_shape = (coords[0].len(), coords[1].len());
        let mut slice_values = unsafe { Array2::uninitialized(slice_shape.f()) };
        let values_buffer = slice_values.as_slice_memory_order_mut().unwrap();

        values_buffer.par_iter_mut().enumerate().for_each(
            |(idx, value)| {
                let [idx_0, idx_1] = self.compute_slice_indices_from_flat_idx(axes, idx);
                let mut point_in_slice = Point3::equal_components(slice_axis_coord);
                point_in_slice[axes[0]] = coords[0][idx_0];
                point_in_slice[axes[1]] = coords[1][idx_1];
                *value = interpolator.interp_scalar_field(self, &point_in_slice).expect_inside();
            }
        );
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
where F: BFloat,
      G: Grid3<F>
{
    name: String,
    grid: Arc<G>,
    components: In3D<ScalarField3<F, G>>
}

impl<F, G> VectorField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In3D<ScalarField3<F, G>>) -> Self {
        components[X].set_grid(Arc::clone(&grid));
        components[Y].set_grid(Arc::clone(&grid));
        components[Z].set_grid(Arc::clone(&grid));
        VectorField3{
            name,
            grid,
            components
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { self.grid.as_ref() }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> { Arc::clone(&self.grid) }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim3) -> &ScalarField3<F, G> { &self.components[dim] }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim3) -> CoordRefs3<F> { self.components[dim].coords() }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In3D<CoordRefs3<F>> {
        In3D::new(self.coords(X), self.coords(Y), self.coords(Z))
    }

    /// Returns a reference to the 3D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim3) -> &Array3<F> { self.components[dim].values() }

    /// Returns a reference to the 3D array of field values for each component.
    pub fn all_values(&self) -> In3D<&Array3<F>> {
        In3D::new(self.values(X), self.values(Y), self.values(Z))
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim3) -> &In3D<CoordLocation> { self.components[dim].locations() }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> { self.grid.shape() }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    pub fn resampled_to_grid<H, I>(&self, grid: Arc<H>, interpolator: &I) -> VectorField3<F, H>
    where H: Grid3<F>,
          I: Interpolator3
    {
        let components = In3D::new(
            self.components[X].resampled_to_grid(Arc::clone(&grid), interpolator),
            self.components[Y].resampled_to_grid(Arc::clone(&grid), interpolator),
            self.components[Z].resampled_to_grid(Arc::clone(&grid), interpolator)
        );
        VectorField3::new(
            self.name.clone(),
            grid,
            components
        )
    }

    /// Returns a view of the 2D slice located at the given index along the given dimension,
    /// for each component of the field.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> In3D<ArrayView2<F>> {
        In3D::new(
            self.components[X].slice_across_axis_at_idx(axis, idx),
            self.components[Y].slice_across_axis_at_idx(axis, idx),
            self.components[Z].slice_across_axis_at_idx(axis, idx)
        )
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(&self, interpolator: &I, x_coord: F, resampled_locations: ResampledCoordLocations) -> PlaneVectorField3<F, G::XSliceGrid>
    where I: Interpolator3
    {
        let slice_grid = Arc::new(self.grid.slice_across_x());
        let slice_field_components = In3D::new(
            self.components[X].create_slice_across_x(Arc::clone(&slice_grid), interpolator, x_coord, resampled_locations.clone()),
            self.components[Y].create_slice_across_x(Arc::clone(&slice_grid), interpolator, x_coord, resampled_locations.clone()),
            self.components[Z].create_slice_across_x(Arc::clone(&slice_grid), interpolator, x_coord, resampled_locations)
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(&self, interpolator: &I, y_coord: F, resampled_locations: ResampledCoordLocations) -> PlaneVectorField3<F, G::YSliceGrid>
    where I: Interpolator3
    {
        let slice_grid = Arc::new(self.grid.slice_across_y());
        let slice_field_components = In3D::new(
            self.components[X].create_slice_across_y(Arc::clone(&slice_grid), interpolator, y_coord, resampled_locations.clone()),
            self.components[Y].create_slice_across_y(Arc::clone(&slice_grid), interpolator, y_coord, resampled_locations.clone()),
            self.components[Z].create_slice_across_y(Arc::clone(&slice_grid), interpolator, y_coord, resampled_locations)
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(&self, interpolator: &I, z_coord: F, resampled_locations: ResampledCoordLocations) -> PlaneVectorField3<F, G::ZSliceGrid>
    where I: Interpolator3
    {
        let slice_grid = Arc::new(self.grid.slice_across_z());
        let slice_field_components = In3D::new(
            self.components[X].create_slice_across_z(Arc::clone(&slice_grid), interpolator, z_coord, resampled_locations.clone()),
            self.components[Y].create_slice_across_z(Arc::clone(&slice_grid), interpolator, z_coord, resampled_locations.clone()),
            self.components[Z].create_slice_across_z(Arc::clone(&slice_grid), interpolator, z_coord, resampled_locations)
        );
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(&self, interpolator: &I, axis: Dim3, coord: F, location: CoordLocation) -> PlaneVectorField3<F, RegularGrid2<F>>
    where I: Interpolator3
    {
        let slice_grid = Arc::new(self.grid.regular_slice_across_axis(axis));
        let slice_field_components = In3D::new(
            self.components[X].create_regular_slice_across_axis(Arc::clone(&slice_grid), interpolator, axis, coord, location),
            self.components[Y].create_regular_slice_across_axis(Arc::clone(&slice_grid), interpolator, axis, coord, location),
            self.components[Z].create_regular_slice_across_axis(Arc::clone(&slice_grid), interpolator, axis, coord, location)
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
where F: BFloat,
      G: Grid2<F>
{
    name: String,
    grid: Arc<G>,
    locations: In2D<CoordLocation>,
    values: Array2<F>
}

#[derive(Serialize)]
struct ScalarFieldSerializeData2<F: BFloat> {
    coords: Coords2<F>,
    values: Array2<F>
}

impl<F, G> ScalarField2<F, G>
where F: BFloat,
      G: Grid2<F>
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate locations specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: Arc<G>, locations: In2D<CoordLocation>, values: Array2<F>) -> Self {
        ScalarField2{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { self.grid.as_ref() }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> { Arc::clone(&self.grid) }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs2<F> {
        CoordRefs2::new(
            &self.grid.coords_by_type(self.locations[Dim2::X])[Dim2::X],
            &self.grid.coords_by_type(self.locations[Dim2::Y])[Dim2::Y]
        )
    }

    /// Returns a reference to the 2D array of field values.
    pub fn values(&self) -> &Array2<F> { &self.values }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values are defined.
    pub fn locations(&self) -> &In2D<CoordLocation> { &self.locations }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> { self.grid.shape() }

    /// Serializes the field data into pickle format and save at the given path.
    pub fn save_as_pickle<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()>
    where F: Serialize
    {
        let data = ScalarFieldSerializeData2{
            coords: self.coords().into_owned(),
            values: self.values().clone()
        };
        save_data_as_pickle(file_path, &data)
    }

    fn set_grid(&mut self, new_grid: Arc<G>) {
        let grid_shape = new_grid.shape();
        let values_shape = self.values.shape();
        assert!(grid_shape[Dim2::X] == values_shape[0] &&
                grid_shape[Dim2::Y] == values_shape[1],
                "Shape of new grid does not match shape of array of values.");
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
where F: BFloat,
      G: Grid2<F>
{
    name: String,
    grid: Arc<G>,
    components: In2D<ScalarField2<F, G>>
}

impl<F, G> VectorField2<F, G>
where F: BFloat,
      G: Grid2<F>
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In2D<ScalarField2<F, G>>) -> Self {
        components[Dim2::X].set_grid(Arc::clone(&grid));
        components[Dim2::Y].set_grid(Arc::clone(&grid));
        VectorField2{ name, grid, components }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { self.grid.as_ref() }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> { Arc::clone(&self.grid) }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim2) -> &ScalarField2<F, G> { &self.components[dim] }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim2) -> CoordRefs2<F> { self.components[dim].coords() }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In2D<CoordRefs2<F>> {
        In2D::new(self.coords(Dim2::X), self.coords(Dim2::Y))
    }

    /// Returns a reference to the 2D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim2) -> &Array2<F> { self.components[dim].values() }

    /// Returns a reference to the 2D array of field values for each component.
    pub fn all_values(&self) -> In2D<&Array2<F>> {
        In2D::new(self.values(Dim2::X), self.values(Dim2::Y))
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim2) -> &In2D<CoordLocation> { self.components[dim].locations() }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> { self.grid.shape() }
}

/// A field of 3D vectors in a 2D plane.
///
/// Holds the grid and values of the three components of a vector field in a 2D plane,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct PlaneVectorField3<F, G>
where F: BFloat,
      G: Grid2<F>
{
    name: String,
    grid: Arc<G>,
    components: In3D<ScalarField2<F, G>>
}

impl<F, G> PlaneVectorField3<F, G>
where F: BFloat,
      G: Grid2<F>
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In3D<ScalarField2<F, G>>) -> Self {
        components[X].set_grid(Arc::clone(&grid));
        components[Y].set_grid(Arc::clone(&grid));
        components[Z].set_grid(Arc::clone(&grid));
        PlaneVectorField3{ name, grid, components }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { self.grid.as_ref() }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> { Arc::clone(&self.grid) }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim3) -> &ScalarField2<F, G> { &self.components[dim] }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim3) -> CoordRefs2<F> { self.components[dim].coords() }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In3D<CoordRefs2<F>> {
        In3D::new(self.coords(X), self.coords(Y), self.coords(Z))
    }

    /// Returns a reference to the 3D array of field values for each component.
    pub fn all_values(&self) -> In3D<&Array2<F>> {
        In3D::new(self.values(X), self.values(Y), self.values(Z))
    }

    /// Returns a reference to the 2D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim3) -> &Array2<F> { self.components[dim].values() }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim3) -> &In2D<CoordLocation> { self.components[dim].locations() }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> { self.grid.shape() }
}

/// Computes the 3D array indices corresponding to a given index into the flattened version of the array,
/// assuming the array is laid out in column-major order.
pub fn compute_3d_array_indices_from_flat_idx(shape: &In3D<usize>, idx: usize) -> Idx3<usize> {
    let i = idx % shape[X];
    let j = idx/shape[X] % shape[Y];
    let k = idx/(shape[X]*shape[Y]);
    Idx3::new(i, j, k)
}

/// Computes the 2D array indices corresponding to a given index into the flattened version of the array,
/// assuming the array is laid out in column-major order.
pub fn compute_2d_array_indices_from_flat_idx(shape: &In2D<usize>, idx: usize) -> Idx2<usize> {
    let i = idx % shape[Dim2::X];
    let j = idx/shape[Dim2::X];
    Idx2::new(i, j)
}
