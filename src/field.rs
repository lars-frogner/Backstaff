//! Scalar and vector fields.

use std::{io, path};
use ndarray::prelude::*;
use serde::Serialize;
use crate::num::BFloat;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Point3, Coords2, CoordRefs3, CoordRefs2};
use crate::grid::{CoordLocation, Grid3, Grid2};
use crate::grid::regular::RegularGrid2;
use crate::interpolation::Interpolator3;
use crate::io::utils::save_data_as_pickle;
use Dim3::{X, Y, Z};

/// Locations in the grid cell for resampled field values.
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
#[derive(Debug, Clone)]
pub struct ScalarField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    name: String,
    grid: G,
    locations: In3D<CoordLocation>,
    values: Array3<F>
}

impl<F, G> ScalarField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate types specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: G, locations: In3D<CoordLocation>, values: Array3<F>) -> Self {
        ScalarField3{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

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

    /// Returns a view of the 2D slice of the field located at the given index along the given axis.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> ArrayView2<F> {
        self.values.index_axis(Axis(axis as usize), idx)
    }

    /// Returns a 2D scalar field corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(&self, interpolator: &I, x_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::XSliceGrid>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_x();
        let slice_locations = self.select_slice_locations([Y, Z], &resampled_locations);
        let slice_values = self.compute_slice_values(interpolator, X, x_coord, resampled_locations, false);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    /// Returns a 2D scalar field corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(&self, interpolator: &I, y_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::YSliceGrid>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_y();
        let slice_locations = self.select_slice_locations([X, Z], &resampled_locations);
        let slice_values = self.compute_slice_values(interpolator, Y, y_coord, resampled_locations, false);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    /// Returns a 2D scalar field corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(&self, interpolator: &I, z_coord: F, resampled_locations: ResampledCoordLocations) -> ScalarField2<F, G::ZSliceGrid>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_z();
        let slice_locations = self.select_slice_locations([X, Y], &resampled_locations);
        let slice_values = self.compute_slice_values(interpolator, Z, z_coord, resampled_locations, false);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    /// Returns a 2D scalar field corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(&self, interpolator: &I, axis: Dim3, coord: F, location: CoordLocation) -> ScalarField2<F, RegularGrid2<F>>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.regular_slice_across_axis(axis);
        let slice_locations = In2D::same(location);
        let slice_values = self.compute_slice_values(interpolator, axis, coord, ResampledCoordLocations::Equal(location), true);
        ScalarField2::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
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

    fn compute_slice_values<I: Interpolator3>(&self, interpolator: &I, axis: Dim3, coord: F, resampled_locations: ResampledCoordLocations, regular: bool) -> Array2<F>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
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

        let mut point_in_slice = Point3::origin();
        point_in_slice[axis] = coord;

        self.interpolate_slice_values(interpolator, axes, &coords, point_in_slice)
    }

    fn interpolate_slice_values<I>(&self, interpolator: &I, axes: [Dim3; 2], coords: &[&[F]; 2], mut point_in_slice: Point3<F>) -> Array2<F>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_shape = (coords[0].len(), coords[1].len());
        let mut slice_values = unsafe { Array2::uninitialized(slice_shape.f()) };

        for idx_1 in 0..slice_shape.1 {
            point_in_slice[axes[1]] = coords[1][idx_1];
            for idx_0 in  0..slice_shape.0 {
                point_in_slice[axes[0]] = coords[0][idx_0];
                slice_values[[idx_0, idx_1]] = interpolator.interp_scalar_field(self, &point_in_slice).unwrap();
            }
        }
        slice_values
    }
}

/// A 3D vector field.
///
/// Holds the grid and values of the three components of a 3D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct VectorField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    name: String,
    grid: G,
    locations: In3D<In3D<CoordLocation>>,
    values: In3D<Array3<F>>
}

impl<F, G> VectorField3<F, G>
where F: BFloat,
      G: Grid3<F>
{
    /// Creates a new vector field given a name, a grid, the component values and
    /// coordinate types specifying where in the grid cell the component values are defined.
    pub fn new(name: String, grid: G, locations: In3D<In3D<CoordLocation>>, values: In3D<Array3<F>>) -> Self {
        VectorField3{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// component values are defined.
    pub fn coords<'a>(&'a self) -> In3D<CoordRefs3<'a, F>> {
        In3D::new(CoordRefs3::new(
             &self.grid.coords_by_type(self.locations[X][X])[X],
             &self.grid.coords_by_type(self.locations[X][Y])[Y],
             &self.grid.coords_by_type(self.locations[X][Z])[Z]
         ),
         CoordRefs3::new(
             &self.grid.coords_by_type(self.locations[Y][X])[X],
             &self.grid.coords_by_type(self.locations[Y][Y])[Y],
             &self.grid.coords_by_type(self.locations[Y][Z])[Z]
         ),
         CoordRefs3::new(
             &self.grid.coords_by_type(self.locations[Z][X])[X],
             &self.grid.coords_by_type(self.locations[Z][Y])[Y],
             &self.grid.coords_by_type(self.locations[Z][Z])[Z]
         ))
    }

    /// Returns a reference to the three 3D arrays of field component values.
    pub fn values(&self) -> &In3D<Array3<F>> { &self.values }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> { self.grid.shape() }

    /// Creates a new scalar field from the specified vector field component.
    pub fn component_as_scalar_field(&self, dim: Dim3) -> ScalarField3<F, G> {
        let dim_names = In3D::new('x', 'y', 'z');
        let name = format!("{}{}", self.name(), dim_names[dim]);
        ScalarField3::new(name, self.grid.clone(), self.locations[dim].clone(), self.values[dim].clone())
    }

    /// Returns a view of the 2D slice located at the given index along the given dimension,
    /// for each component of the field.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> In3D<ArrayView2<F>> {
        let axis = Axis(axis as usize);
        In3D::new(
            self.values[X].index_axis(axis, idx),
            self.values[Y].index_axis(axis, idx),
            self.values[Z].index_axis(axis, idx)
        )
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(&self, interpolator: &I, x_coord: F, location: CoordLocation) -> PlaneVectorField3<F, G::XSliceGrid>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_x();
        let slice_locations = In3D::same_cloned(In2D::same(location));
        let slice_values = self.compute_slice_values(interpolator, X, x_coord, location, false);
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(&self, interpolator: &I, y_coord: F, location: CoordLocation) -> PlaneVectorField3<F, G::YSliceGrid>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_y();
        let slice_locations = In3D::same_cloned(In2D::same(location));
        let slice_values = self.compute_slice_values(interpolator, Y, y_coord, location, false);
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(&self, interpolator: &I, z_coord: F, location: CoordLocation) -> PlaneVectorField3<F, G::ZSliceGrid>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.slice_across_z();
        let slice_locations = In3D::same_cloned(In2D::same(location));
        let slice_values = self.compute_slice_values(interpolator, Z, z_coord, location, false);
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(&self, interpolator: &I, axis: Dim3, coord: F, location: CoordLocation) -> PlaneVectorField3<F, RegularGrid2<F>>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_grid = self.grid.regular_slice_across_axis(axis);
        let slice_locations = In3D::same_cloned(In2D::same(location));
        let slice_values = self.compute_slice_values(interpolator, axis, coord, location, true);
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_locations, slice_values)
    }

    fn select_slice_coords(&self, axes: [Dim3; 2], location: CoordLocation) -> [&[F]; 2] {
        let coords = self.grid.coords_by_type(location);
        [&coords[axes[0]], &coords[axes[1]]]
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

    fn compute_slice_values<I>(&self, interpolator: &I, axis: Dim3, coord: F, location: CoordLocation, regular: bool) -> In3D<Array2<F>>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let lower_bound = self.grid.lower_bounds()[axis];
        let upper_bound = self.grid.upper_bounds()[axis];
        if coord < lower_bound || coord >= upper_bound{
            panic!("Slicing coordinate is outside the grid bounds.");
        }

        let axes = Dim3::slice_except(axis);

        let coords = if regular {
            self.select_regular_slice_coords(axes, location)
        } else {
            self.select_slice_coords(axes, location)
        };

        let mut point_in_slice = Point3::origin();
        point_in_slice[axis] = coord;

        self.interpolate_slice_values(interpolator, axes, &coords, point_in_slice)
    }

    fn interpolate_slice_values<I>(&self, interpolator: &I, axes: [Dim3; 2], coords: &[&[F]; 2], mut point_in_slice: Point3<F>) -> In3D<Array2<F>>
    where F: num::cast::FromPrimitive,
          I: Interpolator3
    {
        let slice_shape = (coords[0].len(), coords[1].len());
        let mut slice_values = unsafe {
            In3D::new(Array2::uninitialized(slice_shape.f()),
                      Array2::uninitialized(slice_shape.f()),
                      Array2::uninitialized(slice_shape.f()))
        };
        for idx_1 in 0..slice_shape.1 {
            point_in_slice[axes[1]] = coords[1][idx_1];
            for idx_0 in  0..slice_shape.0 {
                point_in_slice[axes[0]] = coords[0][idx_0];
                let vector = interpolator.interp_vector_field(self, &point_in_slice).unwrap();
                slice_values[X][[idx_0, idx_1]] = vector[X];
                slice_values[Y][[idx_0, idx_1]] = vector[Y];
                slice_values[Z][[idx_0, idx_1]] = vector[Z];
            }
        }
        slice_values
    }
}

/// A 2D scalar field.
///
/// Holds the grid and values of a 2D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct ScalarField2<F, G>
where F: BFloat,
      G: Grid2<F> + Clone
{
    name: String,
    grid: G,
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
      G: Grid2<F> + Clone
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate types specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: G, locations: In2D<CoordLocation>, values: Array2<F>) -> Self {
        ScalarField2{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

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

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> { self.grid.shape() }

    /// Serializes the field data into pickle format and save at the given path.
    pub fn save_as_pickle(&self, file_path: &path::Path) -> io::Result<()>
    where F: Serialize
    {
        let data = ScalarFieldSerializeData2{
            coords: self.coords().into_owned(),
            values: self.values().clone()
        };
        save_data_as_pickle(file_path, &data)
    }
}

/// A 2D vector field.
///
/// Holds the grid and values of the two components of a 2D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct VectorField2<F, G>
where F: BFloat,
      G: Grid2<F> + Clone
{
    name: String,
    grid: G,
    locations: In2D<In2D<CoordLocation>>,
    values: In2D<Array2<F>>
}

impl<F, G> VectorField2<F, G>
where F: BFloat,
      G: Grid2<F> + Clone
{
    /// Creates a new vector field given a name, a grid, the component values and
    /// coordinate types specifying where in the grid cell the component values are defined.
    pub fn new(name: String, grid: G, locations: In2D<In2D<CoordLocation>>, values: In2D<Array2<F>>) -> Self {
        VectorField2{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// component values are defined.
    pub fn coords<'a>(&'a self) -> In2D<CoordRefs2<'a, F>> {
        In2D::new(CoordRefs2::new(
             &self.grid.coords_by_type(self.locations[Dim2::X][Dim2::X])[Dim2::X],
             &self.grid.coords_by_type(self.locations[Dim2::X][Dim2::Y])[Dim2::Y]
         ),
         CoordRefs2::new(
             &self.grid.coords_by_type(self.locations[Dim2::Y][Dim2::X])[Dim2::X],
             &self.grid.coords_by_type(self.locations[Dim2::Y][Dim2::Y])[Dim2::Y]
         ))
    }

    /// Returns a reference to the two 2D arrays of field component values.
    pub fn values(&self) -> &In2D<Array2<F>> { &self.values }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> { self.grid.shape() }

    /// Creates a new scalar field from the specified vector field component.
    pub fn component_as_scalar_field(&self, dim: Dim2) -> ScalarField2<F, G> {
        let dim_names = In2D::new('x', 'y');
        let name = format!("{}{}", self.name(), dim_names[dim]);
        ScalarField2::new(name, self.grid.clone(), self.locations[dim].clone(), self.values[dim].clone())
    }
}

/// A field of 3D vectors in a 2D plane.
///
/// Holds the grid and values of the three components of a vector field in a 2D plane,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct PlaneVectorField3<F, G>
where F: BFloat,
      G: Grid2<F> + Clone
{
    name: String,
    grid: G,
    locations: In3D<In2D<CoordLocation>>,
    values: In3D<Array2<F>>
}

impl<F, G> PlaneVectorField3<F, G>
where F: BFloat,
      G: Grid2<F> + Clone
{
    /// Creates a new vector field given a name, a grid, the component values and
    /// coordinate types specifying where in the grid cell the component values are defined.
    pub fn new(name: String, grid: G, locations: In3D<In2D<CoordLocation>>, values: In3D<Array2<F>>) -> Self {
        PlaneVectorField3{ name, grid, locations, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// component values are defined.
    pub fn coords<'a>(&'a self) -> In3D<CoordRefs2<'a, F>> {
        In3D::new(CoordRefs2::new(
             &self.grid.coords_by_type(self.locations[X][Dim2::X])[Dim2::X],
             &self.grid.coords_by_type(self.locations[X][Dim2::Y])[Dim2::Y]
         ),
         CoordRefs2::new(
             &self.grid.coords_by_type(self.locations[Y][Dim2::X])[Dim2::X],
             &self.grid.coords_by_type(self.locations[Y][Dim2::Y])[Dim2::Y]
         ),
         CoordRefs2::new(
             &self.grid.coords_by_type(self.locations[Z][Dim2::X])[Dim2::X],
             &self.grid.coords_by_type(self.locations[Z][Dim2::Y])[Dim2::Y]
         ))
    }

    /// Returns a reference to the three 2D arrays of field component values.
    pub fn values(&self) -> &In3D<Array2<F>> { &self.values }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> { self.grid.shape() }

    /// Creates a new scalar field from the specified vector field component.
    pub fn component_as_scalar_field(&self, dim: Dim3) -> ScalarField2<F, G> {
        let dim_names = In3D::new('x', 'y', 'z');
        let name = format!("{}{}", self.name(), dim_names[dim]);
        ScalarField2::new(name, self.grid.clone(), self.locations[dim].clone(), self.values[dim].clone())
    }
}
