//! Fields representing Bifrost simulation variables.

use num;
use ndarray::prelude::*;
use crate::geometry::{Dim3, In3D, Point3, CoordRefs3};
use crate::grid::{CoordsType, Grid3};
use crate::interpolation::Interpolator3;
use Dim3::{X, Y, Z};

/// A 3D scalar field.
///
/// Holds the grid and values of a 3D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct ScalarField3<F, G>
where F: num::Float,
      G: Grid3<F> + Clone
{
    name: String,
    grid: G,
    coord_types: In3D<CoordsType>,
    values: Array3<F>
}

/// Locations for resampled field values.
pub enum ResampleLocations {
    Original,
    Center,
    LowerEdge,
    UniformCenter
}

impl<F, G> ScalarField3<F, G>
where F: num::Float + std::fmt::Display,
      G: Grid3<F> + Clone
{
    /// Creates a new 3D scalar field given a name, a grid, the values and
    /// coordinate types specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: G, coord_types: In3D<CoordsType>, values: Array3<F>) -> Self {
        ScalarField3{ name, grid, coord_types, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs3<F> {
        CoordRefs3::new(
            &self.grid.coords_by_type(self.coord_types[X])[X],
            &self.grid.coords_by_type(self.coord_types[Y])[Y],
            &self.grid.coords_by_type(self.coord_types[Z])[Z]
        )
    }

    /// Returns a reference to the 3D array of field values.
    pub fn values(&self) -> &Array3<F> { &self.values }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> { self.grid.shape() }

    /// Returns a view of the 2D slice of the field located at the given index along the given dimension.
    pub fn slice_at_idx(&self, dim: Dim3, idx: usize) -> ArrayView2<F> {
        self.values.index_axis(Axis(dim as usize), idx)
    }

    /// Returns a 2D slice of the field located at the given coordinate along the given dimension.
    pub fn slice_at_coord<I>(&self, interpolator: &I, dim: Dim3, coord: F, resample_locations: ResampleLocations) -> Array2<F>
    where I: Interpolator3
    {
        let grid_shape = self.shape();
        let [dim_0, dim_1] = Dim3::slice_except(dim);

        let lower_bound = self.grid.lower_bounds()[dim];
        let upper_bound = self.grid.upper_bounds()[dim];
        if coord < lower_bound || coord >= upper_bound{
            panic!("`coord` is outside the bounds [{}, {})", lower_bound, upper_bound);
        }

        let mut point = Point3::origin();
        point[dim] = coord;

        let (coords_0, coords_1) = match resample_locations {
            ResampleLocations::Original => {
                let coords = self.coords();
                (coords[dim_0], coords[dim_1])
            },
            ResampleLocations::Center => {
                let centers = self.grid.centers();
                (&centers[dim_0], &centers[dim_1])
            },
            ResampleLocations::LowerEdge => {
                let lower_edges = self.grid.lower_edges();
                (&lower_edges[dim_0], &lower_edges[dim_1])
            },
            ResampleLocations::UniformCenter => {
                let uniform_centers = self.grid.uniform_centers();
                (uniform_centers[dim_0], uniform_centers[dim_1])
            }
        };

        let slice_shape = (grid_shape[dim_0], grid_shape[dim_1]);
        let mut slice = unsafe { Array2::uninitialized(slice_shape.f()) };

        for idx_1 in 0..grid_shape[dim_1] {
            point[dim_1] = coords_1[idx_1];
            for idx_0 in  0..grid_shape[dim_0] {
                point[dim_0] = coords_0[idx_0];
                slice[[idx_0, idx_1]] = interpolator.interp_scalar_field(self, &point).unwrap();
            }
        }

        slice
    }
}

/// A 3D vector field.
///
/// Holds the grid and values of the three components of a 3D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct VectorField3<F, G>
where F: num::Float,
      G: Grid3<F> + Clone
{
    name: String,
    grid: G,
    coord_types: In3D<In3D<CoordsType>>,
    values: In3D<Array3<F>>
}

impl<F, G> VectorField3<F, G>
where F: num::Float + std::fmt::Display,
      G: Grid3<F> + Clone
{
    /// Creates a new 3D vector field given a name, a grid, the component values and
    /// coordinate types specifying where in the grid cell the component values are defined.
    pub fn new(name: String, grid: G, coord_types: In3D<In3D<CoordsType>>, values: In3D<Array3<F>>) -> Self {
        VectorField3{ name, grid, coord_types, values }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str { &self.name }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// component values are defined.
    pub fn coords<'a>(&'a self) -> In3D<CoordRefs3<'a, F>> {
        In3D::new(CoordRefs3::new(
             &self.grid.coords_by_type(self.coord_types[X][X])[X],
             &self.grid.coords_by_type(self.coord_types[X][Y])[Y],
             &self.grid.coords_by_type(self.coord_types[X][Z])[Z]
         ),
         CoordRefs3::new(
             &self.grid.coords_by_type(self.coord_types[Y][X])[X],
             &self.grid.coords_by_type(self.coord_types[Y][Y])[Y],
             &self.grid.coords_by_type(self.coord_types[Y][Z])[Z]
         ),
         CoordRefs3::new(
             &self.grid.coords_by_type(self.coord_types[Z][X])[X],
             &self.grid.coords_by_type(self.coord_types[Z][Y])[Y],
             &self.grid.coords_by_type(self.coord_types[Z][Z])[Z]
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
        ScalarField3::new(name, self.grid.clone(), self.coord_types[dim].clone(), self.values[dim].clone())
    }
}
