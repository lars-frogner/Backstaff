//! Fields for representing Bifrost simulation variables.

use num;
use ndarray::prelude::*;
use crate::grid::{Dim, In3D, CoordsType, CoordRefs3, Grid3};
use Dim::{X, Y, Z};

/// A 3D scalar field.
///
/// Holds the grid and values of a 3D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct ScalarField3<T, G>
where T: num::Float,
      G: Grid3<T> + Clone
{
    grid: G,
    coord_types: In3D<CoordsType>,
    values: Array3<T>
}

impl<T, G> ScalarField3<T, G>
where T: num::Float,
      G: Grid3<T> + Clone
{
    /// Creates a new 3D scalar field given the grid, the values and
    /// coordinate types specifying where in the grid cell the values are defined.
    pub fn new(grid: G, coord_types: In3D<CoordsType>, values: Array3<T>) -> Self {
        ScalarField3{ grid, coord_types, values }
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords<'a>(&'a self) -> CoordRefs3<'a, T> {
        CoordRefs3::new(
            &self.grid.coords_by_type(self.coord_types[X])[X],
            &self.grid.coords_by_type(self.coord_types[Y])[Y],
            &self.grid.coords_by_type(self.coord_types[Z])[Z],
        )
    }

    /// Returns a reference to the 3D array of field values.
    pub fn values(&self) -> &Array3<T> { &self.values }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> { self.grid.shape() }
}

/// A 3D vector field.
///
/// Holds the grid and values of the three components of a 3D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Debug, Clone)]
pub struct VectorField3<T, G>
where T: num::Float,
      G: Grid3<T> + Clone
{
    grid: G,
    coord_types: In3D<In3D<CoordsType>>,
    values: In3D<Array3<T>>
}

impl<T, G> VectorField3<T, G>
where T: num::Float,
      G: Grid3<T> + Clone
{
    /// Creates a new 3D vector field given the grid, the component values and
    /// coordinate types specifying where in the grid cell the component values are defined.
    pub fn new(grid: G, coord_types: In3D<In3D<CoordsType>>, values: In3D<Array3<T>>) -> Self {
        VectorField3{ grid, coord_types, values }
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { &self.grid }

    /// Returns a set of references to the coordinates where the field
    /// component values are defined.
    pub fn coords<'a>(&'a self) -> In3D<CoordRefs3<'a, T>> {
        In3D::new(CoordRefs3::new(
             &self.grid.coords_by_type(self.coord_types[X][X])[X],
             &self.grid.coords_by_type(self.coord_types[X][Y])[Y],
             &self.grid.coords_by_type(self.coord_types[X][Z])[Z],
         ),
         CoordRefs3::new(
             &self.grid.coords_by_type(self.coord_types[Y][X])[X],
             &self.grid.coords_by_type(self.coord_types[Y][Y])[Y],
             &self.grid.coords_by_type(self.coord_types[Y][Z])[Z],
         ),
         CoordRefs3::new(
             &self.grid.coords_by_type(self.coord_types[Z][X])[X],
             &self.grid.coords_by_type(self.coord_types[Z][Y])[Y],
             &self.grid.coords_by_type(self.coord_types[Z][Z])[Z],
         ))
    }

    /// Returns a reference to the three 3D arrays of field component values.
    pub fn values(&self) -> &In3D<Array3<T>> { &self.values }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> { self.grid.shape() }
}
