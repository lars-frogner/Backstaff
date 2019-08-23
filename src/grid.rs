//! Grids used in Bifrost.

pub mod regular;
pub mod hor_regular;

use num;
use crate::geometry::{Dim3, In3D, Point3, Idx3, Coords3};

/// A potential crossing of the lower or upper bounds of a grid dimension.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BoundsCrossing {
    None,
    Upper,
    Lower
}

/// A found 3D index inside the grid or a bounds crossing for each dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum FoundIdx3 {
    Inside(Idx3<usize>),
    Outside(In3D<BoundsCrossing>)
}

/// Coordinates located at center or lower edge of grid cell.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CoordsType {
    Center = 0,
    Lower = 1
}

/// Regular grid or non-uniform in z-direction.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Grid3Type {
    Regular,
    HorRegular
}

/// Defines the properties of a 3D grid.
pub trait Grid3<T: num::Float> {

    /// The specific type of the grid.
    const TYPE: Grid3Type;

    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    fn new(center_coords: Coords3<T>, lower_edge_coords: Coords3<T>, is_periodic: In3D<bool>) -> Self;

    /// Returns the 3D shape of the grid.
    fn shape(&self) -> &In3D<usize>;

    /// Whether the grid is periodic along the given dimension.
    fn is_periodic(&self, dim: Dim3) -> bool;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, coord_type: CoordsType) -> &Coords3<T>;

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &Coords3<T> { self.coords_by_type(CoordsType::Center) }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &Coords3<T> { self.coords_by_type(CoordsType::Lower) }

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// or specifies on which side of the grid the coordinate lies if outside.
    fn find_grid_cell(&self, point: &Point3<T>) -> FoundIdx3;
}
