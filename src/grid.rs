//! Grids used in Bifrost.

pub mod regular;
pub mod hor_regular;

use num;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Vec3, Vec2, Point3, Point2, Idx3, Idx2, Coords3, Coords2, CoordRefs3, CoordRefs2};
use self::regular::RegularGrid2;
use Dim3::{X, Y, Z};

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
pub enum CoordLocation {
    Center = 0,
    LowerEdge = 1
}

/// Regular grid or only uniform in the horizontal direction.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum GridType {
    Regular,
    HorRegular
}

/// Defines the properties of a 3D grid.
pub trait Grid3<F: num::Float>: Clone {
    type XSliceGrid: Grid2<F>;
    type YSliceGrid: Grid2<F>;
    type ZSliceGrid: Grid2<F>;

    /// The specific type of the grid.
    const TYPE: GridType;

    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    fn new(center_coords: Coords3<F>, lower_edge_coords: Coords3<F>, is_periodic: In3D<bool>) -> Self;

    /// Returns the 3D shape of the grid.
    fn shape(&self) -> &In3D<usize>;

    /// Whether the grid is periodic along the given dimension.
    fn is_periodic(&self, dim: Dim3) -> bool;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, location: CoordLocation) -> &Coords3<F>;

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &Coords3<F> { self.coords_by_type(CoordLocation::Center) }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &Coords3<F> { self.coords_by_type(CoordLocation::LowerEdge) }

    /// Returns a reference to the central coordinates in a regular version of the grid.
    fn regular_centers(&self) -> CoordRefs3<F>;

    /// Returns a reference to the lower coordinates in a regular version of the grid.
    fn regular_lower_edges(&self) -> CoordRefs3<F>;

    /// Returns a reference to the lower coordinate bounds of each dimension.
    fn lower_bounds(&self) -> &Vec3<F>;

    /// Returns a reference to the upper coordinate bounds of each dimension.
    fn upper_bounds(&self) -> &Vec3<F>;

    /// Returns a reference to the full coordinate extent of each dimension.
    fn extents(&self) -> &Vec3<F>;

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// or specifies on which side of the grid the coordinate lies if outside.
    fn find_grid_cell(&self, point: &Point3<F>) -> FoundIdx3;

    /// Given a point that may be outside the grid boundaries, returns a new point
    /// wrapped around the boundaries to the inside of the grid, or `None` if the
    /// point is outside a non-periodic boundary.
    fn wrap_point(&self, point: &Point3<F>) -> Option<Point3<F>> {
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();
        let mut wrapped_point = point.clone();
        for dim in Dim3::slice().iter() {
            if self.is_periodic(*dim) {
                if wrapped_point[*dim] < lower_bounds[*dim] {
                    wrapped_point[*dim] = upper_bounds[*dim] - ((upper_bounds[*dim] - point[*dim]) % extents[*dim]);
                } else if wrapped_point[*dim] >= upper_bounds[*dim] {
                    wrapped_point[*dim] = lower_bounds[*dim] + ((point[*dim] - lower_bounds[*dim]) % extents[*dim]);
                }
            } else if wrapped_point[*dim] < lower_bounds[*dim] || wrapped_point[*dim] >= upper_bounds[*dim] {
                return None
            }
        }
        Some(wrapped_point)
    }

    /// Slices the grid across the x-axis and returns the corresponding 2D grid.
    fn slice_across_x(&self) -> Self::XSliceGrid {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::XSliceGrid::new(
            Coords2::new(centers[Y].clone(), centers[Z].clone()),
            Coords2::new(lower_edges[Y].clone(), lower_edges[Z].clone()),
            In2D::new(self.is_periodic(Y), self.is_periodic(Z))
        )
    }

    /// Slices the grid across the y-axis and returns the corresponding 2D grid.
    fn slice_across_y(&self) -> Self::YSliceGrid {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::YSliceGrid::new(
            Coords2::new(centers[X].clone(), centers[Z].clone()),
            Coords2::new(lower_edges[X].clone(), lower_edges[Z].clone()),
            In2D::new(self.is_periodic(X), self.is_periodic(Z))
        )
    }

    /// Slices the grid across the z-axis and returns the corresponding 2D grid.
    fn slice_across_z(&self) -> Self::ZSliceGrid {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::ZSliceGrid::new(
            Coords2::new(centers[X].clone(), centers[Y].clone()),
            Coords2::new(lower_edges[X].clone(), lower_edges[Y].clone()),
            In2D::new(self.is_periodic(X), self.is_periodic(Y))
        )
    }

    /// Slices the grid across the given axis and returns the corresponding regular 2D grid.
    fn regular_slice_across_axis(&self, axis: Dim3) -> RegularGrid2<F>
    where F: num::cast::FromPrimitive
    {
        let regular_centers = self.regular_centers();
        let regular_lower_edges = self.regular_lower_edges();
        let [axis_0, axis_1] = Dim3::slice_except(axis);
        RegularGrid2::new(
            Coords2::new(regular_centers[axis_0].clone(), regular_centers[axis_1].clone()),
            Coords2::new(regular_lower_edges[axis_0].clone(), regular_lower_edges[axis_1].clone()),
            In2D::new(self.is_periodic(axis_0), self.is_periodic(axis_1))
        )
    }
}

/// A found 2D index inside the grid or a bounds crossing for each dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum FoundIdx2 {
    Inside(Idx2<usize>),
    Outside(In2D<BoundsCrossing>)
}

/// Defines the properties of a 2D grid.
pub trait Grid2<F: num::Float>: Clone {

    /// The specific type of the grid.
    const TYPE: GridType;

    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    fn new(center_coords: Coords2<F>, lower_edge_coords: Coords2<F>, is_periodic: In2D<bool>) -> Self;

    /// Returns the 2D shape of the grid.
    fn shape(&self) -> &In2D<usize>;

    /// Whether the grid is periodic along the given dimension.
    fn is_periodic(&self, dim: Dim2) -> bool;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, location: CoordLocation) -> &Coords2<F>;

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &Coords2<F> { self.coords_by_type(CoordLocation::Center) }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &Coords2<F> { self.coords_by_type(CoordLocation::LowerEdge) }

    /// Returns a reference to the central coordinates in a regular version of the grid.
    fn regular_centers(&self) -> CoordRefs2<F>;

    /// Returns a reference to the lower coordinates in a regular version of the grid.
    fn regular_lower_edges(&self) -> CoordRefs2<F>;

    /// Returns a reference to the lower coordinate bounds of each dimension.
    fn lower_bounds(&self) -> &Vec2<F>;

    /// Returns a reference to the upper coordinate bounds of each dimension.
    fn upper_bounds(&self) -> &Vec2<F>;

    /// Returns a reference to the full coordinate extent of each dimension.
    fn extents(&self) -> &Vec2<F>;

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// or specifies on which side of the grid the coordinate lies if outside.
    fn find_grid_cell(&self, point: &Point2<F>) -> FoundIdx2;

    /// Given a point that may be outside the grid boundaries, returns a new point
    /// wrapped around the boundaries to the inside of the grid, or `None` if the
    /// point is outside a non-periodic boundary.
    fn wrap_point(&self, point: &Point2<F>) -> Option<Point2<F>> {
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();
        let mut wrapped_point = point.clone();
        for dim in Dim2::slice().iter() {
            if self.is_periodic(*dim) {
                if wrapped_point[*dim] < lower_bounds[*dim] {
                    wrapped_point[*dim] = upper_bounds[*dim] - ((upper_bounds[*dim] - point[*dim]) % extents[*dim]);
                } else if wrapped_point[*dim] >= upper_bounds[*dim] {
                    wrapped_point[*dim] = lower_bounds[*dim] + ((point[*dim] - lower_bounds[*dim]) % extents[*dim]);
                }
            } else if wrapped_point[*dim] < lower_bounds[*dim] || wrapped_point[*dim] >= upper_bounds[*dim] {
                return None
            }
        }
        Some(wrapped_point)
    }
}