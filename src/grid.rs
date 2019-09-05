//! Grids used in Bifrost.

pub mod regular;
pub mod hor_regular;

use num;
use ndarray::prelude::*;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Vec3, Vec2, Point3, Point2, Idx3, Idx2, Coords3, Coords2, CoordRefs3, CoordRefs2};
use self::regular::RegularGrid2;
use Dim3::{X, Y, Z};

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
    fn from_coords(center_coords: Coords3<F>, lower_edge_coords: Coords3<F>, is_periodic: In3D<bool>) -> Self
    where F: num::cast::FromPrimitive;

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

    /// Creates a vector of points corresponding to grid cell centers or lower edges,
    /// collapsed with the inner dimension varying the fastest.
    fn create_point_list(&self, location: CoordLocation) -> Vec<Point3<F>> {
        let coords = match location {
            CoordLocation::Center => self.centers(),
            CoordLocation::LowerEdge => self.lower_edges()
        };
        let shape = self.shape();
        let mut points = Vec::with_capacity(shape[X]*shape[Y]*shape[Z]);
        for k in 0..shape[Z] {
            for j in 0..shape[Y] {
                for i in 0..shape[X] {
                    points.push(Point3::new(coords[X][i], coords[Y][j], coords[Z][k]));
                }
            }
        }
        points
    }

    /// Whether the given point is inside the bounds of the grid.
    fn point_is_inside(&self, point: &Point3<F>) -> bool {
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        Dim3::slice().iter().all(|&dim| point[dim] >= lower_bounds[dim] && point[dim] < upper_bounds[dim])
    }

    /// Whether the given point is inside the bounds of the given grid cell.
    fn point_is_inside_cell(&self, point: &Point3<F>, cell_idx: &Idx3<usize>) -> bool {
        let lower_edges = self.lower_edges();
        Dim3::slice().iter().all(|&dim| coord_is_inside_grid_cell(&lower_edges[dim], point[dim], cell_idx[dim]))
    }

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// or specifies on which side of the grid the coordinate lies if outside.
    fn find_grid_cell(&self, point: &Point3<F>) -> Option<Idx3<usize>>
    where F: num::cast::FromPrimitive
    {
        if self.point_is_inside(point) {
            let lower_edges = self.lower_edges();
            let i = search_idx_of_coord(&lower_edges[X], point[X]).unwrap();
            let j = search_idx_of_coord(&lower_edges[Y], point[Y]).unwrap();
            let k = search_idx_of_coord(&lower_edges[Z], point[Z]).unwrap();
            let idx = Idx3::new(i, j, k);
            debug_assert!(self.point_is_inside_cell(point, &idx));
            Some(idx)
        } else {
            None
        }
    }

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
    fn slice_across_x(&self) -> Self::XSliceGrid
    where F: num::cast::FromPrimitive
    {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::XSliceGrid::from_coords(
            Coords2::new(centers[Y].clone(), centers[Z].clone()),
            Coords2::new(lower_edges[Y].clone(), lower_edges[Z].clone()),
            In2D::new(self.is_periodic(Y), self.is_periodic(Z))
        )
    }

    /// Slices the grid across the y-axis and returns the corresponding 2D grid.
    fn slice_across_y(&self) -> Self::YSliceGrid
    where F: num::cast::FromPrimitive
    {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::YSliceGrid::from_coords(
            Coords2::new(centers[X].clone(), centers[Z].clone()),
            Coords2::new(lower_edges[X].clone(), lower_edges[Z].clone()),
            In2D::new(self.is_periodic(X), self.is_periodic(Z))
        )
    }

    /// Slices the grid across the z-axis and returns the corresponding 2D grid.
    fn slice_across_z(&self) -> Self::ZSliceGrid
    where F: num::cast::FromPrimitive
    {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::ZSliceGrid::from_coords(
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
        RegularGrid2::from_coords(
            Coords2::new(regular_centers[axis_0].to_vec(), regular_centers[axis_1].to_vec()),
            Coords2::new(regular_lower_edges[axis_0].to_vec(), regular_lower_edges[axis_1].to_vec()),
            In2D::new(self.is_periodic(axis_0), self.is_periodic(axis_1))
        )
    }
}

/// Defines the properties of a 2D grid.
pub trait Grid2<F: num::Float>: Clone {

    /// The specific type of the grid.
    const TYPE: GridType;

    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    fn from_coords(center_coords: Coords2<F>, lower_edge_coords: Coords2<F>, is_periodic: In2D<bool>) -> Self
    where F: num::cast::FromPrimitive;

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

    /// Creates a vector of points corresponding to grid cell centers or lower edges,
    /// collapsed with the inner dimension varying the fastest.
    fn create_point_list(&self, location: CoordLocation) -> Vec<Point2<F>> {
        let coords = match location {
            CoordLocation::Center => self.centers(),
            CoordLocation::LowerEdge => self.lower_edges()
        };
        let shape = self.shape();
        let mut points = Vec::with_capacity(shape[Dim2::X]*shape[Dim2::Y]);
        for j in 0..shape[Dim2::Y] {
            for i in 0..shape[Dim2::X] {
                points.push(Point2::new(coords[Dim2::X][i], coords[Dim2::Y][j]));
            }
        }
        points
    }

    /// Whether the given point is inside the bounds of the grid.
    fn point_is_inside(&self, point: &Point2<F>) -> bool {
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        Dim2::slice().iter().all(|&dim| point[dim] >= lower_bounds[dim] && point[dim] < upper_bounds[dim])
    }

    /// Whether the given point is inside the bounds of the given grid cell.
    fn point_is_inside_cell(&self, point: &Point2<F>, cell_idx: &Idx2<usize>) -> bool {
        let lower_edges = self.lower_edges();
        Dim2::slice().iter().all(|&dim| coord_is_inside_grid_cell(&lower_edges[dim], point[dim], cell_idx[dim]))
    }

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// or specifies on which side of the grid the coordinate lies if outside.
    fn find_grid_cell(&self, point: &Point2<F>) -> Option<Idx2<usize>>
    where F: num::cast::FromPrimitive
    {
        if self.point_is_inside(point) {
            let lower_edges = self.lower_edges();
            let i = search_idx_of_coord(&lower_edges[Dim2::X], point[Dim2::X]).unwrap();
            let j = search_idx_of_coord(&lower_edges[Dim2::Y], point[Dim2::Y]).unwrap();
            let idx = Idx2::new(i, j);
            debug_assert!(self.point_is_inside_cell(point, &idx));
            Some(idx)
        } else {
            None
        }
    }

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

fn extent_from_bounds<F>(lower_bound: F, upper_bound: F) -> F
where F: num::Float
{
    upper_bound - lower_bound
}

fn cell_extent_from_bounds<F>(size: usize, lower_bound: F, upper_bound: F) -> F
where F: num::Float + num::cast::FromPrimitive
{
    let extent = extent_from_bounds(lower_bound, upper_bound);
    extent/F::from_usize(size).unwrap()
}

fn regular_coords_from_bounds<F>(size: usize, lower_bound: F, upper_bound: F) -> (Vec<F>, Vec<F>)
where F: num::Float + num::cast::FromPrimitive
{
    let cell_extent = cell_extent_from_bounds(size, lower_bound, upper_bound);
    let half_cell_extent = F::from_f32(0.5).unwrap()*cell_extent;
    let centers = Array::linspace(lower_bound + half_cell_extent, upper_bound - half_cell_extent, size).to_vec();
    let lower_edges = Array::linspace(lower_bound, upper_bound - cell_extent, size).to_vec();
    (centers, lower_edges)
}

fn bounds_from_coords<F>(size: usize, centers: &[F], lower_edges: &[F]) -> (F, F)
where F: num::Float + num::cast::FromPrimitive
{
    (lower_edges[0], F::from_f32(2.0).unwrap()*centers[size - 1] - lower_edges[size - 1])
}

fn search_idx_of_coord<F>(lower_edges: &[F], coord: F) -> Option<usize>
where F: num::Float + num::cast::FromPrimitive
{
    let mut low = 0;
    let mut high = lower_edges.len() - 1;
    let mut mid;

    if coord >= lower_edges[high] {
        return Some(high)
    }

    while coord >= lower_edges[low] && coord < lower_edges[high] {

        let low_float  = F::from_usize(low).unwrap();
        let high_float = F::from_usize(high).unwrap();
        let mid_float = (low_float + (coord - lower_edges[low])*(high_float - low_float)/(lower_edges[high] - lower_edges[low])).floor();

        mid = F::to_usize(&mid_float).unwrap();

        if mid >= high {
            // Due to roundoff error, we might get `mid == high` even though `coord < lower_edges[high]`.
            // If this happens, `coord` will be very close to `lower_edges[high]`, so we return `high - 1`.
            return Some(high - 1)
        }

        if lower_edges[mid + 1] <= coord {
            low = mid + 1
        } else if lower_edges[mid] > coord {
            high = mid
        } else {
            return Some(mid)
        }
    }
    None
}

fn coord_is_inside_grid_cell<F: num::Float>(lower_edges: &[F], coord: F, cell_idx: usize) -> bool {
    (cell_idx == (lower_edges.len() - 1) || coord < lower_edges[cell_idx+1]) && coord >= lower_edges[cell_idx]
}