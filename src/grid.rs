//! Structured grids.

pub mod hor_regular;
pub mod regular;

use self::regular::RegularGrid2;
use crate::geometry::{
    CoordRefs2, CoordRefs3, Coords2, Coords3, Dim2, Dim3, Idx2, Idx3, In2D, In3D, Point2, Point3,
    Vec2, Vec3,
};
use crate::num::BFloat;
use ndarray::prelude::*;
use Dim3::{X, Y, Z};

/// Coordinates located at center or lower edge of grid cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoordLocation {
    Center = 0,
    LowerEdge = 1,
}

/// Regular grid or only uniform in the horizontal direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridType {
    Regular,
    HorRegular,
}

/// A query for a result at a 3D grid point.
///
/// * If the point is inside the grid bounds, the query contains a result of type `T`.
/// * If the point is outside a periodic boundary, in contains the result as well as the wrapped position.
/// * If the point is outside a non-periodic boundary, it contains no result.
#[derive(Clone, Debug, PartialEq)]
pub enum GridPointQuery3<F: BFloat, T> {
    Inside(T),
    MovedInside((T, Point3<F>)),
    Outside,
}

/// A query for a result at a 3D grid point.
///
/// * If the point is inside the grid bounds, the query contains a result of type `T`.
/// * If the point is outside a periodic boundary, in contains the result as well as the wrapped position.
/// * If the point is outside a non-periodic boundary, it contains no result.
#[derive(Clone, Debug, PartialEq)]
pub enum GridPointQuery2<F: BFloat, T> {
    Inside(T),
    MovedInside((T, Point2<F>)),
    Outside,
}

/// Defines the properties of a 3D grid.
pub trait Grid3<F: BFloat>: Clone + Sync + Send {
    type XSliceGrid: Grid2<F>;
    type YSliceGrid: Grid2<F>;
    type ZSliceGrid: Grid2<F>;

    /// The specific type of the grid.
    const TYPE: GridType;

    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    ///
    /// Optionally, the upward and downward derivatives of the coordinates can be
    /// passed in to be stored with the grid. They are not used for computations.
    fn from_coords(
        center_coords: Coords3<F>,
        lower_edge_coords: Coords3<F>,
        is_periodic: In3D<bool>,
        up_derivatives: Option<Coords3<F>>,
        down_derivatives: Option<Coords3<F>>,
    ) -> Self;

    /// Returns the 3D shape of the grid.
    fn shape(&self) -> &In3D<usize>;

    /// Whether the grid is periodic along the given dimension.
    fn is_periodic(&self, dim: Dim3) -> bool;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, location: CoordLocation) -> &Coords3<F>;

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &Coords3<F> {
        self.coords_by_type(CoordLocation::Center)
    }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &Coords3<F> {
        self.coords_by_type(CoordLocation::LowerEdge)
    }

    /// Returns a reference to the upward derivatives of the grid coordinates,
    /// if they are available.
    fn up_derivatives(&self) -> Option<&Coords3<F>>;

    /// Returns a reference to the downward derivatives of the grid coordinates,
    /// if they are available.
    fn down_derivatives(&self) -> Option<&Coords3<F>>;

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

    /// Returns the lower and upper corner of the grid cell of the given 3D index.
    fn grid_cell_extremal_corners(&self, indices: &Idx3<usize>) -> (Point3<F>, Point3<F>) {
        let lower_corner = self.lower_edges().point(indices);
        let center = self.centers().point(indices);
        let upper_corner = &center + (&center - &lower_corner);
        (lower_corner, upper_corner)
    }

    /// Returns the coordinate extents of the grid cell at the given 3D index.
    fn grid_cell_extents(&self, indices: &Idx3<usize>) -> Vec3<F> {
        let (lower_corner, upper_corner) = self.grid_cell_extremal_corners(indices);
        upper_corner - lower_corner
    }

    /// Returns the volume of the grid cell at the given 3D index.
    fn grid_cell_volume(&self, indices: &Idx3<usize>) -> F {
        let grid_cell_extents = self.grid_cell_extents(indices);
        grid_cell_extents[X] * grid_cell_extents[Y] * grid_cell_extents[Z]
    }

    /// Creates a vector of points corresponding to grid cell centers or lower edges,
    /// collapsed with the inner dimension varying the fastest.
    fn create_point_list(&self, location: CoordLocation) -> Vec<Point3<F>> {
        let coords = match location {
            CoordLocation::Center => self.centers(),
            CoordLocation::LowerEdge => self.lower_edges(),
        };
        let shape = self.shape();
        let mut points = Vec::with_capacity(shape[X] * shape[Y] * shape[Z]);
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
        Dim3::slice()
            .iter()
            .all(|&dim| point[dim] >= lower_bounds[dim] && point[dim] < upper_bounds[dim])
    }

    /// Whether the given 3D index is inside the bounds of the grid.
    fn indices_are_inside(&self, indices: &Idx3<usize>) -> bool {
        let shape = self.shape();
        Dim3::slice().iter().all(|&dim| indices[dim] < shape[dim])
    }

    /// Whether the given point is inside the bounds of the given grid cell.
    fn point_is_inside_cell(&self, point: &Point3<F>, cell_idx: &Idx3<usize>) -> bool {
        let lower_edges = self.lower_edges();
        Dim3::slice()
            .iter()
            .all(|&dim| coord_is_inside_grid_cell(&lower_edges[dim], point[dim], cell_idx[dim]))
    }

    /// Tries to find the 3D index of the grid cell containing the given coordinate,
    /// and returns the result as a `GridPointQuery3`.
    fn find_grid_cell(&self, point: &Point3<F>) -> GridPointQuery3<F, Idx3<usize>> {
        let lower_edges = self.lower_edges();
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();

        let mut point = point.clone();
        let mut indices = Idx3::origin();
        let mut wrapped = false;

        for &dim in &Dim3::slice() {
            if point[dim] < lower_bounds[dim] {
                if self.is_periodic(dim) {
                    point[dim] = wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                } else {
                    return GridPointQuery3::Outside;
                }
            } else if point[dim] >= upper_bounds[dim] {
                if self.is_periodic(dim) {
                    point[dim] = wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                } else {
                    return GridPointQuery3::Outside;
                }
            };
            indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                .expect("Coordinate index search failed.");
        }
        debug_assert!(
            self.point_is_inside_cell(&point, &indices),
            "Found wrong grid cell."
        );

        if wrapped {
            GridPointQuery3::MovedInside((indices, point))
        } else {
            GridPointQuery3::Inside(indices)
        }
    }

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// wrapping around any periodic boundaries,
    /// or the index of the closest grid cell if the coordinate is outside
    /// a non-periodic boundary.
    ///
    /// The result is returned as a `GridPointQuery3` which is guaranteed not
    /// to be `Outside`.
    fn find_closest_grid_cell(&self, point: &Point3<F>) -> GridPointQuery3<F, Idx3<usize>> {
        let lower_edges = self.lower_edges();
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();
        let shape = self.shape();

        let mut point = point.clone();
        let mut indices = Idx3::origin();
        let mut wrapped = false;

        for &dim in &Dim3::slice() {
            if self.is_periodic(dim) {
                if point[dim] < lower_bounds[dim] {
                    point[dim] = wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                } else if point[dim] >= upper_bounds[dim] {
                    point[dim] = wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                };
                indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                    .expect("Coordinate index search failed.");
            } else if point[dim] < lower_bounds[dim] {
                indices[dim] = 0;
                point[dim] = lower_bounds[dim];
            } else if point[dim] >= upper_bounds[dim] {
                indices[dim] = shape[dim] - 1;
                point[dim] = upper_bounds[dim].prev();
            } else {
                indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                    .expect("Coordinate index search failed.");
            }
        }
        debug_assert!(
            self.indices_are_inside(&indices),
            "Found inside index is actually on the outside."
        );

        if wrapped {
            GridPointQuery3::MovedInside((indices, point))
        } else {
            GridPointQuery3::Inside(indices)
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
        for &dim in &Dim3::slice() {
            if self.is_periodic(dim) {
                if wrapped_point[dim] < lower_bounds[dim] {
                    wrapped_point[dim] =
                        wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                } else if wrapped_point[dim] >= upper_bounds[dim] {
                    wrapped_point[dim] =
                        wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                }
            } else if wrapped_point[dim] < lower_bounds[dim]
                || wrapped_point[dim] >= upper_bounds[dim]
            {
                return None;
            }
        }
        debug_assert!(
            self.point_is_inside(&wrapped_point),
            "Wrapped to outside of bounds."
        );
        Some(wrapped_point)
    }

    /// Given a 1D index that may be outside the grid boundaries for the given dimension,
    /// returns a new index wrapped around the boundaries to the inside of the grid,
    /// or the index of the closest grid cell if the index is outside
    /// a non-periodic boundary.
    fn wrap_idx_to_closest(&self, dim: Dim3, idx: isize) -> usize {
        let length = self.shape()[dim];
        if self.is_periodic(dim) {
            if idx < 0 {
                wrap_idx_lower(length, idx)
            } else if (idx as usize) >= length {
                wrap_idx_upper(length, idx)
            } else {
                idx as usize
            }
        } else if idx < 0 {
            0
        } else if (idx as usize) >= length {
            length - 1
        } else {
            idx as usize
        }
    }

    /// Creates a vector containing 1D indices running from the given lower
    /// (inclusive) to upper (exclusive) index, wrapping around periodic
    /// boundaries and truncating the range if crossing non-periodic
    /// boundaries.
    ///
    /// The given indices are allowed to lie outside the grid, but the lower
    /// index must be smaller than the upper index.
    fn create_idx_range_list<I>(&self, dim: Dim3, lower_idx: I, upper_idx: I) -> Vec<usize>
    where
        I: num::Integer + num::cast::ToPrimitive,
    {
        debug_assert!(upper_idx >= lower_idx);

        let lower_idx = I::to_isize(&lower_idx).expect("Conversion failed.");
        let upper_idx = I::to_isize(&upper_idx).expect("Conversion failed.");

        let wrapped_lower_idx = self.wrap_idx_to_closest(dim, lower_idx);
        let wrapped_upper_idx = self.wrap_idx_to_closest(dim, upper_idx - 1);

        self.create_idx_range_list_wrapped(dim, wrapped_lower_idx, wrapped_upper_idx)
    }

    /// Creates a vector containing 1D indices running from the given lower
    /// (inclusive) to upper (inclusive) index, wrapping around periodic
    /// boundaries and truncating the range if crossing non-periodic
    /// boundaries.
    ///
    /// The given indices must lie inside the grid, but the lower index is
    /// allowed to be larger than the upper index.
    fn create_idx_range_list_wrapped(
        &self,
        dim: Dim3,
        wrapped_lower_idx: usize,
        wrapped_upper_idx: usize,
    ) -> Vec<usize> {
        debug_assert!(
            wrapped_lower_idx < self.shape()[dim] && wrapped_upper_idx < self.shape()[dim]
        );
        if wrapped_upper_idx >= wrapped_lower_idx {
            (wrapped_lower_idx..=wrapped_upper_idx).collect()
        } else {
            let length = self.shape()[dim];
            let mut index_list: Vec<_> = (wrapped_lower_idx..length).collect();
            index_list.append(&mut (0..=wrapped_upper_idx).collect());
            index_list
        }
    }

    /// Slices the grid across the x-axis and returns the corresponding 2D grid.
    fn slice_across_x(&self) -> Self::XSliceGrid {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::XSliceGrid::from_coords(
            Coords2::new(centers[Y].clone(), centers[Z].clone()),
            Coords2::new(lower_edges[Y].clone(), lower_edges[Z].clone()),
            In2D::new(self.is_periodic(Y), self.is_periodic(Z)),
        )
    }

    /// Slices the grid across the y-axis and returns the corresponding 2D grid.
    fn slice_across_y(&self) -> Self::YSliceGrid {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::YSliceGrid::from_coords(
            Coords2::new(centers[X].clone(), centers[Z].clone()),
            Coords2::new(lower_edges[X].clone(), lower_edges[Z].clone()),
            In2D::new(self.is_periodic(X), self.is_periodic(Z)),
        )
    }

    /// Slices the grid across the z-axis and returns the corresponding 2D grid.
    fn slice_across_z(&self) -> Self::ZSliceGrid {
        let centers = self.centers();
        let lower_edges = self.lower_edges();
        Self::ZSliceGrid::from_coords(
            Coords2::new(centers[X].clone(), centers[Y].clone()),
            Coords2::new(lower_edges[X].clone(), lower_edges[Y].clone()),
            In2D::new(self.is_periodic(X), self.is_periodic(Y)),
        )
    }

    /// Slices the grid across the given axis and returns the corresponding regular 2D grid.
    fn regular_slice_across_axis(&self, axis: Dim3) -> RegularGrid2<F> {
        let regular_centers = self.regular_centers();
        let regular_lower_edges = self.regular_lower_edges();
        let [axis_0, axis_1] = Dim3::slice_except(axis);
        RegularGrid2::from_coords(
            Coords2::new(
                regular_centers[axis_0].to_vec(),
                regular_centers[axis_1].to_vec(),
            ),
            Coords2::new(
                regular_lower_edges[axis_0].to_vec(),
                regular_lower_edges[axis_1].to_vec(),
            ),
            In2D::new(self.is_periodic(axis_0), self.is_periodic(axis_1)),
        )
    }
}

/// Defines the properties of a 2D grid.
pub trait Grid2<F: BFloat>: Clone + Sync + Send {
    /// The specific type of the grid.
    const TYPE: GridType;

    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    fn from_coords(
        center_coords: Coords2<F>,
        lower_edge_coords: Coords2<F>,
        is_periodic: In2D<bool>,
    ) -> Self;

    /// Returns the 2D shape of the grid.
    fn shape(&self) -> &In2D<usize>;

    /// Whether the grid is periodic along the given dimension.
    fn is_periodic(&self, dim: Dim2) -> bool;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, location: CoordLocation) -> &Coords2<F>;

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &Coords2<F> {
        self.coords_by_type(CoordLocation::Center)
    }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &Coords2<F> {
        self.coords_by_type(CoordLocation::LowerEdge)
    }

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
            CoordLocation::LowerEdge => self.lower_edges(),
        };
        let shape = self.shape();
        let mut points = Vec::with_capacity(shape[Dim2::X] * shape[Dim2::Y]);
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
        Dim2::slice()
            .iter()
            .all(|&dim| point[dim] >= lower_bounds[dim] && point[dim] < upper_bounds[dim])
    }

    /// Whether the given 2D index is inside the bounds of the grid.
    fn indices_are_inside(&self, indices: &Idx2<usize>) -> bool {
        let shape = self.shape();
        Dim2::slice().iter().all(|&dim| indices[dim] < shape[dim])
    }

    /// Whether the given point is inside the bounds of the given grid cell.
    fn point_is_inside_cell(&self, point: &Point2<F>, cell_idx: &Idx2<usize>) -> bool {
        let lower_edges = self.lower_edges();
        Dim2::slice()
            .iter()
            .all(|&dim| coord_is_inside_grid_cell(&lower_edges[dim], point[dim], cell_idx[dim]))
    }

    /// Tries to find the 2D index of the grid cell containing the given coordinate,
    /// and returns the result as a `GridPointQuery2`.
    fn find_grid_cell(&self, point: &Point2<F>) -> GridPointQuery2<F, Idx2<usize>> {
        let lower_edges = self.lower_edges();
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();

        let mut point = point.clone();
        let mut indices = Idx2::origin();
        let mut wrapped = false;

        for &dim in Dim2::slice().iter() {
            if point[dim] < lower_bounds[dim] {
                if self.is_periodic(dim) {
                    point[dim] = wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                } else {
                    return GridPointQuery2::Outside;
                }
            } else if point[dim] >= upper_bounds[dim] {
                if self.is_periodic(dim) {
                    point[dim] = wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                } else {
                    return GridPointQuery2::Outside;
                }
            };
            indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                .expect("Coordinate index search failed.");
        }
        debug_assert!(
            self.point_is_inside_cell(&point, &indices),
            "Found wrong grid cell."
        );

        if wrapped {
            GridPointQuery2::MovedInside((indices, point))
        } else {
            GridPointQuery2::Inside(indices)
        }
    }

    /// Finds the 2D index of the grid cell containing the given coordinate,
    /// wrapping around any periodic boundaries,
    /// or the index of the closest grid cell if the coordinate is outside
    /// a non-periodic boundary.
    fn find_closest_grid_cell(&self, point: &Point2<F>) -> Idx2<usize> {
        let lower_edges = self.lower_edges();
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();
        let shape = self.shape();

        let mut point = point.clone();
        let mut indices = Idx2::origin();
        for &dim in Dim2::slice().iter() {
            if self.is_periodic(dim) {
                if point[dim] < lower_bounds[dim] {
                    point[dim] = wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                } else if point[dim] >= upper_bounds[dim] {
                    point[dim] = wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                };
                indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                    .expect("Coordinate index search failed.");
            } else {
                indices[dim] = if point[dim] < lower_bounds[dim] {
                    0
                } else if point[dim] >= upper_bounds[dim] {
                    shape[dim] - 1
                } else {
                    search_idx_of_coord(&lower_edges[dim], point[dim])
                        .expect("Coordinate index search failed.")
                };
            }
        }
        debug_assert!(
            self.indices_are_inside(&indices),
            "Found inside index is actually on the outside."
        );
        indices
    }

    /// Given a point that may be outside the grid boundaries, returns a new point
    /// wrapped around the boundaries to the inside of the grid, or `None` if the
    /// point is outside a non-periodic boundary.
    fn wrap_point(&self, point: &Point2<F>) -> Option<Point2<F>> {
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();
        let mut wrapped_point = point.clone();
        for &dim in Dim2::slice().iter() {
            if self.is_periodic(dim) {
                if wrapped_point[dim] < lower_bounds[dim] {
                    wrapped_point[dim] =
                        wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                } else if wrapped_point[dim] >= upper_bounds[dim] {
                    wrapped_point[dim] =
                        wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                }
            } else if wrapped_point[dim] < lower_bounds[dim]
                || wrapped_point[dim] >= upper_bounds[dim]
            {
                return None;
            }
        }
        debug_assert!(
            self.point_is_inside(&wrapped_point),
            "Wrapped to outside of bounds."
        );
        Some(wrapped_point)
    }
}

impl<F: BFloat, T> GridPointQuery3<F, T> {
    /// Returns the query result if the grid point was already inside the grid, otherwise panics.
    pub fn expect_inside(self) -> T {
        match self {
            GridPointQuery3::Inside(result) => result,
            GridPointQuery3::MovedInside(_) => {
                panic!("Grid point query was moved when expected to already be inside.")
            }
            GridPointQuery3::Outside => {
                panic!("Grid point query was outside when expected to be inside.")
            }
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise panics.
    pub fn expect_inside_or_moved(self) -> T {
        match self {
            GridPointQuery3::Inside(result) => result,
            GridPointQuery3::MovedInside((result, _)) => result,
            GridPointQuery3::Outside => {
                panic!("Grid point query was outside when expected to be inside.")
            }
        }
    }

    /// Returns the query result, and updates the given position with the possibly moved position.
    /// Panics if the grid point was outside a non-periodic boundary and not moved inside.
    pub fn unwrap_and_update_position<P: BFloat>(self, position: &mut Point3<P>) -> T {
        match self {
            GridPointQuery3::Inside(result) => result,
            GridPointQuery3::MovedInside((result, moved_position)) => {
                *position = Point3::from(&moved_position);
                result
            }
            GridPointQuery3::Outside => {
                panic!("Grid point query was outside when expected to be inside.")
            }
        }
    }
}

impl<F: BFloat, T> GridPointQuery2<F, T> {
    /// Returns the query result if the grid point was already inside the grid, otherwise panics.
    pub fn expect_inside(self) -> T {
        match self {
            GridPointQuery2::Inside(result) => result,
            GridPointQuery2::MovedInside(_) => {
                panic!("Grid point query was moved when expected to already be inside.")
            }
            GridPointQuery2::Outside => {
                panic!("Grid point query was outside when expected to be inside.")
            }
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise panics.
    pub fn expect_inside_or_moved(self) -> T {
        match self {
            GridPointQuery2::Inside(result) => result,
            GridPointQuery2::MovedInside((result, _)) => result,
            GridPointQuery2::Outside => {
                panic!("Grid point query was outside when expected to be inside.")
            }
        }
    }

    /// Returns the query result, and updates the given position with the possibly moved position.
    /// Panics if the grid point was outside a non-periodic boundary and not moved inside.
    pub fn unwrap_and_update_position<P: BFloat>(self, position: &mut Point2<P>) -> T {
        match self {
            GridPointQuery2::Inside(result) => result,
            GridPointQuery2::MovedInside((result, moved_position)) => {
                *position = Point2::from(&moved_position);
                result
            }
            GridPointQuery2::Outside => {
                panic!("Grid point query was outside when expected to be inside.")
            }
        }
    }
}

fn extent_from_bounds<F: BFloat>(lower_bound: F, upper_bound: F) -> F {
    upper_bound - lower_bound
}

fn cell_extent_from_bounds<F: BFloat>(size: usize, lower_bound: F, upper_bound: F) -> F {
    let extent = extent_from_bounds(lower_bound, upper_bound);
    extent / F::from_usize(size).unwrap()
}

fn regular_coords_from_bounds<F: BFloat>(
    size: usize,
    lower_bound: F,
    upper_bound: F,
) -> (Vec<F>, Vec<F>) {
    let cell_extent = cell_extent_from_bounds(size, lower_bound, upper_bound);
    let half_cell_extent = F::from_f32(0.5).unwrap() * cell_extent;
    let centers = Array::linspace(
        lower_bound + half_cell_extent,
        upper_bound - half_cell_extent,
        size,
    )
    .to_vec();
    let lower_edges = Array::linspace(lower_bound, upper_bound - cell_extent, size).to_vec();
    (centers, lower_edges)
}

fn bounds_from_coords<F: BFloat>(size: usize, centers: &[F], lower_edges: &[F]) -> (F, F) {
    (
        lower_edges[0],
        F::from_f32(2.0).unwrap() * centers[size - 1] - lower_edges[size - 1],
    )
}

fn search_idx_of_coord<F: BFloat>(lower_edges: &[F], coord: F) -> Option<usize> {
    let mut low = 0;
    let mut high = lower_edges.len() - 1;
    let mut mid;

    if coord >= lower_edges[high] {
        return Some(high);
    }

    while coord >= lower_edges[low] && coord < lower_edges[high] {
        let low_float = F::from_usize(low).unwrap();
        let high_float = F::from_usize(high).unwrap();
        let mid_float = (low_float
            + (coord - lower_edges[low]) * (high_float - low_float)
                / (lower_edges[high] - lower_edges[low]))
            .floor();

        mid = F::to_usize(&mid_float).expect("Conversion failed.");

        if mid >= high {
            // Due to roundoff error, we might get `mid == high` even though `coord < lower_edges[high]`.
            // If this happens, `coord` will be very close to `lower_edges[high]`, so we return `high - 1`.
            return Some(high - 1);
        }

        if lower_edges[mid + 1] <= coord {
            low = mid + 1
        } else if lower_edges[mid] > coord {
            high = mid
        } else {
            return Some(mid);
        }
    }
    None
}

fn coord_is_inside_grid_cell<F: BFloat>(lower_edges: &[F], coord: F, cell_idx: usize) -> bool {
    (cell_idx == (lower_edges.len() - 1) || coord < lower_edges[cell_idx + 1])
        && coord >= lower_edges[cell_idx]
}

fn wrap_coordinate_lower<F: BFloat>(upper_bound: F, extent: F, coord: F) -> F {
    F::min(
        upper_bound - ((upper_bound - coord) % extent),
        upper_bound.prev(),
    )
}

fn wrap_coordinate_upper<F: BFloat>(lower_bound: F, extent: F, coord: F) -> F {
    F::max(lower_bound + ((coord - lower_bound) % extent), lower_bound)
}

fn wrap_idx_lower(length: usize, idx: isize) -> usize {
    debug_assert!(idx < 0);
    length - (((length as isize - idx) as usize) % length)
}

fn wrap_idx_upper(length: usize, idx: isize) -> usize {
    debug_assert!(idx >= (length as isize));
    (idx as usize) % length
}
