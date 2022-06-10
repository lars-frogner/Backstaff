//! Structured grids.

pub mod hor_regular;
pub mod regular;

use self::{hor_regular::NonUniformGrid1, regular::RegularGrid2};
use crate::{
    field::ScalarField1,
    geometry::{
        CoordRefs2, CoordRefs3, Coords2, Coords3, Dim2,
        Dim3::{self, X, Y, Z},
        Idx2, Idx3, In2D, In3D, Point2, Point3, PointTransformation2, SimplePolygon2, Vec2, Vec3,
    },
    interpolation::Interpolator1,
    num::BFloat,
};
use ndarray::prelude::*;
use std::{borrow::Cow, io, iter, sync::Arc};

/// Default floating-point precision to use for grids.
#[allow(non_camel_case_types)]
pub type fgr = f64;

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

/// A query for a result at a 2D grid point.
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

/// A query for a result at a 1D grid point.
///
/// * If the point is inside the grid bounds, the query contains a result of type `T`.
/// * If the point is outside a periodic boundary, in contains the result as well as the wrapped position.
/// * If the point is outside a non-periodic boundary, it contains no result.
#[derive(Clone, Debug, PartialEq)]
pub enum GridPointQuery1<F: BFloat, T> {
    Inside(T),
    MovedInside((T, F)),
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

    /// Returns the type of the grid.
    fn grid_type(&self) -> GridType {
        Self::TYPE
    }

    /// Returns the 3D shape of the grid.
    fn shape(&self) -> &In3D<usize>;

    /// Returns a reference to the grid periodicity object.
    fn periodicity(&self) -> &In3D<bool>;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, location: CoordLocation) -> &Coords3<F>;

    /// Whether the grid is periodic along the given dimension.
    fn is_periodic(&self, dim: Dim3) -> bool {
        self.periodicity()[dim]
    }

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

    /// Sets the periodicity of the grid along each dimension.
    fn set_periodicity(&mut self, is_periodic: In3D<bool>);

    /// Sets the upward derivatives of the grid coordinates.
    fn set_up_derivatives(&mut self, up_derivatives: Option<Coords3<F>>);

    /// Sets the downward derivatives of the grid coordinates.
    fn set_down_derivatives(&mut self, down_derivatives: Option<Coords3<F>>);

    /// Creates a new 3D grid restricted to slices of the coordinate arrays of
    /// the original grid.
    fn subgrid(&self, start_indices: &Idx3<usize>, end_indices: &Idx3<usize>) -> Self {
        let shape = self.shape();
        let mut is_periodic = self.periodicity().clone();

        for &dim in &Dim3::slice() {
            assert!(
                start_indices[dim] <= end_indices[dim],
                "Start index is not smaller than or equal to end index"
            );
            assert!(
                end_indices[dim] < shape[dim],
                "End index is outside of grid bound"
            );
            is_periodic[dim] =
                is_periodic[dim] && start_indices[dim] == 0 && end_indices[dim] == shape[dim] - 1;
        }

        Self::from_coords(
            self.centers().subcoords(start_indices, end_indices),
            self.lower_edges().subcoords(start_indices, end_indices),
            is_periodic,
            self.up_derivatives()
                .map(|coords| coords.subcoords(start_indices, end_indices)),
            self.down_derivatives()
                .map(|coords| coords.subcoords(start_indices, end_indices)),
        )
    }

    /// Returns the lower bounds of the grid cell of the given 3D index.
    fn grid_cell_lower_bounds(&self, indices: &Idx3<usize>) -> Vec3<F> {
        self.lower_edges().vector(indices)
    }

    /// Returns the upper bounds of the grid cell of the given 3D index.
    fn grid_cell_upper_bounds(&self, indices: &Idx3<usize>) -> Vec3<F> {
        let shape = self.shape();
        let lower_edges = self.lower_edges();
        let upper_bounds = self.upper_bounds();
        Vec3::with_each_component(|dim| {
            let idx = indices[dim];
            if idx == shape[dim] - 1 {
                upper_bounds[dim]
            } else {
                lower_edges[dim][idx + 1]
            }
        })
    }

    /// Returns the lower corner of the grid cell of the given 3D index.
    fn grid_cell_lower_corner(&self, indices: &Idx3<usize>) -> Point3<F> {
        self.lower_edges().point(indices)
    }

    /// Returns the upper corner of the grid cell of the given 3D index.
    fn grid_cell_upper_corner(&self, indices: &Idx3<usize>) -> Point3<F> {
        let shape = self.shape();
        let lower_edges = self.lower_edges();
        let upper_bounds = self.upper_bounds();
        Point3::with_each_component(|dim| {
            let idx = indices[dim];
            if idx == shape[dim] - 1 {
                upper_bounds[dim]
            } else {
                lower_edges[dim][idx + 1]
            }
            .prev()
        })
    }

    /// Returns the lower and upper corner of the grid cell of the given 3D index.
    fn grid_cell_extremal_corners(&self, indices: &Idx3<usize>) -> (Point3<F>, Point3<F>) {
        (
            self.grid_cell_lower_corner(indices),
            self.grid_cell_upper_corner(indices),
        )
    }

    /// Returns the coordinate extents of the grid cell at the given 3D index.
    fn grid_cell_extents(&self, indices: &Idx3<usize>) -> Vec3<F>;

    /// Returns the average coordinate extents of the grid cells.
    fn average_grid_cell_extents(&self) -> Vec3<F> {
        let shape = self.shape();
        let extents = self.extents();
        Vec3::with_each_component(|dim| extents[dim] / F::from_usize(shape[dim]).unwrap())
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

    /// Whether the given grid bounds are fully inside the non-periodic boundaries of the grid.
    fn contains_bounds(&self, other_lower_bounds: &Vec3<F>, other_upper_bounds: &Vec3<F>) -> bool {
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let is_periodic = self.periodicity();
        Dim3::slice().iter().all(|&dim| {
            is_periodic[dim]
                || (other_lower_bounds[dim] >= lower_bounds[dim]
                    && other_upper_bounds[dim] <= upper_bounds[dim])
        })
    }

    /// Whether the given grid is fully inside the non-periodic boundaries of the grid.
    fn contains_grid<H: Grid3<F>>(&self, other: &H) -> bool {
        self.contains_bounds(other.lower_bounds(), other.upper_bounds())
    }

    /// Whether the given grid after applying the given transformation is fully inside
    /// the non-periodic boundaries of the grid
    fn contains_transformed_grid<H, T>(&self, other: &H, transformation: &T) -> bool
    where
        H: Grid3<F>,
        T: PointTransformation2<F>,
    {
        let other_lower_bounds = other.lower_bounds();
        let other_upper_bounds = other.upper_bounds();

        let other_hor_bound_polygon = SimplePolygon2::rectangle_from_horizontal_bounds(
            other_lower_bounds,
            other_upper_bounds,
        )
        .transformed(transformation);

        let (other_hor_bounding_box_lower_bounds, other_hor_bounding_box_upper_bounds) =
            other_hor_bound_polygon.bounds().unwrap();

        let other_bounding_box_lower_bounds = Vec3::new(
            other_hor_bounding_box_lower_bounds[Dim2::X],
            other_hor_bounding_box_lower_bounds[Dim2::Y],
            other_lower_bounds[Z],
        );
        let other_bounding_box_upper_bounds = Vec3::new(
            other_hor_bounding_box_upper_bounds[Dim2::X],
            other_hor_bounding_box_upper_bounds[Dim2::Y],
            other_upper_bounds[Z],
        );

        self.contains_bounds(
            &other_bounding_box_lower_bounds,
            &other_bounding_box_upper_bounds,
        )
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
                .expect("Coordinate index search failed");
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
                    .expect("Coordinate index search failed");
            } else if point[dim] < lower_bounds[dim] {
                indices[dim] = 0;
                point[dim] = lower_bounds[dim];
            } else if point[dim] >= upper_bounds[dim] {
                indices[dim] = shape[dim] - 1;
                point[dim] = upper_bounds[dim].prev();
            } else {
                indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                    .expect("Coordinate index search failed");
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

    /// Given a point representing a lower boundary in each dimension, returns
    /// the index of the lowermost grid cell that lies completely within the
    /// boundaries (inclusive), or `None` if no such grid cell exists.
    fn find_fist_grid_cell_inside_lower_bounds(
        &self,
        lower_bounds: &Point3<F>,
    ) -> Option<Idx3<usize>> {
        let grid_lower_edges = self.lower_edges();
        let shape = self.shape();

        let mut indices = Idx3::origin();

        for &dim in &Dim3::slice() {
            let lower_bound = lower_bounds[dim];
            let lower_edges = &grid_lower_edges[dim];
            let size = shape[dim];
            if lower_bound > lower_edges[size - 1] {
                return None;
            } else if lower_bound <= lower_edges[0] {
                indices[dim] = 0;
            } else {
                indices[dim] = search_idx_of_coord(lower_edges, lower_bound.prev())
                    .expect("Coordinate index search failed")
                    + 1;
            }
        }
        debug_assert!(
            self.indices_are_inside(&indices),
            "Found inside index is actually on the outside."
        );

        Some(indices)
    }

    /// Given a point representing an upper boundary in each dimension, returns
    /// the index of the uppermost grid cell that lies completely within the
    /// boundaries (inclusive), or `None` if no such grid cell exists.
    fn find_last_grid_cell_inside_upper_bounds(
        &self,
        upper_bounds: &Point3<F>,
    ) -> Option<Idx3<usize>> {
        let grid_centers = self.centers();
        let grid_lower_edges = self.lower_edges();
        let grid_upper_bounds = self.upper_bounds();
        let shape = self.shape();

        let mut indices = Idx3::origin();

        for &dim in &Dim3::slice() {
            let upper_bound = upper_bounds[dim];
            let centers = &grid_centers[dim];
            let lower_edges = &grid_lower_edges[dim];
            let size = shape[dim];
            if upper_bound < upper_edge_from_center_and_lower_edge(centers[0], lower_edges[0]) {
                return None;
            } else if upper_bound >= grid_upper_bounds[dim] {
                indices[dim] = size - 1;
            } else {
                indices[dim] = search_idx_of_coord(lower_edges, upper_bound)
                    .expect("Coordinate index search failed")
                    - 1;
            }
        }
        debug_assert!(
            self.indices_are_inside(&indices),
            "Found inside index is actually on the outside."
        );

        Some(indices)
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

        let lower_idx = I::to_isize(&lower_idx).expect("Conversion failed");
        let upper_idx = I::to_isize(&upper_idx).expect("Conversion failed");

        let wrapped_lower_idx = self.wrap_idx_to_closest(dim, lower_idx);
        let wrapped_upper_idx = self.wrap_idx_to_closest(dim, upper_idx - 1);

        self.create_idx_range_list_wrapped(dim, wrapped_lower_idx, wrapped_upper_idx)
    }

    /// Creates a vector containing 1D indices running from the given lower
    /// (inclusive) to upper (inclusive) index, wrapping around boundaries.
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

    /// Returns the number of grid cells from the given lower index to the given
    /// upper index (inclusive).
    ///
    /// The given indices must lie inside the grid, but the lower index is
    /// allowed to be larger than the upper index (happens when wrapping).
    fn count_grid_cells_between(&self, dim: Dim3, lower_idx: usize, upper_idx: usize) -> usize {
        debug_assert!(lower_idx < self.shape()[dim] && upper_idx < self.shape()[dim]);
        if upper_idx >= lower_idx {
            upper_idx + 1 - lower_idx
        } else {
            // Wrap around boundary
            let length = self.shape()[dim];
            (length - lower_idx) + (upper_idx + 1)
        }
    }

    /// Creates a list of edges for n grid cells starting with the lower edge of the
    /// cell at the given index and ending with the upper edge of the final grid cell,
    /// without wrapping when reaching a boundary.
    ///
    /// The given index must lie inside the grid.
    fn determine_n_monotonic_grid_cell_edges(
        &self,
        dim: Dim3,
        start_idx: usize,
        n_grid_cells: usize,
        offset: F,
    ) -> Vec<F>;

    /// Creates a list of edges starting with the lower edge of the
    /// cell at the given lower index and ending with the upper edge of
    /// the cell a the given upper index, without wrapping when reaching
    /// a boundary. The provided offset will be added to the coordinates.
    ///
    /// The given indices must lie inside the grid, but the lower index is
    /// allowed to be larger than the upper index (happens when wrapping).
    fn determine_monotonic_grid_cell_edges_between(
        &self,
        dim: Dim3,
        lower_idx: usize,
        upper_idx: usize,
        offset: F,
    ) -> Vec<F> {
        self.determine_n_monotonic_grid_cell_edges(
            dim,
            lower_idx,
            self.count_grid_cells_between(dim, lower_idx, upper_idx),
            offset,
        )
    }

    /// Like `determine_monotonic_grid_cell_edges_between`, but also returns the
    /// indices of the grid cells of the lower edges.
    fn determine_indexed_monotonic_grid_cell_edges_between(
        &self,
        dim: Dim3,
        lower_idx: usize,
        upper_idx: usize,
        offset: F,
    ) -> (Vec<usize>, Vec<F>) {
        let idx_range_list = self.create_idx_range_list_wrapped(dim, lower_idx, upper_idx);
        let edges = self.determine_n_monotonic_grid_cell_edges(
            dim,
            lower_idx,
            idx_range_list.len(),
            offset,
        );
        (idx_range_list, edges)
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

    #[cfg(feature = "comparison")]
    fn derivatives_equal<H, C>(&self, other: &H, coords_equal: C) -> bool
    where
        H: Grid3<F>,
        C: Fn(&Coords3<F>, &Coords3<F>) -> bool,
    {
        (match (self.up_derivatives(), other.up_derivatives()) {
            (None, None) => true,
            (None, _) => false,
            (_, None) => false,
            (Some(a), Some(b)) => coords_equal(a, b),
        } && match (self.down_derivatives(), other.down_derivatives()) {
            (None, None) => true,
            (None, _) => false,
            (_, None) => false,
            (Some(a), Some(b)) => coords_equal(a, b),
        })
    }
}

#[cfg(feature = "comparison")]
#[macro_export]
macro_rules! impl_partial_eq_for_grid {
    ($T:ident <$F:ident>, $GT:ident) => {
        impl<$F, G> ::std::cmp::PartialEq<G> for $T<$F>
        where
            $F: crate::num::BFloat,
            G: $GT<$F>,
        {
            fn eq(&self, other: &G) -> bool {
                self.shape() == other.shape()
                    && self.periodicity() == other.periodicity()
                    && self.lower_edges() == other.lower_edges()
                    && self.centers() == other.centers()
                    && self.up_derivatives() == other.up_derivatives()
                    && self.down_derivatives() == other.down_derivatives()
            }
        }
    };
}

#[cfg(feature = "comparison")]
#[macro_export]
macro_rules! impl_abs_diff_eq_for_grid {
    ($T:ident <$F:ident>, $GT:ident) => {
        impl<$F, G> approx::AbsDiffEq<G> for $T<$F>
        where
            $F: crate::num::BFloat + approx::AbsDiffEq,
            $F::Epsilon: Copy,
            G: $GT<$F>,
        {
            type Epsilon = <$F as approx::AbsDiffEq>::Epsilon;

            fn default_epsilon() -> Self::Epsilon {
                $F::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &G, epsilon: Self::Epsilon) -> bool {
                self.shape() == other.shape()
                    && self.periodicity() == other.periodicity()
                    && self.lower_edges().abs_diff_eq(other.lower_edges(), epsilon)
                    && self.centers().abs_diff_eq(other.centers(), epsilon)
                    && self.derivatives_equal(other, |a, b| a.abs_diff_eq(b, epsilon))
            }
        }
    };
}

#[cfg(feature = "comparison")]
#[macro_export]
macro_rules! impl_relative_eq_for_grid {
    ($T:ident <$F:ident>, $GT:ident) => {
        impl<$F, G> approx::RelativeEq<G> for $T<$F>
        where
            $F: crate::num::BFloat + approx::RelativeEq,
            $F::Epsilon: Copy,
            G: $GT<$F>,
        {
            fn default_max_relative() -> Self::Epsilon {
                $F::default_max_relative()
            }

            fn relative_eq(
                &self,
                other: &G,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                if self.shape() != other.shape() {
                    #[cfg(debug_assertions)]
                    {
                        println!("Shapes not equal");
                        dbg!(self.shape(), other.shape());
                    }
                    return false;
                }
                if self.periodicity() != other.periodicity() {
                    #[cfg(debug_assertions)]
                    {
                        println!("Periodicities not equal");
                        dbg!(self.periodicity(), other.periodicity());
                    }
                    return false;
                }
                if self
                    .lower_edges()
                    .relative_ne(other.lower_edges(), epsilon, max_relative)
                {
                    #[cfg(debug_assertions)]
                    {
                        println!("Lower edges not equal");
                        dbg!(self.lower_edges(), other.lower_edges());
                    }
                    return false;
                }
                if self
                    .centers()
                    .relative_ne(other.centers(), epsilon, max_relative)
                {
                    #[cfg(debug_assertions)]
                    {
                        println!("Centers not equal");
                        dbg!(self.centers(), other.centers());
                    }
                    return false;
                }
                if !self.derivatives_equal(other, |a, b| a.relative_eq(b, epsilon, max_relative)) {
                    #[cfg(debug_assertions)]
                    {
                        println!("Derivatives not equal");
                        dbg!(self.up_derivatives(), other.up_derivatives());
                        dbg!(self.down_derivatives(), other.down_derivatives());
                    }
                    return false;
                }
                true
            }
        }
    };
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

    /// Returns a reference to the grid periodicity object.
    fn periodicity(&self) -> &In2D<bool>;

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

    /// Returns a reference to the upward derivatives of the grid coordinates,
    /// if they are available.
    fn up_derivatives(&self) -> Option<&Coords2<F>> {
        None
    }

    /// Returns a reference to the downward derivatives of the grid coordinates,
    /// if they are available.
    fn down_derivatives(&self) -> Option<&Coords2<F>> {
        None
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
                .expect("Coordinate index search failed");
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
    ///
    /// The result is returned as a `GridPointQuery2` which is guaranteed not
    /// to be `Outside`.
    fn find_closest_grid_cell(&self, point: &Point2<F>) -> GridPointQuery2<F, Idx2<usize>> {
        let lower_edges = self.lower_edges();
        let lower_bounds = self.lower_bounds();
        let upper_bounds = self.upper_bounds();
        let extents = self.extents();
        let shape = self.shape();

        let mut point = point.clone();
        let mut indices = Idx2::origin();
        let mut wrapped = false;

        for &dim in &Dim2::slice() {
            if self.is_periodic(dim) {
                if point[dim] < lower_bounds[dim] {
                    point[dim] = wrap_coordinate_lower(upper_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                } else if point[dim] >= upper_bounds[dim] {
                    point[dim] = wrap_coordinate_upper(lower_bounds[dim], extents[dim], point[dim]);
                    wrapped = true;
                };
                indices[dim] = search_idx_of_coord(&lower_edges[dim], point[dim])
                    .expect("Coordinate index search failed");
            } else {
                indices[dim] = if point[dim] < lower_bounds[dim] {
                    0
                } else if point[dim] >= upper_bounds[dim] {
                    shape[dim] - 1
                } else {
                    search_idx_of_coord(&lower_edges[dim], point[dim])
                        .expect("Coordinate index search failed")
                };
            }
        }
        debug_assert!(
            self.indices_are_inside(&indices),
            "Found inside index is actually on the outside."
        );

        if wrapped {
            GridPointQuery2::MovedInside((indices, point))
        } else {
            GridPointQuery2::Inside(indices)
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
        for &dim in &Dim2::slice() {
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

    #[cfg(feature = "comparison")]
    fn derivatives_equal<H, C>(&self, other: &H, coords_equal: C) -> bool
    where
        H: Grid2<F>,
        C: Fn(&Coords2<F>, &Coords2<F>) -> bool,
    {
        (match (self.up_derivatives(), other.up_derivatives()) {
            (None, None) => true,
            (None, _) => false,
            (_, None) => false,
            (Some(a), Some(b)) => coords_equal(a, b),
        } && match (self.down_derivatives(), other.down_derivatives()) {
            (None, None) => true,
            (None, _) => false,
            (_, None) => false,
            (Some(a), Some(b)) => coords_equal(a, b),
        })
    }
}

/// Defines the properties of a 1D grid.
pub trait Grid1<F: BFloat>: Clone + Sync + Send {
    /// Creates a new grid given the coordinates of the cell centers and lower edges,
    /// as well as which dimensions are periodic.
    fn from_coords(center_coords: Vec<F>, lower_edge_coords: Vec<F>, is_periodic: bool) -> Self;

    /// Returns the size of the grid.
    fn size(&self) -> usize;

    /// Whether the grid is periodic.
    fn is_periodic(&self) -> bool;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, location: CoordLocation) -> &[F];

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &[F] {
        self.coords_by_type(CoordLocation::Center)
    }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &[F] {
        self.coords_by_type(CoordLocation::LowerEdge)
    }

    /// Returns a reference to the central coordinates in a regular version of the grid.
    fn regular_centers(&self) -> &[F];

    /// Returns a reference to the lower coordinates in a regular version of the grid.
    fn regular_lower_edges(&self) -> &[F];

    /// Returns the lower coordinate bound.
    fn lower_bound(&self) -> F;

    /// Returns the upper coordinate bound.
    fn upper_bound(&self) -> F;

    /// Returns the full coordinate extent.
    fn extent(&self) -> F;

    /// Whether the given coordinate is inside the bounds of the grid.
    fn coord_is_inside(&self, coord: F) -> bool {
        coord >= self.lower_bound() && coord < self.upper_bound()
    }

    /// Whether the given index is inside the bounds of the grid.
    fn index_is_inside(&self, idx: usize) -> bool {
        idx < self.size()
    }

    /// Whether the given coordinate is inside the bounds of the given grid cell.
    fn coord_is_inside_cell(&self, coord: F, cell_idx: usize) -> bool {
        coord_is_inside_grid_cell(self.lower_edges(), coord, cell_idx)
    }

    /// Tries to find the index of the grid cell containing the given coordinate,
    /// and returns the result as a `GridPointQuery1`.
    fn find_grid_cell(&self, mut coord: F) -> GridPointQuery1<F, usize> {
        let lower_edges = self.lower_edges();
        let lower_bound = self.lower_bound();
        let upper_bound = self.upper_bound();
        let extent = self.extent();

        let mut wrapped = false;

        if coord < lower_bound {
            if self.is_periodic() {
                coord = wrap_coordinate_lower(upper_bound, extent, coord);
                wrapped = true;
            } else {
                return GridPointQuery1::Outside;
            }
        } else if coord >= upper_bound {
            if self.is_periodic() {
                coord = wrap_coordinate_upper(lower_bound, extent, coord);
                wrapped = true;
            } else {
                return GridPointQuery1::Outside;
            }
        };
        let index =
            search_idx_of_coord(lower_edges, coord).expect("Coordinate index search failed");

        debug_assert!(
            self.coord_is_inside_cell(coord, index),
            "Found wrong grid cell."
        );

        if wrapped {
            GridPointQuery1::MovedInside((index, coord))
        } else {
            GridPointQuery1::Inside(index)
        }
    }

    /// Finds the index of the grid cell containing the given coordinate,
    /// wrapping around any periodic boundaries,
    /// or the index of the closest grid cell if the coordinate is outside
    /// a non-periodic boundary.
    ///
    /// The result is returned as a `GridPointQuery1` which is guaranteed not
    /// to be `Outside`.
    fn find_closest_grid_cell(&self, mut coord: F) -> GridPointQuery1<F, usize> {
        let lower_edges = self.lower_edges();
        let lower_bound = self.lower_bound();
        let upper_bound = self.upper_bound();
        let extent = self.extent();
        let size = self.size();

        let index;
        let mut wrapped = false;

        if self.is_periodic() {
            if coord < lower_bound {
                coord = wrap_coordinate_lower(upper_bound, extent, coord);
                wrapped = true;
            } else if coord >= upper_bound {
                coord = wrap_coordinate_upper(lower_bound, extent, coord);
                wrapped = true;
            };
            index =
                search_idx_of_coord(lower_edges, coord).expect("Coordinate index search failed");
        } else {
            index = if coord < lower_bound {
                0
            } else if coord >= upper_bound {
                size - 1
            } else {
                search_idx_of_coord(lower_edges, coord).expect("Coordinate index search failed")
            };
        }
        debug_assert!(
            self.index_is_inside(index),
            "Found inside index is actually on the outside."
        );

        if wrapped {
            GridPointQuery1::MovedInside((index, coord))
        } else {
            GridPointQuery1::Inside(index)
        }
    }

    /// Given a coordinate that may be outside the grid boundaries, returns a new coordinate
    /// wrapped around the boundaries to the inside of the grid, or `None` if the
    /// coordinate is outside a non-periodic boundary.
    fn wrap_coord(&self, coord: F) -> Option<F> {
        let lower_bound = self.lower_bound();
        let upper_bound = self.upper_bound();
        let extent = self.extent();
        let mut wrapped_coord = coord;

        if self.is_periodic() {
            if wrapped_coord < lower_bound {
                wrapped_coord = wrap_coordinate_lower(upper_bound, extent, coord);
            } else if wrapped_coord >= upper_bound {
                wrapped_coord = wrap_coordinate_upper(lower_bound, extent, coord);
            }
        } else if wrapped_coord < lower_bound || wrapped_coord >= upper_bound {
            return None;
        }
        debug_assert!(
            self.coord_is_inside(wrapped_coord),
            "Wrapped to outside of bounds."
        );
        Some(wrapped_coord)
    }
}

impl<F: BFloat, T> GridPointQuery3<F, T> {
    /// Returns the query result if the grid point was already inside the grid, otherwise panics.
    pub fn expect_inside(self) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside(_) => {
                panic!("Grid point query was moved when expected to already be inside.")
            }
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise panics.
    pub fn expect_inside_or_moved(self) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, _)) => result,
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise returns the provided default value.
    pub fn inside_or_moved_or_default(self, default: T) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, _)) => result,
            Self::Outside => default,
        }
    }

    /// Returns the query result, and updates the given position with the possibly moved position.
    /// Panics if the grid point was outside a non-periodic boundary and not moved inside.
    pub fn unwrap_and_update_position<P: BFloat>(self, position: &mut Point3<P>) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, moved_position)) => {
                *position = Point3::from(&moved_position);
                result
            }
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }
}

impl<F: BFloat, T> GridPointQuery2<F, T> {
    /// Returns the query result if the grid point was already inside the grid, otherwise panics.
    pub fn expect_inside(self) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside(_) => {
                panic!("Grid point query was moved when expected to already be inside.")
            }
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise panics.
    pub fn expect_inside_or_moved(self) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, _)) => result,
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise returns the provided default value.
    pub fn inside_or_moved_or_default(self, default: T) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, _)) => result,
            Self::Outside => default,
        }
    }

    /// Returns the query result, and updates the given position with the possibly moved position.
    /// Panics if the grid point was outside a non-periodic boundary and not moved inside.
    pub fn unwrap_and_update_position<P: BFloat>(self, position: &mut Point2<P>) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, moved_position)) => {
                *position = Point2::from(&moved_position);
                result
            }
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }
}

impl<F: BFloat, T> GridPointQuery1<F, T> {
    /// Returns the query result if the grid point was already inside the grid, otherwise panics.
    pub fn expect_inside(self) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside(_) => {
                panic!("Grid point query was moved when expected to already be inside.")
            }
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise panics.
    pub fn expect_inside_or_moved(self) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, _)) => result,
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }

    /// Returns the query result if the grid point was already inside the grid or was moved inside,
    /// otherwise returns the provided default value.
    pub fn inside_or_moved_or_default(self, default: T) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, _)) => result,
            Self::Outside => default,
        }
    }

    /// Returns the query result, and updates the given position with the possibly moved position.
    /// Panics if the grid point was outside a non-periodic boundary and not moved inside.
    pub fn unwrap_and_update_position<P: BFloat>(self, position: &mut P) -> T {
        match self {
            Self::Inside(result) => result,
            Self::MovedInside((result, moved_position)) => {
                *position = P::from(moved_position).unwrap();
                result
            }
            Self::Outside => panic!("Grid point query was outside when expected to be inside."),
        }
    }
}

/// Verifies that the given coordinate arrays conform to an available grid type, and returns this type.
pub fn verify_coordinate_arrays<F: BFloat>(
    center_coords: &Coords3<F>,
    lower_coords: &Coords3<F>,
    print_grid_type: bool,
) -> io::Result<GridType> {
    let nonuniformity_threshold_factor = F::from_f32(1e-3).unwrap();

    let mut is_uniform = In3D::same(true);

    for &dim in &Dim3::slice() {
        let center_vec = &center_coords[dim];
        let lower_vec = &lower_coords[dim];

        let length = center_vec.len();

        if lower_vec.len() != length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Inconsistent number of {}-coordinates", dim),
            ));
        }

        if length < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Insufficient number of {}-coordinates (must be at least 2)",
                    dim
                ),
            ));
        }

        if !center_vec
            .iter()
            .zip(center_vec.iter().skip(1))
            .all(|(&lower, &upper)| upper >= lower)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Center {}-coordinates do not increase monotonically", dim),
            ));
        }

        if !lower_vec
            .iter()
            .zip(lower_vec.iter().skip(1))
            .all(|(&lower, &upper)| upper >= lower)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Lower edge {}-coordinates do not increase monotonically",
                    dim
                ),
            ));
        }

        let nonuniformity_threshold = nonuniformity_threshold_factor
            * (F::from_f32(2.0).unwrap() * *center_vec.last().unwrap()
                - *lower_vec.last().unwrap()
                - *lower_vec.first().unwrap())
            / F::from_usize(length).unwrap();

        let center_differences: Vec<_> = center_vec
            .iter()
            .zip(center_vec.iter().skip(1))
            .map(|(&lower, &upper)| <F as num::Float>::abs(upper - lower))
            .collect();

        let uniform_centers = center_differences
            .iter()
            .zip(center_differences.iter().skip(1))
            .all(|(&first, &second)| {
                <F as num::Float>::abs(second - first) < nonuniformity_threshold
            });

        let lower_edge_differences: Vec<_> = lower_vec
            .iter()
            .zip(lower_vec.iter().skip(1))
            .map(|(&lower, &upper)| <F as num::Float>::abs(upper - lower))
            .collect();

        let uniform_lower_edges = lower_edge_differences
            .iter()
            .zip(lower_edge_differences.iter().skip(1))
            .all(|(&first, &second)| {
                <F as num::Float>::abs(second - first) < nonuniformity_threshold
            });

        if uniform_centers != uniform_lower_edges {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Inconsistent uniformity of {}-coordinates", dim),
            ));
        }

        is_uniform[dim] = uniform_centers;

        if !lower_vec
            .iter()
            .zip(center_vec.iter())
            .all(|(&lower, &upper)| upper > lower)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Found grid cell where center {}-coordinate is not larger than lower edge coordinate",
                    dim
                ),
            ));
        }
    }

    let detected_grid_type = match is_uniform.to_tuple() {
        (true, true, true) => {
            if print_grid_type {
                println!("Detected regular grid");
            }
            GridType::Regular
        }
        (true, true, false) => {
            if print_grid_type {
                println!("Detected horizontally regular grid");
            }
            GridType::HorRegular
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Non-uniform x- or y-coordinates not supported",
            ))
        }
    };

    Ok(detected_grid_type)
}

/// Compute lower edges and centers of grid cells in a regular 1D grid.
pub fn regular_coords_from_bounds<F: BFloat>(
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

/// Determine center and lower edge coordinates for a grid with the given
/// size and bounds using the given control values for grid cell extents.
pub fn create_new_grid_coords_from_control_extents<F: BFloat, I: Interpolator1>(
    target_number_of_grid_cells: usize,
    lower_bound: F,
    upper_bound: F,
    control_coords: &[F],
    control_grid_cell_extents: &[F],
    interpolator: &I,
) -> Result<(Vec<F>, Vec<F>), Cow<'static, str>> {
    const UNIFORM_CELLS: usize = 4;
    const MAX_ITERATIONS: usize = 100000;
    const TOLERANCE: f64 = 1e-9;

    assert!(upper_bound > lower_bound);
    assert!(target_number_of_grid_cells >= 2 * UNIFORM_CELLS);
    assert!(control_coords.len() >= 2);
    assert_eq!(control_grid_cell_extents.len(), control_coords.len());
    assert!(
        *control_coords.first().unwrap() == lower_bound
            && *control_coords.last().unwrap() == upper_bound
    );

    let lower_bound = F::to_f64(&lower_bound).unwrap();
    let upper_bound = F::to_f64(&upper_bound).unwrap();

    let extent = upper_bound - lower_bound;
    let target_mean_grid_cell_extent = extent / (target_number_of_grid_cells as f64);

    // Normalize control coordinates to go from 0 to 1
    let control_coords: Vec<_> = control_coords
        .iter()
        .map(|edge_coord| (F::to_f64(edge_coord).unwrap() - lower_bound) / extent)
        .collect();

    // Scale control grid cell extents to target mean value
    let mut control_grid_cell_extents: Vec<_> = control_grid_cell_extents
        .iter()
        .map(|grid_extent| F::to_f64(grid_extent).unwrap())
        .collect();
    let control_grid_cell_extent_sum: f64 = control_grid_cell_extents.iter().sum();
    let mean_control_grid_cell_extent =
        control_grid_cell_extent_sum / (control_grid_cell_extents.len() as f64);
    control_grid_cell_extents
        .iter_mut()
        .for_each(|grid_cell_extent| {
            *grid_cell_extent *= target_mean_grid_cell_extent / mean_control_grid_cell_extent
        });

    // Compute centers in a grid with edges defined by the control coordinates (although they will not be used)
    let mut centers: Vec<f64> = control_coords
        .iter()
        .zip(control_coords.iter().skip(1))
        .map(|(lower, upper)| lower + 0.5 * (upper - lower))
        .collect();
    centers.push(2.0 * control_coords.last().unwrap() - centers.last().unwrap());

    // Define 1D "field" of control grid scale extents for interpolation
    let control_grid_cell_extent_field = ScalarField1::new(
        String::new(),
        Arc::new(NonUniformGrid1::from_coords(centers, control_coords, false)),
        CoordLocation::LowerEdge,
        Array1::from(control_grid_cell_extents),
    );

    let evaluate_grid_cell_extent = |coord| -> Result<_, Cow<_>> {
        let grid_cell_extent = interpolator
            .interp_extrap_scalar_field(&control_grid_cell_extent_field, coord)
            .expect_inside();
        if grid_cell_extent >= 0.0 {
            Ok(grid_cell_extent)
        } else {
            Err("Encountered negative grid cell extent\n\
                 (probably caused by too strong variations in control points)"
                .into())
        }
    };

    // Adjust scaling of grid cell extents until the cumulative sum of discretized
    // grid cell extents hits the upper boundary

    let mut grid_cell_edges = vec![0.0; target_number_of_grid_cells + 1];
    let temp_upper_boundary_index = target_number_of_grid_cells - UNIFORM_CELLS;

    let mut correction_scale = 1.0 / extent;
    let mut iterations = 0;
    loop {
        // Make sure grid cells near lower boundary have uniform extent
        let grid_cell_extent = evaluate_grid_cell_extent(0.0)? * correction_scale;
        for i in 1..=UNIFORM_CELLS {
            grid_cell_edges[i] = grid_cell_edges[i - 1] + grid_cell_extent;
        }
        // Make sure interpolation to follow starts at coordinate 0.0
        let offset = -grid_cell_edges[UNIFORM_CELLS];

        for i in UNIFORM_CELLS..temp_upper_boundary_index {
            if offset + grid_cell_edges[i] > 1.0 {
                // If we pass the upper boundary too early, stop interpolation
                // and simply repeat the last grid cell extent
                let grid_cell_extent = evaluate_grid_cell_extent(1.0)? * correction_scale;
                for j in i..temp_upper_boundary_index {
                    grid_cell_edges[j + 1] = grid_cell_edges[j] + grid_cell_extent;
                }
                break;
            }

            let grid_cell_extent_1 =
                evaluate_grid_cell_extent(offset + grid_cell_edges[i])? * correction_scale;
            let grid_cell_extent_2 =
                evaluate_grid_cell_extent(offset + grid_cell_edges[i] - 0.5 * grid_cell_extent_1)?
                    * correction_scale;
            let grid_cell_extent_3 =
                evaluate_grid_cell_extent(offset + grid_cell_edges[i] + 0.5 * grid_cell_extent_1)?
                    * correction_scale;
            let grid_cell_extent = 0.5 * (grid_cell_extent_2 + grid_cell_extent_3);

            grid_cell_edges[i + 1] = grid_cell_edges[i] + grid_cell_extent;
        }

        iterations += 1;

        // Check whether we hit the upper boundary (1.0).
        // If not, adjust the correction scale and retry.
        let upper_boundary = offset + grid_cell_edges[temp_upper_boundary_index - 1];
        if f64::abs(upper_boundary - 1.0) > TOLERANCE {
            correction_scale *= 1.0 / upper_boundary;
        } else {
            break;
        }

        if iterations > MAX_ITERATIONS {
            return Err(format!(
                "Exceeded limit of {} iterations for determining grid cell extents\n\
                 (probably caused by too strong variations in control points)",
                MAX_ITERATIONS
            )
            .into());
        }
    }

    // Make sure grid cells near upper boundary have uniform extent
    let grid_cell_extent =
        grid_cell_edges[temp_upper_boundary_index] - grid_cell_edges[temp_upper_boundary_index - 1];
    for i in temp_upper_boundary_index..target_number_of_grid_cells {
        grid_cell_edges[i + 1] = grid_cell_edges[i] + grid_cell_extent;
    }
    // Computed coordinates must be corrected so that the final upper boundary is at 1.0
    let correction_scale = 1.0 / grid_cell_edges[target_number_of_grid_cells];

    // Rescale computed grid cell edges from [0, 1] to actual coordinates
    grid_cell_edges
        .iter_mut()
        .for_each(|coord| *coord = (*coord) * correction_scale * extent + lower_bound);

    // Compute centers and lower edges of discretized grid

    let grid_cell_extents = grid_cell_edges
        .iter()
        .zip(grid_cell_edges.iter().skip(1))
        .map(|(lower, upper)| upper - lower);

    let centers: Vec<_> = grid_cell_extents
        .zip(grid_cell_edges.iter())
        .map(|(grid_cell_extent, lower_edge)| {
            F::from_f64(lower_edge + 0.5 * grid_cell_extent).unwrap()
        })
        .collect();

    grid_cell_edges.pop().unwrap();

    let mut lower_edges: Vec<_> = grid_cell_edges
        .iter()
        .map(|&lower_edge| F::from_f64(lower_edge).unwrap())
        .collect();

    // Ensure derived upper bound will have the exact specified value
    *lower_edges.last_mut().unwrap() =
        F::from_f32(2.0).unwrap() * *centers.last().unwrap() - F::from_f64(upper_bound).unwrap();

    Ok((centers, lower_edges))
}

/// Compute upward and downward weighted derivatives of the given grid cell centers.
pub fn compute_up_and_down_derivatives<F: BFloat>(centers: &[F]) -> (Vec<F>, Vec<F>) {
    const D: f64 = -75.0 / 107520.0;
    const C: f64 = 1029.0 / 107520.0;
    const B: f64 = -8575.0 / 107520.0;
    const A: f64 = 1.0 - 3.0 * B - 5.0 * C - 7.0 * D;

    let n = centers.len();
    assert!(n >= 8);

    let centers: Vec<_> = centers
        .iter()
        .map(|center| F::to_f64(center).unwrap())
        .collect();

    let mut up_derivatives = vec![0.0; n];
    let mut down_derivatives = vec![0.0; n];

    for i in 3..n - 4 {
        up_derivatives[i] = A * (centers[i + 1] - centers[i])
            + B * (centers[i + 2] - centers[i - 1])
            + C * (centers[i + 3] - centers[i - 2])
            + D * (centers[i + 4] - centers[i - 3]);
    }
    for i in 0..3 {
        up_derivatives[i] = up_derivatives[3];
    }
    for i in n - 4..n {
        up_derivatives[i] = up_derivatives[n - 5];
    }

    for i in 4..n - 3 {
        down_derivatives[i] = A * (centers[i] - centers[i - 1])
            + B * (centers[i + 1] - centers[i - 2])
            + C * (centers[i + 2] - centers[i - 3])
            + D * (centers[i + 3] - centers[i - 4]);
    }
    for i in 0..4 {
        down_derivatives[i] = down_derivatives[4];
    }
    for i in n - 3..n {
        down_derivatives[i] = down_derivatives[n - 4];
    }

    (
        up_derivatives
            .into_iter()
            .map(|d| F::from_f64(1.0 / d).unwrap())
            .collect(),
        down_derivatives
            .into_iter()
            .map(|d| F::from_f64(1.0 / d).unwrap())
            .collect(),
    )
}

/// Computes coordinate derivatives for regular coordinates with the given cell extent.
pub fn compute_regular_derivatives<F: BFloat>(size: usize, cell_extent: F) -> Vec<F> {
    iter::repeat(F::one() / cell_extent).take(size).collect()
}

pub fn compute_grid_cell_extents<F: BFloat>(centers: &[F], lower_edges: &[F]) -> Vec<F> {
    assert_eq!(centers.len(), lower_edges.len());
    let mut grid_cell_extents = Vec::with_capacity(lower_edges.len());
    grid_cell_extents.extend(
        lower_edges
            .iter()
            .zip(lower_edges.iter().skip(1))
            .map(|(&lower, &upper)| upper - lower),
    );
    grid_cell_extents.push(cell_extent_from_center_and_lower_edge(
        *centers.last().unwrap(),
        *lower_edges.last().unwrap(),
    ));
    grid_cell_extents
}

pub fn extent_from_bounds<F: BFloat>(lower_bound: F, upper_bound: F) -> F {
    upper_bound - lower_bound
}

pub fn cell_extent_from_bounds<F: BFloat>(size: usize, lower_bound: F, upper_bound: F) -> F {
    let extent = extent_from_bounds(lower_bound, upper_bound);
    extent / F::from_usize(size).unwrap()
}

fn bounds_from_coords<F: BFloat>(size: usize, centers: &[F], lower_edges: &[F]) -> (F, F) {
    (
        lower_edges[0],
        upper_edge_from_center_and_lower_edge(centers[size - 1], lower_edges[size - 1]),
    )
}

fn cell_extent_from_center_and_lower_edge<F: BFloat>(center: F, lower_edge: F) -> F {
    F::from_f32(2.0).unwrap() * (center - lower_edge)
}

fn upper_edge_from_center_and_lower_edge<F: BFloat>(center: F, lower_edge: F) -> F {
    lower_edge + cell_extent_from_center_and_lower_edge(center, lower_edge)
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

        mid = F::to_usize(&mid_float).expect("Conversion failed");

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
