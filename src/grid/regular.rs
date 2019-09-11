//! Structured grids with uniform spacing in all dimensions.

use crate::num::BFloat;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Vec3, Vec2, Coords3, Coords2, CoordRefs3, CoordRefs2};
use super::{CoordLocation, GridType, Grid3, Grid2};
use Dim3::{X, Y, Z};

/// A regular 3D grid.
#[derive(Clone, Debug)]
pub struct RegularGrid3<F: BFloat> {
    coords: [Coords3<F>; 2],
    is_periodic: In3D<bool>,
    shape: In3D<usize>,
    lower_bounds: Vec3<F>,
    upper_bounds: Vec3<F>,
    extents: Vec3<F>,
    cell_extents: Vec3<F>
}

impl<F: BFloat> RegularGrid3<F> {
    /// Creates a new regular grid given the shape and bounds, as well as which dimensions are periodic.
    pub fn from_bounds(shape: In3D<usize>, lower_bounds: Vec3<F>, upper_bounds: Vec3<F>, is_periodic: In3D<bool>) -> Self {
        let size_x = shape[X];
        let size_y = shape[Y];
        let size_z = shape[Z];

        assert_ne!(size_x, 0, "Cannot create grid with size zero along any dimension.");
        assert_ne!(size_y, 0, "Cannot create grid with size zero along any dimension.");
        assert_ne!(size_z, 0, "Cannot create grid with size zero along any dimension.");

        let (centers_x, lower_edges_x) = super::regular_coords_from_bounds(size_x, lower_bounds[X], upper_bounds[X]);
        let (centers_y, lower_edges_y) = super::regular_coords_from_bounds(size_y, lower_bounds[Y], upper_bounds[Y]);
        let (centers_z, lower_edges_z) = super::regular_coords_from_bounds(size_z, lower_bounds[Z], upper_bounds[Z]);

        let extent_x = super::extent_from_bounds(lower_bounds[X], upper_bounds[X]);
        let extent_y = super::extent_from_bounds(lower_bounds[Y], upper_bounds[Y]);
        let extent_z = super::extent_from_bounds(lower_bounds[Z], upper_bounds[Z]);

        let cell_extent_x = super::cell_extent_from_bounds(size_x, lower_bounds[X], upper_bounds[X]);
        let cell_extent_y = super::cell_extent_from_bounds(size_y, lower_bounds[Y], upper_bounds[Y]);
        let cell_extent_z = super::cell_extent_from_bounds(size_z, lower_bounds[Z], upper_bounds[Z]);

        RegularGrid3{
            coords: [Coords3::new(centers_x, centers_y, centers_z), Coords3::new(lower_edges_x, lower_edges_y, lower_edges_z)],
            is_periodic,
            shape,
            lower_bounds,
            upper_bounds,
            extents: Vec3::new(extent_x, extent_y, extent_z),
            cell_extents: Vec3::new(cell_extent_x, cell_extent_y, cell_extent_z)
        }
    }

    /// Constructs a reshaped version of the given grid, with the same bounds and periodicity.
    pub fn reshaped(&self, new_shape: In3D<usize>) -> Self {
        RegularGrid3::from_bounds(
            new_shape,
            self.lower_bounds.clone(),
            self.upper_bounds.clone(),
            self.is_periodic.clone()
        )
    }

    /// Returns a reference to the extent of a grid cell in each dimension.
    pub fn cell_extents(&self) -> &Vec3<F> { &self.cell_extents }
}

impl<F: BFloat> Grid3<F> for RegularGrid3<F> {
    type XSliceGrid = RegularGrid2<F>;
    type YSliceGrid = RegularGrid2<F>;
    type ZSliceGrid = RegularGrid2<F>;

    const TYPE: GridType = GridType::Regular;

    fn from_coords(centers: Coords3<F>, lower_edges: Coords3<F>, is_periodic: In3D<bool>) -> Self {
        let size_x = centers[X].len();
        let size_y = centers[Y].len();
        let size_z = centers[Z].len();

        assert_ne!(size_x, 0, "Cannot create grid with size zero along any dimension.");
        assert_ne!(size_y, 0, "Cannot create grid with size zero along any dimension.");
        assert_ne!(size_z, 0, "Cannot create grid with size zero along any dimension.");

        let (lower_bound_x, upper_bound_x) = super::bounds_from_coords(size_x, &centers[X], &lower_edges[X]);
        let (lower_bound_y, upper_bound_y) = super::bounds_from_coords(size_y, &centers[Y], &lower_edges[Y]);
        let (lower_bound_z, upper_bound_z) = super::bounds_from_coords(size_z, &centers[Z], &lower_edges[Z]);

        let extent_x = super::extent_from_bounds(lower_bound_x, upper_bound_x);
        let extent_y = super::extent_from_bounds(lower_bound_y, upper_bound_y);
        let extent_z = super::extent_from_bounds(lower_bound_z, upper_bound_z);

        let cell_extent_x = super::cell_extent_from_bounds(size_x, lower_bound_x, upper_bound_x);
        let cell_extent_y = super::cell_extent_from_bounds(size_y, lower_bound_y, upper_bound_y);
        let cell_extent_z = super::cell_extent_from_bounds(size_z, lower_bound_z, upper_bound_z);

        RegularGrid3{
            coords: [centers, lower_edges],
            is_periodic,
            shape: In3D::new(size_x, size_y, size_z),
            lower_bounds: Vec3::new(lower_bound_x, lower_bound_y, lower_bound_z),
            upper_bounds: Vec3::new(upper_bound_x, upper_bound_y, upper_bound_z),
            extents: Vec3::new(extent_x, extent_y, extent_z),
            cell_extents: Vec3::new(cell_extent_x, cell_extent_y, cell_extent_z)
        }
    }

    fn shape(&self) -> &In3D<usize> { &self.shape }
    fn is_periodic(&self, dim: Dim3) -> bool { self.is_periodic[dim] }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords3<F> { &self.coords[location as usize] }

    fn regular_centers(&self) -> CoordRefs3<F> {
        let centers = self.centers();
        CoordRefs3::new(
            &centers[X],
            &centers[Y],
            &centers[Z]
        )
    }

    fn regular_lower_edges(&self) -> CoordRefs3<F> {
        let lower_edges = self.lower_edges();
        CoordRefs3::new(
            &lower_edges[X],
            &lower_edges[Y],
            &lower_edges[Z]
        )
    }

    fn lower_bounds(&self) -> &Vec3<F> { &self.lower_bounds }
    fn upper_bounds(&self) -> &Vec3<F> { &self.upper_bounds }
    fn extents(&self) -> &Vec3<F> { &self.extents }
}

/// A regular 2D grid.
#[derive(Clone, Debug)]
pub struct RegularGrid2<F: BFloat> {
    coords: [Coords2<F>; 2],
    is_periodic: In2D<bool>,
    shape: In2D<usize>,
    lower_bounds: Vec2<F>,
    upper_bounds: Vec2<F>,
    extents: Vec2<F>,
    cell_extents: Vec2<F>
}

impl<F: BFloat> RegularGrid2<F> {
    /// Creates a new regular grid given the shape and bounds, as well as which dimensions are periodic.
    pub fn from_bounds(shape: In2D<usize>, lower_bounds: Vec2<F>, upper_bounds: Vec2<F>, is_periodic: In2D<bool>) -> Self {
        let size_x = shape[Dim2::X];
        let size_y = shape[Dim2::Y];

        assert_ne!(size_x, 0, "Cannot create grid with size zero along any dimension.");
        assert_ne!(size_y, 0, "Cannot create grid with size zero along any dimension.");

        let (centers_x, lower_edges_x) = super::regular_coords_from_bounds(shape[Dim2::X], lower_bounds[Dim2::X], upper_bounds[Dim2::X]);
        let (centers_y, lower_edges_y) = super::regular_coords_from_bounds(shape[Dim2::Y], lower_bounds[Dim2::Y], upper_bounds[Dim2::Y]);

        let extent_x = super::extent_from_bounds(lower_bounds[Dim2::X], upper_bounds[Dim2::X]);
        let extent_y = super::extent_from_bounds(lower_bounds[Dim2::Y], upper_bounds[Dim2::Y]);

        let cell_extent_x = super::cell_extent_from_bounds(size_x, lower_bounds[Dim2::X], upper_bounds[Dim2::X]);
        let cell_extent_y = super::cell_extent_from_bounds(size_y, lower_bounds[Dim2::Y], upper_bounds[Dim2::Y]);

        RegularGrid2{
            coords: [Coords2::new(centers_x, centers_y), Coords2::new(lower_edges_x, lower_edges_y)],
            is_periodic,
            shape,
            lower_bounds,
            upper_bounds,
            extents: Vec2::new(extent_x, extent_y),
            cell_extents: Vec2::new(cell_extent_x, cell_extent_y)
        }
    }

    /// Constructs a reshaped version of the grid, with the same bounds and periodicity.
    pub fn reshaped(&self, new_shape: In2D<usize>) -> Self {
        RegularGrid2::from_bounds(
            new_shape,
            self.lower_bounds.clone(),
            self.upper_bounds.clone(),
            self.is_periodic.clone()
        )
    }

    /// Returns a reference to the extent of a grid cell in each dimension.
    pub fn cell_extents(&self) -> &Vec2<F> { &self.cell_extents }
}

impl<F: BFloat> Grid2<F> for RegularGrid2<F> {
    const TYPE: GridType = GridType::Regular;

    fn from_coords(centers: Coords2<F>, lower_edges: Coords2<F>, is_periodic: In2D<bool>) -> Self {
        let size_x = centers[Dim2::X].len();
        let size_y = centers[Dim2::Y].len();

        assert_ne!(size_x, 0, "Cannot create grid with size zero along any dimension.");
        assert_ne!(size_y, 0, "Cannot create grid with size zero along any dimension.");

        let (lower_bound_x, upper_bound_x) = super::bounds_from_coords(size_x, &centers[Dim2::X], &lower_edges[Dim2::X]);
        let (lower_bound_y, upper_bound_y) = super::bounds_from_coords(size_y, &centers[Dim2::Y], &lower_edges[Dim2::Y]);

        let extent_x = super::extent_from_bounds(lower_bound_x, upper_bound_x);
        let extent_y = super::extent_from_bounds(lower_bound_y, upper_bound_y);

        let cell_extent_x = super::cell_extent_from_bounds(size_x, lower_bound_x, upper_bound_x);
        let cell_extent_y = super::cell_extent_from_bounds(size_y, lower_bound_y, upper_bound_y);

        RegularGrid2{
            coords: [centers, lower_edges],
            is_periodic,
            shape: In2D::new(size_x, size_y),
            lower_bounds: Vec2::new(lower_bound_x, lower_bound_y),
            upper_bounds: Vec2::new(upper_bound_x, upper_bound_y),
            extents: Vec2::new(extent_x, extent_y),
            cell_extents: Vec2::new(cell_extent_x, cell_extent_y)
        }
    }

    fn shape(&self) -> &In2D<usize> { &self.shape }
    fn is_periodic(&self, dim: Dim2) -> bool { self.is_periodic[dim] }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords2<F> { &self.coords[location as usize] }

    fn regular_centers(&self) -> CoordRefs2<F> {
        let centers = self.centers();
        CoordRefs2::new(
            &centers[Dim2::X],
            &centers[Dim2::Y]
        )
    }

    fn regular_lower_edges(&self) -> CoordRefs2<F> {
        let lower_edges = self.lower_edges();
        CoordRefs2::new(
            &lower_edges[Dim2::X],
            &lower_edges[Dim2::Y]
        )
    }

    fn lower_bounds(&self) -> &Vec2<F> { &self.lower_bounds }
    fn upper_bounds(&self) -> &Vec2<F> { &self.upper_bounds }
    fn extents(&self) -> &Vec2<F> { &self.extents }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::prelude::*;
    use crate::geometry::{Point3, Idx3};
    use crate::grid::GridPointQuery3;

    #[test]
    fn regular_grid_index_search_works() {

        let (mx, my, mz) = (17, 5, 29);

        let xc = Array::linspace( -1.0,  1.0, mx);
        let yc = Array::linspace(  1.0,  5.2, my);
        let zc = Array::linspace(-10.0, 10.0, mz);

        let (dx, dy, dz) = (xc[1] - xc[0], yc[1] - yc[0], zc[1] - zc[0]);

        let xdn = Array::linspace(xc[0] - dx/2.0, xc[mx-1] - dx/2.0, mx);
        let ydn = Array::linspace(yc[0] - dy/2.0, yc[my-1] - dy/2.0, my);
        let zdn = Array::linspace(zc[0] - dz/2.0, zc[mz-1] - dz/2.0, mz);

        let centers = Coords3::new(xc.to_vec(), yc.to_vec(), zc.to_vec());
        let lower_edges = Coords3::new(xdn.to_vec(), ydn.to_vec(), zdn.to_vec());

        let grid = RegularGrid3::from_coords(centers, lower_edges, In3D::new(false, false, false));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[mx-1] + dx + 1e-12, ydn[my-1] + dy + 1e-12, zdn[mz-1] + dz + 1e-12)), GridPointQuery3::Outside);
        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)), GridPointQuery3::Inside(Idx3::new(0, 0, 0)));
        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] - 1e-9, zdn[0] + 1e-12)), GridPointQuery3::Outside);
        assert_eq!(grid.find_grid_cell(&Point3::new(-0.68751, 1.5249, 7.5)), GridPointQuery3::Inside(Idx3::new(2, 0, 25)));
    }
}
