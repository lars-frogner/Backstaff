//! Structured grids with uniform spacing in all dimensions.

use super::{CoordLocation, Grid1, Grid2, Grid3, GridType};
use crate::{
    geometry::{
        CoordRefs2, CoordRefs3, Coords2, Coords3, Dim2,
        Dim3::{self, X, Y, Z},
        Idx3, In2D, In3D, Vec2, Vec3,
    },
    num::BFloat,
};
use std::iter;

/// A regular 3D grid.
#[derive(Clone, Debug)]
pub struct RegularGrid3<F: BFloat> {
    coords: [Coords3<F>; 2],
    is_periodic: In3D<bool>,
    shape: In3D<usize>,
    lower_bounds: Vec3<F>,
    upper_bounds: Vec3<F>,
    extents: Vec3<F>,
    cell_extents: Vec3<F>,
    coord_derivatives: [Option<Coords3<F>>; 2],
}

impl<F: BFloat> RegularGrid3<F> {
    /// Creates a new regular grid given the shape and bounds, as well as which dimensions are periodic.
    pub fn from_bounds(
        shape: In3D<usize>,
        lower_bounds: Vec3<F>,
        upper_bounds: Vec3<F>,
        is_periodic: In3D<bool>,
    ) -> Self {
        let size_x = shape[X];
        let size_y = shape[Y];
        let size_z = shape[Z];

        assert_ne!(
            size_x, 0,
            "Cannot create grid with size zero along any dimension."
        );
        assert_ne!(
            size_y, 0,
            "Cannot create grid with size zero along any dimension."
        );
        assert_ne!(
            size_z, 0,
            "Cannot create grid with size zero along any dimension."
        );

        let (centers_x, lower_edges_x) =
            super::regular_coords_from_bounds(size_x, lower_bounds[X], upper_bounds[X]);
        let (centers_y, lower_edges_y) =
            super::regular_coords_from_bounds(size_y, lower_bounds[Y], upper_bounds[Y]);
        let (centers_z, lower_edges_z) =
            super::regular_coords_from_bounds(size_z, lower_bounds[Z], upper_bounds[Z]);

        let extent_x = super::extent_from_bounds(lower_bounds[X], upper_bounds[X]);
        let extent_y = super::extent_from_bounds(lower_bounds[Y], upper_bounds[Y]);
        let extent_z = super::extent_from_bounds(lower_bounds[Z], upper_bounds[Z]);

        let cell_extent_x =
            super::cell_extent_from_bounds(size_x, lower_bounds[X], upper_bounds[X]);
        let cell_extent_y =
            super::cell_extent_from_bounds(size_y, lower_bounds[Y], upper_bounds[Y]);
        let cell_extent_z =
            super::cell_extent_from_bounds(size_z, lower_bounds[Z], upper_bounds[Z]);

        let derivatives_x = iter::repeat(F::one() / cell_extent_x)
            .take(size_x)
            .collect();
        let derivatives_y = iter::repeat(F::one() / cell_extent_y)
            .take(size_y)
            .collect();
        let derivatives_z = iter::repeat(F::one() / cell_extent_z)
            .take(size_z)
            .collect();
        let derivatives = Some(Coords3::new(derivatives_x, derivatives_y, derivatives_z));

        Self {
            coords: [
                Coords3::new(centers_x, centers_y, centers_z),
                Coords3::new(lower_edges_x, lower_edges_y, lower_edges_z),
            ],
            is_periodic,
            shape,
            lower_bounds,
            upper_bounds,
            extents: Vec3::new(extent_x, extent_y, extent_z),
            cell_extents: Vec3::new(cell_extent_x, cell_extent_y, cell_extent_z),
            coord_derivatives: [derivatives.clone(), derivatives],
        }
    }

    /// Constructs a reshaped version of the given grid, with the same bounds and periodicity.
    pub fn reshaped(&self, new_shape: In3D<usize>) -> Self {
        RegularGrid3::from_bounds(
            new_shape,
            self.lower_bounds.clone(),
            self.upper_bounds.clone(),
            self.is_periodic.clone(),
        )
    }

    /// Returns a reference to the extent of a grid cell in each dimension.
    pub fn cell_extents(&self) -> &Vec3<F> {
        &self.cell_extents
    }
}

impl<F: BFloat> Grid3<F> for RegularGrid3<F> {
    type XSliceGrid = RegularGrid2<F>;
    type YSliceGrid = RegularGrid2<F>;
    type ZSliceGrid = RegularGrid2<F>;

    const TYPE: GridType = GridType::Regular;

    fn from_coords(
        centers: Coords3<F>,
        lower_edges: Coords3<F>,
        is_periodic: In3D<bool>,
        up_derivatives: Option<Coords3<F>>,
        down_derivatives: Option<Coords3<F>>,
    ) -> Self {
        let size_x = centers[X].len();
        let size_y = centers[Y].len();
        let size_z = centers[Z].len();

        assert_ne!(
            size_x, 0,
            "Cannot create grid with size zero along any dimension"
        );
        assert_ne!(
            size_y, 0,
            "Cannot create grid with size zero along any dimension"
        );
        assert_ne!(
            size_z, 0,
            "Cannot create grid with size zero along any dimension"
        );
        assert_eq!(
            lower_edges[X].len(),
            size_x,
            "Centers and lower edges must have the same shape"
        );
        assert_eq!(
            lower_edges[Y].len(),
            size_y,
            "Centers and lower edges must have the same shape"
        );
        assert_eq!(
            lower_edges[Z].len(),
            size_z,
            "Centers and lower edges must have the same shape"
        );

        let (lower_bound_x, upper_bound_x) =
            super::bounds_from_coords(size_x, &centers[X], &lower_edges[X]);
        let (lower_bound_y, upper_bound_y) =
            super::bounds_from_coords(size_y, &centers[Y], &lower_edges[Y]);
        let (lower_bound_z, upper_bound_z) =
            super::bounds_from_coords(size_z, &centers[Z], &lower_edges[Z]);

        let extent_x = super::extent_from_bounds(lower_bound_x, upper_bound_x);
        let extent_y = super::extent_from_bounds(lower_bound_y, upper_bound_y);
        let extent_z = super::extent_from_bounds(lower_bound_z, upper_bound_z);

        let cell_extent_x = super::cell_extent_from_bounds(size_x, lower_bound_x, upper_bound_x);
        let cell_extent_y = super::cell_extent_from_bounds(size_y, lower_bound_y, upper_bound_y);
        let cell_extent_z = super::cell_extent_from_bounds(size_z, lower_bound_z, upper_bound_z);

        Self {
            coords: [centers, lower_edges],
            is_periodic,
            shape: In3D::new(size_x, size_y, size_z),
            lower_bounds: Vec3::new(lower_bound_x, lower_bound_y, lower_bound_z),
            upper_bounds: Vec3::new(upper_bound_x, upper_bound_y, upper_bound_z),
            extents: Vec3::new(extent_x, extent_y, extent_z),
            cell_extents: Vec3::new(cell_extent_x, cell_extent_y, cell_extent_z),
            coord_derivatives: [up_derivatives, down_derivatives],
        }
    }

    fn shape(&self) -> &In3D<usize> {
        &self.shape
    }
    fn is_periodic(&self, dim: Dim3) -> bool {
        self.is_periodic[dim]
    }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords3<F> {
        &self.coords[location as usize]
    }

    fn up_derivatives(&self) -> Option<&Coords3<F>> {
        self.coord_derivatives[0].as_ref()
    }

    fn down_derivatives(&self) -> Option<&Coords3<F>> {
        self.coord_derivatives[1].as_ref()
    }

    fn regular_centers(&self) -> CoordRefs3<F> {
        let centers = self.centers();
        CoordRefs3::new(&centers[X], &centers[Y], &centers[Z])
    }

    fn regular_lower_edges(&self) -> CoordRefs3<F> {
        let lower_edges = self.lower_edges();
        CoordRefs3::new(&lower_edges[X], &lower_edges[Y], &lower_edges[Z])
    }

    fn lower_bounds(&self) -> &Vec3<F> {
        &self.lower_bounds
    }
    fn upper_bounds(&self) -> &Vec3<F> {
        &self.upper_bounds
    }
    fn extents(&self) -> &Vec3<F> {
        &self.extents
    }
    fn set_periodicity(&mut self, is_periodic: In3D<bool>) {
        self.is_periodic = is_periodic;
    }
    fn set_up_derivatives(&mut self, up_derivatives: Option<Coords3<F>>) {
        if let Some(ref up_derivatives) = up_derivatives {
            for &dim in &Dim3::slice() {
                assert_eq!(
                    up_derivatives[dim].len(),
                    self.shape[dim],
                    "Upward derivatives must have the same shape as the grid"
                );
            }
        }
        self.coord_derivatives[0] = up_derivatives;
    }
    fn set_down_derivatives(&mut self, down_derivatives: Option<Coords3<F>>) {
        if let Some(ref down_derivatives) = down_derivatives {
            for &dim in &Dim3::slice() {
                assert_eq!(
                    down_derivatives[dim].len(),
                    self.shape[dim],
                    "Downward derivatives must have the same shape as the grid"
                );
            }
        }
        self.coord_derivatives[1] = down_derivatives;
    }

    fn grid_cell_extents(&self, _indices: &Idx3<usize>) -> Vec3<F> {
        self.cell_extents().clone()
    }
    fn average_grid_cell_extents(&self) -> Vec3<F> {
        self.cell_extents().clone()
    }
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
    cell_extents: Vec2<F>,
}

impl<F: BFloat> RegularGrid2<F> {
    /// Creates a new regular grid given the shape and bounds, as well as which dimensions are periodic.
    pub fn from_bounds(
        shape: In2D<usize>,
        lower_bounds: Vec2<F>,
        upper_bounds: Vec2<F>,
        is_periodic: In2D<bool>,
    ) -> Self {
        let size_x = shape[Dim2::X];
        let size_y = shape[Dim2::Y];

        assert_ne!(
            size_x, 0,
            "Cannot create grid with size zero along any dimension."
        );
        assert_ne!(
            size_y, 0,
            "Cannot create grid with size zero along any dimension."
        );

        let (centers_x, lower_edges_x) = super::regular_coords_from_bounds(
            shape[Dim2::X],
            lower_bounds[Dim2::X],
            upper_bounds[Dim2::X],
        );
        let (centers_y, lower_edges_y) = super::regular_coords_from_bounds(
            shape[Dim2::Y],
            lower_bounds[Dim2::Y],
            upper_bounds[Dim2::Y],
        );

        let extent_x = super::extent_from_bounds(lower_bounds[Dim2::X], upper_bounds[Dim2::X]);
        let extent_y = super::extent_from_bounds(lower_bounds[Dim2::Y], upper_bounds[Dim2::Y]);

        let cell_extent_x =
            super::cell_extent_from_bounds(size_x, lower_bounds[Dim2::X], upper_bounds[Dim2::X]);
        let cell_extent_y =
            super::cell_extent_from_bounds(size_y, lower_bounds[Dim2::Y], upper_bounds[Dim2::Y]);

        Self {
            coords: [
                Coords2::new(centers_x, centers_y),
                Coords2::new(lower_edges_x, lower_edges_y),
            ],
            is_periodic,
            shape,
            lower_bounds,
            upper_bounds,
            extents: Vec2::new(extent_x, extent_y),
            cell_extents: Vec2::new(cell_extent_x, cell_extent_y),
        }
    }

    /// Constructs a reshaped version of the grid, with the same bounds and periodicity.
    pub fn reshaped(&self, new_shape: In2D<usize>) -> Self {
        RegularGrid2::from_bounds(
            new_shape,
            self.lower_bounds.clone(),
            self.upper_bounds.clone(),
            self.is_periodic.clone(),
        )
    }

    /// Returns a reference to the extent of a grid cell in each dimension.
    pub fn cell_extents(&self) -> &Vec2<F> {
        &self.cell_extents
    }
}

impl<F: BFloat> Grid2<F> for RegularGrid2<F> {
    const TYPE: GridType = GridType::Regular;

    fn from_coords(centers: Coords2<F>, lower_edges: Coords2<F>, is_periodic: In2D<bool>) -> Self {
        let size_x = centers[Dim2::X].len();
        let size_y = centers[Dim2::Y].len();

        assert_ne!(
            size_x, 0,
            "Cannot create grid with size zero along any dimension."
        );
        assert_ne!(
            size_y, 0,
            "Cannot create grid with size zero along any dimension."
        );

        let (lower_bound_x, upper_bound_x) =
            super::bounds_from_coords(size_x, &centers[Dim2::X], &lower_edges[Dim2::X]);
        let (lower_bound_y, upper_bound_y) =
            super::bounds_from_coords(size_y, &centers[Dim2::Y], &lower_edges[Dim2::Y]);

        let extent_x = super::extent_from_bounds(lower_bound_x, upper_bound_x);
        let extent_y = super::extent_from_bounds(lower_bound_y, upper_bound_y);

        let cell_extent_x = super::cell_extent_from_bounds(size_x, lower_bound_x, upper_bound_x);
        let cell_extent_y = super::cell_extent_from_bounds(size_y, lower_bound_y, upper_bound_y);

        Self {
            coords: [centers, lower_edges],
            is_periodic,
            shape: In2D::new(size_x, size_y),
            lower_bounds: Vec2::new(lower_bound_x, lower_bound_y),
            upper_bounds: Vec2::new(upper_bound_x, upper_bound_y),
            extents: Vec2::new(extent_x, extent_y),
            cell_extents: Vec2::new(cell_extent_x, cell_extent_y),
        }
    }

    fn shape(&self) -> &In2D<usize> {
        &self.shape
    }
    fn is_periodic(&self, dim: Dim2) -> bool {
        self.is_periodic[dim]
    }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords2<F> {
        &self.coords[location as usize]
    }

    fn regular_centers(&self) -> CoordRefs2<F> {
        let centers = self.centers();
        CoordRefs2::new(&centers[Dim2::X], &centers[Dim2::Y])
    }

    fn regular_lower_edges(&self) -> CoordRefs2<F> {
        let lower_edges = self.lower_edges();
        CoordRefs2::new(&lower_edges[Dim2::X], &lower_edges[Dim2::Y])
    }

    fn lower_bounds(&self) -> &Vec2<F> {
        &self.lower_bounds
    }
    fn upper_bounds(&self) -> &Vec2<F> {
        &self.upper_bounds
    }
    fn extents(&self) -> &Vec2<F> {
        &self.extents
    }
}

/// A regular 1D grid.
#[derive(Clone, Debug)]
pub struct RegularGrid1<F: BFloat> {
    coords: [Vec<F>; 2],
    is_periodic: bool,
    size: usize,
    lower_bound: F,
    upper_bound: F,
    extent: F,
    cell_extent: F,
}

impl<F: BFloat> Grid1<F> for RegularGrid1<F> {
    fn from_coords(centers: Vec<F>, lower_edges: Vec<F>, is_periodic: bool) -> Self {
        let size = centers.len();

        assert_ne!(size, 0, "Cannot create grid with size zero.");

        let (lower_bound, upper_bound) = super::bounds_from_coords(size, &centers, &lower_edges);

        let extent = super::extent_from_bounds(lower_bound, upper_bound);

        let cell_extent = super::cell_extent_from_bounds(size, lower_bound, upper_bound);

        Self {
            coords: [centers, lower_edges],
            is_periodic,
            size,
            lower_bound,
            upper_bound,
            extent,
            cell_extent,
        }
    }

    fn size(&self) -> usize {
        self.size
    }
    fn is_periodic(&self) -> bool {
        self.is_periodic
    }
    fn coords_by_type(&self, location: CoordLocation) -> &[F] {
        &self.coords[location as usize]
    }

    fn regular_centers(&self) -> &[F] {
        self.centers()
    }

    fn regular_lower_edges(&self) -> &[F] {
        self.lower_edges()
    }

    fn lower_bound(&self) -> F {
        self.lower_bound
    }
    fn upper_bound(&self) -> F {
        self.upper_bound
    }
    fn extent(&self) -> F {
        self.extent
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::geometry::{Idx3, Point3};
    use crate::grid::GridPointQuery3;
    use ndarray::prelude::*;

    #[test]
    fn regular_grid_index_search_works() {
        let (mx, my, mz) = (17, 5, 29);

        let xc = Array::linspace(-1.0, 1.0, mx);
        let yc = Array::linspace(1.0, 5.2, my);
        let zc = Array::linspace(-10.0, 10.0, mz);

        let (dx, dy, dz) = (xc[1] - xc[0], yc[1] - yc[0], zc[1] - zc[0]);

        let xdn = Array::linspace(xc[0] - dx / 2.0, xc[mx - 1] - dx / 2.0, mx);
        let ydn = Array::linspace(yc[0] - dy / 2.0, yc[my - 1] - dy / 2.0, my);
        let zdn = Array::linspace(zc[0] - dz / 2.0, zc[mz - 1] - dz / 2.0, mz);

        let centers = Coords3::new(xc.to_vec(), yc.to_vec(), zc.to_vec());
        let lower_edges = Coords3::new(xdn.to_vec(), ydn.to_vec(), zdn.to_vec());

        let grid = RegularGrid3::from_coords(
            centers,
            lower_edges,
            In3D::new(false, false, false),
            None,
            None,
        );

        assert_eq!(
            grid.find_grid_cell(&Point3::new(
                xdn[mx - 1] + dx + 1e-12,
                ydn[my - 1] + dy + 1e-12,
                zdn[mz - 1] + dz + 1e-12
            )),
            GridPointQuery3::Outside
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)),
            GridPointQuery3::Inside(Idx3::new(0, 0, 0))
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] - 1e-9, zdn[0] + 1e-12)),
            GridPointQuery3::Outside
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(-0.68751, 1.5249, 7.5)),
            GridPointQuery3::Inside(Idx3::new(2, 0, 25))
        );
    }
}
