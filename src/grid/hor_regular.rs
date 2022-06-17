//! Structured grids with uniform spacing in the horizontal dimensions.

use super::{regular::RegularGrid2, CoordLocation, Grid1, Grid2, Grid3, GridType};
use crate::{
    geometry::{
        CoordRefs2, CoordRefs3, Coords2, Coords3, Dim2,
        Dim3::{self, X, Y, Z},
        Idx3, In2D, In3D, Vec2, Vec3,
    },
    num::BFloat,
};
use std::iter;

#[cfg(feature = "for-testing")]
use crate::{impl_abs_diff_eq_for_grid, impl_partial_eq_for_grid, impl_relative_eq_for_grid};

/// A 3D grid which is regular in x and y but non-uniform in z.
#[derive(Clone, Debug)]
pub struct HorRegularGrid3<F> {
    coords: [Coords3<F>; 2],
    grid_cell_extents_z: Vec<F>,
    regular_z_coords: [Vec<F>; 2],
    is_periodic: In3D<bool>,
    grid_type: GridType,
    shape: In3D<usize>,
    lower_bounds: Vec3<F>,
    upper_bounds: Vec3<F>,
    extents: Vec3<F>,
    hor_grid_cell_extents: Vec2<F>,
    coord_derivatives: [Option<Coords3<F>>; 2],
}

impl<F: BFloat> HorRegularGrid3<F> {
    pub fn from_regular_grid_data(
        coords: [Coords3<F>; 2],
        is_periodic: In3D<bool>,
        shape: In3D<usize>,
        lower_bounds: Vec3<F>,
        upper_bounds: Vec3<F>,
        extents: Vec3<F>,
        cell_extents: Vec3<F>,
        coord_derivatives: [Option<Coords3<F>>; 2],
    ) -> Self {
        let grid_cell_extents_z = vec![cell_extents[Z]; shape[Z]];
        let regular_z_coords = [coords[0][Z].clone(), coords[1][Z].clone()];
        let hor_grid_cell_extents = Vec2::new(cell_extents[X], cell_extents[Y]);
        let grid_type = GridType::Regular;
        Self {
            coords,
            grid_cell_extents_z,
            regular_z_coords,
            is_periodic,
            grid_type,
            shape,
            lower_bounds,
            upper_bounds,
            extents,
            hor_grid_cell_extents,
            coord_derivatives,
        }
    }
}

impl<F: BFloat> Grid3<F> for HorRegularGrid3<F> {
    type XSliceGrid = HorRegularGrid2<F>;
    type YSliceGrid = HorRegularGrid2<F>;
    type ZSliceGrid = RegularGrid2<F>;

    fn from_coords_unchecked(
        centers: Coords3<F>,
        lower_edges: Coords3<F>,
        is_periodic: In3D<bool>,
        up_derivatives: Option<Coords3<F>>,
        down_derivatives: Option<Coords3<F>>,
        grid_type: GridType,
    ) -> Self {
        assert!(
            !is_periodic[Z],
            "This grid type cannot be periodic in the z-direction."
        );

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

        let grid_cell_extents_z = super::compute_grid_cell_extents(&centers[Z], &lower_edges[Z]);

        let extent_x = super::extent_from_bounds(lower_bound_x, upper_bound_x);
        let extent_y = super::extent_from_bounds(lower_bound_y, upper_bound_y);
        let extent_z = super::extent_from_bounds(lower_bound_z, upper_bound_z);

        let (regular_centers_z, regular_lower_edges_z) =
            super::regular_coords_from_bounds(size_z, lower_bound_z, upper_bound_z);

        let grid_cell_extent_x = extent_x / F::from_usize(size_x).unwrap();
        let grid_cell_extent_y = extent_y / F::from_usize(size_y).unwrap();

        Self {
            coords: [centers, lower_edges],
            grid_cell_extents_z,
            regular_z_coords: [regular_centers_z, regular_lower_edges_z],
            is_periodic,
            grid_type,
            shape: In3D::new(size_x, size_y, size_z),
            lower_bounds: Vec3::new(lower_bound_x, lower_bound_y, lower_bound_z),
            upper_bounds: Vec3::new(upper_bound_x, upper_bound_y, upper_bound_z),
            extents: Vec3::new(extent_x, extent_y, extent_z),
            hor_grid_cell_extents: Vec2::new(grid_cell_extent_x, grid_cell_extent_y),
            coord_derivatives: [up_derivatives, down_derivatives],
        }
    }

    fn detected_grid_type(&self) -> GridType {
        self.grid_type
    }

    fn shape(&self) -> &In3D<usize> {
        &self.shape
    }
    fn periodicity(&self) -> &In3D<bool> {
        &self.is_periodic
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
        CoordRefs3::new(&centers[X], &centers[Y], &self.regular_z_coords[0])
    }

    fn regular_lower_edges(&self) -> CoordRefs3<F> {
        let lower_edges = self.lower_edges();
        CoordRefs3::new(&lower_edges[X], &lower_edges[Y], &self.regular_z_coords[1])
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
        assert!(
            !is_periodic[Z],
            "This grid type cannot be periodic in the z-direction."
        );
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

    fn grid_cell_extents(&self, indices: &Idx3<usize>) -> Vec3<F> {
        Vec3::new(
            self.hor_grid_cell_extents[Dim2::X],
            self.hor_grid_cell_extents[Dim2::Y],
            self.grid_cell_extents_z[indices[Z]],
        )
    }

    fn determine_n_monotonic_grid_cell_edges(
        &self,
        dim: Dim3,
        start_idx: usize,
        n_grid_cells: usize,
        offset: F,
    ) -> Vec<F> {
        debug_assert!(start_idx < self.shape[dim]);
        let start_lower_edge = self.lower_edges()[dim][start_idx] + offset;
        if dim == Z {
            iter::once(start_lower_edge)
                .chain(
                    self.grid_cell_extents_z[start_idx..start_idx + n_grid_cells]
                        .iter()
                        .scan(start_lower_edge, |edge, &grid_cell_extent| {
                            *edge = *edge + grid_cell_extent;
                            Some(*edge)
                        }),
                )
                .collect()
        } else {
            let cell_extent = self.hor_grid_cell_extents[Dim2::from_dim3(dim).unwrap()];
            iter::once(start_lower_edge)
                .chain(
                    (0..n_grid_cells)
                        .into_iter()
                        .scan(start_lower_edge, |edge, _| {
                            *edge = *edge + cell_extent;
                            Some(*edge)
                        }),
                )
                .collect()
        }
    }
}

#[cfg(feature = "for-testing")]
impl_partial_eq_for_grid!(HorRegularGrid3<F>, Grid3);

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_for_grid!(HorRegularGrid3<F>, Grid3);

#[cfg(feature = "for-testing")]
impl_relative_eq_for_grid!(HorRegularGrid3<F>, Grid3);

/// A 2D grid which is regular in x but non-uniform in y.
#[derive(Clone, Debug)]
pub struct HorRegularGrid2<F> {
    coords: [Coords2<F>; 2],
    regular_y_coords: [Vec<F>; 2],
    is_periodic: In2D<bool>,
    shape: In2D<usize>,
    lower_bounds: Vec2<F>,
    upper_bounds: Vec2<F>,
    extents: Vec2<F>,
}

impl<F: BFloat> HorRegularGrid2<F> {
    pub fn from_regular_grid_data(
        coords: [Coords2<F>; 2],
        is_periodic: In2D<bool>,
        shape: In2D<usize>,
        lower_bounds: Vec2<F>,
        upper_bounds: Vec2<F>,
        extents: Vec2<F>,
    ) -> Self {
        let regular_y_coords = [coords[0][Dim2::Y].clone(), coords[1][Dim2::Y].clone()];
        Self {
            coords,
            regular_y_coords,
            is_periodic,
            shape,
            lower_bounds,
            upper_bounds,
            extents,
        }
    }
}

impl<F: BFloat> Grid2<F> for HorRegularGrid2<F> {
    const TYPE: GridType = GridType::HorRegular;

    fn from_coords(centers: Coords2<F>, lower_edges: Coords2<F>, is_periodic: In2D<bool>) -> Self {
        assert!(
            !is_periodic[Dim2::Y],
            "This grid type cannot be periodic in the y-direction."
        );

        let size_x = centers[Dim2::X].len();
        let size_y = centers[Dim2::Y].len();

        let (lower_bound_x, upper_bound_x) =
            super::bounds_from_coords(size_x, &centers[Dim2::X], &lower_edges[Dim2::X]);
        let (lower_bound_y, upper_bound_y) =
            super::bounds_from_coords(size_y, &centers[Dim2::Y], &lower_edges[Dim2::Y]);

        let extent_x = super::extent_from_bounds(lower_bound_x, upper_bound_x);
        let extent_y = super::extent_from_bounds(lower_bound_y, upper_bound_y);

        let (regular_centers_y, regular_lower_edges_y) =
            super::regular_coords_from_bounds(size_y, lower_bound_y, upper_bound_y);

        Self {
            coords: [centers, lower_edges],
            regular_y_coords: [regular_centers_y, regular_lower_edges_y],
            is_periodic,
            shape: In2D::new(size_x, size_y),
            lower_bounds: Vec2::new(lower_bound_x, lower_bound_y),
            upper_bounds: Vec2::new(upper_bound_x, upper_bound_y),
            extents: Vec2::new(extent_x, extent_y),
        }
    }

    fn shape(&self) -> &In2D<usize> {
        &self.shape
    }

    fn periodicity(&self) -> &In2D<bool> {
        &self.is_periodic
    }

    fn is_periodic(&self, dim: Dim2) -> bool {
        self.is_periodic[dim]
    }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords2<F> {
        &self.coords[location as usize]
    }

    fn regular_centers(&self) -> CoordRefs2<F> {
        let centers = self.centers();
        CoordRefs2::new(&centers[Dim2::X], &self.regular_y_coords[0])
    }

    fn regular_lower_edges(&self) -> CoordRefs2<F> {
        let lower_edges = self.lower_edges();
        CoordRefs2::new(&lower_edges[Dim2::X], &self.regular_y_coords[1])
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

#[cfg(feature = "for-testing")]
impl_partial_eq_for_grid!(HorRegularGrid2<F>, Grid2);

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_for_grid!(HorRegularGrid2<F>, Grid2);

#[cfg(feature = "for-testing")]
impl_relative_eq_for_grid!(HorRegularGrid2<F>, Grid2);

/// A 1D non-uniform grid.
#[derive(Clone, Debug)]
pub struct NonUniformGrid1<F> {
    coords: [Vec<F>; 2],
    regular_coords: [Vec<F>; 2],
    size: usize,
    lower_bound: F,
    upper_bound: F,
    extent: F,
}

impl<F: BFloat> NonUniformGrid1<F> {
    pub fn from_regular_grid_data(
        coords: [Vec<F>; 2],
        size: usize,
        lower_bound: F,
        upper_bound: F,
        extent: F,
    ) -> Self {
        let regular_coords = coords.clone();
        Self {
            coords,
            regular_coords,
            size,
            lower_bound,
            upper_bound,
            extent,
        }
    }
}

impl<F: BFloat> Grid1<F> for NonUniformGrid1<F> {
    fn from_coords(centers: Vec<F>, lower_edges: Vec<F>, is_periodic: bool) -> Self {
        assert!(!is_periodic, "This grid type cannot be periodic.");

        let size = centers.len();

        let (lower_bound, upper_bound) = super::bounds_from_coords(size, &centers, &lower_edges);

        let extent = super::extent_from_bounds(lower_bound, upper_bound);

        let (regular_centers, regular_lower_edges) =
            super::regular_coords_from_bounds(size, lower_bound, upper_bound);

        Self {
            coords: [centers, lower_edges],
            regular_coords: [regular_centers, regular_lower_edges],
            size,
            lower_bound,
            upper_bound,
            extent,
        }
    }

    fn size(&self) -> usize {
        self.size
    }
    fn is_periodic(&self) -> bool {
        false
    }
    fn coords_by_type(&self, location: CoordLocation) -> &[F] {
        &self.coords[location as usize]
    }

    fn regular_centers(&self) -> &[F] {
        &self.regular_coords[0]
    }

    fn regular_lower_edges(&self) -> &[F] {
        &self.regular_coords[1]
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
    use ndarray::s;

    #[test]
    fn varying_z_grid_index_search_works() {
        #![allow(clippy::deref_addrof)] // Mutes warning due to workings of s! macro
        let (mx, my, mz) = (17, 5, 29);

        let xc = Array::linspace(-1.0, 1.0, mx);
        let yc = Array::linspace(1.0, 5.2, my);

        let (dx, dy) = (xc[1] - xc[0], yc[1] - yc[0]);

        let xdn = Array::linspace(xc[0] - dx / 2.0, xc[mx - 1] - dx / 2.0, mx);
        let ydn = Array::linspace(yc[0] - dy / 2.0, yc[my - 1] - dy / 2.0, my);

        let zdn = Array::linspace(-2.0, 2.0, mz + 1)
            + Array::linspace(1.0, 2.0, mz + 1).mapv(|a| a * a * a * a);
        let zc = (zdn.slice(s![1..]).into_owned() + zdn.slice(s![..mz])) * 0.5;
        let zdn = zdn.slice(s![..mz]).into_owned();

        let z_max = 2.0 * zc[mz - 1] - zdn[mz - 1];

        let centers = Coords3::new(xc.to_vec(), yc.to_vec(), zc.to_vec());
        let lower_edges = Coords3::new(xdn.to_vec(), ydn.to_vec(), zdn.to_vec());

        let grid = HorRegularGrid3::from_coords(
            centers,
            lower_edges,
            In3D::new(false, false, false),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            grid.find_grid_cell(&Point3::new(
                xdn[mx - 1] + dx + 1e-12,
                ydn[my - 1] + dy + 1e-12,
                z_max + 1e-12
            )),
            GridPointQuery3::Outside
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)),
            GridPointQuery3::Inside(Idx3::new(0, 0, 0))
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] - 1e-9)),
            GridPointQuery3::Outside
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(-0.68751, 1.5249, 3.0)),
            GridPointQuery3::Inside(Idx3::new(2, 0, 10))
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(0.0, 2.0, 16.7)),
            GridPointQuery3::Inside(Idx3::new(8, 1, 27))
        );
        assert_eq!(
            grid.find_grid_cell(&Point3::new(0.0, 2.0, -0.7)),
            GridPointQuery3::Inside(Idx3::new(8, 1, 1))
        );
    }
}
