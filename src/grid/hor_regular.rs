//! Grid with uniform spacing in the horizontal dimensions.

use num;
use ndarray::prelude::*;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Vec3, Vec2, Point3, Point2, Idx3, Idx2, Coords3, Coords2, CoordRefs3, CoordRefs2};
use super::{BoundsCrossing, FoundIdx3, FoundIdx2, CoordLocation, GridType, Grid3, Grid2};
use super::regular::RegularGrid2;
use Dim3::{X, Y, Z};

/// A 3D grid which is regular in x and y but non-uniform in z.
#[derive(Debug, Clone)]
pub struct HorRegularGrid3<F: num::Float> {
    coords: [Coords3<F>; 2],
    regular_z_coords: [Array1<F>; 2],
    is_periodic: In3D<bool>,
    grid_shape: In3D<usize>,
    lower_bounds: Vec3<F>,
    upper_bounds: Vec3<F>,
    extents: Vec3<F>,
    coord_to_idx_scales_xy: In2D<F>
}

impl<F> HorRegularGrid3<F>
where F: num::Float + num::cast::FromPrimitive
{
    fn compute_bounds(grid_shape: &In3D<usize>, centers: &Coords3<F>, lower_edges: &Coords3<F>) -> (Vec3<F>, Vec3<F>) {
        let lower_bounds = Vec3::new(
            lower_edges[X][0],
            lower_edges[Y][0],
            lower_edges[Z][0]
        );
        let upper_bounds = Vec3::new(
            F::from_f32(2.0).unwrap()*centers[X][grid_shape[X] - 1] - lower_edges[X][grid_shape[X] - 1],
            F::from_f32(2.0).unwrap()*centers[Y][grid_shape[Y] - 1] - lower_edges[Y][grid_shape[Y] - 1],
            F::from_f32(2.0).unwrap()*centers[Z][grid_shape[Z] - 1] - lower_edges[Z][grid_shape[Z] - 1]
        );
        (lower_bounds, upper_bounds)
    }

    fn compute_extents(lower_bounds: &Vec3<F>, upper_bounds: &Vec3<F>) -> Vec3<F> {
        Vec3::new(
            upper_bounds[X] - lower_bounds[X],
            upper_bounds[Y] - lower_bounds[Y],
            upper_bounds[Z] - lower_bounds[Z]
        )
    }

    fn compute_coord_to_idx_scales_xy(grid_shape: &In3D<usize>, lower_bounds: &Vec3<F>, upper_bounds: &Vec3<F>) -> In2D<F> {
        In2D::new(
            F::from_usize(grid_shape[X]).unwrap()/(upper_bounds[X] - lower_bounds[X]),
            F::from_usize(grid_shape[Y]).unwrap()/(upper_bounds[Y] - lower_bounds[Y])
        )
    }

    fn find_idx_with_interpolation_search(&self, point: &Point3<F>, dim: Dim3) -> Option<usize> {
        find_1d_grid_idx_with_interpolation_search(&self.coords[1][dim], point[dim])
    }
}

impl<F> Grid3<F> for HorRegularGrid3<F>
where F: num::Float + num::cast::FromPrimitive + std::fmt::Debug
{
    type XSliceGrid = HorRegularGrid2<F>;
    type YSliceGrid = HorRegularGrid2<F>;
    type ZSliceGrid = RegularGrid2<F>;

    const TYPE: GridType = GridType::HorRegular;

    fn new(centers: Coords3<F>, lower_edges: Coords3<F>, is_periodic: In3D<bool>) -> Self {

        assert!(!is_periodic[Z], "This grid type cannot be periodic in the z-direction.");

        let grid_shape = In3D::new(centers[X].len(), centers[Y].len(), centers[Z].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let extents = Self::compute_extents(&lower_bounds, &upper_bounds);
        let coord_to_idx_scales_xy = Self::compute_coord_to_idx_scales_xy(&grid_shape, &lower_bounds, &upper_bounds);

        let regular_z_lower_edges = Array::linspace(lower_edges[Z][0], lower_edges[Z][grid_shape[Z] - 1], grid_shape[Z]);
        let regular_half_dz = F::from_f32(0.5).unwrap()*(regular_z_lower_edges[1] - regular_z_lower_edges[0]);
        let regular_z_centers = Array::linspace(regular_z_lower_edges[0] + regular_half_dz, regular_z_lower_edges[grid_shape[Z] - 1] + regular_half_dz, grid_shape[Z]);

        let coords = [centers, lower_edges];
        let regular_z_coords = [regular_z_centers, regular_z_lower_edges];

        HorRegularGrid3{
            coords,
            regular_z_coords,
            is_periodic,
            grid_shape,
            lower_bounds,
            upper_bounds,
            extents,
            coord_to_idx_scales_xy
        }
    }

    fn shape(&self) -> &In3D<usize> { &self.grid_shape }
    fn is_periodic(&self, dim: Dim3) -> bool { self.is_periodic[dim] }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords3<F> { &self.coords[location as usize] }

    fn regular_centers(&self) -> CoordRefs3<F> {
        let centers = self.centers();
        CoordRefs3::new(
            &centers[X],
            &centers[Y],
            &self.regular_z_coords[0]
        )
    }

    fn regular_lower_edges(&self) -> CoordRefs3<F> {
        let lower_edges = self.lower_edges();
        CoordRefs3::new(
            &lower_edges[X],
            &lower_edges[Y],
            &self.regular_z_coords[1]
        )
    }

    fn lower_bounds(&self) -> &Vec3<F> { &self.lower_bounds }
    fn upper_bounds(&self) -> &Vec3<F> { &self.upper_bounds }
    fn extents(&self) -> &Vec3<F> { &self.extents }

    fn find_grid_cell(&self, point: &Point3<F>) -> FoundIdx3 {

        let mut crossings = In3D::new(BoundsCrossing::None, BoundsCrossing::None, BoundsCrossing::None);
        let mut is_outside = false;

        for dim in Dim3::slice().iter() {
            if point[*dim] < self.lower_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Lower;
            } else if point[*dim] >= self.upper_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Upper;
            }
        }

        if is_outside {
            FoundIdx3::Outside(crossings)
        } else {
            let i = F::to_usize(&(self.coord_to_idx_scales_xy[Dim2::X]*(point[X] - self.lower_bounds[X])).floor()).unwrap();
            let j = F::to_usize(&(self.coord_to_idx_scales_xy[Dim2::Y]*(point[Y] - self.lower_bounds[Y])).floor()).unwrap();
            let k = self.find_idx_with_interpolation_search(point, Z).unwrap();
            FoundIdx3::Inside(Idx3::new(i, j, k))
        }
    }
}

/// A 2D grid which is regular in x but non-uniform in y.
#[derive(Debug, Clone)]
pub struct HorRegularGrid2<F: num::Float> {
    coords: [Coords2<F>; 2],
    regular_y_coords: [Array1<F>; 2],
    is_periodic: In2D<bool>,
    grid_shape: In2D<usize>,
    lower_bounds: Vec2<F>,
    upper_bounds: Vec2<F>,
    extents: Vec2<F>,
    coord_to_idx_scale_x: F
}

impl<F> HorRegularGrid2<F>
where F: num::Float + num::cast::FromPrimitive
{
    fn compute_bounds(grid_shape: &In2D<usize>, centers: &Coords2<F>, lower_edges: &Coords2<F>) -> (Vec2<F>, Vec2<F>) {
        let lower_bounds = Vec2::new(
            lower_edges[Dim2::X][0],
            lower_edges[Dim2::Y][0]
        );
        let upper_bounds = Vec2::new(
            F::from_f32(2.0).unwrap()*centers[Dim2::X][grid_shape[Dim2::X] - 1] - lower_edges[Dim2::X][grid_shape[Dim2::X] - 1],
            F::from_f32(2.0).unwrap()*centers[Dim2::Y][grid_shape[Dim2::Y] - 1] - lower_edges[Dim2::Y][grid_shape[Dim2::Y] - 1]
        );
        (lower_bounds, upper_bounds)
    }

    fn compute_extents(lower_bounds: &Vec2<F>, upper_bounds: &Vec2<F>) -> Vec2<F> {
        Vec2::new(
            upper_bounds[Dim2::X] - lower_bounds[Dim2::X],
            upper_bounds[Dim2::Y] - lower_bounds[Dim2::Y]
        )
    }

    fn compute_coord_to_idx_scale_x(grid_shape: &In2D<usize>, lower_bounds: &Vec2<F>, upper_bounds: &Vec2<F>) -> F {
        F::from_usize(grid_shape[Dim2::X]).unwrap()/(upper_bounds[Dim2::X] - lower_bounds[Dim2::X])
    }

    fn find_idx_with_interpolation_search(&self, point: &Point2<F>, dim: Dim2) -> Option<usize> {
        find_1d_grid_idx_with_interpolation_search(&self.coords[1][dim], point[dim])
    }
}

impl<F> Grid2<F> for HorRegularGrid2<F>
where F: num::Float + num::cast::FromPrimitive + std::fmt::Debug
{
    const TYPE: GridType = GridType::HorRegular;

    fn new(centers: Coords2<F>, lower_edges: Coords2<F>, is_periodic: In2D<bool>) -> Self {

        assert!(!is_periodic[Dim2::Y], "This grid type cannot be periodic in the y-direction.");

        let grid_shape = In2D::new(centers[Dim2::X].len(), centers[Dim2::Y].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let extents = Self::compute_extents(&lower_bounds, &upper_bounds);
        let coord_to_idx_scale_x = Self::compute_coord_to_idx_scale_x(&grid_shape, &lower_bounds, &upper_bounds);

        let regular_y_lower_edges = Array::linspace(lower_edges[Dim2::Y][0], lower_edges[Dim2::Y][grid_shape[Dim2::Y] - 1], grid_shape[Dim2::Y]);
        let regular_half_dy = F::from_f32(0.5).unwrap()*(regular_y_lower_edges[1] - regular_y_lower_edges[0]);
        let regular_y_centers = Array::linspace(regular_y_lower_edges[0] + regular_half_dy, regular_y_lower_edges[grid_shape[Dim2::Y] - 1] + regular_half_dy, grid_shape[Dim2::Y]);

        let coords = [centers, lower_edges];
        let regular_y_coords = [regular_y_centers, regular_y_lower_edges];

        HorRegularGrid2{
            coords,
            regular_y_coords,
            is_periodic,
            grid_shape,
            lower_bounds,
            upper_bounds,
            extents,
            coord_to_idx_scale_x
        }
    }

    fn shape(&self) -> &In2D<usize> { &self.grid_shape }
    fn is_periodic(&self, dim: Dim2) -> bool { self.is_periodic[dim] }
    fn coords_by_type(&self, location: CoordLocation) -> &Coords2<F> { &self.coords[location as usize] }

    fn regular_centers(&self) -> CoordRefs2<F> {
        let centers = self.centers();
        CoordRefs2::new(
            &centers[Dim2::X],
            &self.regular_y_coords[0]
        )
    }

    fn regular_lower_edges(&self) -> CoordRefs2<F> {
        let lower_edges = self.lower_edges();
        CoordRefs2::new(
            &lower_edges[Dim2::X],
            &self.regular_y_coords[1]
        )
    }

    fn lower_bounds(&self) -> &Vec2<F> { &self.lower_bounds }
    fn upper_bounds(&self) -> &Vec2<F> { &self.upper_bounds }
    fn extents(&self) -> &Vec2<F> { &self.extents }

    fn find_grid_cell(&self, point: &Point2<F>) -> FoundIdx2 {

        let mut crossings = In2D::new(BoundsCrossing::None, BoundsCrossing::None);
        let mut is_outside = false;

        for dim in Dim2::slice().iter() {
            if point[*dim] < self.lower_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Lower;
            } else if point[*dim] >= self.upper_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Upper;
            }
        }

        if is_outside {
            FoundIdx2::Outside(crossings)
        } else {
            let i = F::to_usize(&(self.coord_to_idx_scale_x*(point[Dim2::X] - self.lower_bounds[Dim2::X])).floor()).unwrap();
            let j = self.find_idx_with_interpolation_search(point, Dim2::Y).unwrap();
            FoundIdx2::Inside(Idx2::new(i, j))
        }
    }
}

fn find_1d_grid_idx_with_interpolation_search<F>(lower_edges: &Array1<F>, point_coord: F) -> Option<usize>
where F: num::Float + num::cast::FromPrimitive
{
    let mut low = 0;
    let mut high = lower_edges.len() - 1;
    let mut mid;

    if point_coord >= lower_edges[high] {
        return Some(high)
    }

    while (point_coord >= lower_edges[low]) && (point_coord <= lower_edges[high]) {

        let low_float  = F::from_usize(low).unwrap();
        let high_float = F::from_usize(high).unwrap();
        let mid_float = (low_float + (point_coord - lower_edges[low])*(high_float - low_float)/(lower_edges[high] - lower_edges[low])).floor();

        mid = F::to_usize(&mid_float).unwrap();

        if lower_edges[mid + 1] <= point_coord {
            low = mid + 1
        } else if lower_edges[mid] > point_coord {
            high = mid
        } else {
            return Some(mid)
        }
    }
    None
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::s;

    #[test]
    fn varying_z_grid_index_search_works() {
        #![allow(clippy::deref_addrof)] // Mutes warning due to workings of s! macro
        let (mx, my, mz) = (17, 5, 29);

        let xc = Array::linspace(-1.0,  1.0, mx);
        let yc = Array::linspace( 1.0,  5.2, my);

        let (dx, dy) = (xc[1] - xc[0], yc[1] - yc[0]);

        let xdn = Array::linspace(xc[0] - dx/2.0, xc[mx-1] - dx/2.0, mx);
        let ydn = Array::linspace(yc[0] - dy/2.0, yc[my-1] - dy/2.0, my);

        let zdn = Array::linspace(-2.0, 2.0, mz+1) + Array::linspace(1.0, 2.0, mz+1).mapv(|a| a*a*a*a);
        let zc = (zdn.slice(s![1..]).into_owned() + zdn.slice(s![..mz]))*0.5;
        let zdn = zdn.slice(s![..mz]).into_owned();

        let z_max = 2.0*zc[mz-1] - zdn[mz-1];

        let centers = Coords3::new(xc.clone(), yc.clone(), zc.clone());
        let lower_edges = Coords3::new(xdn.clone(), ydn.clone(), zdn.clone());

        let grid = HorRegularGrid3::new(centers, lower_edges, In3D::new(false, false, false));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[mx-1] + dx + 1e-12, ydn[my-1] + dy + 1e-12, z_max + 1e-12)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::Upper, BoundsCrossing::Upper, BoundsCrossing::Upper)));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)),
                   FoundIdx3::Inside(Idx3::new(0, 0, 0)));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] - 1e-9)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::None, BoundsCrossing::None, BoundsCrossing::Lower)));

        assert_eq!(grid.find_grid_cell(&Point3::new(-0.68751, 1.5249, 3.0)),
                   FoundIdx3::Inside(Idx3::new(2, 0, 10)));

        assert_eq!(grid.find_grid_cell(&Point3::new(0.0, 2.0, 16.7)),
                   FoundIdx3::Inside(Idx3::new(8, 1, 27)));

        assert_eq!(grid.find_grid_cell(&Point3::new(0.0, 2.0, -0.7)),
                   FoundIdx3::Inside(Idx3::new(8, 1, 1)));
    }
}
