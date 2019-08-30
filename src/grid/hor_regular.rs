//! Grid with uniform spacing in the horizontal dimensions.

use num;
use ndarray::prelude::*;
use super::{BoundsCrossing, FoundIdx3, CoordsType, Grid3Type, Grid3};
use crate::geometry::{Dim3, In3D, In2D, Vec3, Point3, Idx3, Coords3, CoordRefs3};
use Dim3::{X, Y, Z};

/// A 3D grid which is regular in x and y but non-uniform in z.
#[derive(Debug, Clone)]
pub struct HorRegularGrid3<F: num::Float> {
    coords: [Coords3<F>; 2],
    uniform_z_coords: Array1<F>,
    is_periodic: In3D<bool>,
    grid_shape: In3D<usize>,
    lower_bounds: Vec3<F>,
    upper_bounds: Vec3<F>,
    extents: Vec3<F>,
    coord_to_idx_scales: In2D<F>
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

    fn compute_coord_to_idx_scales(grid_shape: &In3D<usize>, lower_bounds: &Vec3<F>, upper_bounds: &Vec3<F>) -> In2D<F> {
        In2D::new(
            F::from_usize(grid_shape[X]).unwrap()/(upper_bounds[X] - lower_bounds[X]),
            F::from_usize(grid_shape[Y]).unwrap()/(upper_bounds[Y] - lower_bounds[Y])
        )
    }

    fn find_idx_with_interpolation_search(&self, point: &Point3<F>, dim: Dim3) -> Option<usize> {

        let lower_edges = &self.coords[1][dim];
        let c = point[dim];

        let mut low = 0;
        let mut high = self.grid_shape[dim] - 1;
        let mut mid;

        if c >= lower_edges[high] {
            return Some(high)
        }

        while (c >= lower_edges[low]) && (c <= lower_edges[high]) {

            let low_float  = F::from_usize(low).unwrap();
            let high_float = F::from_usize(high).unwrap();
            let mid_float = (low_float + (c - lower_edges[low])*(high_float - low_float)/(lower_edges[high] - lower_edges[low])).floor();

            mid = F::to_usize(&mid_float).unwrap();

            if lower_edges[mid + 1] <= c {
                low = mid + 1
            } else if lower_edges[mid] > c {
                high = mid
            } else {
                return Some(mid)
            }
        }

        None
    }
}

impl<F> Grid3<F> for HorRegularGrid3<F>
where F: num::Float + num::cast::FromPrimitive + std::fmt::Debug
{
    const TYPE: Grid3Type = Grid3Type::HorRegular;

    fn new(centers: Coords3<F>, lower_edges: Coords3<F>, is_periodic: In3D<bool>) -> Self {

        assert!(!is_periodic[Z], "This grid type cannot be periodic in the z-direction.");

        let grid_shape = In3D::new(centers[X].len(), centers[Y].len(), centers[Z].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let extents = Self::compute_extents(&lower_bounds, &upper_bounds);
        let coord_to_idx_scales = Self::compute_coord_to_idx_scales(&grid_shape, &lower_bounds, &upper_bounds);

        let uniform_z_coords = Array::linspace(centers[Z][0], centers[Z][grid_shape[Z] - 1], grid_shape[Z]);

        let coords = [centers, lower_edges];

        HorRegularGrid3{
            coords,
            uniform_z_coords,
            is_periodic,
            grid_shape,
            lower_bounds,
            upper_bounds,
            extents,
            coord_to_idx_scales
        }
    }

    fn shape(&self) -> &In3D<usize> { &self.grid_shape }
    fn is_periodic(&self, dim: Dim3) -> bool { self.is_periodic[dim] }
    fn coords_by_type(&self, coord_type: CoordsType) -> &Coords3<F> { &self.coords[coord_type as usize] }

    fn uniform_centers(&self) -> CoordRefs3<F> {
        let centers = self.centers();
        CoordRefs3::new(
            &centers[X],
            &centers[Y],
            &self.uniform_z_coords
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
            let i = F::to_usize(&(self.coord_to_idx_scales[X]*(point[X] - self.lower_bounds[X])).floor()).unwrap();
            let j = F::to_usize(&(self.coord_to_idx_scales[Y]*(point[Y] - self.lower_bounds[Y])).floor()).unwrap();
            let k = self.find_idx_with_interpolation_search(point, Z).unwrap();
            FoundIdx3::Inside(Idx3::new(i, j, k))
        }
    }

    fn wrap_point(&self, point: &Point3<F>) -> Option<Point3<F>> {
        let mut wrapped_point = point.clone();
        for dim in Dim3::slice().iter() {
            if self.is_periodic[*dim] {
                if wrapped_point[*dim] < self.lower_bounds[*dim] {
                    wrapped_point[*dim] = self.upper_bounds[*dim] - ((self.upper_bounds[*dim] - point[*dim]) % self.extents[*dim]);
                } else if wrapped_point[*dim] >= self.upper_bounds[*dim] {
                    wrapped_point[*dim] = self.lower_bounds[*dim] + ((point[*dim] - self.lower_bounds[*dim]) % self.extents[*dim]);
                }
            } else if wrapped_point[*dim] < self.lower_bounds[*dim] || wrapped_point[*dim] >= self.upper_bounds[*dim] {
                return None
            }
        }
        Some(wrapped_point)
    }
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
