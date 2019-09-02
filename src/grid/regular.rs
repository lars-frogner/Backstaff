//! Grid with uniform spacing in all dimensions.

use num;
use crate::geometry::{Dim3, Dim2, In3D, In2D, Vec3, Vec2, Point3, Point2, Idx3, Idx2, Coords3, Coords2, CoordRefs3, CoordRefs2};
use super::{BoundsCrossing, FoundIdx3, FoundIdx2, CoordLocation, GridType, Grid3, Grid2};
use Dim3::{X, Y, Z};

/// A regular 3D grid.
#[derive(Debug, Clone)]
pub struct RegularGrid3<F: num::Float> {
    coords: [Coords3<F>; 2],
    is_periodic: In3D<bool>,
    grid_shape: In3D<usize>,
    lower_bounds: Vec3<F>,
    upper_bounds: Vec3<F>,
    extents: Vec3<F>,
    coord_to_idx_scales: In3D<F>
}

impl<F> RegularGrid3<F>
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

    fn compute_coord_to_idx_scales(grid_shape: &In3D<usize>, lower_bounds: &Vec3<F>, upper_bounds: &Vec3<F>) -> In3D<F> {
        In3D::new(
            F::from_usize(grid_shape[X]).unwrap()/(upper_bounds[X] - lower_bounds[X]),
            F::from_usize(grid_shape[Y]).unwrap()/(upper_bounds[Y] - lower_bounds[Y]),
            F::from_usize(grid_shape[Z]).unwrap()/(upper_bounds[Z] - lower_bounds[Z])
        )
    }
}

impl<F> Grid3<F> for RegularGrid3<F>
where F: num::Float + num::cast::FromPrimitive
{
    type XSliceGrid = RegularGrid2<F>;
    type YSliceGrid = RegularGrid2<F>;
    type ZSliceGrid = RegularGrid2<F>;

    const TYPE: GridType = GridType::Regular;

    fn new(centers: Coords3<F>, lower_edges: Coords3<F>, is_periodic: In3D<bool>) -> Self {

        let grid_shape = In3D::new(centers[X].len(), centers[Y].len(), centers[Z].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let extents = Self::compute_extents(&lower_bounds, &upper_bounds);
        let coord_to_idx_scales = Self::compute_coord_to_idx_scales(&grid_shape, &lower_bounds, &upper_bounds);

        let coords = [centers, lower_edges];

        RegularGrid3{
            coords,
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
            FoundIdx3::Inside(Idx3::new(
                F::to_usize(&(self.coord_to_idx_scales[X]*(point[X] - self.lower_bounds[X])).floor()).unwrap(),
                F::to_usize(&(self.coord_to_idx_scales[Y]*(point[Y] - self.lower_bounds[Y])).floor()).unwrap(),
                F::to_usize(&(self.coord_to_idx_scales[Z]*(point[Z] - self.lower_bounds[Z])).floor()).unwrap()
            ))
        }
    }
}

/// A regular 2D grid.
#[derive(Debug, Clone)]
pub struct RegularGrid2<F: num::Float> {
    coords: [Coords2<F>; 2],
    is_periodic: In2D<bool>,
    grid_shape: In2D<usize>,
    lower_bounds: Vec2<F>,
    upper_bounds: Vec2<F>,
    extents: Vec2<F>,
    coord_to_idx_scales: In2D<F>
}

impl<F> RegularGrid2<F>
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

    fn compute_coord_to_idx_scales(grid_shape: &In2D<usize>, lower_bounds: &Vec2<F>, upper_bounds: &Vec2<F>) -> In2D<F> {
        In2D::new(
            F::from_usize(grid_shape[Dim2::X]).unwrap()/(upper_bounds[Dim2::X] - lower_bounds[Dim2::X]),
            F::from_usize(grid_shape[Dim2::Y]).unwrap()/(upper_bounds[Dim2::Y] - lower_bounds[Dim2::Y])
        )
    }
}

impl<F> Grid2<F> for RegularGrid2<F>
where F: num::Float + num::cast::FromPrimitive
{
    const TYPE: GridType = GridType::Regular;

    fn new(centers: Coords2<F>, lower_edges: Coords2<F>, is_periodic: In2D<bool>) -> Self {

        let grid_shape = In2D::new(centers[Dim2::X].len(), centers[Dim2::Y].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let extents = Self::compute_extents(&lower_bounds, &upper_bounds);
        let coord_to_idx_scales = Self::compute_coord_to_idx_scales(&grid_shape, &lower_bounds, &upper_bounds);

        let coords = [centers, lower_edges];

        RegularGrid2{
            coords,
            is_periodic,
            grid_shape,
            lower_bounds,
            upper_bounds,
            extents,
            coord_to_idx_scales
        }
    }

    fn shape(&self) -> &In2D<usize> { &self.grid_shape }
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
            FoundIdx2::Inside(Idx2::new(
                F::to_usize(&(self.coord_to_idx_scales[Dim2::X]*(point[Dim2::X] - self.lower_bounds[Dim2::X])).floor()).unwrap(),
                F::to_usize(&(self.coord_to_idx_scales[Dim2::Y]*(point[Dim2::Y] - self.lower_bounds[Dim2::Y])).floor()).unwrap()
            ))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::prelude::*;

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

        let centers = Coords3::new(xc.clone(), yc.clone(), zc.clone());
        let lower_edges = Coords3::new(xdn.clone(), ydn.clone(), zdn.clone());

        let grid = RegularGrid3::new(centers, lower_edges, In3D::new(false, false, false));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[mx-1] + dx + 1e-12, ydn[my-1] + dy + 1e-12, zdn[mz-1] + dz + 1e-12)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::Upper, BoundsCrossing::Upper, BoundsCrossing::Upper)));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)),
                   FoundIdx3::Inside(Idx3::new(0, 0, 0)));

        assert_eq!(grid.find_grid_cell(&Point3::new(xdn[0] + 1e-12, ydn[0] - 1e-9, zdn[0] + 1e-12)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::None, BoundsCrossing::Lower, BoundsCrossing::None)));

        assert_eq!(grid.find_grid_cell(&Point3::new(-0.68751, 1.5249, 7.5)),
                   FoundIdx3::Inside(Idx3::new(2, 0, 25)));
    }
}
