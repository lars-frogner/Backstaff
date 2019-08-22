//! Grids used in Bifrost.

use num;
use crate::geometry::{Dim, In3D, Coord3, Idx3, Coords3};

/// A potential crossing of the lower or upper bounds of a grid dimension.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BoundsCrossing {
    None,
    Upper,
    Lower
}

/// A found 3D index inside the grid or a bounds crossing for each dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum FoundIdx3<T: num::Integer> {
    Inside(Idx3<T>),
    Outside(In3D<BoundsCrossing>)
}

/// Coordinates located at center or lower edge of grid cell.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CoordsType {
    Center = 0,
    Lower = 1
}

/// Regular grid or non-uniform in z-direction.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Grid3Type {
    Regular,
    VaryingZ
}

/// Defines the properties of a 3D grid.
pub trait Grid3<T: num::Float> {

    /// The specific type of the grid.
    const TYPE: Grid3Type;

    /// Creates a new grid given the coordinates of the cell centers and lower edges.
    fn new(center_coords: Coords3<T>, lower_edge_coords: Coords3<T>) -> Self;

    /// Returns the 3D shape of the grid.
    fn shape(&self) -> &In3D<usize>;

    /// Returns a reference to either the central or lower coordinates depending on the given type value.
    fn coords_by_type(&self, coord_type: CoordsType) -> &Coords3<T>;

    /// Returns a reference to the central coordinates.
    fn centers(&self) -> &Coords3<T> { self.coords_by_type(CoordsType::Center) }

    /// Returns a reference to the lower coordinates.
    fn lower_edges(&self) -> &Coords3<T> { self.coords_by_type(CoordsType::Lower) }

    /// Finds the 3D index of the grid cell containing the given coordinate,
    /// or specifies on which side of the grid the coordinate lies if outside.
    fn find_grid_cell(&self, coord: &Coord3<T>) -> FoundIdx3<usize>;
}

/// A regular 3D grid.
#[derive(Debug, Clone)]
pub struct Grid3Regular<T: num::Float> {
    coords: [Coords3<T>; 2],
    grid_shape: In3D<usize>,
    lower_bounds: In3D<T>,
    upper_bounds: In3D<T>,
    coord_to_idx_scales: In3D<T>
}

impl<T> Grid3Regular<T>
where T: num::Float + num::cast::FromPrimitive
{
    fn compute_bounds(grid_shape: &In3D<usize>, centers: &Coords3<T>, lower_edges: &Coords3<T>) -> (In3D<T>, In3D<T>) {
        let lower_bounds = In3D::new(
            lower_edges[X][0],
            lower_edges[Y][0],
            lower_edges[Z][0]
        );
        let upper_bounds = In3D::new(
            T::from_f32(2.0).unwrap()*centers[X][grid_shape[X] - 1] - lower_edges[X][grid_shape[X] - 1],
            T::from_f32(2.0).unwrap()*centers[Y][grid_shape[Y] - 1] - lower_edges[Y][grid_shape[Y] - 1],
            T::from_f32(2.0).unwrap()*centers[Z][grid_shape[Z] - 1] - lower_edges[Z][grid_shape[Z] - 1]
        );
        (lower_bounds, upper_bounds)
    }

    fn compute_coord_to_idx_scales(grid_shape: &In3D<usize>, lower_bounds: &In3D<T>, upper_bounds: &In3D<T>) -> In3D<T> {
        In3D::new(
            T::from_usize(grid_shape[X]).unwrap()/(upper_bounds[X] - lower_bounds[X]),
            T::from_usize(grid_shape[Y]).unwrap()/(upper_bounds[Y] - lower_bounds[Y]),
            T::from_usize(grid_shape[Z]).unwrap()/(upper_bounds[Z] - lower_bounds[Z])
        )
    }
}

impl<T> Grid3<T> for Grid3Regular<T>
where T: num::Float + num::cast::FromPrimitive
{
    const TYPE: Grid3Type = Grid3Type::Regular;

    fn new(centers: Coords3<T>, lower_edges: Coords3<T>) -> Self {

        let grid_shape = In3D::new(centers[X].len(), centers[Y].len(), centers[Z].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let coord_to_idx_scales = Self::compute_coord_to_idx_scales(&grid_shape, &lower_bounds, &upper_bounds);

        let coords = [centers, lower_edges];

        Grid3Regular{
            coords,
            grid_shape,
            lower_bounds,
            upper_bounds,
            coord_to_idx_scales
        }
    }

    fn shape(&self) -> &In3D<usize> { &self.grid_shape }
    fn coords_by_type(&self, coord_type: CoordsType) -> &Coords3<T> { &self.coords[coord_type as usize] }

    fn find_grid_cell(&self, coord: &Coord3<T>) -> FoundIdx3<usize> {

        let mut crossings = In3D::new(BoundsCrossing::None, BoundsCrossing::None, BoundsCrossing::None);
        let mut is_outside = false;

        for dim in Dim::xyz_slice().iter() {
            if coord[*dim] < self.lower_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Lower;
            } else if coord[*dim] >= self.upper_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Upper;
            }
        }

        if is_outside {
            FoundIdx3::Outside(crossings)
        } else {
            FoundIdx3::Inside(Idx3::new(
                T::to_usize(&(self.coord_to_idx_scales[X]*(coord[X] - self.lower_bounds[X])).floor()).unwrap(),
                T::to_usize(&(self.coord_to_idx_scales[Y]*(coord[Y] - self.lower_bounds[Y])).floor()).unwrap(),
                T::to_usize(&(self.coord_to_idx_scales[Z]*(coord[Z] - self.lower_bounds[Z])).floor()).unwrap()
            ))
        }
    }
}

/// A 3D grid which is regular in x and y but non-uniform in z.
#[derive(Debug, Clone)]
pub struct Grid3VaryingZ<T: num::Float> {
    coords: [Coords3<T>; 2],
    grid_shape: In3D<usize>,
    lower_bounds: In3D<T>,
    upper_bounds: In3D<T>,
    coord_to_idx_scales: In2D<T>
}

impl<T> Grid3VaryingZ<T>
where T: num::Float + num::cast::FromPrimitive
{
    fn compute_bounds(grid_shape: &In3D<usize>, centers: &Coords3<T>, lower_edges: &Coords3<T>) -> (In3D<T>, In3D<T>) {
        let lower_bounds = In3D::new(
            lower_edges[X][0],
            lower_edges[Y][0],
            lower_edges[Z][0]
        );
        let upper_bounds = In3D::new(
            T::from_f32(2.0).unwrap()*centers[X][grid_shape[X] - 1] - lower_edges[X][grid_shape[X] - 1],
            T::from_f32(2.0).unwrap()*centers[Y][grid_shape[Y] - 1] - lower_edges[Y][grid_shape[Y] - 1],
            T::from_f32(2.0).unwrap()*centers[Z][grid_shape[Z] - 1] - lower_edges[Z][grid_shape[Z] - 1]
        );
        (lower_bounds, upper_bounds)
    }

    fn compute_coord_to_idx_scales(grid_shape: &In3D<usize>, lower_bounds: &In3D<T>, upper_bounds: &In3D<T>) -> In2D<T> {
        In2D::new(
            T::from_usize(grid_shape[X]).unwrap()/(upper_bounds[X] - lower_bounds[X]),
            T::from_usize(grid_shape[Y]).unwrap()/(upper_bounds[Y] - lower_bounds[Y])
        )
    }

    fn find_idx_with_interpolation_search(&self, coord: &Coord3<T>, dim: Dim) -> Option<usize> {

        let lower_edges = &self.coords[1][dim];
        let c = coord[dim];

        let mut low = 0;
        let mut high = self.grid_shape[dim] - 1;
        let mut mid;

        if c >= lower_edges[high] {
            return Some(high)
        }

        while (c >= lower_edges[low]) && (c <= lower_edges[high]) {

            let low_float  = T::from_usize(low).unwrap();
            let high_float = T::from_usize(high).unwrap();
            let mid_float = (low_float + (c - lower_edges[low])*(high_float - low_float)/(lower_edges[high] - lower_edges[low])).floor();

            mid = T::to_usize(&mid_float).unwrap();

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

impl<T> Grid3<T> for Grid3VaryingZ<T>
where T: num::Float + num::cast::FromPrimitive + std::fmt::Debug
{
    const TYPE: Grid3Type = Grid3Type::VaryingZ;

    fn new(centers: Coords3<T>, lower_edges: Coords3<T>) -> Self {

        let grid_shape = In3D::new(centers[X].len(), centers[Y].len(), centers[Z].len());

        let (lower_bounds, upper_bounds) = Self::compute_bounds(&grid_shape, &centers, &lower_edges);
        let coord_to_idx_scales = Self::compute_coord_to_idx_scales(&grid_shape, &lower_bounds, &upper_bounds);

        let coords = [centers, lower_edges];

        Grid3VaryingZ{
            coords,
            grid_shape,
            lower_bounds,
            upper_bounds,
            coord_to_idx_scales
        }
    }

    fn shape(&self) -> &In3D<usize> { &self.grid_shape }
    fn coords_by_type(&self, coord_type: CoordsType) -> &Coords3<T> { &self.coords[coord_type as usize] }

    fn find_grid_cell(&self, coord: &Coord3<T>) -> FoundIdx3<usize> {

        let mut crossings = In3D::new(BoundsCrossing::None, BoundsCrossing::None, BoundsCrossing::None);
        let mut is_outside = false;

        for dim in Dim::xyz_slice().iter() {
            if coord[*dim] < self.lower_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Lower;
            } else if coord[*dim] >= self.upper_bounds[*dim] {
                is_outside = true;
                crossings[*dim] = BoundsCrossing::Upper;
            }
        }

        if is_outside {
            FoundIdx3::Outside(crossings)
        } else {
            let i = T::to_usize(&(self.coord_to_idx_scales[X]*(coord[X] - self.lower_bounds[X])).floor()).unwrap();
            let j = T::to_usize(&(self.coord_to_idx_scales[Y]*(coord[Y] - self.lower_bounds[Y])).floor()).unwrap();
            let k = self.find_idx_with_interpolation_search(coord, Z).unwrap();
            FoundIdx3::Inside(Idx3::new(i, j, k))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::prelude::*;
    use ndarray::s;

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

        let grid = Grid3Regular::new(centers, lower_edges);

        assert_eq!(grid.find_grid_cell(&Coord3::new(xdn[mx-1] + dx + 1e-12, ydn[my-1] + dy + 1e-12, zdn[mz-1] + dz + 1e-12)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::Upper, BoundsCrossing::Upper, BoundsCrossing::Upper)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)),
                   FoundIdx3::Inside(Idx3::new(0, 0, 0)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(xdn[0] + 1e-12, ydn[0] - 1e-9, zdn[0] + 1e-12)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::None, BoundsCrossing::Lower, BoundsCrossing::None)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(-0.68751, 1.5249, 7.5)),
                   FoundIdx3::Inside(Idx3::new(2, 0, 25)));
    }

    #[test]
    fn varying_z_grid_index_search_works() {

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

        let grid = Grid3VaryingZ::new(centers, lower_edges);

        assert_eq!(grid.find_grid_cell(&Coord3::new(xdn[mx-1] + dx + 1e-12, ydn[my-1] + dy + 1e-12, z_max + 1e-12)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::Upper, BoundsCrossing::Upper, BoundsCrossing::Upper)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] + 1e-12)),
                   FoundIdx3::Inside(Idx3::new(0, 0, 0)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(xdn[0] + 1e-12, ydn[0] + 1e-12, zdn[0] - 1e-9)),
                   FoundIdx3::Outside(In3D::new(BoundsCrossing::None, BoundsCrossing::None, BoundsCrossing::Lower)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(-0.68751, 1.5249, 3.0)),
                   FoundIdx3::Inside(Idx3::new(2, 0, 10)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(0.0, 2.0, 16.7)),
                   FoundIdx3::Inside(Idx3::new(8, 1, 27)));

        assert_eq!(grid.find_grid_cell(&Coord3::new(0.0, 2.0, -0.7)),
                   FoundIdx3::Inside(Idx3::new(8, 1, 1)));
    }
}
