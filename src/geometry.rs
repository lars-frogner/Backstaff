//! Geometric utility objects.

use num;
use ndarray::prelude::*;

/// Denotes the x- y- or z-dimension.
#[derive(Debug, Copy, Clone)]
pub enum Dim3 {
    X = 0,
    Y = 1,
    Z = 2
}

impl Dim3 {
    /// Creates an array for iterating over the x- y- and z-dimensions.
    pub fn slice() -> [Self; 3] { [Dim3::X, Dim3::Y, Dim3::Z] }

    /// Creates an array for iterating over the x- and y-dimensions.
    pub fn slice_xy() -> [Self; 2] { [Dim3::X, Dim3::Y] }

    /// Creates an array for iterating over all three dimensions except the given one.
    pub fn slice_except(dim: Self) -> [Self; 2] {
        match dim {
            Dim3::X => [Dim3::Y, Dim3::Z],
            Dim3::Y => [Dim3::X, Dim3::Z],
            Dim3::Z => [Dim3::X, Dim3::Y]
        }
    }
}

use Dim3::{X, Y, Z};

/// Represents any quantity with three dimensional components.
#[derive(Debug, Clone, PartialEq)]
pub struct In3D<T>([T; 3]);

impl<T> In3D<T> {
    /// Creates a new 3D quantity given the three components.
    pub fn new(x: T, y: T, z: T) -> Self { In3D([x, y, z]) }
}

impl<T> std::ops::Index<Dim3> for In3D<T> {
    type Output = T;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim as usize] }
}

impl<T> std::ops::IndexMut<Dim3> for In3D<T> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output { &mut self.0[dim as usize] }
}

/// Represents any quantity with two dimensional components.
#[derive(Debug, Clone, PartialEq)]
pub struct In2D<T>([T; 2]);

impl<T> In2D<T> {
    /// Creates a new 2D quantity given the two components.
    pub fn new(x: T, y: T) -> Self { In2D([x, y]) }
}

impl<T> std::ops::Index<Dim3> for In2D<T> {
    type Output = T;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim as usize] }
}

impl<T> std::ops::IndexMut<Dim3> for In2D<T> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output { &mut self.0[dim as usize] }
}

/// A 3D vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec3<T: num::Float>(In3D<T>);

impl<T: num::Float> Vec3<T> {
    /// Creates a new 3D vector given the three components.
    pub fn new(x: T, y: T, z: T) -> Self { Vec3(In3D::new(x, y, z)) }

    /// Computes the dot product of the vector with another vector.
    pub fn dot(&self, other: &Self) -> T {
        self[X]*other[X] +
        self[Y]*other[Y] +
        self[Z]*other[Z]
    }
}

impl<T: num::Float> std::ops::Index<Dim3> for Vec3<T> {
    type Output = T;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim] }
}

impl<T: num::Float> std::ops::IndexMut<Dim3> for Vec3<T> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output { &mut self.0[dim] }
}

impl<T: num::Float> std::ops::Add<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Vec3::new(self[X] + other[X],
                  self[Y] + other[Y],
                  self[Z] + other[Z])
    }
}

impl<T: num::Float> std::ops::Sub<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Vec3::new(self[X] - other[X],
                  self[Y] - other[Y],
                  self[Z] - other[Z])
    }
}

impl<T: num::Float> std::ops::Mul<T> for Vec3<T> {
    type Output = Self;
    fn mul(self, factor: T) -> Self {
        Vec3::new(factor*self[X],
                  factor*self[Y],
                  factor*self[Z])
    }
}

/// A 3D spatial coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3<T: num::Float>(In3D<T>);

impl<T: num::Float> Point3<T> {
    /// Creates a new 3D point given the three components.
    pub fn new(x: T, y: T, z: T) -> Self { Point3(In3D::new(x, y, z)) }

    /// Creates a new 3D point with all components set to zero.
    pub fn origin() -> Self { Self::new(T::zero(), T::zero(), T::zero()) }
}

impl<T: num::Float> std::ops::Index<Dim3> for Point3<T> {
    type Output = T;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim] }
}

impl<T: num::Float> std::ops::IndexMut<Dim3> for Point3<T> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output { &mut self.0[dim] }
}

impl<T: num::Float> std::ops::Add<Vec3<T>> for Point3<T> {
    type Output = Self;
    fn add(self, vector: Vec3<T>) -> Self {
        Point3::new(self[X] + vector[X],
                    self[Y] + vector[Y],
                    self[Z] + vector[Z])
    }
}

impl<T: num::Float> std::ops::Sub<Point3<T>> for Point3<T> {
    type Output = Vec3<T>;
    fn sub(self, other: Self) -> Vec3<T> {
        Vec3::new(self[X] - other[X],
                  self[Y] - other[Y],
                  self[Z] - other[Z])
    }
}

/// A 3D index.
#[derive(Debug, Clone, PartialEq)]
pub struct Idx3<T: num::Integer>(In3D<T>);

impl<T: num::Integer> Idx3<T> {
    /// Creates a new 3D index given the three components.
    pub fn new(i: T, j: T, k: T) -> Self { Idx3(In3D::new(i, j, k)) }
}

impl<T: num::Integer> std::ops::Index<Dim3> for Idx3<T> {
    type Output = T;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim] }
}

impl<T: num::Integer> std::ops::IndexMut<Dim3> for Idx3<T> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output { &mut self.0[dim] }
}

/// 3D spatial coordinate arrays.
#[derive(Debug, Clone)]
pub struct Coords3<T: num::Float>(In3D<Array1<T>>);

impl<T: num::Float> Coords3<T> {
    /// Creates a new 3D set of coordinates given the component 1D coordinates.
    pub fn new(x: Array1<T>, y: Array1<T>, z: Array1<T>) -> Self {
        Coords3(In3D::new(x, y, z))
    }

    /// Creates a 3D point from the coordinates at the given index.
    pub fn point(&self, idx: usize) -> Point3<T> {
        Point3::new(self[X][idx], self[Y][idx], self[Y][idx])
    }
}

impl<T: num::Float> std::ops::Index<Dim3> for Coords3<T> {
    type Output = Array1<T>;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim] }
}

/// References to 3D spatial coordinate arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordRefs3<'a, T: num::Float>(In3D<&'a Array1<T>>);

impl<'a, T: num::Float> CoordRefs3<'a, T> {
    /// Creates a new 3D set of coordinate references given references to the component arrays.
    pub fn new(x: &'a Array1<T>, y: &'a Array1<T>, z: &'a Array1<T>) -> Self {
        CoordRefs3(In3D::new(x, y, z))
    }

    /// Creates a 3D point from the coordinates at the given index.
    pub fn point(&self, idx: usize) -> Point3<T> {
        Point3::new(self[X][idx], self[Y][idx], self[Y][idx])
    }
}

impl<'a, T: num::Float> std::ops::Index<Dim3> for CoordRefs3<'a, T> {
    type Output = &'a Array1<T>;
    fn index(&self, dim: Dim3) -> &Self::Output { &self.0[dim] }
}
