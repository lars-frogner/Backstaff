//! Geometric utility objects.

use num;
use ndarray::prelude::*;

/// Denotes the x- y- or z-dimension.
#[derive(Debug, Copy, Clone)]
pub enum Dim {
    X = 0,
    Y = 1,
    Z = 2
}

impl Dim {
    /// Creates an array for iterating over the x- y- and z-dimensions.
    pub fn xyz_slice() -> [Self; 3] { [Dim::X, Dim::Y, Dim::Z] }

    /// Creates an array for iterating over the x- and y-dimensions.
    pub fn xy_slice() -> [Self; 2] { [Dim::X, Dim::Y] }
}

use self::Dim::{X, Y, Z};

/// Represents any quantity with three dimensional components.
#[derive(Debug, Clone, PartialEq)]
pub struct In3D<T>([T; 3]);

impl<T> In3D<T> {
    /// Creates a new 3D quantity given the three components.
    pub fn new(x: T, y: T, z: T) -> Self { In3D([x, y, z]) }
}

impl<T> std::ops::Index<Dim> for In3D<T> {
    type Output = T;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}

impl<T> std::ops::IndexMut<Dim> for In3D<T> {
    fn index_mut(&mut self, dim: Dim) -> &mut Self::Output { &mut self.0[dim as usize] }
}

/// Represents any quantity with two dimensional components.
#[derive(Debug, Clone, PartialEq)]
pub struct In2D<T>([T; 2]);

impl<T> In2D<T> {
    /// Creates a new 2D quantity given the two components.
    pub fn new(x: T, y: T) -> Self { In2D([x, y]) }
}

impl<T> std::ops::Index<Dim> for In2D<T> {
    type Output = T;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}

impl<T> std::ops::IndexMut<Dim> for In2D<T> {
    fn index_mut(&mut self, dim: Dim) -> &mut Self::Output { &mut self.0[dim as usize] }
}

/// A 3D coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct Coord3<T: num::Float>([T; 3]);

impl<T: num::Float> Coord3<T> {
    /// Creates a new 3D coordinate given the three components.
    pub fn new(x: T, y: T, z: T) -> Self { Coord3([x, y, z]) }
}

impl<T: num::Float> std::ops::Index<Dim> for Coord3<T> {
    type Output = T;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}

impl<T: num::Float> std::ops::IndexMut<Dim> for Coord3<T> {
    fn index_mut(&mut self, dim: Dim) -> &mut Self::Output { &mut self.0[dim as usize] }
}

impl<T: num::Float, > std::ops::Add<Coord3<T>> for Coord3<T> {
    type Output = Self;
    fn add(self, other: Coord3<T>) -> Self {
        Self([self.0[0] + other.0[0],
              self.0[1] + other.0[1],
              self.0[2] + other.0[2]])
    }
}

impl<T: num::Float> std::ops::Sub<Coord3<T>> for Coord3<T> {
    type Output = Self;
    fn sub(self, other: Coord3<T>) -> Self {
        Self([self.0[0] - other.0[0],
              self.0[1] - other.0[1],
              self.0[2] - other.0[2]])
    }
}

/// A 3D index.
#[derive(Debug, Clone, PartialEq)]
pub struct Idx3<T: num::Integer>([T; 3]);

impl<T: num::Integer> Idx3<T> {
    /// Creates a new 3D index given the three components.
    pub fn new(i: T, j: T, k: T) -> Self { Idx3([i, j, k]) }
}

impl<T: num::Integer> std::ops::Index<Dim> for Idx3<T> {
    type Output = T;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}

impl<T: num::Integer> std::ops::IndexMut<Dim> for Idx3<T> {
    fn index_mut(&mut self, dim: Dim) -> &mut Self::Output { &mut self.0[dim as usize] }
}

/// A 3D vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec3<T: num::Float>([T; 3]);

impl<T: num::Float> Vec3<T> {
    /// Creates a new 3D coordinate given the three components.
    pub fn new(x: T, y: T, z: T) -> Self { Vec3([x, y, z]) }
}

impl<T: num::Float> std::ops::Index<Dim> for Vec3<T> {
    type Output = T;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}

impl<T: num::Float> std::ops::IndexMut<Dim> for Vec3<T> {
    fn index_mut(&mut self, dim: Dim) -> &mut Self::Output { &mut self.0[dim as usize] }
}

impl<T: num::Float> std::ops::Add<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn add(self, other: Vec3<T>) -> Self {
        Self([self.0[0] + other.0[0],
              self.0[1] + other.0[1],
              self.0[2] + other.0[2]])
    }
}

impl<T: num::Float> std::ops::Sub<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn sub(self, other: Vec3<T>) -> Self {
        Self([self.0[0] - other.0[0],
              self.0[1] - other.0[1],
              self.0[2] - other.0[2]])
    }
}

/// 3D spatial coordinate arrays.
#[derive(Debug, Clone)]
pub struct Coords3<T: num::Float>([Array1<T>; 3]);

impl<T: num::Float> Coords3<T> {
    /// Creates a new 3D set of coordinates given the component arrays.
    pub fn new(x: Array1<T>, y: Array1<T>, z: Array1<T>) -> Self { Coords3([x, y, z]) }
}

impl<T: num::Float> std::ops::Index<Dim> for Coords3<T> {
    type Output = Array1<T>;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}

/// References to 3D spatial coordinate arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordRefs3<'a, T: num::Float>([&'a Array1<T>; 3]);

impl<'a, T: num::Float> CoordRefs3<'a, T> {
    /// Creates a new 3D set of coordinate references given references to the component arrays.
    pub fn new(x: &'a Array1<T>, y: &'a Array1<T>, z: &'a Array1<T>) -> Self { CoordRefs3([x, y, z]) }
}

impl<'a, T: num::Float> std::ops::Index<Dim> for CoordRefs3<'a, T> {
    type Output = &'a Array1<T>;
    fn index(&self, dim: Dim) -> &Self::Output { &self.0[dim as usize] }
}
