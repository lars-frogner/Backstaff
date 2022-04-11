//! Geometric utility objects.

use crate::num::BFloat;
use num;
use std::{
    fmt,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

#[cfg(feature = "serialization")]
use serde::Serialize;

/// Denotes the x-, y- or z-dimension.
#[derive(Clone, Copy, Debug)]
pub enum Dim3 {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Dim3 {
    /// Creates an array for iterating over the x-, y- and z-dimensions.
    pub fn slice() -> [Self; 3] {
        [Self::X, Self::Y, Self::Z]
    }

    /// Creates an array for iterating over the x- and y-dimensions.
    pub fn slice_xy() -> [Self; 2] {
        [Self::X, Self::Y]
    }

    /// Creates an array for iterating over all three dimensions except the given one.
    pub fn slice_except(dim: Self) -> [Self; 2] {
        match dim {
            Self::X => [Self::Y, Self::Z],
            Self::Y => [Self::X, Self::Z],
            Self::Z => [Self::X, Self::Y],
        }
    }
}

impl fmt::Display for Dim3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::X => "x",
                Self::Y => "y",
                Self::Z => "z",
            }
        )
    }
}

use Dim3::{X, Y, Z};

/// Denotes the x- or y-dimension.
#[derive(Clone, Copy, Debug)]
pub enum Dim2 {
    X = 0,
    Y = 1,
}

impl Dim2 {
    /// Creates an array for iterating over the x- and y-dimensions.
    pub fn slice() -> [Self; 2] {
        [Self::X, Self::Y]
    }
}

impl fmt::Display for Dim2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::X => "x",
                Self::Y => "y",
            }
        )
    }
}

/// Represents any quantity with three dimensional components.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct In3D<T>([T; 3]);

impl<T> In3D<T> {
    /// Creates a new 3D quantity given the three components.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self([x, y, z])
    }

    /// Creates a new 3D quantity with the given value copied into all components.
    pub fn same(a: T) -> Self
    where
        T: Copy,
    {
        Self([a, a, a])
    }

    /// Creates a new 3D quantity with the given value cloned into all components.
    pub fn same_cloned(a: T) -> Self
    where
        T: Clone,
    {
        Self([a.clone(), a.clone(), a])
    }

    /// Creates a new tuple containing copies of the three components.
    pub fn to_tuple(&self) -> (T, T, T)
    where
        T: Copy,
    {
        (self[X], self[Y], self[Z])
    }
}

impl<T> Index<Dim3> for In3D<T> {
    type Output = T;
    fn index(&self, dim: Dim3) -> &Self::Output {
        &self.0[dim as usize]
    }
}

impl<T> IndexMut<Dim3> for In3D<T> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output {
        &mut self.0[dim as usize]
    }
}

impl<'a, T> IntoIterator for &'a In3D<T> {
    type Item = &'a T;
    type IntoIter = ::std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Represents any quantity with two dimensional components.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct In2D<T>([T; 2]);

impl<T> In2D<T> {
    /// Creates a new 2D quantity given the two components.
    pub fn new(x: T, y: T) -> Self {
        Self([x, y])
    }

    /// Creates a new 2D quantity with the given value copied into both components.
    pub fn same(a: T) -> Self
    where
        T: Copy,
    {
        Self([a, a])
    }

    /// Creates a new 2D quantity with the given value cloned into all components.
    pub fn same_cloned(a: T) -> Self
    where
        T: Clone,
    {
        Self([a.clone(), a])
    }

    /// Creates a new tuple containing copies of the three components.
    pub fn to_tuple(&self) -> (T, T)
    where
        T: Copy,
    {
        (self[Dim2::X], self[Dim2::Y])
    }
}

impl<T> Index<Dim2> for In2D<T> {
    type Output = T;
    fn index(&self, dim: Dim2) -> &Self::Output {
        &self.0[dim as usize]
    }
}

impl<T> IndexMut<Dim2> for In2D<T> {
    fn index_mut(&mut self, dim: Dim2) -> &mut Self::Output {
        &mut self.0[dim as usize]
    }
}

impl<'a, T> IntoIterator for &'a In2D<T> {
    type Item = &'a T;
    type IntoIter = ::std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// A 3D vector.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Vec3<F>(In3D<F>);

impl<F: BFloat> Vec3<F> {
    /// Creates a new 3D vector given the three components.
    pub fn new(x: F, y: F, z: F) -> Self {
        Self(In3D::new(x, y, z))
    }

    /// Creates a new zero vector.
    pub fn zero() -> Self {
        Self::new(F::zero(), F::zero(), F::zero())
    }

    /// Creates a new vector with all component equal to the given value.
    pub fn equal_components(a: F) -> Self {
        Self::new(a, a, a)
    }

    /// Creates a new vector from the given vector, which may have a different component type.
    pub fn from<U: BFloat>(other: &Vec3<U>) -> Self {
        Self::new(
            F::from(other[X]).expect("Conversion failed"),
            F::from(other[Y]).expect("Conversion failed"),
            F::from(other[Z]).expect("Conversion failed"),
        )
    }

    /// Constructs a new point from the vector components.
    pub fn to_point3(&self) -> Point3<F> {
        Point3::new(self[X], self[Y], self[Z])
    }

    /// Constructs a new vector from the absolute values of the vector components.
    pub fn abs(&self) -> Self {
        Self::new(
            num::Float::abs(self[X]),
            num::Float::abs(self[Y]),
            num::Float::abs(self[Z]),
        )
    }

    /// Constructs a new vector by taking the component-wise max with the given vector.
    pub fn max_with(&self, other: &Self) -> Self {
        Self::new(
            F::max(self[X], other[X]),
            F::max(self[Y], other[Y]),
            F::max(self[Z], other[Z]),
        )
    }

    /// Constructs a new vector by taking the component-wise min with the given vector.
    pub fn min_with(&self, other: &Self) -> Self {
        Self::new(
            F::min(self[X], other[X]),
            F::min(self[Y], other[Y]),
            F::min(self[Z], other[Z]),
        )
    }

    /// Computes the squared length of the vector.
    pub fn squared_length(&self) -> F {
        self[X] * self[X] + self[Y] * self[Y] + self[Z] * self[Z]
    }

    /// Computes the length of the vector.
    pub fn length(&self) -> F {
        self.squared_length().sqrt()
    }

    /// Whether the vector is the zero vector.
    pub fn is_zero(&self) -> bool {
        self[X] == F::zero() && self[Y] == F::zero() && self[Z] == F::zero()
    }

    /// Computes the dot product of the vector with another vector.
    pub fn dot(&self, other: &Self) -> F {
        self[X] * other[X] + self[Y] * other[Y] + self[Z] * other[Z]
    }

    /// Normalizes the vector to have unit length.
    pub fn normalize(&mut self) {
        let length = self.length();
        assert!(length != F::zero());
        let inv_length = length.recip();
        self[X] = self[X] * inv_length;
        self[Y] = self[Y] * inv_length;
        self[Z] = self[Z] * inv_length;
    }

    /// Reverses the direction of the vector.
    pub fn reverse(&mut self) {
        self[X] = -self[X];
        self[Y] = -self[Y];
        self[Z] = -self[Z];
    }

    /// Determines the maximum component value for the vector.
    pub fn max(&self) -> F {
        F::max(self[X], F::max(self[Y], self[Z]))
    }
}

impl<F: BFloat> Index<Dim3> for Vec3<F> {
    type Output = F;
    fn index(&self, dim: Dim3) -> &Self::Output {
        &self.0[dim]
    }
}

impl<F: BFloat> IndexMut<Dim3> for Vec3<F> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

impl<'a, F: BFloat> Add<&'a Vec3<F>> for &'a Vec3<F> {
    type Output = Vec3<F>;
    fn add(self, other: Self) -> Self::Output {
        Self::Output::new(self[X] + other[X], self[Y] + other[Y], self[Z] + other[Z])
    }
}

impl<F: BFloat> Add<Vec3<F>> for &Vec3<F> {
    type Output = Vec3<F>;
    fn add(self, other: Vec3<F>) -> Self::Output {
        self + &other
    }
}

impl<F: BFloat> Add<Vec3<F>> for Vec3<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<F: BFloat> Add<&Vec3<F>> for Vec3<F> {
    type Output = Self;
    fn add(self, other: &Self) -> Self::Output {
        &self + other
    }
}

impl<'a, F: BFloat> Sub<&'a Vec3<F>> for &'a Vec3<F> {
    type Output = Vec3<F>;
    fn sub(self, other: Self) -> Self::Output {
        Self::Output::new(self[X] - other[X], self[Y] - other[Y], self[Z] - other[Z])
    }
}

impl<F: BFloat> Sub<Vec3<F>> for &Vec3<F> {
    type Output = Vec3<F>;
    fn sub(self, other: Vec3<F>) -> Self::Output {
        self - &other
    }
}

impl<F: BFloat> Sub<Vec3<F>> for Vec3<F> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<F: BFloat> Sub<&Vec3<F>> for Vec3<F> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<F: BFloat> Mul<F> for &Vec3<F> {
    type Output = Vec3<F>;
    fn mul(self, factor: F) -> Self::Output {
        Self::Output::new(factor * self[X], factor * self[Y], factor * self[Z])
    }
}

impl<F: BFloat> Mul<F> for Vec3<F> {
    type Output = Self;
    fn mul(self, factor: F) -> Self::Output {
        &self * factor
    }
}

impl<F: BFloat> Div<F> for &Vec3<F> {
    type Output = Vec3<F>;
    fn div(self, divisor: F) -> Self::Output {
        #![allow(clippy::suspicious_arithmetic_impl)]
        let factor = divisor.recip();
        self * factor
    }
}

impl<F: BFloat> Div<F> for Vec3<F> {
    type Output = Self;
    fn div(self, divisor: F) -> Self::Output {
        &self / divisor
    }
}

impl<'a, F: BFloat> IntoIterator for &'a Vec3<F> {
    type Item = <&'a In3D<F> as IntoIterator>::Item;
    type IntoIter = <&'a In3D<F> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Vec3<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self[X], self[Y], self[Z])
    }
}

/// A 2D vector.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Vec2<F>(In2D<F>);

impl<F: BFloat> Vec2<F> {
    /// Creates a new 2D vector given the three components.
    pub fn new(x: F, y: F) -> Self {
        Self(In2D::new(x, y))
    }

    /// Creates a new zero vector.
    pub fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    /// Creates a new vector with all component equal to the given value.
    pub fn equal_components(a: F) -> Self {
        Self::new(a, a)
    }

    /// Creates a new vector from the given vector, which may have a different component type.
    pub fn from<U: BFloat>(other: &Vec2<U>) -> Self {
        Self::new(
            F::from(other[Dim2::X]).expect("Conversion failed"),
            F::from(other[Dim2::Y]).expect("Conversion failed"),
        )
    }

    /// Constructs a new point from the vector components.
    pub fn to_point2(&self) -> Point2<F> {
        Point2::new(self[Dim2::X], self[Dim2::Y])
    }

    /// Constructs a new vector from the absolute values of the vector components.
    pub fn abs(&self) -> Self {
        Self::new(
            num::Float::abs(self[Dim2::X]),
            num::Float::abs(self[Dim2::Y]),
        )
    }

    /// Constructs a new vector by taking the component-wise max with the given vector.
    pub fn max_with(&self, other: &Self) -> Self {
        Self::new(
            F::max(self[Dim2::X], other[Dim2::X]),
            F::max(self[Dim2::Y], other[Dim2::Y]),
        )
    }

    /// Constructs a new vector by taking the component-wise min with the given vector.
    pub fn min_with(&self, other: &Self) -> Self {
        Self::new(
            F::min(self[Dim2::X], other[Dim2::X]),
            F::min(self[Dim2::Y], other[Dim2::Y]),
        )
    }

    /// Computes the squared length of the vector.
    pub fn squared_length(&self) -> F {
        self[Dim2::X] * self[Dim2::X] + self[Dim2::Y] * self[Dim2::Y]
    }

    /// Computes the length of the vector.
    pub fn length(&self) -> F {
        self.squared_length().sqrt()
    }

    /// Whether the vector is the zero vector.
    pub fn is_zero(&self) -> bool {
        self[Dim2::X] == F::zero() && self[Dim2::Y] == F::zero()
    }

    /// Computes the dot product of the vector with another vector.
    pub fn dot(&self, other: &Self) -> F {
        self[Dim2::X] * other[Dim2::X] + self[Dim2::Y] * other[Dim2::Y]
    }

    /// Normalizes the vector to have unit length.
    pub fn normalize(&mut self) {
        let length = self.length();
        assert!(length != F::zero());
        let inv_length = length.recip();
        self[Dim2::X] = self[Dim2::X] * inv_length;
        self[Dim2::Y] = self[Dim2::Y] * inv_length;
    }

    /// Reverses the direction of the vector.
    pub fn reverse(&mut self) {
        self[Dim2::X] = -self[Dim2::X];
        self[Dim2::Y] = -self[Dim2::Y];
    }
}

impl<F: BFloat> Index<Dim2> for Vec2<F> {
    type Output = F;
    fn index(&self, dim: Dim2) -> &Self::Output {
        &self.0[dim]
    }
}

impl<F: BFloat> IndexMut<Dim2> for Vec2<F> {
    fn index_mut(&mut self, dim: Dim2) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

impl<'a, F: BFloat> Add<&'a Vec2<F>> for &'a Vec2<F> {
    type Output = Vec2<F>;
    fn add(self, other: Self) -> Self::Output {
        Self::Output::new(
            self[Dim2::X] + other[Dim2::X],
            self[Dim2::Y] + other[Dim2::Y],
        )
    }
}

impl<F: BFloat> Add<Vec2<F>> for &Vec2<F> {
    type Output = Vec2<F>;
    fn add(self, other: Vec2<F>) -> Self::Output {
        self + &other
    }
}

impl<F: BFloat> Add<Vec2<F>> for Vec2<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<F: BFloat> Add<&Vec2<F>> for Vec2<F> {
    type Output = Self;
    fn add(self, other: &Self) -> Self::Output {
        &self + other
    }
}

impl<'a, F: BFloat> Sub<&'a Vec2<F>> for &'a Vec2<F> {
    type Output = Vec2<F>;
    fn sub(self, other: Self) -> Self::Output {
        Self::Output::new(
            self[Dim2::X] - other[Dim2::X],
            self[Dim2::Y] - other[Dim2::Y],
        )
    }
}

impl<F: BFloat> Sub<Vec2<F>> for &Vec2<F> {
    type Output = Vec2<F>;
    fn sub(self, other: Vec2<F>) -> Self::Output {
        self - &other
    }
}

impl<F: BFloat> Sub<Vec2<F>> for Vec2<F> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<F: BFloat> Sub<&Vec2<F>> for Vec2<F> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<F: BFloat> Mul<F> for &Vec2<F> {
    type Output = Vec2<F>;
    fn mul(self, factor: F) -> Self::Output {
        Self::Output::new(factor * self[Dim2::X], factor * self[Dim2::Y])
    }
}

impl<F: BFloat> Mul<F> for Vec2<F> {
    type Output = Self;
    fn mul(self, factor: F) -> Self::Output {
        &self * factor
    }
}

impl<F: BFloat> Div<F> for &Vec2<F> {
    type Output = Vec2<F>;
    fn div(self, divisor: F) -> Self::Output {
        #![allow(clippy::suspicious_arithmetic_impl)]
        let factor = divisor.recip();
        self * factor
    }
}

impl<F: BFloat> Div<F> for Vec2<F> {
    type Output = Self;
    fn div(self, divisor: F) -> Self::Output {
        &self / divisor
    }
}

impl<'a, F: BFloat> IntoIterator for &'a Vec2<F> {
    type Item = <&'a In2D<F> as IntoIterator>::Item;
    type IntoIter = <&'a In2D<F> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Vec2<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self[Dim2::X], self[Dim2::Y])
    }
}

/// A 3D spatial coordinate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Point3<F>(In3D<F>);

impl<F: BFloat> Point3<F> {
    /// Creates a new 3D point given the three components.
    pub fn new(x: F, y: F, z: F) -> Self {
        Self(In3D::new(x, y, z))
    }

    /// Creates a new 3D point with all components set to zero.
    pub fn origin() -> Self {
        Self::new(F::zero(), F::zero(), F::zero())
    }

    /// Creates a new point with all component equal to the given value.
    pub fn equal_components(a: F) -> Self {
        Self::new(a, a, a)
    }

    /// Creates a new point from the given point, which may have a different component type.
    pub fn from<U: BFloat>(other: &Point3<U>) -> Self {
        Self::from_components(other[X], other[Y], other[Z])
    }

    /// Creates a new point from the given components, which may have different types.
    pub fn from_components<U: BFloat, V: BFloat, W: BFloat>(x: U, y: V, z: W) -> Self {
        Self::new(
            F::from(x).expect("Conversion failed"),
            F::from(y).expect("Conversion failed"),
            F::from(z).expect("Conversion failed"),
        )
    }

    /// Constructs a new vector from the point components.
    pub fn to_vec3(&self) -> Vec3<F> {
        Vec3::new(self[X], self[Y], self[Z])
    }
}

impl<F: BFloat> Index<Dim3> for Point3<F> {
    type Output = F;
    fn index(&self, dim: Dim3) -> &Self::Output {
        &self.0[dim]
    }
}

impl<F: BFloat> IndexMut<Dim3> for Point3<F> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

impl<'a, F: BFloat> Sub<&'a Self> for &'a Point3<F> {
    type Output = Vec3<F>;
    fn sub(self, other: &Self) -> Self::Output {
        Self::Output::new(self[X] - other[X], self[Y] - other[Y], self[Z] - other[Z])
    }
}

impl<F: BFloat> Sub<Self> for Point3<F> {
    type Output = Vec3<F>;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<F: BFloat> Sub<Self> for &Point3<F> {
    type Output = Vec3<F>;
    fn sub(self, other: Self) -> Self::Output {
        #![allow(clippy::op_ref)]
        self - &other
    }
}

impl<F: BFloat> Sub<&Self> for Point3<F> {
    type Output = Vec3<F>;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<'a, F: BFloat> Add<&'a Vec3<F>> for &'a Point3<F> {
    type Output = Point3<F>;
    fn add(self, vector: &Vec3<F>) -> Self::Output {
        Self::Output::new(
            self[X] + vector[X],
            self[Y] + vector[Y],
            self[Z] + vector[Z],
        )
    }
}

impl<F: BFloat> Add<Vec3<F>> for Point3<F> {
    type Output = Self;
    fn add(self, vector: Vec3<F>) -> Self::Output {
        &self + &vector
    }
}

impl<F: BFloat> Add<Vec3<F>> for &Point3<F> {
    type Output = Point3<F>;
    fn add(self, vector: Vec3<F>) -> Self::Output {
        self + &vector
    }
}

impl<F: BFloat> Add<&Vec3<F>> for Point3<F> {
    type Output = Self;
    fn add(self, vector: &Vec3<F>) -> Self::Output {
        &self + vector
    }
}

impl<'a, F: BFloat> Sub<&'a Vec3<F>> for &'a Point3<F> {
    type Output = Point3<F>;
    fn sub(self, vector: &Vec3<F>) -> Self::Output {
        Self::Output::new(
            self[X] - vector[X],
            self[Y] - vector[Y],
            self[Z] - vector[Z],
        )
    }
}

impl<F: BFloat> Sub<Vec3<F>> for Point3<F> {
    type Output = Self;
    fn sub(self, vector: Vec3<F>) -> Self::Output {
        &self - &vector
    }
}

impl<F: BFloat> Sub<Vec3<F>> for &Point3<F> {
    type Output = Point3<F>;
    fn sub(self, vector: Vec3<F>) -> Self::Output {
        self - &vector
    }
}

impl<F: BFloat> Sub<&Vec3<F>> for Point3<F> {
    type Output = Self;
    fn sub(self, vector: &Vec3<F>) -> Self::Output {
        &self - vector
    }
}

impl<'a, F: BFloat> IntoIterator for &'a Point3<F> {
    type Item = <&'a In3D<F> as IntoIterator>::Item;
    type IntoIter = <&'a In3D<F> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Point3<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self[X], self[Y], self[Z])
    }
}

/// A 2D spatial coordinate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Point2<F>(In2D<F>);

impl<F: BFloat> Point2<F> {
    /// Creates a new 2D point given the three components.
    pub fn new(x: F, y: F) -> Self {
        Self(In2D::new(x, y))
    }

    /// Creates a new 2D point with all components set to zero.
    pub fn origin() -> Self {
        Self::new(F::zero(), F::zero())
    }

    /// Creates a new point with all component equal to the given value.
    pub fn equal_components(a: F) -> Self {
        Self::new(a, a)
    }

    /// Creates a new point from the given components, which may have a different type.
    pub fn from_components<U: BFloat, V: BFloat>(x: U, y: V) -> Self {
        Self::new(
            F::from(x).expect("Conversion failed"),
            F::from(y).expect("Conversion failed"),
        )
    }

    /// Creates a new point from the given point, which may have a different component type.
    pub fn from<U: BFloat>(other: &Point2<U>) -> Self {
        Self::from_components(other[Dim2::X], other[Dim2::Y])
    }

    /// Constructs a new vector from the point components.
    pub fn to_vec2(&self) -> Vec2<F> {
        Vec2::new(self[Dim2::X], self[Dim2::Y])
    }
}

impl<F: BFloat> Index<Dim2> for Point2<F> {
    type Output = F;
    fn index(&self, dim: Dim2) -> &Self::Output {
        &self.0[dim]
    }
}

impl<F: BFloat> IndexMut<Dim2> for Point2<F> {
    fn index_mut(&mut self, dim: Dim2) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

impl<'a, F: BFloat> Sub<&'a Self> for &'a Point2<F> {
    type Output = Vec2<F>;
    fn sub(self, other: &Self) -> Self::Output {
        Self::Output::new(
            self[Dim2::X] - other[Dim2::X],
            self[Dim2::Y] - other[Dim2::Y],
        )
    }
}

impl<F: BFloat> Sub<Self> for Point2<F> {
    type Output = Vec2<F>;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<F: BFloat> Sub<Self> for &Point2<F> {
    type Output = Vec2<F>;
    fn sub(self, other: Self) -> Self::Output {
        #![allow(clippy::op_ref)]
        self - &other
    }
}

impl<F: BFloat> Sub<&Self> for Point2<F> {
    type Output = Vec2<F>;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<'a, F: BFloat> Add<&'a Vec2<F>> for &'a Point2<F> {
    type Output = Point2<F>;
    fn add(self, vector: &Vec2<F>) -> Self::Output {
        Self::Output::new(
            self[Dim2::X] + vector[Dim2::X],
            self[Dim2::Y] + vector[Dim2::Y],
        )
    }
}

impl<F: BFloat> Add<Vec2<F>> for Point2<F> {
    type Output = Self;
    fn add(self, vector: Vec2<F>) -> Self::Output {
        &self + &vector
    }
}

impl<F: BFloat> Add<Vec2<F>> for &Point2<F> {
    type Output = Point2<F>;
    fn add(self, vector: Vec2<F>) -> Self::Output {
        self + &vector
    }
}

impl<F: BFloat> Add<&Vec2<F>> for Point2<F> {
    type Output = Self;
    fn add(self, vector: &Vec2<F>) -> Self::Output {
        &self + vector
    }
}

impl<'a, F: BFloat> Sub<&'a Vec2<F>> for &'a Point2<F> {
    type Output = Point2<F>;
    fn sub(self, vector: &Vec2<F>) -> Self::Output {
        Self::Output::new(
            self[Dim2::X] - vector[Dim2::X],
            self[Dim2::Y] - vector[Dim2::Y],
        )
    }
}

impl<F: BFloat> Sub<Vec2<F>> for Point2<F> {
    type Output = Self;
    fn sub(self, vector: Vec2<F>) -> Self::Output {
        &self - &vector
    }
}

impl<F: BFloat> Sub<Vec2<F>> for &Point2<F> {
    type Output = Point2<F>;
    fn sub(self, vector: Vec2<F>) -> Self::Output {
        self - &vector
    }
}

impl<F: BFloat> Sub<&Vec2<F>> for Point2<F> {
    type Output = Self;
    fn sub(self, vector: &Vec2<F>) -> Self::Output {
        &self - vector
    }
}

impl<'a, F: BFloat> IntoIterator for &'a Point2<F> {
    type Item = <&'a In2D<F> as IntoIterator>::Item;
    type IntoIter = <&'a In2D<F> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Point2<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self[Dim2::X], self[Dim2::Y])
    }
}

/// A 3D index.
#[derive(Clone, Debug, PartialEq)]
pub struct Idx3<I>(In3D<I>);

impl<I: num::Integer> Idx3<I> {
    /// Creates a new 3D index given the three components.
    pub fn new(i: I, j: I, k: I) -> Self {
        Self(In3D::new(i, j, k))
    }

    /// Creates a new 3D index with all components set to zero.
    pub fn origin() -> Self {
        Self::new(I::zero(), I::zero(), I::zero())
    }

    /// Creates a new 3D index from the given index, which may have a different component type.
    pub fn from<U>(other: &Idx3<U>) -> Self
    where
        I: num::NumCast + Copy,
        U: num::Integer + num::NumCast + Copy,
    {
        Self::new(
            I::from(other[X]).expect("Conversion failed"),
            I::from(other[Y]).expect("Conversion failed"),
            I::from(other[Z]).expect("Conversion failed"),
        )
    }

    /// Constructs a new 3D index by taking the component-wise max with the given index.
    pub fn max_with(&self, other: &Self) -> Self
    where
        I: Copy,
    {
        Self::new(
            I::max(self[X], other[X]),
            I::max(self[Y], other[Y]),
            I::max(self[Z], other[Z]),
        )
    }

    /// Constructs a new 3D index by taking the component-wise min with the given index.
    pub fn min_with(&self, other: &Self) -> Self
    where
        I: Copy,
    {
        Self::new(
            I::min(self[X], other[X]),
            I::min(self[Y], other[Y]),
            I::min(self[Z], other[Z]),
        )
    }
}

impl<I: num::Integer> Index<Dim3> for Idx3<I> {
    type Output = I;
    fn index(&self, dim: Dim3) -> &Self::Output {
        &self.0[dim]
    }
}

impl<I: num::Integer> IndexMut<Dim3> for Idx3<I> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

impl<'a, I: num::Integer> IntoIterator for &'a Idx3<I> {
    type Item = <&'a In3D<I> as IntoIterator>::Item;
    type IntoIter = <&'a In3D<I> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<I: num::Integer + fmt::Display> fmt::Display for Idx3<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self[X], self[Y], self[Z])
    }
}

/// A 2D index.
#[derive(Clone, Debug, PartialEq)]
pub struct Idx2<I>(In2D<I>);

impl<I: num::Integer> Idx2<I> {
    /// Creates a new 2D index given the three components.
    pub fn new(i: I, j: I) -> Self {
        Self(In2D::new(i, j))
    }

    /// Creates a new 2D index with all components set to zero.
    pub fn origin() -> Self {
        Self::new(I::zero(), I::zero())
    }

    /// Creates a new 2D index from the given index, which may have a different component type.
    pub fn from<U>(other: &Idx2<U>) -> Self
    where
        I: num::NumCast + Copy,
        U: num::Integer + num::NumCast + Copy,
    {
        Self::new(
            I::from(other[Dim2::X]).expect("Conversion failed"),
            I::from(other[Dim2::Y]).expect("Conversion failed"),
        )
    }

    /// Constructs a new 2D index by taking the component-wise max with the given index.
    pub fn max_with(&self, other: &Self) -> Self
    where
        I: Copy,
    {
        Self::new(
            I::max(self[Dim2::X], other[Dim2::X]),
            I::max(self[Dim2::Y], other[Dim2::Y]),
        )
    }

    /// Constructs a new 2D index by taking the component-wise min with the given index.
    pub fn min_with(&self, other: &Self) -> Self
    where
        I: Copy,
    {
        Self::new(
            I::min(self[Dim2::X], other[Dim2::X]),
            I::min(self[Dim2::Y], other[Dim2::Y]),
        )
    }
}

impl<I: num::Integer> Index<Dim2> for Idx2<I> {
    type Output = I;
    fn index(&self, dim: Dim2) -> &Self::Output {
        &self.0[dim]
    }
}

impl<I: num::Integer> IndexMut<Dim2> for Idx2<I> {
    fn index_mut(&mut self, dim: Dim2) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

impl<'a, I: num::Integer> IntoIterator for &'a Idx2<I> {
    type Item = <&'a In2D<I> as IntoIterator>::Item;
    type IntoIter = <&'a In2D<I> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<I: num::Integer + fmt::Display> fmt::Display for Idx2<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self[Dim2::X], self[Dim2::Y])
    }
}

/// 3D spatial coordinate arrays.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Coords3<F>(In3D<Vec<F>>);

impl<F: BFloat> Coords3<F> {
    /// Creates a new 3D set of coordinates given the component 1D coordinates.
    pub fn new(x: Vec<F>, y: Vec<F>, z: Vec<F>) -> Self {
        Self(In3D::new(x, y, z))
    }

    /// Creates a 3D point from the coordinates at the given indices.
    pub fn point(&self, indices: &Idx3<usize>) -> Point3<F> {
        Point3::new(
            self[X][indices[X]],
            self[Y][indices[Y]],
            self[Z][indices[Z]],
        )
    }

    /// Creates a new 3D set of coordinates restricted to slices of the original
    /// coordinate arrays.
    pub fn subcoords(&self, start_indices: &Idx3<usize>, end_indices: &Idx3<usize>) -> Self {
        Self::new(
            self[X][start_indices[X]..=end_indices[X]].to_vec(),
            self[Y][start_indices[Y]..=end_indices[Y]].to_vec(),
            self[Z][start_indices[Z]..=end_indices[Z]].to_vec(),
        )
    }
}

impl<F: BFloat> Index<Dim3> for Coords3<F> {
    type Output = Vec<F>;
    fn index(&self, dim: Dim3) -> &Self::Output {
        &self.0[dim]
    }
}

impl<F: BFloat> IndexMut<Dim3> for Coords3<F> {
    fn index_mut(&mut self, dim: Dim3) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

/// 2D spatial coordinate arrays.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Coords2<F>(In2D<Vec<F>>);

impl<F: BFloat> Coords2<F> {
    /// Creates a new 2D set of coordinates given the component 1D coordinates.
    pub fn new(x: Vec<F>, y: Vec<F>) -> Self {
        Self(In2D::new(x, y))
    }

    /// Creates a 2D point from the coordinates at the given indices.
    pub fn point(&self, indices: &Idx2<usize>) -> Point2<F> {
        Point2::new(
            self[Dim2::X][indices[Dim2::X]],
            self[Dim2::Y][indices[Dim2::Y]],
        )
    }
}

impl<F: BFloat> Index<Dim2> for Coords2<F> {
    type Output = Vec<F>;
    fn index(&self, dim: Dim2) -> &Self::Output {
        &self.0[dim]
    }
}

impl<F: BFloat> IndexMut<Dim2> for Coords2<F> {
    fn index_mut(&mut self, dim: Dim2) -> &mut Self::Output {
        &mut self.0[dim]
    }
}

/// References to 3D spatial coordinate arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordRefs3<'a, F: BFloat>(In3D<&'a [F]>);

impl<'a, F: BFloat> CoordRefs3<'a, F> {
    /// Creates a new 3D set of coordinate references given references to the component arrays.
    pub fn new(x: &'a [F], y: &'a [F], z: &'a [F]) -> Self {
        Self(In3D::new(x, y, z))
    }

    /// Clones the coordinate references to produce a set of owned coordinate arrays.
    pub fn into_owned(self) -> Coords3<F> {
        Coords3::new(self[X].to_vec(), self[Y].to_vec(), self[Z].to_vec())
    }

    /// Creates a 3D point from the coordinates at the given indices.
    pub fn point(&self, indices: &Idx3<usize>) -> Point3<F> {
        Point3::new(
            self[X][indices[X]],
            self[Y][indices[Y]],
            self[Z][indices[Z]],
        )
    }
}

impl<'a, F: BFloat> Index<Dim3> for CoordRefs3<'a, F> {
    type Output = &'a [F];
    fn index(&self, dim: Dim3) -> &Self::Output {
        &self.0[dim]
    }
}

/// References to 2D spatial coordinate arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordRefs2<'a, F: BFloat>(In2D<&'a [F]>);

impl<'a, F: BFloat> CoordRefs2<'a, F> {
    /// Creates a new 2D set of coordinate references given references to the component arrays.
    pub fn new(x: &'a [F], y: &'a [F]) -> Self {
        Self(In2D::new(x, y))
    }

    /// Clones the coordinate references to produce a set of owned coordinate arrays.
    pub fn into_owned(self) -> Coords2<F> {
        Coords2::new(self[Dim2::X].to_vec(), self[Dim2::Y].to_vec())
    }

    /// Creates a 2D point from the coordinates at the given indices.
    pub fn point(&self, indices: &Idx2<usize>) -> Point2<F> {
        Point2::new(
            self[Dim2::X][indices[Dim2::X]],
            self[Dim2::Y][indices[Dim2::Y]],
        )
    }
}

impl<'a, F: BFloat> Index<Dim2> for CoordRefs2<'a, F> {
    type Output = &'a [F];
    fn index(&self, dim: Dim2) -> &Self::Output {
        &self.0[dim]
    }
}
