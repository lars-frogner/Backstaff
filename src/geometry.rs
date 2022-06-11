//! Geometric utility objects.

use crate::num::BFloat;
use num;
use std::{
    borrow::Cow,
    fmt, iter,
    marker::PhantomData,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

#[cfg(feature = "serialization")]
use serde::Serialize;

#[cfg(feature = "for-testing")]
use approx::{AbsDiffEq, RelativeEq};

#[cfg(feature = "for-testing")]
use crate::num::ComparableSlice;

/// Denotes the x-, y- or z-dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
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

    /// Returns the number of the dimension.
    pub fn num(self) -> usize {
        self as usize
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dim2 {
    X = 0,
    Y = 1,
}

impl Dim2 {
    /// Creates an array for iterating over the x- and y-dimensions.
    pub fn slice() -> [Self; 2] {
        [Self::X, Self::Y]
    }

    /// Returns an `Option` with the corresponding 2D dimension from a 3D dimension,
    /// or `None` if the 3D dimension is `Z`.
    pub fn from_dim3(dim: Dim3) -> Option<Self> {
        match dim {
            X => Some(Self::X),
            Y => Some(Self::Y),
            _ => None,
        }
    }

    /// Returns the number of the dimension.
    pub fn num(self) -> usize {
        self as usize
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

    /// Creates a new 3D quantity by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim3) -> T,
    {
        Self::new(
            create_component(X),
            create_component(Y),
            create_component(Z),
        )
    }

    /// Converts to a new 3D quantity with a different component type.
    pub fn converted<U>(&self) -> In3D<U>
    where
        U: From<T>,
        T: Copy,
    {
        In3D::<U>::new(self[X].into(), self[Y].into(), self[Z].into())
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

impl<T> FromIterator<T> for In3D<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        Self([
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        ])
    }
}

impl<T: fmt::Display> fmt::Display for In3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        fmt::Display::fmt(&self[X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Y], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Z], f)?;
        f.write_str("]")
    }
}

#[cfg(feature = "for-testing")]
impl<T> AbsDiffEq for In3D<T>
where
    T: AbsDiffEq,
    T::Epsilon: Copy,
{
    type Epsilon = <T as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        T::abs_diff_eq(&self[X], &other[X], epsilon)
            && T::abs_diff_eq(&self[Y], &other[Y], epsilon)
            && T::abs_diff_eq(&self[Z], &other[Z], epsilon)
    }
}

#[cfg(feature = "for-testing")]
impl<T> RelativeEq for In3D<T>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        T::relative_eq(&self[X], &other[X], epsilon, max_relative)
            && T::relative_eq(&self[Y], &other[Y], epsilon, max_relative)
            && T::relative_eq(&self[Z], &other[Z], epsilon, max_relative)
    }
}

#[cfg(feature = "for-testing")]
macro_rules! impl_abs_diff_eq {
    ($T:ident <$F:ident>, $SUBT:ident) => {
        impl<$F> AbsDiffEq for $T<$F>
        where
            $F: BFloat + AbsDiffEq,
            $F::Epsilon: Copy,
        {
            type Epsilon = <$SUBT<$F> as AbsDiffEq>::Epsilon;

            fn default_epsilon() -> Self::Epsilon {
                $SUBT::<$F>::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $SUBT::<$F>::abs_diff_eq(&self.0, &other.0, epsilon)
            }
        }
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_relative_eq {
    ($T:ident <$F:ident>, $SUBT:ident) => {
        impl<$F> RelativeEq for $T<$F>
        where
            $F: BFloat + RelativeEq,
            $F::Epsilon: Copy,
        {
            fn default_max_relative() -> Self::Epsilon {
                $SUBT::<$F>::default_max_relative()
            }

            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                $SUBT::<$F>::relative_eq(&self.0, &other.0, epsilon, max_relative)
            }
        }
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_abs_diff_eq_3d {
    ($T:ident <$F:ident>) => {
        impl_abs_diff_eq!($T<$F>, In3D);
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_relative_eq_3d {
    ($T:ident <$F:ident>) => {
        impl_relative_eq!($T<$F>, In3D);
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_abs_diff_eq_2d {
    ($T:ident <$F:ident>) => {
        impl_abs_diff_eq!($T<$F>, In2D);
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_relative_eq_2d {
    ($T:ident <$F:ident>) => {
        impl_relative_eq!($T<$F>, In2D);
    };
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

    /// Creates a new 2D quantity by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim2) -> T,
    {
        Self::new(create_component(Dim2::X), create_component(Dim2::Y))
    }

    /// Converts to a new 2D quantity with a different component type.
    pub fn converted<U>(&self) -> In2D<U>
    where
        U: From<T>,
        T: Copy,
    {
        In2D::<U>::new(self[Dim2::X].into(), self[Dim2::Y].into())
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

impl<T> FromIterator<T> for In2D<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        Self([iter.next().unwrap(), iter.next().unwrap()])
    }
}

impl<T: fmt::Display> fmt::Display for In2D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        fmt::Display::fmt(&self[Dim2::X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Dim2::Y], f)?;
        f.write_str("]")
    }
}

#[cfg(feature = "for-testing")]
impl<T> AbsDiffEq for In2D<T>
where
    T: AbsDiffEq,
    T::Epsilon: Copy,
{
    type Epsilon = <T as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        T::abs_diff_eq(&self[Dim2::X], &other[Dim2::X], epsilon)
            && T::abs_diff_eq(&self[Dim2::Y], &other[Dim2::Y], epsilon)
    }
}

#[cfg(feature = "for-testing")]
impl<T> RelativeEq for In2D<T>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        T::relative_eq(&self[Dim2::X], &other[Dim2::X], epsilon, max_relative)
            && T::relative_eq(&self[Dim2::Y], &other[Dim2::Y], epsilon, max_relative)
    }
}

/// A 3D vector.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Vec3<F>(In3D<F>);

impl<F: BFloat> Vec3<F> {
    /// Creates a new 3D vector given the three components.
    pub fn new(x: F, y: F, z: F) -> Self {
        Self::from_in3d(In3D::new(x, y, z))
    }

    /// Creates a new 3D vector given an `In3D` object of components.
    pub fn from_in3d(components: In3D<F>) -> Self {
        Self(components)
    }

    /// Creates a new 3D vector by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim3) -> F,
    {
        Self(In3D::with_each_component(create_component))
    }

    /// Converts to a new vector with a different component type.
    pub fn converted<U>(&self) -> Vec3<U>
    where
        U: From<F>,
        F: Copy,
    {
        Vec3(self.0.converted())
    }

    /// Creates a new vector with components cast to the specified floating point type.
    pub fn cast<FNEW>(&self) -> Vec3<FNEW>
    where
        FNEW: BFloat,
    {
        Vec3::<FNEW>::new(
            num::cast(self[X]).unwrap(),
            num::cast(self[Y]).unwrap(),
            num::cast(self[Z]).unwrap(),
        )
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
        Point3::with_each_component(|dim| self[dim])
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

    /// Constructs a 2D vector by discarding the z-component.
    pub fn without_z(&self) -> Vec2<F> {
        Vec2::new(self[X], self[Y])
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

impl<F> FromIterator<F> for Vec3<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Vec3<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        fmt::Display::fmt(&self[X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Y], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Z], f)?;
        f.write_str(")")
    }
}

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_3d!(Vec3<F>);

#[cfg(feature = "for-testing")]
impl_relative_eq_3d!(Vec3<F>);

/// A 2D vector.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Vec2<F>(In2D<F>);

impl<F: BFloat> Vec2<F> {
    /// Creates a new 2D vector given the three components.
    pub fn new(x: F, y: F) -> Self {
        Self::from_in2d(In2D::new(x, y))
    }

    /// Creates a new 3D vector given an `In2D` object of components.
    pub fn from_in2d(components: In2D<F>) -> Self {
        Self(components)
    }

    /// Creates a new 2D vector by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim2) -> F,
    {
        Self(In2D::with_each_component(create_component))
    }

    /// Converts to a new vector with a different component type.
    pub fn converted<U>(&self) -> Vec2<U>
    where
        U: From<F>,
        F: Copy,
    {
        Vec2(self.0.converted())
    }

    /// Creates a new vector with components cast to the specified floating point type.
    pub fn cast<FNEW>(&self) -> Vec2<FNEW>
    where
        FNEW: BFloat,
    {
        Vec2::<FNEW>::new(
            num::cast(self[Dim2::X]).unwrap(),
            num::cast(self[Dim2::Y]).unwrap(),
        )
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
        Point2::with_each_component(|dim| self[dim])
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

    /// Returns a version of the vector rotated 90 degrees counter-clockwise.
    pub fn rotated_90(&self) -> Self {
        Self::new(-self[Dim2::Y], self[Dim2::X])
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

impl<F> FromIterator<F> for Vec2<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Vec2<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        fmt::Display::fmt(&self[Dim2::X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Dim2::Y], f)?;
        f.write_str(")")
    }
}

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_2d!(Vec2<F>);

#[cfg(feature = "for-testing")]
impl_relative_eq_2d!(Vec2<F>);

/// A 3D spatial coordinate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Point3<F>(In3D<F>);

impl<F: BFloat> Point3<F> {
    /// Creates a new 3D point given the three components.
    pub fn new(x: F, y: F, z: F) -> Self {
        Self(In3D::new(x, y, z))
    }

    /// Creates a new 3D point by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim3) -> F,
    {
        Self(In3D::with_each_component(create_component))
    }

    /// Converts to a new point with a different component type.
    pub fn converted<U>(&self) -> Point3<U>
    where
        U: From<F>,
        F: Copy,
    {
        Point3(self.0.converted())
    }

    /// Creates a new point with components cast to the specified floating point type.
    pub fn cast<FNEW>(&self) -> Point3<FNEW>
    where
        FNEW: BFloat,
    {
        Point3::<FNEW>::new(
            num::cast(self[X]).unwrap(),
            num::cast(self[Y]).unwrap(),
            num::cast(self[Z]).unwrap(),
        )
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
        Vec3::with_each_component(|dim| self[dim])
    }

    /// Constructs a 2D point by discarding the z-component.
    pub fn without_z(&self) -> Point2<F> {
        Point2::new(self[X], self[Y])
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

impl<F> FromIterator<F> for Point3<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Point3<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        fmt::Display::fmt(&self[X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Y], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Z], f)?;
        f.write_str(")")
    }
}

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_3d!(Point3<F>);

#[cfg(feature = "for-testing")]
impl_relative_eq_3d!(Point3<F>);

/// A 2D spatial coordinate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Point2<F>(In2D<F>);

impl<F: BFloat> Point2<F> {
    /// Creates a new 2D point given the three components.
    pub fn new(x: F, y: F) -> Self {
        Self(In2D::new(x, y))
    }

    /// Creates a new 2D point by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim2) -> F,
    {
        Self(In2D::with_each_component(create_component))
    }

    /// Converts to a new point with a different component type.
    pub fn converted<U>(&self) -> Point2<U>
    where
        U: From<F>,
        F: Copy,
    {
        Point2(self.0.converted())
    }

    /// Creates a new point with components cast to the specified floating point type.
    pub fn cast<FNEW>(&self) -> Point2<FNEW>
    where
        FNEW: BFloat,
    {
        Point2::<FNEW>::new(
            num::cast(self[Dim2::X]).unwrap(),
            num::cast(self[Dim2::Y]).unwrap(),
        )
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
        Vec2::with_each_component(|dim| self[dim])
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

impl<F> FromIterator<F> for Point2<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<F: BFloat + fmt::Display> fmt::Display for Point2<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        fmt::Display::fmt(&self[Dim2::X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Dim2::Y], f)?;
        f.write_str(")")
    }
}

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_2d!(Point2<F>);

#[cfg(feature = "for-testing")]
impl_relative_eq_2d!(Point2<F>);

/// A 3D index.
#[derive(Clone, Debug, PartialEq)]
pub struct Idx3<I>(In3D<I>);

impl<I: num::Integer> Idx3<I> {
    /// Creates a new 3D index given the three components.
    pub fn new(i: I, j: I, k: I) -> Self {
        Self(In3D::new(i, j, k))
    }

    /// Creates a new 3D index by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim3) -> I,
    {
        Self(In3D::with_each_component(create_component))
    }

    /// Converts to a new 3D index with a different component type.
    pub fn converted<U>(&self) -> Idx3<U>
    where
        U: From<I>,
        I: Copy,
    {
        Idx3(self.0.converted())
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

impl<F> FromIterator<F> for Idx3<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<I: num::Integer + fmt::Display> fmt::Display for Idx3<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        fmt::Display::fmt(&self[X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Y], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Z], f)?;
        f.write_str("]")
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

    /// Creates a new 2D index by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim2) -> I,
    {
        Self(In2D::with_each_component(create_component))
    }

    /// Converts to a new 2D index with a different component type.
    pub fn converted<U>(&self) -> Idx2<U>
    where
        U: From<I>,
        I: Copy,
    {
        Idx2(self.0.converted())
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

impl<F> FromIterator<F> for Idx2<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<I: num::Integer + fmt::Display> fmt::Display for Idx2<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        fmt::Display::fmt(&self[Dim2::X], f)?;
        f.write_str(", ")?;
        fmt::Display::fmt(&self[Dim2::Y], f)?;
        f.write_str("]")
    }
}

/// 3D spatial coordinate arrays.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Coords3<F>(In3D<Vec<F>>);

impl<F: BFloat> Coords3<F> {
    /// Creates a new 3D set of coordinates given the component 1D coordinates.
    pub fn new(x: Vec<F>, y: Vec<F>, z: Vec<F>) -> Self {
        Self(In3D::new(x, y, z))
    }

    /// Creates a new 3D set of coordinates by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim3) -> Vec<F>,
    {
        Self(In3D::with_each_component(create_component))
    }

    /// Returns the shape of the 3D set of coordinates.
    pub fn shape(&self) -> In3D<usize> {
        In3D::with_each_component(|dim| self[dim].len())
    }

    /// Creates a 3D point from the coordinates at the given indices.
    pub fn point(&self, indices: &Idx3<usize>) -> Point3<F> {
        Point3::new(
            self[X][indices[X]],
            self[Y][indices[Y]],
            self[Z][indices[Z]],
        )
    }

    /// Creates a 3D vector from the coordinates at the given indices.
    pub fn vector(&self, indices: &Idx3<usize>) -> Vec3<F> {
        Vec3::new(
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

#[cfg(feature = "for-testing")]
impl<F> AbsDiffEq for Coords3<F>
where
    F: AbsDiffEq,
    F::Epsilon: Copy,
{
    type Epsilon = <F as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        F::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let x = ComparableSlice(&self.0[X]);
        let y = ComparableSlice(&self.0[Y]);
        let z = ComparableSlice(&self.0[Z]);
        x.abs_diff_eq(&ComparableSlice(&other.0[X]), epsilon)
            && y.abs_diff_eq(&ComparableSlice(&other.0[Y]), epsilon)
            && z.abs_diff_eq(&ComparableSlice(&other.0[Z]), epsilon)
    }
}

#[cfg(feature = "for-testing")]
impl<F> RelativeEq for Coords3<F>
where
    F: RelativeEq,
    F::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        F::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let x = ComparableSlice(&self.0[X]);
        let y = ComparableSlice(&self.0[Y]);
        let z = ComparableSlice(&self.0[Z]);
        x.relative_eq(&ComparableSlice(&other.0[X]), epsilon, max_relative)
            && y.relative_eq(&ComparableSlice(&other.0[Y]), epsilon, max_relative)
            && z.relative_eq(&ComparableSlice(&other.0[Z]), epsilon, max_relative)
    }
}

/// 2D spatial coordinate arrays.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize))]
pub struct Coords2<F>(In2D<Vec<F>>);

impl<F: BFloat> Coords2<F> {
    /// Creates a new 2D set of coordinates given the component 1D coordinates.
    pub fn new(x: Vec<F>, y: Vec<F>) -> Self {
        Self(In2D::new(x, y))
    }

    /// Creates a new 2D set of coordinates by evaluating the given component
    /// constructor for each dimension.
    pub fn with_each_component<C>(create_component: C) -> Self
    where
        C: Fn(Dim2) -> Vec<F>,
    {
        Self(In2D::with_each_component(create_component))
    }

    /// Returns the shape of the 2D set of coordinates.
    pub fn shape(&self) -> In2D<usize> {
        In2D::with_each_component(|dim| self[dim].len())
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

#[cfg(feature = "for-testing")]
impl<F> AbsDiffEq for Coords2<F>
where
    F: AbsDiffEq,
    F::Epsilon: Copy,
{
    type Epsilon = <F as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        F::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let x = ComparableSlice(&self.0[Dim2::X]);
        let y = ComparableSlice(&self.0[Dim2::Y]);
        x.abs_diff_eq(&ComparableSlice(&other.0[Dim2::X]), epsilon)
            && y.abs_diff_eq(&ComparableSlice(&other.0[Dim2::Y]), epsilon)
    }
}

#[cfg(feature = "for-testing")]
impl<F> RelativeEq for Coords2<F>
where
    F: RelativeEq,
    F::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        F::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let x = ComparableSlice(&self.0[Dim2::X]);
        let y = ComparableSlice(&self.0[Dim2::Y]);
        x.relative_eq(&ComparableSlice(&other.0[Dim2::X]), epsilon, max_relative)
            && y.relative_eq(&ComparableSlice(&other.0[Dim2::Y]), epsilon, max_relative)
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
        Coords3::with_each_component(|dim| self[dim].to_vec())
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
        Coords2::with_each_component(|dim| self[dim].to_vec())
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

/// Line in 2D, specified by its coefficients in the general line equation
///  `a*x + b*y + c = 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Line2<F> {
    a: F,
    b: F,
    c: F,
}

impl<F: BFloat> Line2<F> {
    /// Creates a new 2D line from the given coefficients.
    pub fn new(a: F, b: F, c: F) -> Self {
        Self { a, b, c }
    }

    /// Creates a new 2D line given two points on the line.
    pub fn new_from_points(point_1: &Point2<F>, point_2: &Point2<F>) -> Self {
        let diff = point_2 - point_1;
        let a = diff[Dim2::Y];
        let b = -diff[Dim2::X];
        let c = point_2[Dim2::X] * point_1[Dim2::Y] - point_2[Dim2::Y] * point_1[Dim2::X];
        Self::new(a, b, c)
    }

    /// Evaluates the line equation for the given point.
    /// Evaluates to zero for points on the line, and to
    /// non-zero values with opposite sign for points on
    /// either side of the line.
    pub fn evaluate(&self, point: &Point2<F>) -> F {
        self.a * point[Dim2::X] + self.b * point[Dim2::Y] + self.c
    }

    /// Returns an `Option` with the point where this line intersects with the given line,
    /// or `None` if the lines are parallel.
    pub fn intersection(&self, other: &Self) -> Option<Point2<F>> {
        let w = self.a * other.b - self.b * other.a;
        if w == F::zero() {
            None
        } else {
            Some(Point2::new(
                (self.b * other.c - self.c * other.b) / w,
                (self.c * other.a - self.a * other.c) / w,
            ))
        }
    }
}

/// A polygon in 2D, assumed to be non-intersecting and closed.
#[derive(Clone, Debug, PartialEq)]
pub struct SimplePolygon2<F> {
    vertices: Vec<Point2<F>>,
}

macro_rules! vertex_pair_iter {
    ($vertices:expr) => {
        $vertices.iter().zip(
            $vertices
                .iter()
                .skip(1)
                .chain(iter::once($vertices.first().unwrap())),
        )
    };
}

macro_rules! checked_vertex_pair_iter {
    ($vertices:expr) => {
        if $vertices.len() > 2 {
            Some(vertex_pair_iter!($vertices))
        } else {
            None
        }
    };
}

impl<F: BFloat> SimplePolygon2<F> {
    /// Creates a new 2D polygon from the given list of vertices.
    pub fn new(vertices: Vec<Point2<F>>) -> Self {
        Self { vertices }
    }

    /// Creates a new polygon with no vertices.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Creates a new polygon with no vertices, but with allocated
    /// capacity for the given number of vertices.
    pub fn empty_with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity))
    }

    /// Creates a rectangle polygon from the given lower and upper bounds.
    pub fn rectangle_from_bounds(lower_bounds: &Vec2<F>, upper_bounds: &Vec2<F>) -> Self {
        Self::new(vec![
            lower_bounds.to_point2(),
            Point2::new(upper_bounds[Dim2::X], lower_bounds[Dim2::Y]),
            upper_bounds.to_point2(),
            Point2::new(lower_bounds[Dim2::X], upper_bounds[Dim2::Y]),
        ])
    }

    /// Creates a rectangle polygon from the horizontal components of the given
    /// lower and upper bounds.
    pub fn rectangle_from_horizontal_bounds(
        lower_bounds: &Vec3<F>,
        upper_bounds: &Vec3<F>,
    ) -> Self {
        Self::rectangle_from_bounds(&lower_bounds.without_z(), &upper_bounds.without_z())
    }

    /// Returns the number of vertices of the polygon.
    pub fn n_vertices(&self) -> usize {
        self.vertices().len()
    }

    /// Returns a slice with the vertices of the polygon.
    pub fn vertices(&self) -> &[Point2<F>] {
        &self.vertices
    }

    /// Returns an `Option` with the lower and upper bounds of the polygon,
    /// or `None` if the polygon is empty.
    pub fn bounds(&self) -> Option<(Vec2<F>, Vec2<F>)> {
        self.vertices().first().map(|first_vertex| {
            self.vertices().iter().skip(1).map(Point2::to_vec2).fold(
                (first_vertex.to_vec2(), first_vertex.to_vec2()),
                |(lower_bounds, upper_bounds), vertex| {
                    (
                        vertex.min_with(&lower_bounds),
                        vertex.max_with(&upper_bounds),
                    )
                },
            )
        })
    }

    /// Returns an `Option` with the area of the polygon,
    /// or `None` if the polygon has no area.
    pub fn area(&self) -> Option<F> {
        let mut signed_area = checked_vertex_pair_iter!(self.vertices())?.fold(
            F::zero(),
            |signed_area, (current_vertex, next_vertex)| {
                signed_area + current_vertex[Dim2::X] * next_vertex[Dim2::Y]
                    - next_vertex[Dim2::X] * current_vertex[Dim2::Y]
            },
        );
        if signed_area == F::zero() {
            None
        } else {
            signed_area = signed_area * F::from_f32(0.5).unwrap();
            Some(<F as num::Float>::abs(signed_area))
        }
    }

    /// Returns an `Option` with the area and centroid of the polygon,
    /// or `None` if the polygon has no area.
    pub fn area_and_centroid(&self) -> Option<(F, Point2<F>)> {
        let (mut signed_area, centroid_x, centroid_y) = checked_vertex_pair_iter!(self.vertices())?
            .fold(
                (F::zero(), F::zero(), F::zero()),
                |(signed_area, centroid_x, centroid_y), (current_vertex, next_vertex)| {
                    let area_contribution = current_vertex[Dim2::X] * next_vertex[Dim2::Y]
                        - next_vertex[Dim2::X] * current_vertex[Dim2::Y];
                    (
                        signed_area + area_contribution,
                        centroid_x
                            + (current_vertex[Dim2::X] + next_vertex[Dim2::X]) * area_contribution,
                        centroid_y
                            + (current_vertex[Dim2::Y] + next_vertex[Dim2::Y]) * area_contribution,
                    )
                },
            );
        signed_area = signed_area * F::from_f32(0.5).unwrap();
        let area = <F as num::Float>::abs(signed_area);

        if signed_area == F::zero() {
            None
        // Avoid numerical instability for almost degenerate polygons
        } else if area < F::from_f32(1e-6).unwrap() {
            Some((area, self.vertex_centroid().unwrap()))
        } else {
            let centroid_norm = F::one() / (F::from_i32(6).unwrap() * signed_area);
            let centroid = Point2::new(centroid_norm * centroid_x, centroid_norm * centroid_y);
            Some((area, centroid))
        }
    }

    /// Returns an `Option` with the centroid of the set of polygon vertices,
    /// or `None` if the polygon is empty.
    pub fn vertex_centroid(&self) -> Option<Point2<F>> {
        self.vertices
            .iter()
            .map(Point2::to_vec2)
            .reduce(|summed_vertices, vertex| summed_vertices + vertex)
            .map(|summed_vertices| {
                (summed_vertices / F::from_usize(self.n_vertices()).unwrap()).to_point2()
            })
    }

    /// Returns and `Option` with the polygon defined by the intersection between this
    /// and the given polygon, or `None` if the polygons do not intersect.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        // Use this polygon as first guess for intersection polygon
        let mut intersection_polygon = Cow::Borrowed(self);

        // Loop through edges of the other polygon
        for (current_vertex_other, next_vertex_other) in
            checked_vertex_pair_iter!(other.vertices())?
        {
            // No intersection if the intersection polygon has fewer than 3 vertices
            if intersection_polygon.n_vertices() <= 2 {
                return None;
            }

            // Find line corresponding to the current edge of the other polygon
            let edge_line_other = Line2::new_from_points(current_vertex_other, next_vertex_other);

            // Evaluate edge line equation for all vertices of the intersection polygon.
            // Vertices with line values > 0 are outside the other polygon.
            let vertex_line_values: Vec<_> = intersection_polygon
                .vertices()
                .iter()
                .map(|vertex| edge_line_other.evaluate(vertex))
                .collect();

            // Next intersection polygon candidate will correspond to the current candidate
            // except that any part outside the current edge of the other polygon will be cut away
            let mut new_intersection_polygon = Self::empty_with_capacity(5);

            // Loop through edges of the current intersection polygon candidate
            for (
                (current_vertex, next_vertex),
                (&current_vertex_line_value, &next_vertex_line_value),
            ) in vertex_pair_iter!(intersection_polygon.vertices())
                .zip(vertex_pair_iter!(vertex_line_values))
            {
                if current_vertex_line_value <= F::zero() {
                    // Vertex where the edge starts is not outside the other polygon, so keep it in the next candidate
                    new_intersection_polygon.add_vertex(current_vertex.clone());
                }
                if current_vertex_line_value * next_vertex_line_value < F::zero() {
                    // Edge crosses the other polygon, so find the intersection and add as vertex in the next candidate
                    let edge_line = Line2::new_from_points(current_vertex, next_vertex);
                    new_intersection_polygon
                        .add_vertex(edge_line_other.intersection(&edge_line).unwrap());
                }
            }
            intersection_polygon = Cow::Owned(new_intersection_polygon);
        }

        if intersection_polygon.n_vertices() <= 2 {
            None
        } else {
            Some(intersection_polygon.into_owned())
        }
    }

    /// Adds the given vertex to the polygon.
    pub fn add_vertex(&mut self, vertex: Point2<F>) {
        self.vertices.push(vertex);
    }

    /// Creates a version of the polygon where the vertices have been transformed
    /// with the given point transformation.
    pub fn transformed<T>(&self, transformation: &T) -> Self
    where
        T: PointTransformation2<F>,
    {
        Self::new(
            self.vertices()
                .iter()
                .map(|point| transformation.transform(point))
                .collect(),
        )
    }
}

/// Defines the properties of a transformation of 2D points.
pub trait PointTransformation2<F: BFloat>: Sync {
    /// Whether the transformation is the identity transformation.
    const IS_IDENTITY: bool = false;

    /// Returns the transformed version of the given 2D point.
    fn transform(&self, point: &Point2<F>) -> Point2<F>;

    /// Returns the horizontally transformed version of the given 3D point.
    fn transform_horizontally(&self, point: &Point3<F>) -> Point3<F> {
        let transformed_hor_point = self.transform(&point.without_z());
        Point3::new(
            transformed_hor_point[Dim2::X],
            transformed_hor_point[Dim2::Y],
            point[Z],
        )
    }
}

/// Identity transformation for 2D points.
pub struct IdentityTransformation2<F> {
    _phantom: PhantomData<F>,
}

impl<F> IdentityTransformation2<F> {
    /// Creates a new identity transformation.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F> Default for IdentityTransformation2<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BFloat> PointTransformation2<F> for IdentityTransformation2<F> {
    const IS_IDENTITY: bool = true;

    fn transform(&self, point: &Point2<F>) -> Point2<F> {
        point.clone()
    }
}

/// Transformation for rotating 2D points about the origin.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationTransformation2<F> {
    rotated_x_axis_unit_vec: Vec2<F>,
    rotated_y_axis_unit_vec: Vec2<F>,
}

impl<F: BFloat> RotationTransformation2<F> {
    /// Creates a new 2D point rotation transformation from the given rotated vector.
    pub fn new_from_rotated_vec(mut rotated_vec: Vec2<F>) -> Self {
        rotated_vec.normalize();
        let rotated_y_axis_unit_vec = rotated_vec.rotated_90();
        Self {
            rotated_x_axis_unit_vec: rotated_vec,
            rotated_y_axis_unit_vec,
        }
    }

    /// Creates a new 2D point rotation transformation from the given angle (in radians).
    pub fn new_from_angle(angle_rad: F) -> Self {
        Self::new_from_rotated_vec(Vec2::new(F::cos(angle_rad), F::sin(angle_rad)))
    }
}

impl<F: BFloat> PointTransformation2<F> for RotationTransformation2<F> {
    fn transform(&self, point: &Point2<F>) -> Point2<F> {
        (&self.rotated_x_axis_unit_vec * point[Dim2::X]
            + &self.rotated_y_axis_unit_vec * point[Dim2::Y])
            .to_point2()
    }
}

/// Transformation for translating 2D points.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationTransformation2<F> {
    translation: Vec2<F>,
}

impl<F: BFloat> TranslationTransformation2<F> {
    /// Creates a new 2D point translation transformation from the given translation vector.
    pub fn new(translation: Vec2<F>) -> Self {
        Self { translation }
    }
}

impl<F: BFloat> PointTransformation2<F> for TranslationTransformation2<F> {
    fn transform(&self, point: &Point2<F>) -> Point2<F> {
        point + &self.translation
    }
}

/// Transformation for rotating 2D points about the origin and then translating the origin.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationAndTranslationTransformation2<F> {
    rotation: RotationTransformation2<F>,
    translation: TranslationTransformation2<F>,
}

impl<F: BFloat> RotationAndTranslationTransformation2<F> {
    pub fn new(
        rotation: RotationTransformation2<F>,
        translation: TranslationTransformation2<F>,
    ) -> Self {
        Self {
            rotation,
            translation,
        }
    }
}

impl<F: BFloat> PointTransformation2<F> for RotationAndTranslationTransformation2<F> {
    fn transform(&self, point: &Point2<F>) -> Point2<F> {
        self.translation.transform(&self.rotation.transform(point))
    }
}
