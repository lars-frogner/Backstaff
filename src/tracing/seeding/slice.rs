//! Generation of seed points along a slice through a field.

use std::{fmt, vec};
use num;
use crate::geometry::{Dim3, Point3};
use crate::grid::{Grid3, CoordLocation};
use crate::field::{ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use super::super::ftr;

pub enum SeedDistribution {
    Uniform,
    Stratified,
    ValueRelative
}

pub struct SliceSeeder3 {
    seed_points: Vec<Point3<ftr>>
}

impl SliceSeeder3 {
    pub fn for_scalar_field<F, G, I>(field: ScalarField3<F, G>, interpolator: &I, axis: Dim3, coord: ftr)
    where F: num::Float + num::FromPrimitive + fmt::Display,
          G: Grid3<F>,
          I: Interpolator3
    {
        let _slice_field = field.regular_slice_across_axis(interpolator, axis, F::from(coord).unwrap(), CoordLocation::Center);

        // flatten slice
        // normalize to unit integral to get probability distribution as function of index
        // compute cumulative distribution
        // draw random number between 0 and 1
        // find index corresponding to this value in the cumulative distribution
    }

    pub fn for_vector_field<F, G, I>(field: VectorField3<F, G>, interpolator: &I, axis: Dim3, coord: ftr)
    where F: num::Float + num::FromPrimitive + fmt::Display,
          G: Grid3<F>,
          I: Interpolator3
    {
        let _slice_field = field.regular_slice_across_axis(interpolator, axis, F::from(coord).unwrap(), CoordLocation::Center);
    }
}

impl IntoIterator for SliceSeeder3 {
    type Item = Point3<ftr>;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.seed_points.into_iter()
    }
}
