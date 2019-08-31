//! Field lines in Bifrost vector fields.

pub mod natural;
pub mod regular;

use std::{io, path};
use std::collections::HashMap;
use serde::Serialize;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use super::stepping::Stepper3;
use super::ftr;

/// Data associated with a 3D field line.
#[derive(Serialize)]
struct FieldLineData3 {
    positions: Vec<Point3<ftr>>,
    scalar_values: HashMap<String, Vec<ftr>>,
    vector_values: HashMap<String, Vec<Vec3<ftr>>>
}

/// Defines the properties of a field line of a 3D vector field.
pub trait FieldLine3 {
    /// Returns a reference to the positions making up the field line.
    fn positions(&self) -> &Vec<Point3<ftr>>;

    /// Traces the field line through a 3D vector field.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to trace.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper`: Stepper to use (will be consumed).
    /// - `start_position`: Position where the tracing should start.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `S`: Type of stepper.
    fn trace<F, G, I, S>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>)
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3,
            S: Stepper3;

    /// Stores the given scalar values for the field line points.
    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>);

    /// Stores the given vector values for the field line points.
    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>);

    /// Serializes the field line data into a pickle file saved at the given path.
    fn save_as_pickle(&self, file_path: &path::Path) -> io::Result<()>;

    /// Returns the number of points making up the field line.
    fn number_of_points(&self) -> usize { self.positions().len() }

    /// Extracts and stores the value of the given scalar field at each field line point.
    fn extract_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3
    {
        let mut values = Vec::with_capacity(self.number_of_points());
        for pos in self.positions() {
            let value = interpolator.interp_scalar_field(field, &Point3::from(pos)).unwrap();
            values.push(num::NumCast::from(value).unwrap());
        }
        self.add_scalar_values(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at each field line point.
    fn extract_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3
    {
        let mut values = Vec::with_capacity(self.number_of_points());
        for pos in self.positions() {
            let value = interpolator.interp_vector_field(field, &Point3::from(pos)).unwrap();
            values.push(Vec3::from(&value));
        }
        self.add_vector_values(field.name().to_string(), values);
    }
}

impl FieldLineData3 {
    fn new() -> Self {
        FieldLineData3{
            positions: Vec::new(),
            scalar_values: HashMap::new(),
            vector_values: HashMap::new()
        }
    }
}