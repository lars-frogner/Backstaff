//! Field lines in vector fields.

pub mod natural;
pub mod regular;

use std::{io, path};
use serde::Serialize;
use rayon::prelude::*;
use crate::num::BFloat;
use crate::io::utils::save_data_as_pickle;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use super::stepping::{StepperFactory3, Stepper3};
use super::seeding::Seeder3;
use super::{ftr, TracerResult};

/// Collection of 3D field lines.
#[derive(Serialize)]
pub struct FieldLineSet3<L: FieldLine3> {
    field_lines: Vec<L>
}

/// Defines the properties of a field line of a 3D vector field.
pub trait FieldLine3: Serialize {
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
    /// # Returns
    ///
    /// A `TracerResult` which is either:
    ///
    /// - `Ok`: Contains an `Option<StoppingCause>`, possibly indicating why tracing was terminated.
    /// - `Void`: No field line was traced.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `St`: Type of stepper.
    fn trace<F, G, I, St>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: St, start_position: &Point3<ftr>) -> TracerResult
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3,
          St: Stepper3;

    /// Stores the given scalar values for the field line points.
    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>);

    /// Stores the given vector values for the field line points.
    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>);

    /// Returns the number of points making up the field line.
    fn number_of_points(&self) -> usize { self.positions().len() }

    /// Extracts and stores the value of the given scalar field at each field line point.
    fn extract_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        let mut values = Vec::with_capacity(self.number_of_points());
        for pos in self.positions() {
            let value = interpolator.interp_scalar_field(field, &Point3::from(pos)).expect_inside();
            values.push(num::NumCast::from(value).expect("Conversion failed."));
        }
        self.add_scalar_values(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at each field line point.
    fn extract_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        let mut values = Vec::with_capacity(self.number_of_points());
        for pos in self.positions() {
            let value = interpolator.interp_vector_field(field, &Point3::from(pos)).expect_inside();
            values.push(Vec3::from(&value));
        }
        self.add_vector_values(field.name().to_string(), values);
    }

    /// Serializes the field line data into pickle format and save at the given path.
    fn save_as_pickle<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        save_data_as_pickle(file_path, &self)
    }
}

impl<L: FieldLine3> FieldLineSet3<L> {
    /// Traces all the field lines in the set from positions generated by the given seeder.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to trace.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper_factory`: Factory structure to use for producing steppers.
    /// - `seeder`: Seeder to use for generating start positions.
    /// - `field_line_initializer`: Closure for initializing empty field lines.
    ///
    /// # Returns
    ///
    /// An `Option` which is either:
    ///
    /// - `Some`: Contains a new `FieldLineSet3` with traced field lines.
    /// - `None`: No field lines were traced.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `StF`: Type of stepper factory.
    /// - `Sd`: Type of seeder.
    /// - `FI`: Function type with no parameters returning a value of type `L`.
    pub fn trace<F, G, I, StF, Sd, FI>(field: &VectorField3<F, G>, interpolator: &I, stepper_factory: StF, seeder: Sd, field_line_initializer: &FI) -> Option<Self>
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3,
          StF: StepperFactory3,
          Sd: Seeder3,
          FI: Fn() -> L
    {
        let seed_iter = seeder.into_iter();

        let field_lines: Vec<L> = seed_iter.filter_map(
            |start_position| {
                let mut field_line = field_line_initializer();
                if let TracerResult::Ok(_) = field_line.trace(field, interpolator, stepper_factory.produce(), &start_position) {
                    Some(field_line)
                } else {
                    None
                }
            }
        ).collect();

        if field_lines.is_empty() {
            None
        } else {
            Some(FieldLineSet3{ field_lines })
        }
    }

    /// Traces all the field lines in the set from positions generated by the given seeder, in parallel.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to trace.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper_factory`: Factory structure to use for producing steppers.
    /// - `seeder`: Seeder to use for generating start positions.
    /// - `field_line_initializer`: Closure for initializing empty field lines.
    ///
    /// # Returns
    ///
    /// An `Option` which is either:
    ///
    /// - `Some`: Contains a new `FieldLineSet3` with traced field lines.
    /// - `None`: No field lines were traced.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `StF`: Type of stepper factory.
    /// - `Sd`: Type of seeder.
    /// - `FI`: Function type with no parameters returning a value of type `L`.
    pub fn par_trace<F, G, I, StF, Sd, FI>(field: &VectorField3<F, G>, interpolator: &I, stepper_factory: StF, seeder: Sd, field_line_initializer: &FI) -> Option<Self>
    where L: Send,
          F: BFloat + Sync,
          G: Grid3<F> + Sync + Send,
          I: Interpolator3 + Sync,
          StF: StepperFactory3 + Sync,
          Sd: Seeder3 + IntoParallelIterator<Item = Point3<ftr>>,
          FI: Fn() -> L + Sync
    {
        let seed_iter = seeder.into_par_iter();

        let field_lines: Vec<L> = seed_iter.filter_map(
            |start_position| {
                let mut field_line = field_line_initializer();
                if let TracerResult::Ok(_) = field_line.trace(field, interpolator, stepper_factory.produce(), &start_position) {
                    Some(field_line)
                } else {
                    None
                }
            }
        ).collect();

        if field_lines.is_empty() {
            None
        } else {
            Some(FieldLineSet3{ field_lines })
        }
    }

    /// Serializes the field line data into pickle format and save at the given path.
    pub fn save_as_pickle<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        save_data_as_pickle(file_path, &self)
    }
}