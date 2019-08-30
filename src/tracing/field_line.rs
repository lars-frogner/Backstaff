//! Field lines in Bifrost vector fields.

use std::{io, path, fs};
use std::collections::HashMap;
use serde::Serialize;
use serde_pickle;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use super::stepping::{Stepper3, StepperInstruction};
use super::{ftr, trace, trace_dense};

/// A field line of a vector field.
#[derive(Serialize)]
pub struct FieldLine {
    positions: Vec<Point3<ftr>>,
    scalar_values: HashMap<String, Vec<ftr>>,
    vector_values: HashMap<String, Vec<Vec3<ftr>>>
}

impl FieldLine {
    /// Creates a new field line by tracing the given vector field from the given start position
    /// using the given interpolator and stepper, and storing the natural position of each step.
    pub fn trace<F, G, I, S>(field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>) -> Self
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3,
            S: Stepper3
    {
        let mut positions = Vec::new();
        let mut add_position = |pos: &Point3<_>| {
            positions.push(pos.clone());
            StepperInstruction::Continue
        };

        trace(field, interpolator, stepper, start_position, &mut add_position);

        FieldLine{
            positions,
            scalar_values: HashMap::new(),
            vector_values: HashMap::new()
        }
    }

    /// Creates a new field line by tracing the given vector field from the given start position
    /// using the given interpolator and stepper, and storing regularly spaced positions along
    /// the field line.
    pub fn trace_dense<F, G, I, S>(field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>) -> Self
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3,
            S: Stepper3
    {
        let mut positions = Vec::new();
        let mut add_position = |pos: &Point3<_>| {
            positions.push(pos.clone());
            StepperInstruction::Continue
        };

        trace_dense(field, interpolator, stepper, start_position, &mut add_position);

        FieldLine{
            positions,
            scalar_values: HashMap::new(),
            vector_values: HashMap::new()
        }
    }

    /// Returns the number of points making up the field line.
    pub fn number_of_points(&self) -> usize { self.positions.len() }

    /// Extracts and stores the value of the given scalar field at each field line point.
    pub fn extract_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3
    {
        let mut values = Vec::with_capacity(self.number_of_points());
        for pos in &self.positions {
            let value = interpolator.interp_scalar_field(field, &Point3::from(pos)).unwrap();
            values.push(num::NumCast::from(value).unwrap());
        }
        self.scalar_values.insert(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at each field line point.
    pub fn extract_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3
    {
        let mut values = Vec::with_capacity(self.number_of_points());
        for pos in &self.positions {
            let value = interpolator.interp_vector_field(field, &Point3::from(pos)).unwrap();
            values.push(Vec3::from(&value));
        }
        self.vector_values.insert(field.name().to_string(), values);
    }

    /// Serializes the field line data into a protocol 3 pickle file saved at the given path.
    pub fn save_as_pickle(&self, file_path: &path::Path) -> io::Result<()> {
        let mut file = fs::File::create(file_path)?;
        match serde_pickle::to_writer(&mut file, self, true) {
            Ok(_) => Ok(()),
            Err(serde_pickle::Error::Io(err)) => Err(err),
            Err(_) => Err(io::Error::new(io::ErrorKind::Other, "Unexpected error while serializing field line to pickle file"))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::reading::{SnapshotReader, Endianness};
    use crate::interpolation::poly_fit::PolyFitInterpolator3;
    use crate::tracing::stepping::rkf::RKFStepperConfig;
    use crate::tracing::stepping::rkf::rkf23::RKF23Stepper3;
    use crate::tracing::stepping::rkf::rkf45::RKF45Stepper3;

    #[test]
    fn tracing_and_saving_field_line_works() {
        let params_path = path::PathBuf::from("data/en024031_emer3.0sml_ebeam_631.idl");
        let reader: SnapshotReader<HorRegularGrid3<f32>> = SnapshotReader::new(&params_path, Endianness::Little).unwrap();
        let magnetic_field = reader.read_3d_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3;

        let stepper_23 = RKF23Stepper3::new(RKFStepperConfig::default());
        let stepper_45 = RKF45Stepper3::new(RKFStepperConfig::default());
        let stepper_23_dense = RKF23Stepper3::new(RKFStepperConfig{ dense_step_size: 0.1, ..RKFStepperConfig::default()});
        let stepper_45_dense = RKF45Stepper3::new(RKFStepperConfig{ dense_step_size: 0.1, ..RKFStepperConfig::default()});

        let start_position = Point3::new(12.0, 12.0, -10.0);

        let field_line_23 = FieldLine::trace(&magnetic_field, &interpolator, stepper_23, &start_position);
        let field_line_45 = FieldLine::trace(&magnetic_field, &interpolator, stepper_45, &start_position);
        let field_line_23_dense = FieldLine::trace_dense(&magnetic_field, &interpolator, stepper_23_dense, &start_position);
        let field_line_45_dense = FieldLine::trace_dense(&magnetic_field, &interpolator, stepper_45_dense, &start_position);

        field_line_23.save_as_pickle(&path::PathBuf::from("data/test_field_line_23.pickle")).unwrap();
        field_line_45.save_as_pickle(&path::PathBuf::from("data/test_field_line_45.pickle")).unwrap();
        field_line_23_dense.save_as_pickle(&path::PathBuf::from("data/test_field_line_23_dense.pickle")).unwrap();
        field_line_45_dense.save_as_pickle(&path::PathBuf::from("data/test_field_line_45_dense.pickle")).unwrap();
    }
}
