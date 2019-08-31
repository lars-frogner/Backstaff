//! Field lines with natural spacing between points.

use std::{io, path};
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{VectorField3};
use crate::interpolation::Interpolator3;
use crate::io::utils::save_data_as_pickle;
use super::super::{ftr, trace_3d_field_line};
use super::super::stepping::{Stepper3, SteppingSense, StepperInstruction};
use super::{FieldLineData3, FieldLine3};

/// A field line of a 3D vector field with points corresponding to the natural position of each step.
pub struct NaturalFieldLine3 {
    sense: SteppingSense,
    data: FieldLineData3
}

impl NaturalFieldLine3 {
    /// Creates a new empty field line.
    pub fn new(sense: SteppingSense) -> Self {
        NaturalFieldLine3{ sense, data: FieldLineData3::new() }
    }
}

impl FieldLine3 for NaturalFieldLine3 {
    fn positions(&self) -> &Vec<Point3<ftr>> { &self.data.positions }

    fn trace<F, G, I, S>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>)
    where F: num::Float + std::fmt::Display,
            G: Grid3<F> + Clone,
            I: Interpolator3,
            S: Stepper3
    {
        let sense = self.sense;
        let mut add_position = |pos: &Point3<_>| {
            self.data.positions.push(pos.clone());
            StepperInstruction::Continue
        };
        trace_3d_field_line(field, interpolator, stepper, start_position, sense, &mut add_position);
    }

    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>) {
        self.data.scalar_values.insert(field_name, values);
    }

    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>) {
        self.data.vector_values.insert(field_name, values);
    }

    fn save_as_pickle(&self, file_path: &path::Path) -> io::Result<()> {
        save_data_as_pickle(&self.data, file_path)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::io::Endianness;
    use crate::io::snapshot::SnapshotReader;
    use crate::interpolation::poly_fit::PolyFitInterpolator3;
    use crate::tracing::stepping::rkf::RKFStepperConfig;
    use crate::tracing::stepping::rkf::rkf23::RKF23Stepper3;
    use crate::tracing::stepping::rkf::rkf45::RKF45Stepper3;

    #[test]
    fn tracing_and_saving_natural_field_line_works() {
        let params_path = path::PathBuf::from("data/en024031_emer3.0sml_ebeam_631.idl");
        let reader: SnapshotReader<HorRegularGrid3<f32>> = SnapshotReader::new(&params_path, Endianness::Little).unwrap();
        let magnetic_field = reader.read_3d_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3;

        let stepper_23 = RKF23Stepper3::new(RKFStepperConfig::default());
        let stepper_45 = RKF45Stepper3::new(RKFStepperConfig::default());

        let start_position = Point3::new(12.0, 12.0, -10.0);

        let mut field_line_23 = NaturalFieldLine3::new(SteppingSense::Same);
        let mut field_line_45 = NaturalFieldLine3::new(SteppingSense::Same);

        field_line_23.trace(&magnetic_field, &interpolator, stepper_23, &start_position);
        field_line_45.trace(&magnetic_field, &interpolator, stepper_45, &start_position);

        field_line_23.save_as_pickle(&path::PathBuf::from("data/test_field_line_23.pickle")).unwrap();
        field_line_45.save_as_pickle(&path::PathBuf::from("data/test_field_line_45.pickle")).unwrap();
    }
}