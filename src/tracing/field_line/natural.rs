//! Field lines with natural spacing between points.

use std::fmt;
use std::collections::VecDeque;
use num;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{VectorField3};
use crate::interpolation::Interpolator3;
use super::super::{ftr, TracerResult, trace_3d_field_line};
use super::super::stepping::{Stepper3, SteppingSense, StepperInstruction};
use super::{FieldLineData3, FieldLine3};

/// A field line of a 3D vector field with points corresponding to the natural position of each step.
pub struct NaturalFieldLine3 {
    sense: SteppingSense,
    data: FieldLineData3
}

/// A field line of a 3D vector field with points corresponding to the natural position of each step,
/// traced both forward and backward along the field direction.
#[derive(Default)]
pub struct DualNaturalFieldLine3 {
    data: FieldLineData3
}

impl NaturalFieldLine3 {
    /// Creates a new empty field line.
    pub fn new(sense: SteppingSense) -> Self {
        NaturalFieldLine3{ sense, data: FieldLineData3::new() }
    }
}

impl DualNaturalFieldLine3 {
    /// Creates a new empty field line.
    pub fn new() -> Self {
        DualNaturalFieldLine3{ data: FieldLineData3::new() }
    }
}

impl FieldLine3 for NaturalFieldLine3 {
    type Data = FieldLineData3;

    fn data(&self) -> &Self::Data { &self.data }
    fn positions(&self) -> &Vec<Point3<ftr>> { &self.data.positions }

    fn trace<F, G, I, S>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>) -> TracerResult
    where F: num::Float + fmt::Display,
          G: Grid3<F>,
          I: Interpolator3,
          S: Stepper3
    {
        let sense = self.sense;
        let mut add_position = |pos: &Point3<_>| {
            self.data.positions.push(pos.clone());
            StepperInstruction::Continue
        };
        trace_3d_field_line(field, interpolator, stepper, start_position, sense, &mut add_position)
    }

    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>) {
        self.data.scalar_values.insert(field_name, values);
    }

    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>) {
        self.data.vector_values.insert(field_name, values);
    }
}

impl FieldLine3 for DualNaturalFieldLine3 {
    type Data = FieldLineData3;

    fn data(&self) -> &Self::Data { &self.data }
    fn positions(&self) -> &Vec<Point3<ftr>> { &self.data.positions }

    fn trace<F, G, I, S>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>) -> TracerResult
    where F: num::Float + fmt::Display,
          G: Grid3<F>,
          I: Interpolator3,
          S: Stepper3
    {
        let mut backward_positions = VecDeque::new();
        let mut add_backward_position = |pos: &Point3<_>| {
            backward_positions.push_front(pos.clone());
            StepperInstruction::Continue
        };
        if let TracerResult::Void = trace_3d_field_line(field, interpolator, stepper.clone(), start_position, SteppingSense::Opposite, &mut add_backward_position) {
            return TracerResult::Void
        }
        backward_positions.pop_back().unwrap(); // Remove start position

        let mut forward_positions = Vec::new();
        let mut add_forward_position = |pos: &Point3<_>| {
            forward_positions.push(pos.clone());
            StepperInstruction::Continue
        };
        if let TracerResult::Void = trace_3d_field_line(field, interpolator, stepper, start_position, SteppingSense::Same, &mut add_forward_position) {
            return TracerResult::Void
        }

        self.data.positions = Vec::from(backward_positions);
        self.data.positions.extend(forward_positions);

        TracerResult::Ok(None)
    }

    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>) {
        self.data.scalar_values.insert(field_name, values);
    }

    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>) {
        self.data.vector_values.insert(field_name, values);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::path;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::io::Endianness;
    use crate::io::snapshot::SnapshotReader3;
    use crate::interpolation::poly_fit::PolyFitInterpolator3;
    use crate::tracing::stepping::rkf::RKFStepperConfig;
    use crate::tracing::stepping::rkf::rkf23::{RKF23Stepper3};
    use crate::tracing::stepping::rkf::rkf45::{RKF45Stepper3, RKF45StepperFactory3};
    use crate::tracing::field_line::FieldLineSet3;

    #[test]
    fn tracing_and_saving_natural_field_line_works() {
        let params_path = path::PathBuf::from("data/en024031_emer3.0sml_ebeam_631.idl");
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new(&params_path, Endianness::Little).unwrap();
        let magnetic_field = reader.read_3d_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3;

        let stepper_23 = RKF23Stepper3::new(RKFStepperConfig::default());
        let stepper_45 = RKF45Stepper3::new(RKFStepperConfig::default());

        let start_position = Point3::new(12.0, 12.0, -10.0);

        let mut field_line_23 = NaturalFieldLine3::new(SteppingSense::Same);
        let mut field_line_45 = NaturalFieldLine3::new(SteppingSense::Same);

        field_line_23.trace(&magnetic_field, &interpolator, stepper_23, &start_position);
        field_line_45.trace(&magnetic_field, &interpolator, stepper_45, &start_position);

        field_line_23.save_as_pickle(&path::PathBuf::from("data/natural_field_line_23.pickle")).unwrap();
        field_line_45.save_as_pickle(&path::PathBuf::from("data/natural_field_line_45.pickle")).unwrap();
    }

    #[test]
    fn tracing_and_saving_natural_field_line_set_works() {
        let params_path = path::PathBuf::from("data/en024031_emer3.0sml_ebeam_631.idl");
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new(&params_path, Endianness::Little).unwrap();
        let magnetic_field = reader.read_3d_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3;
        let stepper_factory = RKF45StepperFactory3::new(RKFStepperConfig::default());
        let seeder = vec![Point3::new(12.0, 12.0, -10.0), Point3::new(12.0, 12.0, -7.0)];

        let field_line_set = FieldLineSet3::trace(&magnetic_field, &interpolator, stepper_factory, seeder, &|| DualNaturalFieldLine3::new() ).unwrap();
        field_line_set.save_as_pickle(&path::PathBuf::from("data/natural_field_line_set.pickle")).unwrap();
    }
}