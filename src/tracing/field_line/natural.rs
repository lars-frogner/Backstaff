//! Field lines with natural spacing between points.

use std::collections::{HashMap, VecDeque};
use serde::ser::{Serialize, Serializer, SerializeStruct};
use crate::num::BFloat;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{VectorField3};
use crate::interpolation::Interpolator3;
use super::super::{ftr, TracerResult, trace_3d_field_line};
use super::super::stepping::{Stepper3, SteppingSense, StepperInstruction};
use super::FieldLine3;

/// A field line of a 3D vector field with points corresponding to the natural position of each step.
pub struct NaturalFieldLine3 {
    sense: SteppingSense,
    max_length: Option<ftr>,
    positions: Vec<Point3<ftr>>,
    scalar_values: HashMap<String, Vec<ftr>>,
    vector_values: HashMap<String, Vec<Vec3<ftr>>>
}

/// A field line of a 3D vector field with points corresponding to the natural position of each step,
/// traced both forward and backward along the field direction.
pub struct DualNaturalFieldLine3 {
    max_length: Option<ftr>,
    positions: Vec<Point3<ftr>>,
    scalar_values: HashMap<String, Vec<ftr>>,
    vector_values: HashMap<String, Vec<Vec3<ftr>>>
}

impl NaturalFieldLine3 {
    /// Creates a new empty field line.
    pub fn new(sense: SteppingSense, max_length: Option<ftr>) -> Self {
        if let Some(length) = max_length {
            assert!(length >= 0.0, "Maximum field line length must be non-negative.");
        }
        NaturalFieldLine3{
            sense,
            max_length,
            positions: Vec::new(),
            scalar_values: HashMap::new(),
            vector_values: HashMap::new()
        }
    }
}

impl DualNaturalFieldLine3 {
    /// Creates a new empty field line.
    pub fn new(max_length: Option<ftr>) -> Self {
        if let Some(length) = max_length {
            assert!(length >= 0.0, "Maximum field line length must be non-negative.");
        }
        DualNaturalFieldLine3{
            max_length,
            positions: Vec::new(),
            scalar_values: HashMap::new(),
            vector_values: HashMap::new()
        }
    }
}

impl FieldLine3 for NaturalFieldLine3 {
    fn positions(&self) -> &Vec<Point3<ftr>> { &self.positions }

    fn trace<F, G, I, S>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>) -> TracerResult
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3,
          S: Stepper3
    {
        let sense = self.sense;
        if let Some(length) = self.max_length {
            trace_3d_field_line(field, interpolator, stepper, start_position, sense, &mut |dist, pos| {
                if dist <= length {
                    self.positions.push(pos.clone());
                    StepperInstruction::Continue
                } else {
                    StepperInstruction::Terminate
                }
            })
        } else {
            trace_3d_field_line(field, interpolator, stepper, start_position, sense, &mut |_, pos| {
                self.positions.push(pos.clone());
                StepperInstruction::Continue
            })
        }
    }

    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>) {
        self.scalar_values.insert(field_name, values);
    }

    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>) {
        self.vector_values.insert(field_name, values);
    }
}

impl FieldLine3 for DualNaturalFieldLine3 {
    fn positions(&self) -> &Vec<Point3<ftr>> { &self.positions }

    fn trace<F, G, I, S>(&mut self, field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>) -> TracerResult
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3,
          S: Stepper3
    {
        let mut backward_positions = VecDeque::new();

        let tracer_result = if let Some(length) = self.max_length {
            trace_3d_field_line(field, interpolator, stepper.clone(), start_position, SteppingSense::Opposite, &mut |dist, pos| {
                if dist <= length {
                    backward_positions.push_front(pos.clone());
                    StepperInstruction::Continue
                } else {
                    StepperInstruction::Terminate
                }
            })
        } else {
            trace_3d_field_line(field, interpolator, stepper.clone(), start_position, SteppingSense::Opposite, &mut |_, pos| {
                backward_positions.push_front(pos.clone());
                StepperInstruction::Continue
            })
        };

        if let TracerResult::Void = tracer_result {
            return TracerResult::Void
        }
        backward_positions.pop_back().unwrap(); // Remove start position

        let mut forward_positions = Vec::new();

        let tracer_result = if let Some(length) = self.max_length {
            trace_3d_field_line(field, interpolator, stepper, start_position, SteppingSense::Same, &mut |dist, pos| {
                if dist <= length {
                    forward_positions.push(pos.clone());
                    StepperInstruction::Continue
                } else {
                    StepperInstruction::Terminate
                }
            })
        } else {
            trace_3d_field_line(field, interpolator, stepper, start_position, SteppingSense::Same, &mut |_, pos| {
                forward_positions.push(pos.clone());
                StepperInstruction::Continue
            })
        };

        if let TracerResult::Void = tracer_result {
            return TracerResult::Void
        }

        self.positions = Vec::from(backward_positions);
        self.positions.extend(forward_positions);

        TracerResult::Ok(None)
    }

    fn add_scalar_values(&mut self, field_name: String, values: Vec<ftr>) {
        self.scalar_values.insert(field_name, values);
    }

    fn add_vector_values(&mut self, field_name: String, values: Vec<Vec3<ftr>>) {
        self.vector_values.insert(field_name, values);
    }
}

impl Serialize for NaturalFieldLine3 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("NaturalFieldLine3", 3)?;
        s.serialize_field("positions", &self.positions)?;
        s.serialize_field("scalar_values", &self.scalar_values)?;
        s.serialize_field("vector_values", &self.vector_values)?;
        s.end()
    }
}

impl Serialize for DualNaturalFieldLine3 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("DualNaturalFieldLine3", 3)?;
        s.serialize_field("positions", &self.positions)?;
        s.serialize_field("scalar_values", &self.scalar_values)?;
        s.serialize_field("vector_values", &self.vector_values)?;
        s.end()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::geometry::{Dim3, In2D};
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::io::Endianness;
    use crate::io::snapshot::SnapshotReader3;
    use crate::interpolation::poly_fit::PolyFitInterpolator3;
    use crate::tracing::stepping::rkf::RKFStepperConfig;
    use crate::tracing::stepping::rkf::rkf23::{RKF23Stepper3};
    use crate::tracing::stepping::rkf::rkf45::{RKF45Stepper3, RKF45StepperFactory3};
    use crate::tracing::seeding::slice::SliceSeeder3;
    use crate::tracing::field_line::FieldLineSet3;

    #[test]
    fn tracing_and_saving_natural_field_line_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new("data/en024031_emer3.0sml_ebeam_631.idl", Endianness::Little).unwrap();
        let magnetic_field = reader.read_3d_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3;

        let stepper_23 = RKF23Stepper3::new(RKFStepperConfig::default());
        let stepper_45 = RKF45Stepper3::new(RKFStepperConfig::default());

        let start_position = Point3::new(12.0, 12.0, -10.0);

        let mut field_line_23 = NaturalFieldLine3::new(SteppingSense::Same, None);
        let mut field_line_45 = NaturalFieldLine3::new(SteppingSense::Same, None);

        field_line_23.trace(&magnetic_field, &interpolator, stepper_23, &start_position);
        field_line_45.trace(&magnetic_field, &interpolator, stepper_45, &start_position);

        field_line_23.save_as_pickle("data/natural_field_line_23.pickle").unwrap();
        field_line_45.save_as_pickle("data/natural_field_line_45.pickle").unwrap();
    }

    #[test]
    fn tracing_and_saving_natural_field_line_set_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new("data/en024031_emer3.0sml_ebeam_631.idl", Endianness::Little).unwrap();
        let magnetic_field = reader.read_3d_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3;
        let stepper_factory = RKF45StepperFactory3::new(RKFStepperConfig::default());
        let seeder = SliceSeeder3::stratified(magnetic_field.grid(), Dim3::Z, 0.0, In2D::same(3), 1, 0.6);

        let field_line_set = FieldLineSet3::trace(&magnetic_field, &interpolator, stepper_factory, seeder, &|| DualNaturalFieldLine3::new(None) ).unwrap();
        field_line_set.save_as_combined_pickles("data/natural_field_line_set.pickle").unwrap();
    }
}
