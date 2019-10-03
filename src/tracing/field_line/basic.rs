//! Basic field line tracing.

use super::super::stepping::{Stepper3, StepperInstruction, SteppingSense};
use super::{FieldLineData3, FieldLineTracer3};
use crate::field::VectorField3;
use crate::geometry::{Dim3, Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::num::BFloat;
use crate::tracing::{self, ftr, TracerResult};
use std::collections::VecDeque;
use Dim3::{X, Y, Z};

/// Whether to trace a field line a specified direction or in both directions.
#[derive(Clone, Copy, Debug)]
pub enum FieldLineTracingSense {
    Both,
    One(SteppingSense),
}

impl FieldLineTracingSense {
    pub fn same() -> Self {
        FieldLineTracingSense::One(SteppingSense::Same)
    }
    pub fn opposite() -> Self {
        FieldLineTracingSense::One(SteppingSense::Opposite)
    }
}

/// Whether to trace a field line with regular or natural spacing between points.
#[derive(Clone, Copy, Debug)]
pub enum FieldLinePointSpacing {
    Regular,
    Natural,
}

/// Configuration parameters for basic field line tracer.
#[derive(Clone, Debug)]
pub struct BasicFieldLineTracerConfig {
    /// Direction(s) to trace the field line relative to the field direction.
    pub tracing_sense: FieldLineTracingSense,
    /// Form of spacing between field line points.
    pub point_spacing: FieldLinePointSpacing,
    /// Field lines reaching lengths larger than this will be terminated.
    pub max_length: Option<ftr>,
}

/// A basic field line tracer for a 3D vector fields.
#[derive(Clone, Debug)]
pub struct BasicFieldLineTracer3 {
    config: BasicFieldLineTracerConfig,
}

impl BasicFieldLineTracer3 {
    /// Creates a new basic field line tracer.
    pub fn new(config: BasicFieldLineTracerConfig) -> Self {
        config.validate();
        BasicFieldLineTracer3 { config }
    }
}

impl FieldLineTracer3 for BasicFieldLineTracer3 {
    fn trace<F, G, I, S>(
        &self,
        field: &VectorField3<F, G>,
        interpolator: &I,
        stepper: S,
        start_position: &Point3<ftr>,
    ) -> Option<FieldLineData3>
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        S: Stepper3,
    {
        let mut backward_path = (VecDeque::new(), VecDeque::new(), VecDeque::new());
        let mut backward_length = 0.0;

        if let FieldLineTracingSense::Both = self.config.tracing_sense {
            let tracer_result = if let Some(length) = self.config.max_length {
                let mut callback = |_: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
                    if distance <= length {
                        backward_path.0.push_front(position[X]);
                        backward_path.1.push_front(position[Y]);
                        backward_path.2.push_front(position[Z]);
                        backward_length = distance;
                        StepperInstruction::Continue
                    } else {
                        StepperInstruction::Terminate
                    }
                };
                match self.config.point_spacing {
                    FieldLinePointSpacing::Regular => tracing::trace_3d_field_line_dense(
                        field,
                        interpolator,
                        stepper.clone(),
                        start_position,
                        SteppingSense::Opposite,
                        &mut callback,
                    ),
                    FieldLinePointSpacing::Natural => tracing::trace_3d_field_line(
                        field,
                        interpolator,
                        stepper.clone(),
                        start_position,
                        SteppingSense::Opposite,
                        &mut callback,
                    ),
                }
            } else {
                let mut callback = |_: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
                    backward_path.0.push_front(position[X]);
                    backward_path.1.push_front(position[Y]);
                    backward_path.2.push_front(position[Z]);
                    backward_length = distance;
                    StepperInstruction::Continue
                };
                match self.config.point_spacing {
                    FieldLinePointSpacing::Regular => tracing::trace_3d_field_line_dense(
                        field,
                        interpolator,
                        stepper.clone(),
                        start_position,
                        SteppingSense::Opposite,
                        &mut callback,
                    ),
                    FieldLinePointSpacing::Natural => tracing::trace_3d_field_line(
                        field,
                        interpolator,
                        stepper.clone(),
                        start_position,
                        SteppingSense::Opposite,
                        &mut callback,
                    ),
                }
            };

            if let TracerResult::Void = tracer_result {
                return None;
            }

            // Remove start position
            backward_path.0.pop_back().unwrap();
            backward_path.1.pop_back().unwrap();
            backward_path.2.pop_back().unwrap();
        }

        let mut forward_path = (Vec::new(), Vec::new(), Vec::new());
        let mut forward_length = 0.0;

        let sense = match self.config.tracing_sense {
            FieldLineTracingSense::Both => SteppingSense::Same,
            FieldLineTracingSense::One(sense) => sense,
        };

        let tracer_result = if let Some(length) = self.config.max_length {
            let mut callback = |_: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
                if distance <= length {
                    forward_path.0.push(position[X]);
                    forward_path.1.push(position[Y]);
                    forward_path.2.push(position[Z]);
                    forward_length = distance;
                    StepperInstruction::Continue
                } else {
                    StepperInstruction::Terminate
                }
            };
            match self.config.point_spacing {
                FieldLinePointSpacing::Regular => tracing::trace_3d_field_line_dense(
                    field,
                    interpolator,
                    stepper,
                    start_position,
                    sense,
                    &mut callback,
                ),
                FieldLinePointSpacing::Natural => tracing::trace_3d_field_line(
                    field,
                    interpolator,
                    stepper,
                    start_position,
                    sense,
                    &mut callback,
                ),
            }
        } else {
            let mut callback = |_: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
                forward_path.0.push(position[X]);
                forward_path.1.push(position[Y]);
                forward_path.2.push(position[Z]);
                forward_length = distance;
                StepperInstruction::Continue
            };
            match self.config.point_spacing {
                FieldLinePointSpacing::Regular => tracing::trace_3d_field_line_dense(
                    field,
                    interpolator,
                    stepper,
                    start_position,
                    sense,
                    &mut callback,
                ),
                FieldLinePointSpacing::Natural => tracing::trace_3d_field_line(
                    field,
                    interpolator,
                    stepper,
                    start_position,
                    sense,
                    &mut callback,
                ),
            }
        };

        if let TracerResult::Void = tracer_result {
            return None;
        }

        let path = if let FieldLineTracingSense::Both = self.config.tracing_sense {
            let mut path = (
                Vec::from(backward_path.0),
                Vec::from(backward_path.1),
                Vec::from(backward_path.2),
            );
            path.0.extend(forward_path.0);
            path.1.extend(forward_path.1);
            path.2.extend(forward_path.2);
            path
        } else {
            forward_path
        };

        let total_length = backward_length + forward_length;

        Some(FieldLineData3 { path, total_length })
    }
}

impl BasicFieldLineTracerConfig {
    const DEFAULT_TRACING_SENSE: FieldLineTracingSense = FieldLineTracingSense::Both;
    const DEFAULT_POINT_SPACING: FieldLinePointSpacing = FieldLinePointSpacing::Regular;
    const DEFAULT_MAX_LENGTH: Option<ftr> = None;

    fn validate(&self) {
        if let Some(length) = self.max_length {
            assert!(
                length >= 0.0,
                "Maximum field line length must be non-negative."
            );
        }
    }
}

impl Default for BasicFieldLineTracerConfig {
    fn default() -> Self {
        BasicFieldLineTracerConfig {
            tracing_sense: Self::DEFAULT_TRACING_SENSE,
            point_spacing: Self::DEFAULT_POINT_SPACING,
            max_length: Self::DEFAULT_MAX_LENGTH,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::geometry::{Dim3, In2D};
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig};
    use crate::io::snapshot::SnapshotReader3;
    use crate::io::{Endianness, Verbose};
    use crate::tracing::field_line::FieldLineSet3;
    use crate::tracing::seeding::slice::SliceSeeder3;
    use crate::tracing::stepping::rkf::rkf45::RKF45StepperFactory3;
    use crate::tracing::stepping::rkf::RKFStepperConfig;

    #[test]
    fn tracing_and_saving_basic_field_line_set_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new(
            "data/en024031_emer3.0sml_ebeam_631.idl",
            Endianness::Little,
        )
        .unwrap();
        let magnetic_field = reader.read_vector_field("b").unwrap();

        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());
        let stepper_factory = RKF45StepperFactory3::new(RKFStepperConfig::default());
        let seeder =
            SliceSeeder3::stratified(magnetic_field.grid(), Dim3::Z, 0.0, In2D::same(3), 1, 0.6);

        let tracer = BasicFieldLineTracer3::new(BasicFieldLineTracerConfig::default());

        let field_line_set = FieldLineSet3::trace(
            seeder,
            &tracer,
            &magnetic_field,
            &interpolator,
            stepper_factory,
            Verbose::No,
        );
        field_line_set
            .save_as_combined_pickles("data/natural_field_line_set.pickle")
            .unwrap();
    }
}
