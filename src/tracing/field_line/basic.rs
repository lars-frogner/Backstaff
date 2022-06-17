//! Basic field line tracing.

use super::{
    super::stepping::{Stepper3, StepperInstruction, SteppingSense},
    FieldLinePath3, FieldLineSetProperties3, FieldLineTracer3,
};
use crate::{
    field::CachingScalarFieldProvider3,
    geometry::{
        Dim3::{X, Y, Z},
        Point3, Vec3,
    },
    interpolation::Interpolator3,
    io::snapshot::fdt,
    tracing::{self, ftr, TracerResult},
};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

/// Data required to represent a basic 3D field line.
pub struct BasicFieldLineData3 {
    path: FieldLinePath3,
    total_length: ftr,
}

/// Whether to trace a field line a specified direction or in both directions.
#[derive(Clone, Copy, Debug)]
pub enum FieldLineTracingSense {
    Both,
    One(SteppingSense),
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
    type Data = BasicFieldLineData3;

    fn trace<P, I, S>(
        &self,
        field_name: &str,
        snapshot: &P,
        interpolator: &I,
        stepper: S,
        start_position: &Point3<ftr>,
    ) -> Option<Self::Data>
    where
        P: CachingScalarFieldProvider3<fdt>,
        I: Interpolator3<fdt>,
        S: Stepper3,
    {
        let field = snapshot.cached_vector_field(field_name);

        let mut backward_path = (VecDeque::new(), VecDeque::new(), VecDeque::new());
        let mut backward_length = 0.0;

        if let FieldLineTracingSense::Both = self.config.tracing_sense {
            let tracer_result = if let Some(length) = self.config.max_length {
                let mut callback =
                    |_: &Vec3<ftr>, _: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
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
                let mut callback =
                    |_: &Vec3<ftr>, _: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
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
            let mut callback =
                |_: &Vec3<ftr>, _: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
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
            let mut callback =
                |_: &Vec3<ftr>, _: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
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

        Some(Self::Data { path, total_length })
    }
}

impl FromParallelIterator<BasicFieldLineData3> for FieldLineSetProperties3 {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = BasicFieldLineData3>,
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(|field_line| {
            (
                field_line.path.0,
                (
                    field_line.path.1,
                    (field_line.path.2, field_line.total_length),
                ),
            )
        });

        let (paths_x, (paths_y, (paths_z, total_lengths))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) =
            nested_tuples_iter.unzip();

        let number_of_field_lines = paths_x.len();
        let mut fixed_scalar_values = HashMap::new();
        let fixed_vector_values = HashMap::new();
        let mut varying_scalar_values = HashMap::new();
        let varying_vector_values = HashMap::new();

        fixed_scalar_values.insert(
            "x0".to_string(),
            paths_x.par_iter().map(|path_x| path_x[0]).collect(),
        );
        fixed_scalar_values.insert(
            "y0".to_string(),
            paths_y.par_iter().map(|path_y| path_y[0]).collect(),
        );
        fixed_scalar_values.insert(
            "z0".to_string(),
            paths_z.par_iter().map(|path_z| path_z[0]).collect(),
        );
        fixed_scalar_values.insert("total_length".to_string(), total_lengths);

        varying_scalar_values.insert("x".to_string(), paths_x);
        varying_scalar_values.insert("y".to_string(), paths_y);
        varying_scalar_values.insert("z".to_string(), paths_z);

        FieldLineSetProperties3 {
            number_of_field_lines,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
        }
    }
}

impl FieldLineTracingSense {
    pub fn same() -> Self {
        FieldLineTracingSense::One(SteppingSense::Same)
    }
    pub fn opposite() -> Self {
        FieldLineTracingSense::One(SteppingSense::Opposite)
    }
}

impl BasicFieldLineTracerConfig {
    pub const DEFAULT_TRACING_SENSE: FieldLineTracingSense = FieldLineTracingSense::Both;
    pub const DEFAULT_POINT_SPACING: FieldLinePointSpacing = FieldLinePointSpacing::Regular;
    pub const DEFAULT_MAX_LENGTH: Option<ftr> = None;

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
