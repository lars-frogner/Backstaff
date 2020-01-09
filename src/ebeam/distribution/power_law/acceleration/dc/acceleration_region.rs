//! Acceleration region tracer.

use super::super::super::super::feb;
use crate::geometry::{Dim3, Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::tracing::field_line::{FieldLineSetProperties3, FieldLineTracer3};
use crate::tracing::stepping::{Stepper3, StepperInstruction, SteppingSense};
use crate::tracing::{self, ftr, TracerResult};
use crate::units::solar::U_EL;
use rayon::prelude::*;
use std::collections::VecDeque;
use Dim3::{X, Y, Z};

/// Data required to represent an acceleration region.
pub struct AccelerationRegionData {
    path: (Vec<ftr>, Vec<ftr>, Vec<ftr>),
    total_length: ftr,
    parallel_electric_field_strengths: Vec<feb>,
    average_parallel_electric_field_strength: feb,
    average_electric_magnetic_angle_cosine: feb,
}

/// Configuration parameters for acceleration region tracer.
#[derive(Clone, Debug)]
pub struct AccelerationRegionTracerConfig {
    /// The acceleration region ends where the absolute component of the electric field
    /// parallel to the magnetic field direction becomes lower than this value [V/m].
    pub min_parallel_electric_field_strength: feb,
    /// Acceleration regions shorter than this are discarded [Mm].
    pub min_length: ftr,
}

/// Tracer for extracting an acceleration region based on where the electric
/// field component parallel to the magnetic field direction is significant.
#[derive(Clone, Debug)]
pub struct AccelerationRegionTracer {
    config: AccelerationRegionTracerConfig,
    extra_varying_scalar_names: Vec<String>,
    extra_varying_vector_names: Vec<String>,
}

impl AccelerationRegionTracer {
    /// Creates a new acceleration region tracer.
    pub fn new(
        config: AccelerationRegionTracerConfig,
        extra_varying_scalar_names: Vec<String>,
        extra_varying_vector_names: Vec<String>,
    ) -> Self {
        config.validate();
        AccelerationRegionTracer {
            config,
            extra_varying_scalar_names,
            extra_varying_vector_names,
        }
    }

    /// Returns a reference to the configuration parameters for acceleration region tracer.
    pub fn config(&self) -> &AccelerationRegionTracerConfig {
        &self.config
    }

    /// Returns a reference to the list of scalar quantities to extract along acceleration regions.
    pub fn extra_varying_scalar_names(&self) -> &[String] {
        &self.extra_varying_scalar_names
    }

    /// Returns a reference to the list of vector quantities to extract along acceleration regions.
    pub fn extra_varying_vector_names(&self) -> &[String] {
        &self.extra_varying_vector_names
    }
}

impl FieldLineTracer3 for AccelerationRegionTracer {
    type Data = AccelerationRegionData;

    fn trace<G, I, S>(
        &self,
        _field_name: &str,
        snapshot: &SnapshotCacher3<G>,
        interpolator: &I,
        stepper: S,
        start_position: &Point3<ftr>,
    ) -> Option<Self::Data>
    where
        G: Grid3<fdt>,
        I: Interpolator3,
        S: Stepper3,
    {
        let magnetic_field = snapshot.cached_vector_field("b");
        let electric_field = snapshot.cached_vector_field("e");

        let mut average_electric_magnetic_angle_cosine = 0.0;

        let mut backward_path = (VecDeque::new(), VecDeque::new(), VecDeque::new());
        let mut backward_length = 0.0;
        let mut backward_parallel_electric_field_strengths = VecDeque::new();

        let mut callback =
            |_: &Vec3<ftr>, direction: &Vec3<ftr>, position: &Point3<ftr>, distance: ftr| {
                let electric_field_vector = Vec3::from(
                    &interpolator
                        .interp_vector_field(electric_field, &Point3::from(position))
                        .expect_inside(),
                );
                // Use negative sign since we are stepping in the opposite direction of the magnetic field
                let parallel_electric_field_strength =
                    -electric_field_vector.dot(direction) * (*U_EL);
                if parallel_electric_field_strength.abs()
                    >= self.config.min_parallel_electric_field_strength
                {
                    backward_path.0.push_front(position[X]);
                    backward_path.1.push_front(position[Y]);
                    backward_path.2.push_front(position[Z]);
                    backward_length = distance;
                    backward_parallel_electric_field_strengths
                        .push_front(parallel_electric_field_strength);
                    average_electric_magnetic_angle_cosine += parallel_electric_field_strength
                        / (electric_field_vector.length() * (*U_EL));
                    StepperInstruction::Continue
                } else {
                    StepperInstruction::Terminate
                }
            };
        let tracer_result = tracing::trace_3d_field_line_dense(
            magnetic_field,
            interpolator,
            stepper.clone(),
            start_position,
            SteppingSense::Opposite,
            &mut callback,
        );

        if let TracerResult::Void = tracer_result {
            return None;
        }

        // Remove start position
        backward_path.0.pop_back().unwrap();
        backward_path.1.pop_back().unwrap();
        backward_path.2.pop_back().unwrap();
        backward_parallel_electric_field_strengths
            .pop_back()
            .unwrap();

        let mut forward_path = (Vec::new(), Vec::new(), Vec::new());
        let mut forward_length = 0.0;
        let mut forward_parallel_electric_field_strengths = Vec::new();

        let mut callback = |_: &Vec3<ftr>,
                            direction: &Vec3<ftr>,
                            position: &Point3<ftr>,
                            distance: ftr| {
            let electric_field_vector = Vec3::from(
                &interpolator
                    .interp_vector_field(electric_field, &Point3::from(position))
                    .expect_inside(),
            );
            let parallel_electric_field_strength = electric_field_vector.dot(direction) * (*U_EL);
            if parallel_electric_field_strength.abs()
                >= self.config.min_parallel_electric_field_strength
            {
                forward_path.0.push(position[X]);
                forward_path.1.push(position[Y]);
                forward_path.2.push(position[Z]);
                forward_length = distance;
                forward_parallel_electric_field_strengths.push(parallel_electric_field_strength);
                average_electric_magnetic_angle_cosine +=
                    parallel_electric_field_strength / (electric_field_vector.length() * (*U_EL));
                StepperInstruction::Continue
            } else {
                StepperInstruction::Terminate
            }
        };
        let tracer_result = tracing::trace_3d_field_line_dense(
            magnetic_field,
            interpolator,
            stepper,
            start_position,
            SteppingSense::Same,
            &mut callback,
        );

        if let TracerResult::Void = tracer_result {
            return None;
        }

        let total_length = backward_length + forward_length;
        if total_length < self.config.min_length {
            return None;
        }

        average_electric_magnetic_angle_cosine /= (backward_parallel_electric_field_strengths.len()
            + forward_parallel_electric_field_strengths.len())
            as feb;

        let mut path = (
            Vec::from(backward_path.0),
            Vec::from(backward_path.1),
            Vec::from(backward_path.2),
        );
        path.0.extend(forward_path.0);
        path.1.extend(forward_path.1);
        path.2.extend(forward_path.2);

        let mut parallel_electric_field_strengths =
            Vec::from(backward_parallel_electric_field_strengths);
        parallel_electric_field_strengths.extend(forward_parallel_electric_field_strengths);

        let average_parallel_electric_field_strength = parallel_electric_field_strengths
            .iter()
            .copied()
            .sum::<feb>()
            / (parallel_electric_field_strengths.len() as feb);

        Some(Self::Data {
            path,
            total_length,
            parallel_electric_field_strengths,
            average_parallel_electric_field_strength,
            average_electric_magnetic_angle_cosine,
        })
    }
}

impl AccelerationRegionData {
    /// Returns the total length of the acceleration region [Mm].
    pub fn total_length(&self) -> ftr {
        self.total_length
    }

    /// Returns the average value of the component of the electric field parallel
    /// to the magnetic field in the acceleration region [statV/cm].
    pub fn average_parallel_electric_field_strength(&self) -> feb {
        self.average_parallel_electric_field_strength
    }

    /// Returns the average cosine of the angle between the electric and magnetic
    /// field in the acceleration region.
    pub fn average_electric_magnetic_angle_cosine(&self) -> feb {
        self.average_electric_magnetic_angle_cosine
    }

    /// Returns the position where the accelerated electrons will exit the
    /// acceleration region.
    pub fn exit_position(&self) -> Point3<fdt> {
        // Electrons propagate in the opposite direction of the electric field.
        if self.average_parallel_electric_field_strength > 0.0 {
            Point3::from_components(
                *self.path.0.first().unwrap(),
                *self.path.1.first().unwrap(),
                *self.path.2.first().unwrap(),
            )
        } else {
            Point3::from_components(
                *self.path.0.last().unwrap(),
                *self.path.1.last().unwrap(),
                *self.path.2.last().unwrap(),
            )
        }
    }
}

impl ParallelExtend<AccelerationRegionData> for FieldLineSetProperties3 {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = AccelerationRegionData>,
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(|field_line| {
            (
                field_line.path.0,
                (
                    field_line.path.1,
                    (
                        field_line.path.2,
                        (
                            field_line.total_length,
                            (
                                field_line.parallel_electric_field_strengths,
                                (
                                    field_line.average_parallel_electric_field_strength,
                                    field_line.average_electric_magnetic_angle_cosine,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        });

        let (paths_x, (paths_y, (paths_z, nested_tuples))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) =
            nested_tuples_iter.unzip();

        let (
            total_lengths,
            (
                parallel_electric_field_strengths,
                (
                    average_parallel_electric_field_strengths,
                    average_electric_magnetic_angle_cosines,
                ),
            ),
        ): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = nested_tuples.into_par_iter().unzip();

        self.number_of_field_lines += paths_x.len();

        self.fixed_scalar_values
            .entry("x0".to_string())
            .or_insert_with(Vec::new)
            .par_extend(paths_x.par_iter().map(|path_x| path_x[0]));

        self.fixed_scalar_values
            .entry("y0".to_string())
            .or_insert_with(Vec::new)
            .par_extend(paths_y.par_iter().map(|path_y| path_y[0]));

        self.fixed_scalar_values
            .entry("z0".to_string())
            .or_insert_with(Vec::new)
            .par_extend(paths_z.par_iter().map(|path_z| path_z[0]));

        self.fixed_scalar_values
            .entry("total_length".to_string())
            .or_insert_with(Vec::new)
            .par_extend(total_lengths.into_par_iter());

        self.fixed_scalar_values
            .entry("average_parallel_electric_field_strength".to_string())
            .or_insert_with(Vec::new)
            .par_extend(average_parallel_electric_field_strengths.into_par_iter());

        self.fixed_scalar_values
            .entry("average_electric_magnetic_angle_cosine".to_string())
            .or_insert_with(Vec::new)
            .par_extend(average_electric_magnetic_angle_cosines.into_par_iter());

        self.varying_scalar_values
            .entry("x".to_string())
            .or_insert_with(Vec::new)
            .par_extend(paths_x.into_par_iter());

        self.varying_scalar_values
            .entry("y".to_string())
            .or_insert_with(Vec::new)
            .par_extend(paths_y.into_par_iter());

        self.varying_scalar_values
            .entry("z".to_string())
            .or_insert_with(Vec::new)
            .par_extend(paths_z.into_par_iter());

        self.varying_scalar_values
            .entry("parallel_electric_field_strength".to_string())
            .or_insert_with(Vec::new)
            .par_extend(parallel_electric_field_strengths.into_par_iter());
    }
}

impl FromParallelIterator<AccelerationRegionData> for FieldLineSetProperties3 {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = AccelerationRegionData>,
    {
        let mut properties = Self::default();
        properties.par_extend(par_iter);
        properties
    }
}

impl AccelerationRegionTracerConfig {
    pub const DEFAULT_MIN_PARALLEL_ELECTRIC_FIELD_STRENGTH: feb = 1.0; // [V/m]
    pub const DEFAULT_MIN_LENGTH: ftr = 0.0; // [Mm]

    fn validate(&self) {
        assert!(
            self.min_parallel_electric_field_strength >= 0.0,
            "Minimum parallel electric field strength must be non-negative."
        );
        assert!(
            self.min_length >= 0.0,
            "Minimum acceleration region length must be non-negative."
        );
    }
}

impl Default for AccelerationRegionTracerConfig {
    fn default() -> Self {
        AccelerationRegionTracerConfig {
            min_parallel_electric_field_strength:
                Self::DEFAULT_MIN_PARALLEL_ELECTRIC_FIELD_STRENGTH,
            min_length: Self::DEFAULT_MIN_LENGTH,
        }
    }
}
