//! Non-thermal electron beam physics in Bifrost simulations.

pub mod distribution;
pub mod accelerator;

use std::collections::HashMap;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use rayon::prelude::*;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::{self, ftr, TracerResult};
use crate::tracing::seeding::Seeder3;
use crate::tracing::stepping::{SteppingSense, StepperInstruction, Stepper3, StepperFactory3};
use self::distribution::{DepletionStatus, PropagationResult, Distribution};
use self::accelerator::Accelerator;

/// Floating-point precision to use for electron beam physics.
#[allow(non_camel_case_types)]
pub type feb = f64;

/// A beam of non-thermal electrons propagating through the solar atmosphere.
#[derive(Clone, Debug)]
pub struct ElectronBeam {
    positions: Vec<Point3<ftr>>,
    scalar_values: HashMap<String, Vec<feb>>
}

/// A set of non-thermal electron beams propagating through the solar atmosphere.
#[derive(Clone, Debug)]
pub struct ElectronBeamSwarm {
    beams: Vec<ElectronBeam>
}

impl ElectronBeam {
    fn generate<D, G, I, S>(mut distribution: D, snapshot: &SnapshotCacher3<G>, interpolator: &I, stepper: S, sense: SteppingSense) -> Option<Self>
    where D: Distribution,
          G: Grid3<fdt>,
          I: Interpolator3,
          S: Stepper3
    {
        let mut positions = Vec::new();
        let mut deposited_power_densities = Vec::new();

        let magnetic_field = snapshot.cached_vector_field("b");
        let start_position = Point3::from(distribution.acceleration_position());

        let tracer_result = tracing::trace_3d_field_line_dense(magnetic_field, interpolator, stepper, &start_position, sense,
            &mut |displacement, position, _| {
                let PropagationResult{
                    deposited_power_density,
                    deposition_position,
                    depletion_status
                } = distribution.propagate(snapshot, interpolator, displacement, position);

                positions.push(deposition_position);
                deposited_power_densities.push(deposited_power_density);

                match depletion_status {
                    DepletionStatus::Undepleted => StepperInstruction::Continue,
                    DepletionStatus::Depleted => StepperInstruction::Terminate
                }
            }
        );

        let mut scalar_values = HashMap::new();
        scalar_values.insert("qbeam".to_string(), deposited_power_densities);

        match tracer_result {
            TracerResult::Ok(_) => Some(ElectronBeam{ positions, scalar_values }),
            TracerResult::Void => None
        }
    }
}

impl ElectronBeamSwarm {
    pub fn generate<Sd, G, A, I, StF>(seeder: Sd, mut snapshot: SnapshotCacher3<G>, accelerator: A, interpolator: &I, stepper_factory: StF) -> Option<Self>
    where Sd: Seeder3 + IntoParallelIterator<Item = Point3<ftr>>,
          G: Grid3<fdt>,
          A: Accelerator + Sync + Send,
          A::DistributionType: Send,
          I: Interpolator3,
          StF: StepperFactory3 + Sync
    {
        A::prepare_snapshot_for_generation(&mut snapshot).unwrap_or_else(|err| panic!("Snapshot preparation failed: {}", err));

        let seed_iter = seeder.into_par_iter();
        let distributions: Vec<_> = seed_iter.filter_map(
            |position| {
                accelerator.generate_distribution(&snapshot, interpolator, &Point3::from(&position))
            }
        ).collect();

        A::prepare_snapshot_for_propagation(&mut snapshot).unwrap_or_else(|err| panic!("Snapshot preparation failed: {}", err));

        let beams: Vec<_> = distributions.into_par_iter().filter_map(
            |distribution| {
                let magnetic_field = snapshot.cached_vector_field("b");
                let acceleration_position = distribution.acceleration_position();
                let mut magnetic_field_direction = interpolator.interp_vector_field(magnetic_field, acceleration_position).expect_inside();
                magnetic_field_direction.normalize();
                match distribution.determine_propagation_sense(&magnetic_field_direction) {
                    Some(sense) => ElectronBeam::generate(distribution, &snapshot, interpolator, stepper_factory.produce(), sense),
                    None => None
                }
            }
        ).collect();

        if beams.is_empty() {
            None
        } else {
            Some(ElectronBeamSwarm{ beams })
        }
    }
}

impl Serialize for ElectronBeam {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("ElectronBeam", 2)?;
        s.serialize_field("positions", &self.positions)?;
        s.serialize_field("scalar_values", &self.scalar_values)?;
        s.end()
    }
}
