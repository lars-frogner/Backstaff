//! Non-thermal electron distributions.

pub mod power_law;

use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::Vec3;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use super::feb;
use super::acceleration::AccelerationEvent;

/// Holds the deposited power density after propagating the electron distribution
/// and additionally specifies whether the distribution is depleted.
pub enum PropagationResult {
    Ok(feb),
    Depleted(feb)
}

/// Defines the properties of a non-thermal electron distribution.
pub trait Distribution {
    /// Propagates the electron distribution for the given displacement
    /// and returns the power density deposited during the propagation.
    fn propagate<G, I>(&mut self, snapshot: &mut SnapshotCacher3<G>, interpolator: &I, displacement: &Vec3<fdt>) -> PropagationResult
    where G: Grid3<fdt>,
          I: Interpolator3;
}

/// Defines the properties of a generator of non-thermal electron distributions.
pub trait DistributionGenerator {
    type AccelerationEventType: AccelerationEvent;
    type DistributionType: Distribution;

    /// Generates a new distribution from the given acceleration event.
    ///
    /// Returns `None` if the distribution could not be generated.
    fn generate_from_acceleration_event(&self, event: &Self::AccelerationEventType) -> Option<Self::DistributionType>;
}
