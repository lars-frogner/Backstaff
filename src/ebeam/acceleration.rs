//! Acceleration of non-thermal electron beams.

pub mod simple;

use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;

/// Specifies the properties of an electron acceleration event.
pub trait AccelerationEvent {}

/// Specifies the properties of a generator of electron acceleration events.
pub trait AccelerationEventGenerator {
    type AccelerationEventType: AccelerationEvent;

    /// Generates a new acceleration event at the given position in the given snapshot.
    fn generate<G, I>(&self, snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> Self::AccelerationEventType
    where G: Grid3<fdt>,
          I: Interpolator3;
}
