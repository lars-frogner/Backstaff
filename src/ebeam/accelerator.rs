//! Accelerators combining an acceleration process and a resulting distribution.

use std::io;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use super::distribution::Distribution;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;

    /// Makes sure the fields required to generate distributions are cached in the snapshot cacher.
    fn prepare_snapshot_for_generation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()>;

    /// Makes sure the fields required to propagate distributions are cached in the snapshot cacher.
    fn prepare_snapshot_for_propagation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()>;

    /// Generates a new acceleration event at the given position in the given snapshot.
    ///
    /// Returns `None` if the distribution could not be generated.
    fn generate_distribution<G, I>(&self, snapshot: &SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> Option<Self::DistributionType>
    where G: Grid3<fdt>,
          I: Interpolator3;
}
