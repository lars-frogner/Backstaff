//! Accelerators combining an acceleration process and a resulting distribution.

use std::io;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::Idx3;
use crate::grid::Grid3;
use super::distribution::Distribution;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;

    /// Makes sure the fields required to generate distributions are cached in the snapshot cacher.
    fn prepare_snapshot_for_generation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()>;

    /// Makes sure the fields required to propagate distributions are cached in the snapshot cacher.
    fn prepare_snapshot_for_propagation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()>;

    /// Generates a new acceleration event at the given 3D index in the given snapshot.
    ///
    /// Returns `None` if the distribution could not be generated.
    fn generate_distribution<G>(&self, snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> Option<Self::DistributionType>
    where G: Grid3<fdt>;
}
