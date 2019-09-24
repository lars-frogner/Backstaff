//! Accelerators combining an acceleration process and a resulting distribution.

use super::distribution::Distribution;
use crate::geometry::Idx3;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use std::io;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;

    /// Makes sure the fields required to generate distributions are cached in the snapshot cacher.
    fn prepare_snapshot_for_generation<G: Grid3<fdt>>(
        snapshot: &mut SnapshotCacher3<G>,
    ) -> io::Result<()>;

    /// Makes sure the fields required to propagate distributions are cached in the snapshot cacher.
    fn prepare_snapshot_for_propagation<G: Grid3<fdt>>(
        snapshot: &mut SnapshotCacher3<G>,
    ) -> io::Result<()>;

    /// Tries to generate a new distribution at the given 3D index in the given snapshot.
    ///
    /// Returns `None` if the distribution was rejected.
    fn generate_distribution<G, I>(
        &self,
        snapshot: &SnapshotCacher3<G>,
        interpolator: &I,
        indices: &Idx3<usize>,
    ) -> Option<Self::DistributionType>
    where
        G: Grid3<fdt>,
        I: Interpolator3;
}
