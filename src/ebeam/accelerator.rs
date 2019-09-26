//! Accelerators combining an acceleration process and a resulting distribution.

use super::distribution::Distribution;
use super::BeamMetadataCollection;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::io::Verbose;
use crate::tracing::seeding::IndexSeeder3;
use std::io;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;
    type MetadataCollectionType: BeamMetadataCollection;

    /// Generates a set of distributions with associated metadata in the given snapshot,
    /// at the 3D indices produced by the given seeder.
    fn generate_distributions<Sd, G, I>(
        &self,
        seeder: Sd,
        snapshot: &mut SnapshotCacher3<G>,
        interpolator: &I,
        verbose: Verbose,
    ) -> io::Result<(Vec<Self::DistributionType>, Self::MetadataCollectionType)>
    where
        Sd: IndexSeeder3,
        G: Grid3<fdt>,
        I: Interpolator3;
}
