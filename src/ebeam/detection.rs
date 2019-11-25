//! Detection of reconnection sites.

pub mod manual;
pub mod simple;

use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::io::Verbose;
use crate::tracing::seeding::IndexSeeder3;

/// Defines the properties of a reconnection site detector.
pub trait ReconnectionSiteDetector {
    type Seeder: IndexSeeder3;
    /// Detects reconnection sites in the given snapshot and returns
    /// a seeder with the corresponding 3D indices.
    fn detect_reconnection_sites<G: Grid3<fdt>>(
        &self,
        snapshot: &mut SnapshotCacher3<G>,
        verbose: Verbose,
    ) -> Self::Seeder;
}
