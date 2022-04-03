//! Detection of reconnection sites.

pub mod manual;
pub mod simple;

use crate::{
    grid::Grid3,
    io::{
        snapshot::{fdt, SnapshotCacher3, SnapshotProvider3},
        Verbose,
    },
    seeding::IndexSeeder3,
};

/// Defines the properties of a reconnection site detector.
pub trait ReconnectionSiteDetector {
    type Seeder: IndexSeeder3;
    /// Detects reconnection sites in the given snapshot and returns
    /// a seeder with the corresponding 3D indices.
    fn detect_reconnection_sites<G, P>(
        &self,
        snapshot: &mut SnapshotCacher3<G, P>,
        verbose: Verbose,
    ) -> Self::Seeder
    where
        G: Grid3<fdt>,
        P: SnapshotProvider3<G>;
}
