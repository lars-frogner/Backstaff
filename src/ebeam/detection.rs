//! Detection of reconnection sites.

pub mod manual;
pub mod simple;

use crate::{
    field::CachingScalarFieldProvider3,
    io::{snapshot::fdt, Verbosity},
    seeding::IndexSeeder3,
};

/// Defines the properties of a reconnection site detector.
pub trait ReconnectionSiteDetector {
    type Seeder: IndexSeeder3;
    /// Detects reconnection sites in the given snapshot and returns
    /// a seeder with the corresponding 3D indices.
    fn detect_reconnection_sites(
        &self,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        verbosity: &Verbosity,
    ) -> Self::Seeder;
}
