//! Detection of reconnection sites.

pub mod manual;
pub mod simple;

use crate::{
    field::CachingScalarFieldProvider3,
    io::{snapshot::fdt, Verbosity},
    seeding::DynIndexSeeder3,
};

pub type DynReconnectionSiteDetector = Box<dyn ReconnectionSiteDetector>;

/// Defines the properties of a reconnection site detector.
pub trait ReconnectionSiteDetector {
    /// Detects reconnection sites in the given snapshot and returns
    /// a seeder with the corresponding 3D indices.
    fn detect_reconnection_sites(
        &self,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        verbosity: &Verbosity,
    ) -> DynIndexSeeder3;
}
