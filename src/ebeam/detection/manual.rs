//! Detection of reconnection sites by reading positions from an input file.

use super::ReconnectionSiteDetector;
use crate::{
    field::CachingScalarFieldProvider3,
    io::{snapshot::fdt, Verbosity},
    seeding::{manual::ManualSeeder3, DynIndexSeeder3, Seeder3},
};
use std::io;
use std::path::Path;

/// Detector reading the reconnection site positions from an input file.
pub struct ManualReconnectionSiteDetector {
    seeder: ManualSeeder3,
}

impl ManualReconnectionSiteDetector {
    /// Creates a new manual reconnection site detector reading positions from the given file.
    ///
    /// The input file is assumed to be in CSV format, with each line consisting
    /// of the three comma-separated coordinates of a single position.
    pub fn new(input_file_path: &Path) -> io::Result<Self> {
        Ok(Self {
            seeder: ManualSeeder3::new(input_file_path)?,
        })
    }
}

impl ReconnectionSiteDetector for ManualReconnectionSiteDetector {
    fn detect_reconnection_sites(
        &self,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        _verbosity: &Verbosity,
    ) -> DynIndexSeeder3 {
        Box::new(self.seeder.to_index_seeder(snapshot.grid())) as DynIndexSeeder3
    }
}
