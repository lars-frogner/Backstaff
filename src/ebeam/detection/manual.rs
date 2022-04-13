//! Detection of reconnection sites by reading positions from an input file.

use super::ReconnectionSiteDetector;
use crate::{
    field::CachingScalarFieldProvider3,
    geometry::Idx3,
    grid::Grid3,
    io::{snapshot::fdt, Verbose},
    seeding::{manual::ManualSeeder3, Seeder3},
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
    pub fn new<P: AsRef<Path>>(input_file_path: P) -> io::Result<Self> {
        Ok(Self {
            seeder: ManualSeeder3::new(input_file_path)?,
        })
    }
}

impl ReconnectionSiteDetector for ManualReconnectionSiteDetector {
    type Seeder = Vec<Idx3<usize>>;

    fn detect_reconnection_sites<G, P>(&self, snapshot: &mut P, _verbose: Verbose) -> Self::Seeder
    where
        G: Grid3<fdt>,
        P: CachingScalarFieldProvider3<fdt, G>,
    {
        self.seeder.to_index_seeder(snapshot.grid())
    }
}
