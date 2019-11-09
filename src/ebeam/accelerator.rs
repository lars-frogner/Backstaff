//! Accelerators combining an acceleration process and a resulting distribution.

use super::detection::ReconnectionSiteDetector;
use super::distribution::Distribution;
use super::AccelerationDataCollection;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::io::Verbose;
use crate::tracing::stepping::StepperFactory3;
use std::io;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;
    type AccelerationDataCollectionType: AccelerationDataCollection;

    /// Generates a set of distributions with associated acceleration data in the given snapshot,
    /// at the 3D indices produced by the given seeder.
    fn generate_distributions<G, D, I, StF>(
        &self,
        snapshot: &mut SnapshotCacher3<G>,
        detector: D,
        interpolator: &I,
        stepper_factory: &StF,
        verbose: Verbose,
    ) -> io::Result<(
        Vec<Self::DistributionType>,
        Self::AccelerationDataCollectionType,
    )>
    where
        G: Grid3<fdt>,
        D: ReconnectionSiteDetector,
        I: Interpolator3,
        StF: StepperFactory3 + Sync;
}
