//! Accelerators combining an acceleration process and a resulting distribution.

use super::{
    detection::ReconnectionSiteDetector, distribution::Distribution, AccelerationDataCollection,
};
use crate::{
    field::{ScalarFieldCacher3, ScalarFieldProvider3},
    grid::Grid3,
    interpolation::Interpolator3,
    io::{snapshot::fdt, Verbose},
    tracing::stepping::StepperFactory3,
};
use std::io;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;
    type AccelerationDataCollectionType: AccelerationDataCollection;

    /// Generates a set of distributions with associated acceleration data in the given snapshot,
    /// at the 3D indices produced by the given seeder.
    fn generate_distributions<G, P, D, I, StF>(
        &self,
        snapshot: &mut ScalarFieldCacher3<fdt, G, P>,
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
        P: ScalarFieldProvider3<fdt, G>,
        D: ReconnectionSiteDetector,
        I: Interpolator3,
        StF: StepperFactory3 + Sync;
}
