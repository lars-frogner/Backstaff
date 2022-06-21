//! Accelerators combining an acceleration process and a resulting distribution.

use super::{
    detection::ReconnectionSiteDetector, distribution::Distribution, AccelerationDataCollection,
};
use crate::{
    field::CachingScalarFieldProvider3,
    interpolation::Interpolator3,
    io::{snapshot::fdt, Verbosity},
    tracing::stepping::StepperFactory3,
};
use std::io;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;
    type AccelerationDataCollectionType: AccelerationDataCollection;

    /// Generates a set of distributions with associated acceleration data in the given snapshot,
    /// at the 3D indices produced by the given seeder.
    fn generate_distributions<D, StF>(
        &self,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        detector: D,
        interpolator: &dyn Interpolator3<fdt>,
        stepper_factory: &StF,
        verbosity: &Verbosity,
    ) -> io::Result<(
        Vec<Self::DistributionType>,
        Self::AccelerationDataCollectionType,
    )>
    where
        D: ReconnectionSiteDetector,
        StF: StepperFactory3<fdt> + Sync;
}
