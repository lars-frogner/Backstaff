//! Accelerators combining an acceleration process and a resulting distribution.

use super::{
    detection::ReconnectionSiteDetector, distribution::Distribution, propagation::Propagator,
    AccelerationDataCollection,
};
use crate::{
    field::CachingScalarFieldProvider3,
    interpolation::Interpolator3,
    io::{snapshot::fdt, Verbosity},
    tracing::stepping::DynStepper3,
};
use std::io;

/// Specifies the properties of an acceleration process producing non-thermal electrons.
pub trait Accelerator {
    type DistributionType: Distribution;
    type AccelerationDataCollectionType: AccelerationDataCollection;

    /// Generates a set of distributions wrapped in propagators, with
    /// associated acceleration data in the given snapshot, at the
    /// 3D indices produced by the given seeder.
    fn generate_propagators_with_distributions<P>(
        &self,
        propagator_config: P::Config,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        detector: &dyn ReconnectionSiteDetector,
        interpolator: &dyn Interpolator3<fdt>,
        stepper: DynStepper3<fdt>,
        verbosity: &Verbosity,
    ) -> io::Result<(Vec<P>, Self::AccelerationDataCollectionType)>
    where
        P: Propagator<Self::DistributionType>;
}
