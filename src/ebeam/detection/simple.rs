//! Simple model for detection of reconnection sites.

use super::ReconnectionSiteDetector;
use crate::{
    exit_on_error,
    field::CachingScalarFieldProvider3,
    geometry::Dim3,
    io::{
        snapshot::{self, fdt, fpa, SnapshotProvider3},
        Verbosity,
    },
    seeding::{criterion::CriterionSeeder3, IndexSeeder3},
};

/// Configuration parameters for the simple reconnection site detection method.
#[derive(Clone, Debug)]
pub struct SimpleReconnectionSiteDetectorConfig {
    /// Reconnection sites will be detected where the reconnection factor value is larger than this.
    pub reconnection_factor_threshold: fpa,
    /// Smallest depth at which reconnection sites will be detected [Mm].
    pub min_detection_depth: fpa,
    /// Largest depth at which reconnection sites will be detected [Mm].
    pub max_detection_depth: fpa,
}

/// Detector evaluating the topological conservation criterion from Biskamp (2005)
/// and including points where it exceeds a certain threshold.
#[derive(Clone, Debug)]
pub struct SimpleReconnectionSiteDetector {
    config: SimpleReconnectionSiteDetectorConfig,
}

impl SimpleReconnectionSiteDetector {
    /// Creates a new simple reconnection site detector with the given configuration parameters.
    pub fn new(config: SimpleReconnectionSiteDetectorConfig) -> Self {
        config.validate();
        SimpleReconnectionSiteDetector { config }
    }
}

impl ReconnectionSiteDetector for SimpleReconnectionSiteDetector {
    type Seeder = CriterionSeeder3;

    fn detect_reconnection_sites<P>(&self, provider: &mut P, verbosity: &Verbosity) -> Self::Seeder
    where
        P: CachingScalarFieldProvider3<fdt>,
    {
        let reconnection_factor_field = exit_on_error!(
            provider.provide_scalar_field("krec"),
            "Error: Could not read reconnection factor field from snapshot: {}"
        );
        let seeder = CriterionSeeder3::on_scalar_field_values(
            reconnection_factor_field.as_ref(),
            &|reconnection_factor| {
                (reconnection_factor as fpa) >= self.config.reconnection_factor_threshold
            },
            &|point| {
                point[Dim3::Z] >= self.config.min_detection_depth
                    && point[Dim3::Z] <= self.config.max_detection_depth
            },
        );

        if verbosity.print_messages() {
            println!("Found {} acceleration sites", seeder.number_of_indices());
        }
        seeder
    }
}

impl SimpleReconnectionSiteDetectorConfig {
    pub const DEFAULT_RECONNECTION_FACTOR_THRESHOLD: fpa = 5e-4;
    pub const DEFAULT_MIN_DETECTION_DEPTH: fpa = -13.0; // [Mm]
    pub const DEFAULT_MAX_DETECTION_DEPTH: fpa = 0.0; // [Mm]

    /// Creates a set of power law distribution configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file<P>(provider: &P) -> Self
    where
        P: SnapshotProvider3,
    {
        let reconnection_factor_threshold =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "reconnection_factor_threshold",
                "krec_lim",
                &|krec_lim: fpa| krec_lim,
                Self::DEFAULT_RECONNECTION_FACTOR_THRESHOLD,
            );
        let min_detection_depth =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "min_detection_depth",
                "z_rec_ulim",
                &|z_rec_ulim: fpa| z_rec_ulim,
                Self::DEFAULT_MIN_DETECTION_DEPTH,
            );
        let max_detection_depth =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "max_detection_depth",
                "z_rec_llim",
                &|z_rec_llim: fpa| z_rec_llim,
                Self::DEFAULT_MAX_DETECTION_DEPTH,
            );

        SimpleReconnectionSiteDetectorConfig {
            reconnection_factor_threshold,
            min_detection_depth,
            max_detection_depth,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    fn validate(&self) {
        assert!(
            self.reconnection_factor_threshold >= 0.0,
            "Reconnection factor threshold must be larger than or equal to zero."
        );
        assert!(
            self.min_detection_depth <= self.max_detection_depth,
            "Minimum detection depth must be smaller than or equal to maximum detection depth."
        );
    }
}

impl Default for SimpleReconnectionSiteDetectorConfig {
    fn default() -> Self {
        SimpleReconnectionSiteDetectorConfig {
            reconnection_factor_threshold: Self::DEFAULT_RECONNECTION_FACTOR_THRESHOLD,
            min_detection_depth: Self::DEFAULT_MIN_DETECTION_DEPTH,
            max_detection_depth: Self::DEFAULT_MAX_DETECTION_DEPTH,
        }
    }
}
