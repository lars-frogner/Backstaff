//! Simple model for detection of reconnection sites.

use super::ReconnectionSiteDetector;
use crate::geometry::Dim3;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::io::Verbose;
use crate::tracing::seeding::criterion::CriterionSeeder3;
use crate::tracing::seeding::IndexSeeder3;

/// Configuration parameters for the simple reconnection site detection method.
#[derive(Clone, Debug)]
pub struct SimpleReconnectionSiteDetectorConfig {
    /// Whether to use a normalized version of the reconnection factor.
    pub use_normalized_reconnection_factor: bool,
    /// Reconnection sites will be detected where the reconnection factor value is larger than this.
    pub reconnection_factor_threshold: fdt,
    /// Smallest depth at which reconnection sites will be detected [Mm].
    pub min_detection_depth: fdt,
    /// Largest depth at which reconnection sites will be detected [Mm].
    pub max_detection_depth: fdt,
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

    fn detect_reconnection_sites<G: Grid3<fdt>>(
        &self,
        snapshot: &mut SnapshotCacher3<G>,
        verbose: Verbose,
    ) -> Self::Seeder {
        let reconnection_factor_variable = if self.config.use_normalized_reconnection_factor {
            "krec_norm"
        } else {
            "krec"
        };
        let reconnection_factor_field = snapshot
            .obtain_scalar_field(reconnection_factor_variable)
            .unwrap_or_else(|err| {
                panic!(
                    "Could not read reconnection factor field from snapshot: {}",
                    err
                )
            });
        let mut seeder = CriterionSeeder3::on_scalar_field_values(
            reconnection_factor_field,
            &|reconnection_factor| reconnection_factor >= self.config.reconnection_factor_threshold,
        );

        snapshot.drop_scalar_field(reconnection_factor_variable);

        let z_coordinates = &snapshot.reader().grid().centers()[Dim3::Z];
        seeder.retain_indices(|indices| {
            z_coordinates[indices[Dim3::Z]] >= self.config.min_detection_depth
                && z_coordinates[indices[Dim3::Z]] <= self.config.max_detection_depth
        });

        if verbose.is_yes() {
            println!("Found {} acceleration sites", seeder.number_of_indices());
        }
        seeder
    }
}

impl SimpleReconnectionSiteDetectorConfig {
    const DEFAULT_USE_NORMALIZED_RECONNECTION_FACTOR: bool = false;
    const DEFAULT_RECONNECTION_FACTOR_THRESHOLD: fdt = 1e-3;
    const DEFAULT_MIN_DETECTION_DEPTH: fdt = -13.0; // [Mm]
    const DEFAULT_MAX_DETECTION_DEPTH: fdt = 0.0; // [Mm]

    /// Creates a set of power law distribution configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> Self {
        let use_normalized_reconnection_factor = reader
            .get_numerical_param::<u8>("norm_krec")
            .unwrap_or_else(|err| panic!("{}", err))
            > 0;

        let reconnection_factor_threshold = reader
            .get_numerical_param("krec_lim")
            .unwrap_or_else(|err| panic!("{}", err));

        let min_detection_depth = reader
            .get_numerical_param("z_rec_ulim")
            .unwrap_or_else(|err| panic!("{}", err));

        let max_detection_depth = reader
            .get_numerical_param("z_rec_llim")
            .unwrap_or_else(|err| panic!("{}", err));

        SimpleReconnectionSiteDetectorConfig {
            use_normalized_reconnection_factor,
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
            use_normalized_reconnection_factor: Self::DEFAULT_USE_NORMALIZED_RECONNECTION_FACTOR,
            reconnection_factor_threshold: Self::DEFAULT_RECONNECTION_FACTOR_THRESHOLD,
            min_detection_depth: Self::DEFAULT_MIN_DETECTION_DEPTH,
            max_detection_depth: Self::DEFAULT_MAX_DETECTION_DEPTH,
        }
    }
}
