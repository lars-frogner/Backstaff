//! Propagation of a power-law electron distribution using an
//! analytical method.

use crate::{
    constants::{KEV_TO_ERG, M_H, PI, Q_ELECTRON},
    ebeam::{
        distribution::power_law::PowerLawDistribution,
        feb,
        propagation::{DepletionStatus, PropagationResult, Propagator},
    },
    field::CachingScalarFieldProvider3,
    geometry::{
        Dim3::{X, Y, Z},
        Point3, Vec3,
    },
    grid::Grid3,
    interpolation::Interpolator3,
    io::snapshot::{self, fdt, SnapshotParameters},
    math,
    plasma::ionization,
    tracing::ftr,
    units::solar::{U_L, U_L3, U_R},
};
use ndarray::prelude::*;

/// Configuration parameters for the analytical propagator.
#[derive(Clone, Debug)]
pub struct AnalyticalPropagatorConfig {
    /// Distributions with an estimated depletion distance smaller than this value
    /// are discarded [Mm].
    pub min_depletion_distance: feb,
    /// Distributions are considered depleted when the residual energy factor has
    /// decreased below this limit, given that the deposited power per distance is
    /// smaller than its lower limit.
    pub min_residual_factor: feb,
    /// Distributions are considered depleted when the deposited power per distance
    /// [erg/s/cm] has decreased below this limit, given that the residual energy factor
    /// is smaller than its lower limit.
    pub min_deposited_power_per_distance: feb,
    /// Maximum distance the distribution can propagate before propagation should be terminated [Mm].
    pub max_propagation_distance: ftr,
    /// Maximum distance outside the initial extended acceleration region the
    /// distribution can propagate before energy deposition starts [Mm].
    pub outside_deposition_threshold: feb,
    /// Whether to keep propagating beams even after they are considered depleted.
    pub continue_depleted_beams: bool,
}

/// A propagator for a power-law electron distribution using an
/// analytical method.
#[derive(Clone, Debug)]
pub struct AnalyticalPropagator {
    config: AnalyticalPropagatorConfig,
    distribution: PowerLawDistribution,
    /// Coulomb logarithm for interaction with free electrons.
    electron_coulomb_logarithm: feb,
    /// Coulomb logarithm for interaction with neutral hydrogen atoms.
    neutral_hydrogen_coulomb_logarithm: feb,
    /// Depth-independent factor in the beam heating expression.
    heating_scale: feb,
    /// Equivalent ionized column depth where cut-off energy electrons will thermalize [hydrogen/cm^2].
    stopping_ionized_column_depth: feb,
    /// Current column depth of hydrogen atoms (neutral and ionized) traversed by the distribution [hydrogen/cm^2].
    hydrogen_column_depth: feb,
    /// Equivalent ionized column depth (Hawley & Fisher, 1994) traversed by the distribution [hydrogen/cm^2].
    equivalent_ionized_column_depth: feb,
    /// How far outside the acceleration region the distribution has propagated [Mm].
    outside_distance: feb,
}

impl AnalyticalPropagator {
    /// Fraction of a mass of plasma assumed to be made up of hydrogen.
    const HYDROGEN_MASS_FRACTION: feb = 0.735;

    /// `2*pi*(electron charge [esu])^4/(1 keV [erg])^2`
    const COLLISION_SCALE: feb =
        2.0 * PI * Q_ELECTRON * Q_ELECTRON * (Q_ELECTRON / KEV_TO_ERG) * (Q_ELECTRON / KEV_TO_ERG);

    /// `1/2*ln((1 keV [erg])^3/(2*pi*(electron charge [esu])^6))`
    const ELECTRON_COULOMB_OFFSET: feb = 33.073;

    /// `ln(2/(1.105*(hydrogen ionization potential [keV])))`
    const NEUTRAL_HYDROGEN_COULOMB_OFFSET: feb = 4.891;

    /// Smallest mean electron energy that will be used to compute Coulomb logarithms [keV].
    const MIN_COULOMB_LOG_MEAN_ENERGY: feb = 0.1;

    pub fn compute_total_hydrogen_density(mass_density: feb) -> feb {
        (Self::HYDROGEN_MASS_FRACTION / M_H) * mass_density // [hydrogen/cm^3]
    }

    pub fn compute_electron_coulomb_logarithm(electron_density: feb, electron_energy: feb) -> feb {
        feb::max(
            0.0,
            Self::ELECTRON_COULOMB_OFFSET
                + 0.5 * feb::ln(feb::powi(electron_energy, 3) / electron_density),
        )
    }

    pub fn compute_neutral_hydrogen_coulomb_logarithm(electron_energy: feb) -> feb {
        feb::max(
            0.0,
            Self::NEUTRAL_HYDROGEN_COULOMB_OFFSET + feb::ln(electron_energy),
        )
    }

    pub fn compute_effective_coulomb_logarithm(
        ionization_fraction: feb,
        electron_coulomb_logarithm: feb,
        neutral_hydrogen_coulomb_logarithm: feb,
    ) -> feb {
        ionization_fraction * electron_coulomb_logarithm
            + (1.0 - ionization_fraction) * neutral_hydrogen_coulomb_logarithm
    }

    fn estimate_depletion_distance(
        delta: feb,
        min_residual_factor: feb,
        min_deposited_power_per_distance: feb,
        total_hydrogen_density: feb,
        effective_coulomb_logarithm: feb,
        electron_coulomb_logarithm: feb,
        stopping_ionized_column_depth: feb,
        heating_scale: feb,
    ) -> feb {
        let effective_hydrogen_density = effective_coulomb_logarithm * total_hydrogen_density;
        (stopping_ionized_column_depth * electron_coulomb_logarithm / effective_hydrogen_density)
            * feb::powf(
                feb::max(
                    1.0 / min_residual_factor,
                    effective_hydrogen_density * heating_scale * math::beta(0.5 * delta, 1.0 / 3.0)
                        / min_deposited_power_per_distance,
                ),
                2.0 / delta,
            )
    }

    fn compute_heating_scale(
        total_power: feb,
        delta: feb,
        pitch_angle_cosine: feb,
        lower_cutoff_energy: feb,
    ) -> feb {
        Self::COLLISION_SCALE * total_power * (delta - 2.0)
            / (2.0 * feb::abs(pitch_angle_cosine) * feb::powi(lower_cutoff_energy, 2))
    }

    fn compute_stopping_column_depth(
        pitch_angle_cosine: feb,
        electron_energy: feb,
        coulomb_logarithm: feb,
    ) -> feb {
        feb::abs(pitch_angle_cosine) * feb::powi(electron_energy, 2)
            / (3.0 * Self::COLLISION_SCALE * coulomb_logarithm)
    }

    fn compute_uniform_plasma_heating_integral(
        &self,
        total_hydrogen_density: feb,
        effective_coulomb_logarithm: feb,
        step_length: feb,
    ) -> (feb, feb, feb, feb) {
        let coulomb_logarithm_ratio = effective_coulomb_logarithm / self.electron_coulomb_logarithm;

        let hydrogen_column_depth_increase = total_hydrogen_density * step_length;
        let new_hydrogen_column_depth = self.hydrogen_column_depth + hydrogen_column_depth_increase;
        let new_equivalent_ionized_column_depth = self.equivalent_ionized_column_depth
            + hydrogen_column_depth_increase * coulomb_logarithm_ratio;

        let stopping_column_depth = self.stopping_ionized_column_depth / coulomb_logarithm_ratio;

        let start_column_depth_ratio = self.hydrogen_column_depth / stopping_column_depth;
        let start_ionized_column_depth_ratio =
            self.equivalent_ionized_column_depth / self.stopping_ionized_column_depth;

        let column_depth_ratio_increase = hydrogen_column_depth_increase / stopping_column_depth;

        let end_column_depth_ratio = start_column_depth_ratio + column_depth_ratio_increase;
        let end_ionized_column_depth_ratio =
            start_ionized_column_depth_ratio + column_depth_ratio_increase;

        let power = 0.5 * self.distribution.delta;
        let residual_factor = feb::powf(end_ionized_column_depth_ratio, -power);

        let constant_factor =
            self.heating_scale * stopping_column_depth * effective_coulomb_logarithm;

        let mut deposited_power = 0.0;

        if start_column_depth_ratio < 1.0 {
            let end_column_depth_ratio = feb::min(end_column_depth_ratio, 1.0);

            let evaluate_integrand = |increase| {
                math::incomplete_beta(start_column_depth_ratio + increase, power, 1.0 / 3.0)
                    * feb::powf(start_ionized_column_depth_ratio + increase, -power)
            };
            deposited_power += constant_factor
                * math::integrate_three_point_gauss_legendre(
                    evaluate_integrand,
                    0.0,
                    end_column_depth_ratio - start_column_depth_ratio,
                );
        }

        if end_column_depth_ratio > 1.0 {
            let start_ionized_column_depth_ratio = if start_column_depth_ratio < 1.0 {
                1.0 + start_ionized_column_depth_ratio - start_column_depth_ratio
            } else {
                start_ionized_column_depth_ratio
            };

            let shifted_power = power - 1.0;
            deposited_power += constant_factor
                * math::beta(power, 1.0 / 3.0)
                * (feb::powf(start_ionized_column_depth_ratio, -shifted_power)
                    - feb::powf(end_ionized_column_depth_ratio, -shifted_power))
                / shifted_power;
        }

        (
            deposited_power,
            new_hydrogen_column_depth,
            new_equivalent_ionized_column_depth,
            residual_factor,
        )
    }
}

impl Propagator<PowerLawDistribution> for AnalyticalPropagator {
    type Config = AnalyticalPropagatorConfig;

    fn new(config: Self::Config, distribution: PowerLawDistribution) -> Option<Self> {
        let mean_energy = PowerLawDistribution::compute_mean_energy(
            distribution.delta,
            distribution.lower_cutoff_energy,
        );

        let coulomb_logarithm_energy = feb::max(mean_energy, Self::MIN_COULOMB_LOG_MEAN_ENERGY);

        let electron_coulomb_logarithm = Self::compute_electron_coulomb_logarithm(
            distribution.ambient_electron_density,
            coulomb_logarithm_energy,
        );

        let neutral_hydrogen_coulomb_logarithm =
            Self::compute_neutral_hydrogen_coulomb_logarithm(coulomb_logarithm_energy);

        let ionization_fraction = ionization::compute_equilibrium_hydrogen_ionization_fraction(
            distribution.ambient_temperature,
            distribution.ambient_electron_density,
        );

        let total_hydrogen_density =
            Self::compute_total_hydrogen_density(distribution.ambient_mass_density);

        let heating_scale = Self::compute_heating_scale(
            distribution.total_power,
            distribution.delta,
            distribution.initial_pitch_angle_cosine,
            distribution.lower_cutoff_energy,
        );

        let effective_coulomb_logarithm = Self::compute_effective_coulomb_logarithm(
            ionization_fraction,
            electron_coulomb_logarithm,
            neutral_hydrogen_coulomb_logarithm,
        );

        let stopping_ionized_column_depth = Self::compute_stopping_column_depth(
            distribution.initial_pitch_angle_cosine,
            distribution.lower_cutoff_energy,
            electron_coulomb_logarithm,
        );

        let estimated_depletion_distance = Self::estimate_depletion_distance(
            distribution.delta,
            config.min_residual_factor,
            config.min_deposited_power_per_distance,
            total_hydrogen_density,
            effective_coulomb_logarithm,
            electron_coulomb_logarithm,
            stopping_ionized_column_depth,
            heating_scale,
        );

        if estimated_depletion_distance >= config.min_depletion_distance * U_L {
            let hydrogen_column_depth = 0.0;
            let equivalent_ionized_column_depth = 0.0;
            let outside_distance = 0.0;

            Some(Self {
                config,
                distribution,
                electron_coulomb_logarithm,
                neutral_hydrogen_coulomb_logarithm,
                heating_scale,
                stopping_ionized_column_depth,
                hydrogen_column_depth,
                equivalent_ionized_column_depth,
                outside_distance,
            })
        } else {
            None
        }
    }

    fn distribution(&self) -> &PowerLawDistribution {
        &self.distribution
    }

    fn into_distribution(self) -> PowerLawDistribution {
        let Self {
            config: _,
            distribution,
            electron_coulomb_logarithm: _,
            neutral_hydrogen_coulomb_logarithm: _,
            heating_scale: _,
            stopping_ionized_column_depth: _,
            hydrogen_column_depth: _,
            equivalent_ionized_column_depth: _,
            outside_distance: _,
        } = self;
        distribution
    }

    fn max_propagation_distance(&self) -> ftr {
        self.config.max_propagation_distance
    }

    fn propagate(
        &mut self,
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        acceleration_map: &Array3<bool>,
        interpolator: &dyn Interpolator3<fdt>,
        displacement: &Vec3<ftr>,
        new_position: &Point3<ftr>,
    ) -> PropagationResult {
        let mut deposition_position = new_position - displacement * 0.5;

        let deposition_indices = snapshot
            .grid()
            .find_grid_cell(&Point3::from(&deposition_position))
            .unwrap_and_update_position(&mut deposition_position);

        if self.outside_distance < self.config.outside_deposition_threshold {
            if acceleration_map[(
                deposition_indices[X],
                deposition_indices[Y],
                deposition_indices[Z],
            )] {
                self.outside_distance = 0.0;
            } else {
                self.outside_distance += displacement.length();
            }
            PropagationResult {
                residual_factor: 0.0,
                deposited_power: 0.0,
                deposited_power_density: 0.0,
                deposition_position,
                depletion_status: DepletionStatus::Undepleted,
            }
        } else {
            let electron_density_field = snapshot.cached_scalar_field("nel");
            let mass_density_field = snapshot.cached_scalar_field("r");
            let temperature_field = snapshot.cached_scalar_field("tg");

            #[allow(clippy::useless_conversion)]
            let electron_density = feb::from(interpolator.interp_scalar_field_known_cell(
                electron_density_field,
                &Point3::from(&deposition_position),
                &deposition_indices,
            ));

            #[allow(clippy::useless_conversion)]
            let mass_density = feb::from(interpolator.interp_scalar_field_known_cell(
                mass_density_field,
                &Point3::from(&deposition_position),
                &deposition_indices,
            )) * U_R;

            #[allow(clippy::useless_conversion)]
            let temperature = feb::from(interpolator.interp_scalar_field_known_cell(
                temperature_field,
                &Point3::from(&deposition_position),
                &deposition_indices,
            ));

            let total_hydrogen_density = Self::compute_total_hydrogen_density(mass_density);

            let ionization_fraction = ionization::compute_equilibrium_hydrogen_ionization_fraction(
                temperature,
                electron_density,
            );
            let effective_coulomb_logarithm = Self::compute_effective_coulomb_logarithm(
                ionization_fraction,
                self.electron_coulomb_logarithm,
                self.neutral_hydrogen_coulomb_logarithm,
            );

            let step_length = displacement.length() * U_L; // [cm]

            let (
                deposited_power,
                new_hydrogen_column_depth,
                new_equivalent_ionized_column_depth,
                residual_factor,
            ) = self.compute_uniform_plasma_heating_integral(
                total_hydrogen_density,
                effective_coulomb_logarithm,
                step_length,
            );

            self.hydrogen_column_depth = new_hydrogen_column_depth;
            self.equivalent_ionized_column_depth = new_equivalent_ionized_column_depth;

            let volume = snapshot.grid().grid_cell_volume(&deposition_indices) * U_L3;
            let deposited_power_density = deposited_power / volume;

            let depletion_status = if self.config.continue_depleted_beams
                || residual_factor >= self.config.min_residual_factor
                || deposited_power / step_length >= self.config.min_deposited_power_per_distance
            {
                DepletionStatus::Undepleted
            } else {
                DepletionStatus::Depleted
            };

            PropagationResult {
                residual_factor,
                deposited_power,
                deposited_power_density,
                deposition_position,
                depletion_status,
            }
        }
    }
}

impl AnalyticalPropagatorConfig {
    pub const DEFAULT_MIN_DEPLETION_DISTANCE: feb = 0.5; // [Mm]
    pub const DEFAULT_MIN_RESIDUAL_FACTOR: feb = 1e-5;
    pub const DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE: feb = 1e5; // [erg/s/cm]
    pub const DEFAULT_MAX_PROPAGATION_DISTANCE: ftr = 100.0; // [Mm]
    pub const DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD: feb = 0.1; // [Mm]
    pub const DEFAULT_CONTINUE_DEPLETED_BEAMS: bool = false;

    /// Creates a set of analytical propagator configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file(parameters: &dyn SnapshotParameters) -> Self {
        let min_depletion_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_depletion_distance",
                "min_stop_dist",
                &|min_stop_dist: feb| min_stop_dist,
                Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            );
        let min_residual_factor =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_residual_factor",
                "min_residual",
                &|min_residual: feb| min_residual,
                Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            );
        let min_deposited_power_per_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_deposited_power_per_distance",
                "min_dep_en",
                &|min_dep_en: feb| min_dep_en,
                Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            );
        let max_propagation_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "max_propagation_distance",
                "max_dist",
                &|max_dist: feb| max_dist,
                Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            );
        let outside_deposition_threshold =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "outside_deposition_threshold",
                "out_dep_thresh",
                &|out_dep_thresh: feb| out_dep_thresh,
                Self::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
            );
        AnalyticalPropagatorConfig {
            min_depletion_distance,
            min_residual_factor,
            min_deposited_power_per_distance,
            max_propagation_distance,
            outside_deposition_threshold,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(
            self.min_depletion_distance >= 0.0,
            "Minimum stopping distance must be larger than or equal to zero."
        );
        assert!(
            self.min_residual_factor >= 0.0,
            "Minimum residual factor must be larger than or equal to zero."
        );
        assert!(
            self.min_deposited_power_per_distance >= 0.0,
            "Minimum deposited power per distance must be larger than or equal to zero."
        );
        assert!(
            self.max_propagation_distance >= 0.0,
            "Maximum propagation distance must be larger than or equal to zero."
        );
        assert!(
            self.outside_deposition_threshold >= 0.0,
            "Outside deposition threshold must be larger than or equal to zero."
        );
    }
}

impl Default for AnalyticalPropagatorConfig {
    fn default() -> Self {
        AnalyticalPropagatorConfig {
            min_depletion_distance: Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            min_residual_factor: Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            min_deposited_power_per_distance: Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            outside_deposition_threshold: Self::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }
}
