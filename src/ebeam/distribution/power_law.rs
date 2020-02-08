//! Power-law electron distribution.

pub mod acceleration;

use super::super::{feb, BeamPropertiesCollection, FixedBeamScalarValues, FixedBeamVectorValues};
use super::{DepletionStatus, Distribution, PropagationResult};
use crate::constants::{KEV_TO_ERG, M_H, PI, Q_ELECTRON};
use crate::geometry::{Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::math;
use crate::plasma::ionization;
use crate::tracing::ftr;
use crate::tracing::stepping::SteppingSense;
use crate::units::solar::{U_L, U_L3, U_R};
use rayon::prelude::*;

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionConfig {
    /// Distributions are considered thermalized when they have traversed the stopping
    /// column depth of cut-off energy electrons more times than this.
    pub max_stopping_length_traversals: feb,
    /// Maximum distance the distribution can propagate before propagation should be terminated [Mm].
    pub max_propagation_distance: ftr,
    /// Whether to keep propagating beams even after they are considered thermalized.
    pub continue_thermalized_beams: bool,
}

/// Data associated with a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionData {
    /// Exponent of the inverse power-law.
    delta: feb,
    /// Cosine of the initial pitch angle of the electrons.
    initial_pitch_angle_cosine: feb,
    /// Total energy injected into the distribution per time [erg/s].
    total_power: feb,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [keV].
    lower_cutoff_energy: feb,
    /// Position where the distribution originates [Mm].
    acceleration_position: Point3<fdt>,
    /// Volume of the grid cell where the distribution originates [cm^3].
    acceleration_volume: feb,
    /// Direction of propagation of the electrons relative to the magnetic field direction.
    propagation_sense: SteppingSense,
    /// Coulomb logarithm for interactions with free electrons.
    electron_coulomb_logarithm: feb,
    /// Coulomb logarithm for interactions with neutral hydrogen atoms.
    neutral_hydrogen_coulomb_logarithm: feb,
    /// Depth-independent factor in the beam heating expression.
    heating_scale: feb,
    /// Equivalent ionized column depth where cut-off energy electrons will thermalize [hydrogen/cm^2].
    stopping_ionized_column_depth: feb,
    /// estimated thermalization distance of the electrons in the distribution [cm].
    estimated_thermalization_distance: feb,
}

/// Exposed properties of a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionProperties {
    /// Total energy injected into the distribution per time [erg/s].
    total_power: feb,
    /// Lower cut-off energy [keV].
    lower_cutoff_energy: feb,
    /// Volume of the grid cell where the distribution originates [cm^3].
    acceleration_volume: feb,
    /// estimated thermalization distance of the electrons in the distribution [Mm].
    estimated_thermalization_distance: feb,
}

/// Property values of each individual distribution in a set of power-law distributions.
#[derive(Clone, Default, Debug)]
pub struct PowerLawDistributionPropertiesCollection {
    total_powers: Vec<feb>,
    lower_cutoff_energies: Vec<feb>,
    acceleration_volumes: Vec<feb>,
    estimated_thermalization_distances: Vec<feb>,
}

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1)*lower_cutoff_energy^(delta - 1)*E^(-delta)`.
#[derive(Clone, Debug)]
pub struct PowerLawDistribution {
    config: PowerLawDistributionConfig,
    data: PowerLawDistributionData,
    /// Current column depth of hydrogen atoms (neutral and ionized) traversed by the distribution [hydrogen/cm^2].
    hydrogen_column_depth: feb,
    /// Equivalent ionized column depth (Hawley & Fisher, 1994) traversed by the distribution [hydrogen/cm^2].
    equivalent_ionized_column_depth: feb,
}

impl PowerLawDistribution {
    /// Fraction of a mass of plasma assumed to be made up of hydrogen.
    const HYDROGEN_MASS_FRACTION: feb = 0.735;

    /// `2*pi*(electron charge [esu])^4/(1 keV [erg])^2`
    const COLLISION_SCALE: feb =
        2.0 * PI * Q_ELECTRON * Q_ELECTRON * (Q_ELECTRON / KEV_TO_ERG) * (Q_ELECTRON / KEV_TO_ERG);

    /// `1/2*ln((1 keV [erg])^3/(2*pi*(electron charge [esu])^6))`
    const ELECTRON_COULOMB_OFFSET: feb = 33.073;

    /// `ln(2/(1.105*(hydrogen ionization potential [keV])))`
    const NEUTRAL_HYDROGEN_COULOMB_OFFSET: feb = 4.891;

    fn new(config: PowerLawDistributionConfig, data: PowerLawDistributionData) -> Self {
        let hydrogen_column_depth = 0.0;
        let equivalent_ionized_column_depth = 0.0;

        PowerLawDistribution {
            config,
            data,
            hydrogen_column_depth,
            equivalent_ionized_column_depth,
        }
    }

    fn compute_total_power(total_power_density: feb, acceleration_volume: feb) -> feb {
        total_power_density * acceleration_volume // [erg/s]
    }

    fn compute_mean_energy(delta: feb, lower_cutoff_energy: feb) -> feb {
        lower_cutoff_energy * (delta - 1.0) / (delta - 2.0)
    }

    fn compute_total_hydrogen_density(mass_density: feb) -> feb {
        (Self::HYDROGEN_MASS_FRACTION / M_H) * mass_density // [hydrogen/cm^3]
    }

    fn compute_electron_coulomb_logarithm(electron_density: feb, electron_energy: feb) -> feb {
        feb::max(
            0.0,
            Self::ELECTRON_COULOMB_OFFSET
                + 0.5 * feb::ln(feb::powi(electron_energy, 3) / electron_density),
        )
    }

    fn compute_neutral_hydrogen_coulomb_logarithm(electron_energy: feb) -> feb {
        feb::max(
            0.0,
            Self::NEUTRAL_HYDROGEN_COULOMB_OFFSET + feb::ln(electron_energy),
        )
    }

    fn compute_effective_coulomb_logarithm(
        ionization_fraction: feb,
        electron_coulomb_logarithm: feb,
        neutral_hydrogen_coulomb_logarithm: feb,
    ) -> feb {
        ionization_fraction * electron_coulomb_logarithm
            + (1.0 - ionization_fraction) * neutral_hydrogen_coulomb_logarithm
    }

    fn compute_stopping_column_depth(
        pitch_angle_cosine: feb,
        electron_energy: feb,
        coulomb_logarithm: feb,
    ) -> feb {
        feb::abs(pitch_angle_cosine) * feb::powi(electron_energy, 2)
            / (3.0 * Self::COLLISION_SCALE * coulomb_logarithm)
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

    fn estimate_depletion_distance(
        max_stopping_length_traversals: feb,
        total_hydrogen_density: feb,
        effective_coulomb_logarithm: feb,
        electron_coulomb_logarithm: feb,
        stopping_ionized_column_depth: feb,
    ) -> feb {
        max_stopping_length_traversals * stopping_ionized_column_depth * electron_coulomb_logarithm
            / (effective_coulomb_logarithm * total_hydrogen_density)
    }
}

impl BeamPropertiesCollection for PowerLawDistributionPropertiesCollection {
    type Item = PowerLawDistributionProperties;

    fn distribute_into_maps(
        self,
        scalar_values: &mut FixedBeamScalarValues,
        _vector_values: &mut FixedBeamVectorValues,
    ) {
        scalar_values.insert("total_power".to_string(), self.total_powers);
        scalar_values.insert(
            "lower_cutoff_energy".to_string(),
            self.lower_cutoff_energies,
        );
        scalar_values.insert("acceleration_volume".to_string(), self.acceleration_volumes);
        scalar_values.insert(
            "estimated_thermalization_distance".to_string(),
            self.estimated_thermalization_distances,
        );
    }
}

impl ParallelExtend<PowerLawDistributionProperties> for PowerLawDistributionPropertiesCollection {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = PowerLawDistributionProperties>,
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(|data| {
            (
                data.total_power,
                (
                    data.lower_cutoff_energy,
                    (
                        data.acceleration_volume,
                        data.estimated_thermalization_distance,
                    ),
                ),
            )
        });

        let (
            total_powers,
            (lower_cutoff_energies, (acceleration_volumes, estimated_thermalization_distances)),
        ): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = nested_tuples_iter.unzip();

        self.total_powers.par_extend(total_powers);
        self.lower_cutoff_energies.par_extend(lower_cutoff_energies);
        self.acceleration_volumes.par_extend(acceleration_volumes);
        self.estimated_thermalization_distances
            .par_extend(estimated_thermalization_distances);
    }
}

impl Distribution for PowerLawDistribution {
    type PropertiesCollectionType = PowerLawDistributionPropertiesCollection;

    fn acceleration_position(&self) -> &Point3<fdt> {
        &self.data.acceleration_position
    }

    fn propagation_sense(&self) -> SteppingSense {
        self.data.propagation_sense
    }

    fn max_propagation_distance(&self) -> ftr {
        self.config.max_propagation_distance
    }

    fn properties(&self) -> <Self::PropertiesCollectionType as BeamPropertiesCollection>::Item {
        PowerLawDistributionProperties {
            total_power: self.data.total_power,
            lower_cutoff_energy: self.data.lower_cutoff_energy,
            acceleration_volume: self.data.acceleration_volume,
            estimated_thermalization_distance: self.data.estimated_thermalization_distance / U_L,
        }
    }

    fn propagate<G, I>(
        &mut self,
        snapshot: &SnapshotCacher3<G>,
        interpolator: &I,
        displacement: &Vec3<ftr>,
        new_position: &Point3<ftr>,
    ) -> PropagationResult
    where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
        let mut deposition_position = new_position - displacement * 0.5;

        let deposition_indices = snapshot
            .reader()
            .grid()
            .find_grid_cell(&Point3::from(&deposition_position))
            .unwrap_and_update_position(&mut deposition_position);

        let electron_density_field = snapshot.cached_scalar_field("nel");
        let mass_density_field = snapshot.cached_scalar_field("r");
        let temperature_field = snapshot.cached_scalar_field("tg");

        let electron_density = feb::from(interpolator.interp_scalar_field_known_cell(
            electron_density_field,
            &Point3::from(&deposition_position),
            &deposition_indices,
        ));

        let mass_density = feb::from(interpolator.interp_scalar_field_known_cell(
            mass_density_field,
            &Point3::from(&deposition_position),
            &deposition_indices,
        )) * U_R;

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
            self.data.electron_coulomb_logarithm,
            self.data.neutral_hydrogen_coulomb_logarithm,
        );

        let coulomb_logarithm_ratio =
            effective_coulomb_logarithm / self.data.electron_coulomb_logarithm;

        let step_length = displacement.length() * U_L; // [cm]
        let hydrogen_column_depth_increase = total_hydrogen_density * step_length;

        let new_hydrogen_column_depth = self.hydrogen_column_depth + hydrogen_column_depth_increase;
        let new_equivalent_ionized_column_depth = self.equivalent_ionized_column_depth
            + coulomb_logarithm_ratio * hydrogen_column_depth_increase;

        let column_depth_ratio = new_hydrogen_column_depth * coulomb_logarithm_ratio
            / self.data.stopping_ionized_column_depth;
        let beta = if column_depth_ratio < 1.0 {
            math::incomplete_beta(column_depth_ratio, 0.5 * self.data.delta, 1.0 / 3.0)
        } else {
            math::beta(0.5 * self.data.delta, 1.0 / 3.0)
        };

        let equivalent_ionized_column_depth_ratio =
            new_equivalent_ionized_column_depth / self.data.stopping_ionized_column_depth;

        // Compute power deposited through the step [erg/s]
        let deposited_power = self.data.heating_scale
            * beta
            * total_hydrogen_density
            * effective_coulomb_logarithm
            * feb::powf(
                equivalent_ionized_column_depth_ratio,
                -0.5 * self.data.delta,
            )
            * step_length;

        let depletion_status = if self.config.continue_thermalized_beams
            || equivalent_ionized_column_depth_ratio < self.config.max_stopping_length_traversals
        {
            DepletionStatus::Undepleted
        } else {
            DepletionStatus::Depleted
        };

        self.hydrogen_column_depth = new_hydrogen_column_depth;
        self.equivalent_ionized_column_depth = new_equivalent_ionized_column_depth;

        let volume = feb::from(
            snapshot
                .reader()
                .grid()
                .grid_cell_volume(&deposition_indices),
        ) * U_L3;
        let deposited_power_density = deposited_power / volume;

        PropagationResult {
            deposited_power,
            deposited_power_density,
            deposition_position,
            depletion_status,
        }
    }
}

impl PowerLawDistributionConfig {
    pub const DEFAULT_MAX_STOPPING_LENGTH_TRAVERSALS: feb = 1e5;
    pub const DEFAULT_MAX_PROPAGATION_DISTANCE: ftr = 100.0; // [Mm]
    pub const DEFAULT_CONTINUE_THERMALIZED_BEAMS: bool = false;

    /// Creates a set of power law distribution configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> Self {
        let max_stopping_length_traversals = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "max_stopping_length_traversals",
                "max_stop_lens",
                &|max_stop_lens: feb| max_stop_lens,
                Self::DEFAULT_MAX_STOPPING_LENGTH_TRAVERSALS,
            );
        let max_propagation_distance = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "max_propagation_distance",
                "max_dist",
                &|max_dist| max_dist,
                Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            );
        PowerLawDistributionConfig {
            max_stopping_length_traversals,
            max_propagation_distance,
            continue_thermalized_beams: Self::DEFAULT_CONTINUE_THERMALIZED_BEAMS,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    fn validate(&self) {
        assert!(
            self.max_stopping_length_traversals >= 0.0,
            "Maximum number of stopping lengths must be larger than or equal to zero."
        );
        assert!(
            self.max_propagation_distance >= 0.0,
            "Maximum propagation distance must be larger than or equal to zero."
        );
    }
}

impl Default for PowerLawDistributionConfig {
    fn default() -> Self {
        PowerLawDistributionConfig {
            max_stopping_length_traversals: Self::DEFAULT_MAX_STOPPING_LENGTH_TRAVERSALS,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            continue_thermalized_beams: Self::DEFAULT_CONTINUE_THERMALIZED_BEAMS,
        }
    }
}
