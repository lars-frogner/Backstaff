//! Power-law electron distribution.

pub mod acceleration;

use super::super::{feb, BeamPropertiesCollection, FixedBeamScalarValues, FixedBeamVectorValues};
use super::{DepletionStatus, Distribution, PropagationResult};
use crate::constants::{KEV_TO_ERG, MC2_ELECTRON};
use crate::geometry::{Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::plasma::ionization;
use crate::tracing::ftr;
use crate::tracing::stepping::SteppingSense;
use crate::units::solar::{U_E, U_L, U_R, U_T};
use rayon::prelude::*;

/// An electron pitch-angle distribution which is either peaked or isotropic.
#[derive(Clone, Copy, Debug)]
pub enum PitchAngleDistribution {
    Peaked,
    Isotropic,
}

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionConfig {
    /// Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_remaining_power_density: feb,
    // Maximum distance the distribution can propagate before propagation should be terminated [Mm].
    pub max_propagation_distance: ftr,
}

/// Data associated with a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionData {
    /// Exponent of the inverse power-law.
    delta: feb,
    /// Factor which is 2 for a peaked and 4 for an isotropic pitch angle distribution.
    pitch_angle_factor: feb,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [units of electron rest energy].
    lower_cutoff_energy: feb,
    /// Mean energy of the electrons in the distribution [units of electron rest energy].
    mean_energy: feb,
    /// Estimated propagation distance until the remaining power density drops below the minimum value [cm].
    estimated_depletion_distance: feb,
    /// Position where the distribution originates [Mm].
    acceleration_position: Point3<fdt>,
    /// Direction of propagation of the electrons relative to the magnetic field direction.
    propagation_sense: SteppingSense,
}

/// Exposed properties of a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionProperties {
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [keV].
    lower_cutoff_energy: feb,
    /// Estimated propagation distance until the remaining power density drops below the minimum value [Mm].
    estimated_depletion_distance: feb,
}

/// Property values of each individual distribution in a set of power-law distributions.
#[derive(Clone, Default, Debug)]
pub struct PowerLawDistributionPropertiesCollection {
    total_power_densities: Vec<feb>,
    lower_cutoff_energies: Vec<feb>,
    estimated_depletion_distances: Vec<feb>,
}

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power_density`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1)*lower_cutoff_energy^(delta - 1)*E^(-delta)`.
#[derive(Clone, Debug)]
pub struct PowerLawDistribution {
    config: PowerLawDistributionConfig,
    data: PowerLawDistributionData,
    /// Current collisional depth [dimensionless]. Increases as the distribution propagates.
    collisional_depth: feb,
    /// Current remaining energy per volume and time [erg/(cm^3 s)]. Decreases as the distribution propagates.
    remaining_power_density: feb,
}

impl PowerLawDistribution {
    /// Fraction of a mass of plasma assumed to be made up of hydrogen.
    const HYDROGEN_MASS_FRACTION: feb = 0.735;

    /// 2*pi*(classical electron radius)^2 [cm^2]
    const COLLISION_SCALE: feb = 4.989_344e-25;

    /// 1/2*ln( (2*pi*me*c/h)^3/(pi*alpha) [1/cm^3] )
    const ELECTRON_COULOMB_OFFSET: feb = 37.853_791;

    /// -ln( I_H [m_e*c^2] )
    const NEUTRAL_HYDROGEN_COULOMB_OFFSET: feb = 10.53422;

    /// Returns the exponent of the inverse power-law.
    pub fn delta(&self) -> feb {
        self.data.delta
    }

    /// Returns the total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    pub fn total_power_density(&self) -> feb {
        self.data.total_power_density
    }

    /// Returns the lower cut-off energy [units of electron rest energy].
    pub fn lower_cutoff_energy(&self) -> feb {
        self.data.lower_cutoff_energy
    }

    /// Returns the mean energy of the electrons in the distribution [units of electron rest energy].
    pub fn mean_energy(&self) -> feb {
        self.data.mean_energy
    }

    /// Returns the estimated propagation distance until the remaining
    /// power density drops below the minimum value [cm].
    pub fn estimated_depletion_distance(&self) -> feb {
        self.data.estimated_depletion_distance
    }

    /// Returns the current collisional depth.
    ///
    /// Increases as the distribution propagates.
    pub fn collisional_depth(&self) -> feb {
        self.collisional_depth
    }

    /// Returns the current remaining energy per volume and time [erg/(cm^3 s)].
    ///
    /// Decreases as the distribution propagates.
    pub fn remaining_power_density(&self) -> feb {
        self.remaining_power_density
    }

    fn new(config: PowerLawDistributionConfig, data: PowerLawDistributionData) -> Self {
        let collisional_depth = 0.0;
        let remaining_power_density = data.total_power_density;
        PowerLawDistribution {
            config,
            data,
            collisional_depth,
            remaining_power_density,
        }
    }

    fn compute_neutral_hydrogen_density(
        mass_density: feb,
        temperature: feb,
        electron_density: feb,
    ) -> feb {
        ionization::compute_equilibrium_neutral_hydrogen_density(
            Self::HYDROGEN_MASS_FRACTION,
            mass_density,
            temperature,
            electron_density,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn estimate_depletion_distance(
        electron_density: feb,
        neutral_hydrogen_density: feb,
        delta: feb,
        pitch_angle_factor: feb,
        total_power_density: feb,
        lower_cutoff_energy: feb,
        mean_energy: feb,
        depletion_power_density: feb,
    ) -> feb {
        feb::powi(lower_cutoff_energy, 2)
            * (feb::powf(
                depletion_power_density / total_power_density,
                2.0 / (2.0 - delta),
            ) - 1.0)
            / (pitch_angle_factor
                * Self::compute_collisional_coef(
                    electron_density,
                    neutral_hydrogen_density,
                    mean_energy,
                ))
    }

    fn compute_mean_energy(delta: feb, lower_cutoff_energy: feb) -> feb {
        lower_cutoff_energy * (delta - 1.0) / (delta - 2.0)
    }

    fn determine_pitch_angle_factor(pitch_angle_distribution: PitchAngleDistribution) -> feb {
        match pitch_angle_distribution {
            PitchAngleDistribution::Peaked => 2.0,
            PitchAngleDistribution::Isotropic => 4.0,
        }
    }

    fn compute_collisional_depth_increase(
        electron_density: feb,
        neutral_hydrogen_density: feb,
        pitch_angle_factor: feb,
        mean_energy: feb,
        step_length: feb,
    ) -> feb {
        pitch_angle_factor
            * Self::compute_collisional_coef(
                electron_density,
                neutral_hydrogen_density,
                mean_energy,
            )
            * step_length
    }

    fn compute_remaining_power_density(
        delta: feb,
        total_power_density: feb,
        lower_cutoff_energy: feb,
        collisional_depth: feb,
    ) -> feb {
        total_power_density
            * feb::powf(
                1.0 + collisional_depth / feb::powi(lower_cutoff_energy, 2),
                1.0 - 0.5 * delta,
            )
    }

    fn compute_collisional_coef(
        electron_density: feb,
        neutral_hydrogen_density: feb,
        electron_energy: feb,
    ) -> feb {
        Self::COLLISION_SCALE
            * (electron_density
                * Self::compute_electron_coulomb_logarithm(electron_density, electron_energy)
                + neutral_hydrogen_density
                    * Self::compute_neutral_hydrogen_coulomb_logarithm(electron_energy))
    }

    fn compute_electron_coulomb_logarithm(electron_density: feb, electron_energy: feb) -> feb {
        feb::max(
            0.0,
            Self::ELECTRON_COULOMB_OFFSET
                + 0.5
                    * feb::ln(
                        feb::powi(electron_energy * (electron_energy + 2.0), 2) / electron_density,
                    ),
        )
    }

    fn compute_neutral_hydrogen_coulomb_logarithm(electron_energy: feb) -> feb {
        feb::max(
            0.0,
            Self::NEUTRAL_HYDROGEN_COULOMB_OFFSET
                + 0.5 * feb::ln(electron_energy * electron_energy * (electron_energy + 2.0)),
        )
    }
}

impl BeamPropertiesCollection for PowerLawDistributionPropertiesCollection {
    type Item = PowerLawDistributionProperties;

    fn distribute_into_maps(
        self,
        scalar_values: &mut FixedBeamScalarValues,
        _vector_values: &mut FixedBeamVectorValues,
    ) {
        scalar_values.insert(
            "total_power_density".to_string(),
            self.total_power_densities,
        );
        scalar_values.insert(
            "lower_cutoff_energy".to_string(),
            self.lower_cutoff_energies,
        );
        scalar_values.insert(
            "estimated_depletion_distance".to_string(),
            self.estimated_depletion_distances,
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
                data.total_power_density,
                (data.lower_cutoff_energy, data.estimated_depletion_distance),
            )
        });

        let (total_power_densities, (lower_cutoff_energies, estimated_depletion_distances)): (
            Vec<_>,
            (Vec<_>, Vec<_>),
        ) = nested_tuples_iter.unzip();

        self.total_power_densities.par_extend(total_power_densities);
        self.lower_cutoff_energies.par_extend(lower_cutoff_energies);
        self.estimated_depletion_distances
            .par_extend(estimated_depletion_distances);
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
            total_power_density: self.data.total_power_density,
            lower_cutoff_energy: self.data.lower_cutoff_energy * MC2_ELECTRON / KEV_TO_ERG,
            estimated_depletion_distance: self.data.estimated_depletion_distance / U_L,
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

        let electron_density_field = snapshot.cached_scalar_field("nel");
        let mass_density_field = snapshot.cached_scalar_field("r");
        let temperature_field = snapshot.cached_scalar_field("tg");

        let electron_density = interpolator
            .interp_scalar_field(electron_density_field, &Point3::from(&deposition_position))
            .unwrap_and_update_position(&mut deposition_position)
            as feb;

        let mass_density = interpolator
            .interp_scalar_field(mass_density_field, &Point3::from(&deposition_position))
            .expect_inside() as feb
            * U_R;

        let temperature = interpolator
            .interp_scalar_field(temperature_field, &Point3::from(&deposition_position))
            .expect_inside() as feb;

        let neutral_hydrogen_density =
            Self::compute_neutral_hydrogen_density(mass_density, temperature, electron_density);

        let step_length = displacement.length() * U_L; // [cm]

        let collisional_depth_increase = if electron_density > std::f64::EPSILON {
            Self::compute_collisional_depth_increase(
                electron_density,
                neutral_hydrogen_density,
                self.data.pitch_angle_factor,
                self.data.mean_energy,
                step_length,
            )
        } else {
            0.0
        };

        let new_collisional_depth = self.collisional_depth + collisional_depth_increase;

        let new_remaining_power_density = Self::compute_remaining_power_density(
            self.data.delta,
            self.data.total_power_density,
            self.data.lower_cutoff_energy,
            new_collisional_depth,
        );

        let mut deposited_power_density =
            self.remaining_power_density - new_remaining_power_density;

        self.collisional_depth = new_collisional_depth;
        self.remaining_power_density = new_remaining_power_density;

        let depletion_status =
            if self.remaining_power_density >= self.config.min_remaining_power_density {
                DepletionStatus::Undepleted
            } else {
                // Deposit all the remaining power density
                deposited_power_density += self.remaining_power_density;
                self.remaining_power_density = 0.0;
                DepletionStatus::Depleted
            };

        PropagationResult {
            deposited_power_density,
            deposition_position,
            depletion_status,
        }
    }
}

impl PowerLawDistributionConfig {
    pub const DEFAULT_MIN_REMAINING_POWER_DENSITY: feb = 1e-6; // [erg/(cm^3 s)]
    pub const DEFAULT_MAX_PROPAGATION_DISTANCE: ftr = 100.0; // [Mm]

    /// Creates a set of power law distribution configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> Self {
        let min_remaining_power_density = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "min_remaining_power_density",
                "min_stop_en",
                &|min_stop_en: feb| min_stop_en * U_E / U_T,
                Self::DEFAULT_MIN_REMAINING_POWER_DENSITY,
            );
        let max_propagation_distance = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "max_propagation_distance",
                "max_dist",
                &|max_dist| max_dist,
                Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            );
        PowerLawDistributionConfig {
            min_remaining_power_density,
            max_propagation_distance,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    fn validate(&self) {
        assert!(
            self.min_remaining_power_density >= 0.0,
            "Minimum remaining power density must be larger than or equal to zero."
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
            min_remaining_power_density: Self::DEFAULT_MIN_REMAINING_POWER_DENSITY,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
        }
    }
}
