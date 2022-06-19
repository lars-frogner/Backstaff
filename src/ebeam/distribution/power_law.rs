//! Power-law electron distribution.

pub mod acceleration;

use super::{
    super::{feb, BeamPropertiesCollection, FixedBeamScalarValues, FixedBeamVectorValues},
    DepletionStatus, Distribution, PropagationResult,
};
use crate::{
    constants::{KEV_TO_ERG, M_H, PI, Q_ELECTRON},
    field::CachingScalarFieldProvider3,
    geometry::{
        Dim3::{X, Y, Z},
        Idx3, Point3, Vec3,
    },
    grid::{fgr, Grid3},
    interpolation::Interpolator3,
    io::snapshot::{self, fdt, SnapshotProvider3},
    math,
    plasma::ionization,
    tracing::{ftr, stepping::SteppingSense},
    units::solar::{U_L, U_L3, U_R},
};
use ndarray::prelude::*;
use rayon::prelude::*;

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionConfig {
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

/// Data associated with a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionData {
    /// Exponent of the inverse power-law.
    delta: feb,
    /// Total energy injected into the distribution per time [erg/s].
    total_power: feb,
    #[allow(dead_code)]
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Cosine of the initial pitch angle of the electrons.
    initial_pitch_angle_cosine: feb,
    /// Lower cut-off energy [keV].
    lower_cutoff_energy: feb,
    /// Position where the distribution originates [Mm].
    acceleration_position: Point3<fgr>,
    /// Indices of position where the distribution originates [Mm].
    acceleration_indices: Idx3<usize>,
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
    /// Estimated depletion distance of the electrons in the distribution [cm].
    estimated_depletion_distance: feb,
    /// Cosine of the angle between the electric and magnetic field.
    electric_field_angle_cosine: feb,
}

/// Exposed properties of a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionProperties {
    /// Total energy injected into the distribution per time [erg/s].
    total_power: feb,
    /// Cosine of the initial pitch angle of the electrons.
    initial_pitch_angle_cosine: feb,
    /// Lower cut-off energy [keV].
    lower_cutoff_energy: feb,
    /// Volume of the grid cell where the distribution originates [cm^3].
    acceleration_volume: feb,
    /// Estimated depletion distance of the electrons in the distribution [Mm].
    estimated_depletion_distance: feb,
    /// Cosine of the angle between the electric and magnetic field.
    electric_field_angle_cosine: feb,
    /// Direction of propagation of the electrons relative to the magnetic field direction (+1 or -1).
    propagation_sense: feb,
}

/// Property values of each individual distribution in a set of power-law distributions.
#[derive(Clone, Default, Debug)]
pub struct PowerLawDistributionPropertiesCollection {
    total_powers: Vec<feb>,
    initial_pitch_angle_cosines: Vec<feb>,
    lower_cutoff_energies: Vec<feb>,
    acceleration_volumes: Vec<feb>,
    estimated_depletion_distances: Vec<feb>,
    electric_field_angle_cosines: Vec<feb>,
    propagation_senses: Vec<feb>,
}

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1/2)*lower_cutoff_energy^(delta - 1/2)*E^(-(delta + 1/2))`.
#[derive(Clone, Debug)]
pub struct PowerLawDistribution {
    config: PowerLawDistributionConfig,
    data: PowerLawDistributionData,
    /// Current column depth of hydrogen atoms (neutral and ionized) traversed by the distribution [hydrogen/cm^2].
    hydrogen_column_depth: feb,
    /// Equivalent ionized column depth (Hawley & Fisher, 1994) traversed by the distribution [hydrogen/cm^2].
    equivalent_ionized_column_depth: feb,
    /// How far outside the acceleration region the distribution has propagated [Mm].
    outside_distance: feb,
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
        let outside_distance = 0.0;

        PowerLawDistribution {
            config,
            data,
            hydrogen_column_depth,
            equivalent_ionized_column_depth,
            outside_distance,
        }
    }

    fn compute_total_power(total_power_density: feb, acceleration_volume: feb) -> feb {
        total_power_density * acceleration_volume // [erg/s]
    }

    fn compute_mean_energy(delta: feb, lower_cutoff_energy: feb) -> feb {
        lower_cutoff_energy * (delta - 0.5) / (delta - 1.5)
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

    pub fn compute_uniform_plasma_heating_integral(
        &self,
        total_hydrogen_density: feb,
        effective_coulomb_logarithm: feb,
        step_length: feb,
    ) -> (feb, feb, feb, feb) {
        let coulomb_logarithm_ratio =
            effective_coulomb_logarithm / self.data.electron_coulomb_logarithm;

        let hydrogen_column_depth_increase = total_hydrogen_density * step_length;
        let new_hydrogen_column_depth = self.hydrogen_column_depth + hydrogen_column_depth_increase;
        let new_equivalent_ionized_column_depth = self.equivalent_ionized_column_depth
            + hydrogen_column_depth_increase * coulomb_logarithm_ratio;

        let stopping_column_depth =
            self.data.stopping_ionized_column_depth / coulomb_logarithm_ratio;

        let start_column_depth_ratio = self.hydrogen_column_depth / stopping_column_depth;
        let start_ionized_column_depth_ratio =
            self.equivalent_ionized_column_depth / self.data.stopping_ionized_column_depth;

        let column_depth_ratio_increase = hydrogen_column_depth_increase / stopping_column_depth;

        let end_column_depth_ratio = start_column_depth_ratio + column_depth_ratio_increase;
        let end_ionized_column_depth_ratio =
            start_ionized_column_depth_ratio + column_depth_ratio_increase;

        let power = 0.5 * self.data.delta;
        let residual_factor = feb::powf(end_ionized_column_depth_ratio, -power);

        let constant_factor =
            self.data.heating_scale * stopping_column_depth * effective_coulomb_logarithm;

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

impl BeamPropertiesCollection for PowerLawDistributionPropertiesCollection {
    type Item = PowerLawDistributionProperties;

    fn distribute_into_maps(
        self,
        scalar_values: &mut FixedBeamScalarValues,
        _vector_values: &mut FixedBeamVectorValues,
    ) {
        scalar_values.insert("total_power".to_string(), self.total_powers);
        scalar_values.insert(
            "initial_pitch_angle_cosine".to_string(),
            self.initial_pitch_angle_cosines,
        );
        scalar_values.insert(
            "lower_cutoff_energy".to_string(),
            self.lower_cutoff_energies,
        );
        scalar_values.insert("acceleration_volume".to_string(), self.acceleration_volumes);
        scalar_values.insert(
            "estimated_depletion_distance".to_string(),
            self.estimated_depletion_distances,
        );
        scalar_values.insert(
            "electric_field_angle_cosine".to_string(),
            self.electric_field_angle_cosines,
        );
        scalar_values.insert("propagation_sense".to_string(), self.propagation_senses);
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
                    data.initial_pitch_angle_cosine,
                    (
                        data.lower_cutoff_energy,
                        (
                            data.acceleration_volume,
                            (
                                data.estimated_depletion_distance,
                                (data.electric_field_angle_cosine, data.propagation_sense),
                            ),
                        ),
                    ),
                ),
            )
        });

        let (total_powers, (initial_pitch_angle_cosines, nested_tuples)): (
            Vec<_>,
            (Vec<_>, Vec<_>),
        ) = nested_tuples_iter.unzip();

        let (
            lower_cutoff_energies,
            (acceleration_volumes, (estimated_depletion_distances, nested_tuples)),
        ): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = nested_tuples.into_par_iter().unzip();

        let (electric_field_angle_cosines, propagation_senses): (Vec<_>, Vec<_>) =
            nested_tuples.into_par_iter().unzip();

        self.total_powers.par_extend(total_powers);
        self.initial_pitch_angle_cosines
            .par_extend(initial_pitch_angle_cosines);
        self.lower_cutoff_energies.par_extend(lower_cutoff_energies);
        self.acceleration_volumes.par_extend(acceleration_volumes);
        self.estimated_depletion_distances
            .par_extend(estimated_depletion_distances);
        self.electric_field_angle_cosines
            .par_extend(electric_field_angle_cosines);
        self.propagation_senses.par_extend(propagation_senses);
    }
}

impl Distribution for PowerLawDistribution {
    type PropertiesCollectionType = PowerLawDistributionPropertiesCollection;

    fn acceleration_position(&self) -> &Point3<fgr> {
        &self.data.acceleration_position
    }

    fn acceleration_indices(&self) -> &Idx3<usize> {
        &self.data.acceleration_indices
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
            initial_pitch_angle_cosine: self.data.initial_pitch_angle_cosine,
            lower_cutoff_energy: self.data.lower_cutoff_energy,
            acceleration_volume: self.data.acceleration_volume,
            estimated_depletion_distance: self.data.estimated_depletion_distance / U_L,
            electric_field_angle_cosine: self.data.electric_field_angle_cosine,
            propagation_sense: match self.data.propagation_sense {
                SteppingSense::Same => 1.0,
                SteppingSense::Opposite => -1.0,
            },
        }
    }

    fn propagate<P>(
        &mut self,
        snapshot: &P,
        acceleration_map: &Array3<bool>,
        interpolator: &dyn Interpolator3<fdt>,
        displacement: &Vec3<ftr>,
        new_position: &Point3<ftr>,
    ) -> PropagationResult
    where
        P: CachingScalarFieldProvider3<fdt>,
    {
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
                self.data.electron_coulomb_logarithm,
                self.data.neutral_hydrogen_coulomb_logarithm,
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

impl PowerLawDistributionConfig {
    pub const DEFAULT_MIN_RESIDUAL_FACTOR: feb = 1e-5;
    pub const DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE: feb = 1e5; // [erg/s/cm]
    pub const DEFAULT_MAX_PROPAGATION_DISTANCE: ftr = 100.0; // [Mm]
    pub const DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD: feb = 0.1; // [Mm]
    pub const DEFAULT_CONTINUE_DEPLETED_BEAMS: bool = false;

    /// Creates a set of power law distribution configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file<P>(provider: &P) -> Self
    where
        P: SnapshotProvider3,
    {
        let min_residual_factor =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "min_residual_factor",
                "min_residual",
                &|min_residual: feb| min_residual,
                Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            );
        let min_deposited_power_per_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "min_deposited_power_per_distance",
                "min_dep_en",
                &|min_dep_en: feb| min_dep_en,
                Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            );
        let max_propagation_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "max_propagation_distance",
                "max_dist",
                &|max_dist: feb| max_dist,
                Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            );
        let outside_deposition_threshold =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                provider.parameters(),
                "outside_deposition_threshold",
                "out_dep_thresh",
                &|out_dep_thresh: feb| out_dep_thresh,
                Self::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
            );
        PowerLawDistributionConfig {
            min_residual_factor,
            min_deposited_power_per_distance,
            max_propagation_distance,
            outside_deposition_threshold,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    fn validate(&self) {
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

impl Default for PowerLawDistributionConfig {
    fn default() -> Self {
        PowerLawDistributionConfig {
            min_residual_factor: Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            min_deposited_power_per_distance: Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            outside_deposition_threshold: Self::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }
}
