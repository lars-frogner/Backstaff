//! Power-law electron distribution.

pub mod acceleration;

use super::Distribution;
use crate::{
    constants::M_ELECTRON,
    ebeam::{feb, BeamPropertiesCollection, FixedBeamScalarValues, FixedBeamVectorValues},
    geometry::{Idx3, Point3},
    grid::fgr,
    tracing::stepping::SteppingSense,
};
use rayon::prelude::*;

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1/2)*lower_cutoff_energy^(delta - 1/2)*E^(-(delta + 1/2))`.
#[derive(Clone, Debug)]
pub struct PowerLawDistribution {
    /// Exponent of the inverse power-law.
    pub delta: feb,
    /// Total energy injected into the distribution per time [erg/s].
    pub total_power: feb,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    pub total_power_density: feb,
    /// Cosine of the initial pitch angle of the electrons.
    pub initial_pitch_angle_cosine: feb,
    /// Lower cut-off energy [keV].
    pub lower_cutoff_energy: feb,
    /// Direction of propagation of the electrons relative to the magnetic field direction.
    pub propagation_sense: SteppingSense,
    /// Cosine of the angle between the electric and magnetic field.
    pub electric_field_angle_cosine: feb,
    /// Position where the distribution originates [Mm].
    pub acceleration_position: Point3<fgr>,
    /// Indices of position where the distribution originates [Mm].
    pub acceleration_indices: Idx3<usize>,
    /// Volume of the grid cell where the distribution originates [cm^3].
    pub acceleration_volume: feb,
    /// Number density of electrons where the distribution originates [1/cm^3].
    pub ambient_electron_density: feb,
    /// Mass density where the distribution originates [g/cm^3]
    pub ambient_mass_density: feb,
    /// Temperature where the distribution originates [K]
    pub ambient_temperature: feb,
    /// Strength of the electric field where the distribution originates [gauss].
    pub ambient_electric_field_strength: feb,
    /// Strength of the magnetic field where the distribution originates [statV/cm].
    pub ambient_magnetic_field_strength: feb,
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
    electric_field_angle_cosines: Vec<feb>,
    propagation_senses: Vec<feb>,
}

impl PowerLawDistribution {
    fn compute_total_power(total_power_density: feb, acceleration_volume: feb) -> feb {
        total_power_density * acceleration_volume // [erg/s]
    }

    pub fn compute_mean_energy(delta: feb, lower_cutoff_energy: feb) -> feb {
        lower_cutoff_energy * (delta - 0.5) / (delta - 1.5)
    }

    pub fn evaluate_electron_number_per_dist(
        total_power: feb,
        lower_cutoff_energy: feb,
        delta: feb,
        energy: feb,
    ) -> feb {
        (total_power * (delta - 2.0)
            / (lower_cutoff_energy
                * lower_cutoff_energy
                * feb::sqrt(2.0 * lower_cutoff_energy / M_ELECTRON)))
            * (lower_cutoff_energy / energy).powf(delta + 0.5)
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
                            (data.electric_field_angle_cosine, data.propagation_sense),
                        ),
                    ),
                ),
            )
        });

        let (total_powers, (initial_pitch_angle_cosines, nested_tuples)): (
            Vec<_>,
            (Vec<_>, Vec<_>),
        ) = nested_tuples_iter.unzip();

        let (lower_cutoff_energies, (acceleration_volumes, nested_tuples)): (
            Vec<_>,
            (Vec<_>, Vec<_>),
        ) = nested_tuples.into_par_iter().unzip();

        let (electric_field_angle_cosines, propagation_senses): (Vec<_>, Vec<_>) =
            nested_tuples.into_par_iter().unzip();

        self.total_powers.par_extend(total_powers);
        self.initial_pitch_angle_cosines
            .par_extend(initial_pitch_angle_cosines);
        self.lower_cutoff_energies.par_extend(lower_cutoff_energies);
        self.acceleration_volumes.par_extend(acceleration_volumes);
        self.electric_field_angle_cosines
            .par_extend(electric_field_angle_cosines);
        self.propagation_senses.par_extend(propagation_senses);
    }
}

impl Distribution for PowerLawDistribution {
    type PropertiesCollectionType = PowerLawDistributionPropertiesCollection;

    fn acceleration_position(&self) -> &Point3<fgr> {
        &self.acceleration_position
    }

    fn acceleration_indices(&self) -> &Idx3<usize> {
        &self.acceleration_indices
    }

    fn propagation_sense(&self) -> SteppingSense {
        self.propagation_sense
    }

    fn properties(&self) -> <Self::PropertiesCollectionType as BeamPropertiesCollection>::Item {
        PowerLawDistributionProperties {
            total_power: self.total_power,
            initial_pitch_angle_cosine: self.initial_pitch_angle_cosine,
            lower_cutoff_energy: self.lower_cutoff_energy,
            acceleration_volume: self.acceleration_volume,
            electric_field_angle_cosine: self.electric_field_angle_cosine,
            propagation_sense: match self.propagation_sense {
                SteppingSense::Same => 1.0,
                SteppingSense::Opposite => -1.0,
            },
        }
    }
}
