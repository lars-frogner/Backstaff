//! Command line interface for the DC power-law distribution accelerator.

use crate::cli;
use crate::ebeam::distribution::power_law::acceleration::dc::acceleration_region::AccelerationRegionTracerConfig;
use crate::ebeam::distribution::power_law::acceleration::dc::DCPowerLawAccelerationConfig;
use crate::ebeam::distribution::power_law::PitchAngleDistribution;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use crate::units::solar::{U_E, U_L, U_T};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the DC power-law distribution accelerator.
pub fn create_dc_power_law_accelerator_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("dc_power_law_accelerator")
        .about("Use the direct current power-law distribution accelerator model")
        .long_about(
            "Use the direct current power-law distribution accelerator model.\n\
             The total distribution energy is assumed to be a fixed fraction of the\n\
             reconnection energy, and the lower cut-off energy is computed from the\n\
             acceleration region length and average electric field strength parallel\n\
             to the magnetic field direction.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("acceleration-duration")
                .long("acceleration-duration")
                .require_equals(true)
                .value_name("VALUE")
                .help("Duration of the acceleration events [s] [default: from param file]")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("particle-energy-fraction")
                .long("particle-energy-fraction")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Fraction of the released reconnection energy going into\n\
                     acceleration of electrons [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("power-law-delta")
                .long("power-law-delta")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Exponent of the inverse power-law describing the non-thermal\n\
                     electron distribution [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("pitch-angle-distribution")
                .long("pitch-angle-distribution")
                .require_equals(true)
                .value_name("TYPE")
                .help("Type of pitch angle distribution of the non-thermal electrons\n")
                .takes_value(true)
                .possible_values(&["peaked", "isotropic"])
                .default_value("peaked"),
        )
        .arg(
            Arg::with_name("min-total-power-density")
                .long("min-total-power-density")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with total power densities smaller than this value\n\
                     are discarded [erg/(cm^3 s)] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-depletion-distance")
                .long("min-depletion-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with an initial estimated depletion distance smaller\n\
                     than this value are discarded [cm] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-parallel-electric-field-strength")
                .long("min-parallel-electric-field-strength")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "The acceleration region ends where the absolute component of the electric field\n\
                     parallel to the magnetic field direction becomes lower than this value [statV/cm]\n",
                )
                .takes_value(true)
                .default_value("1e-12"),
        )
        .arg(
            Arg::with_name("min-length")
                .long("min-length")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Acceleration regions shorter than this are discarded [Mm]",
                )
                .takes_value(true)
                .default_value("0"),
        )
}

/// Determines DC power-law distribution accelerator parameters
/// based on provided options and values in parameter file.
pub fn construct_dc_power_law_accelerator_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
) -> (DCPowerLawAccelerationConfig, AccelerationRegionTracerConfig) {
    let acceleration_duration = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "acceleration-duration",
        "dt",
        &|dt| dt * U_T,
        DCPowerLawAccelerationConfig::DEFAULT_ACCELERATION_DURATION,
    );

    let particle_energy_fraction = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "particle-energy-fraction",
        "qjoule_acc_frac",
        &|qjoule_acc_frac| qjoule_acc_frac,
        DCPowerLawAccelerationConfig::DEFAULT_PARTICLE_ENERGY_FRACTION,
    );

    let power_law_delta = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "power-law-delta",
        "power_law_index",
        &|power_law_index| power_law_index,
        DCPowerLawAccelerationConfig::DEFAULT_POWER_LAW_DELTA,
    );

    let pitch_angle_distribution = cli::get_value_from_required_constrained_argument(
        arguments,
        "pitch-angle-distribution",
        &["peaked", "isotropic"],
        &[
            PitchAngleDistribution::Peaked,
            PitchAngleDistribution::Isotropic,
        ],
    );

    let min_total_power_density = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-total-power-density",
        "min_beam_en",
        &|min_beam_en| min_beam_en * U_E / U_T,
        DCPowerLawAccelerationConfig::DEFAULT_MIN_TOTAL_POWER_DENSITY,
    );

    let min_estimated_depletion_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-depletion-distance",
        "min_stop_dist",
        &|min_stop_dist| min_stop_dist * U_L,
        DCPowerLawAccelerationConfig::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
    );

    let min_parallel_electric_field_strength = cli::get_value_from_required_parseable_argument(
        arguments,
        "min-parallel-electric-field-strength",
    );
    let min_length = cli::get_value_from_required_parseable_argument(arguments, "min-length");

    (
        DCPowerLawAccelerationConfig {
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            pitch_angle_distribution,
            min_total_power_density,
            min_estimated_depletion_distance,
        },
        AccelerationRegionTracerConfig {
            min_parallel_electric_field_strength,
            min_length,
        },
    )
}
