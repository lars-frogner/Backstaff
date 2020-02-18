//! Command line interface for the simple power-law distribution accelerator.

use crate::cli;
use crate::ebeam::distribution::power_law::acceleration::simple::SimplePowerLawAccelerationConfig;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use crate::units::solar::{U_E, U_T};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the simple power-law distribution accelerator.
pub fn create_simple_power_law_accelerator_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("simple_power_law_accelerator")
        .about("Use the simple power-law distribution accelerator model")
        .long_about(
            "Use the simple power-law distribution accelerator model.\n\
             The total distribution energy is assumed to be a fixed fraction of the\n\
             reconnection energy, and the lower cut-off energy is found from the intersection\n\
             of the non-thermal distribution with the thermal distribution.",
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
            Arg::with_name("min-lower-cutoff-energy")
                .long("min-lower-cutoff-energy")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with lower cut-off energies lower than this value\n\
                     are discarded [keV] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-thermalization-distance")
                .long("min-thermalization-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with an estimated thermalization distance smaller\n\
                     than this value are discarded [Mm] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max-pitch-angle")
                .long("max-pitch-angle")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with initial absolute pitch angles larger than this are discarded\n\
                    [deg] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max-electric-field-angle")
                .long("max-electric-field-angle")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with electric field directions angled more than this\n\
                     away from the magnetic field axis are discarded [deg] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("cutoff-energy-guess")
                .long("cutoff-energy-guess")
                .require_equals(true)
                .value_name("VALUE")
                .help("Initial guess to use when estimating lower cut-off energy [keV]\n")
                .takes_value(true)
                .default_value("4.0"),
        )
        .arg(
            Arg::with_name("root-finding-error")
                .long("root-finding-error")
                .require_equals(true)
                .value_name("VALUE")
                .help("Target relative error when estimating lower cut-off energy\n")
                .takes_value(true)
                .default_value("1e-3"),
        )
        .arg(
            Arg::with_name("root-finding-iterations")
                .long("root-finding-iterations")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Maximum number of iterations when estimating lower cut-off energy\n")
                .takes_value(true)
                .default_value("100"),
        )
}

/// Determines simple power-law distribution accelerator parameters
/// based on provided options and values in parameter file.
pub fn construct_simple_power_law_accelerator_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
) -> SimplePowerLawAccelerationConfig {
    let acceleration_duration = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "acceleration-duration",
        "dt",
        &|dt| dt * U_T,
        SimplePowerLawAccelerationConfig::DEFAULT_ACCELERATION_DURATION,
    );

    let particle_energy_fraction = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "particle-energy-fraction",
        "qjoule_acc_frac",
        &|qjoule_acc_frac| qjoule_acc_frac,
        SimplePowerLawAccelerationConfig::DEFAULT_PARTICLE_ENERGY_FRACTION,
    );

    let power_law_delta = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "power-law-delta",
        "power_law_index",
        &|power_law_index| power_law_index,
        SimplePowerLawAccelerationConfig::DEFAULT_POWER_LAW_DELTA,
    );

    let min_total_power_density = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-total-power-density",
        "min_beam_en",
        &|min_beam_en| min_beam_en * U_E / U_T,
        SimplePowerLawAccelerationConfig::DEFAULT_MIN_TOTAL_POWER_DENSITY,
    );

    let min_lower_cutoff_energy = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-lower-cutoff-energy",
        "min_cutoff_en",
        &|min_cutoff_en| min_cutoff_en,
        SimplePowerLawAccelerationConfig::DEFAULT_MIN_LOWER_CUTOFF_ENERGY,
    );

    let min_thermalization_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-thermalization-distance",
        "min_stop_dist",
        &|min_stop_dist| min_stop_dist,
        SimplePowerLawAccelerationConfig::DEFAULT_MIN_THERMALIZATION_DISTANCE,
    );

    let max_pitch_angle = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "max-pitch-angle",
        "max_pitch_angle",
        &|max_pitch_angle| max_pitch_angle,
        SimplePowerLawAccelerationConfig::DEFAULT_MAX_PITCH_ANGLE,
    );

    let max_electric_field_angle = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "max-electric-field-angle",
        "max_electric_field_angle",
        &|max_electric_field_angle| max_electric_field_angle,
        SimplePowerLawAccelerationConfig::DEFAULT_MAX_ELECTRIC_FIELD_ANGLE,
    );

    let initial_cutoff_energy_guess =
        cli::get_value_from_required_parseable_argument(arguments, "cutoff-energy-guess");
    let acceptable_root_finding_error =
        cli::get_value_from_required_parseable_argument(arguments, "root-finding-error");
    let max_root_finding_iterations =
        cli::get_value_from_required_parseable_argument(arguments, "root-finding-iterations");

    SimplePowerLawAccelerationConfig {
        acceleration_duration,
        particle_energy_fraction,
        power_law_delta,
        min_total_power_density,
        min_lower_cutoff_energy,
        min_thermalization_distance,
        max_pitch_angle,
        max_electric_field_angle,
        initial_cutoff_energy_guess,
        acceptable_root_finding_error,
        max_root_finding_iterations,
    }
}
