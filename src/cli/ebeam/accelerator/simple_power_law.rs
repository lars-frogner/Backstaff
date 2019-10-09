//! Command line interface for the simple power-law distribution accelerator.

use crate::cli;
use crate::ebeam::distribution::power_law::acceleration::simple::SimplePowerLawAccelerationConfig;
use crate::ebeam::distribution::power_law::PitchAngleDistribution;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use crate::units::solar::{U_E, U_L, U_T};
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
        .arg(
            Arg::with_name("acceleration-duration")
                .long("acceleration-duration")
                .require_equals(true)
                .value_name("VALUE")
                .long_help("Duration of the acceleration events [s] [default: from param file]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("particle-energy-fraction")
                .long("particle-energy-fraction")
                .require_equals(true)
                .value_name("VALUE")
                .long_help(
                    "Fraction of the released reconnection energy going into acceleration of\n\
                    electrons [default: from param file]",
                )
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("power-law-delta")
                .long("power-law-delta")
                .require_equals(true)
                .value_name("VALUE")
                .long_help(
                    "Exponent of the inverse power-law describing the non-thermal electron\n\
                    distribution [default: from param file]",
                )
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("pitch-angle-distribution")
                .long("pitch-angle-distribution")
                .require_equals(true)
                .value_name("TYPE")
                .long_help("Type of pitch angle distribution of the non-thermal electrons\n")
                .next_line_help(true)
                .takes_value(true)
                .possible_values(&["peaked", "isotropic"])
                .default_value("peaked"),
        )
        .arg(
            Arg::with_name("min-total-power-density")
                .long("min-total-power-density")
                .require_equals(true)
                .value_name("VALUE")
                .long_help(
                    "Distributions with total power densities smaller than this value are discarded\n\
                    [erg/(cm^3 s)] [default: from param file]",
                )
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-estimated-depletion-distance")
                .long("min-estimated-depletion-distance")
                .require_equals(true)
                .value_name("VALUE")
                .long_help(
                    "Distributions with an initial estimated depletion distance smaller than this\n\
                    value are discarded [cm] [default: from param file]",
                )
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max-acceleration-angle")
                .long("max-acceleration-angle")
                .require_equals(true)
                .value_name("VALUE")
                .long_help(
                    "Distributions with acceleration directions angled more than this away from the\n\
                    magnetic field axis are discarded [deg]",
                )
                .next_line_help(true)
                .takes_value(true)
                .default_value("70.0"),
        )
        .arg(
            Arg::with_name("initial-cutoff-energy-guess")
                .long("initial-cutoff-energy-guess")
                .require_equals(true)
                .value_name("VALUE")
                .long_help("Initial guess to use when estimating lower cut-off energy [keV]")
                .next_line_help(true)
                .takes_value(true)
                .default_value("4.0"),
        )
        .arg(
            Arg::with_name("acceptable-root-finding-error")
                .long("acceptable-root-finding-error")
                .require_equals(true)
                .value_name("VALUE")
                .long_help("Target relative error when estimating lower cut-off energy")
                .next_line_help(true)
                .takes_value(true)
                .default_value("1e-3"),
        )
        .arg(
            Arg::with_name("max-root-finding-iterations")
                .long("max-root-finding-iterations")
                .require_equals(true)
                .value_name("NUMBER")
                .long_help("Maximum number of iterations when estimating lower cut-off energy")
                .next_line_help(true)
                .takes_value(true)
                .default_value("100"),
        )
        .arg(
            Arg::with_name("ignore-rejection")
                .long("ignore-rejection")
                .help("Generate beams even when they meet a rejection condition"),
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
        SimplePowerLawAccelerationConfig::DEFAULT_MIN_TOTAL_POWER_DENSITY,
    );

    let min_estimated_depletion_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-estimated-depletion-distance",
        "min_stop_dist",
        &|min_stop_dist| min_stop_dist * U_L,
        SimplePowerLawAccelerationConfig::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
    );

    let max_acceleration_angle =
        cli::get_value_from_required_parseable_argument(arguments, "max-acceleration-angle");
    let initial_cutoff_energy_guess =
        cli::get_value_from_required_parseable_argument(arguments, "initial-cutoff-energy-guess");
    let acceptable_root_finding_error =
        cli::get_value_from_required_parseable_argument(arguments, "acceptable-root-finding-error");
    let max_root_finding_iterations =
        cli::get_value_from_required_parseable_argument(arguments, "max-root-finding-iterations");

    let ignore_rejection = arguments.is_present("ignore-rejection");

    SimplePowerLawAccelerationConfig {
        acceleration_duration,
        particle_energy_fraction,
        power_law_delta,
        pitch_angle_distribution,
        min_total_power_density,
        min_estimated_depletion_distance,
        max_acceleration_angle,
        initial_cutoff_energy_guess,
        acceptable_root_finding_error,
        max_root_finding_iterations,
        ignore_rejection,
    }
}
