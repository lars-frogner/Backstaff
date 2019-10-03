//! Command line interface for the simple power-law distribution accelerator.

use crate::cli;
use crate::ebeam::distribution::power_law::acceleration::simple::SimplePowerLawAccelerationConfig;
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by the simple power-law distribution accelerator.
pub fn add_simple_power_law_accelerator_options_to_subcommand<'a, 'b>(
    app: App<'a, 'b>,
) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("acceleration-duration")
            .long("acceleration-duration")
            .value_name("VALUE")
            .long_help("Duration of the acceleration events [s] [default: from param file]")
            .takes_value(true),
    )
    .arg(
        Arg::with_name("particle-energy-fraction")
            .long("particle-energy-fraction")
            .value_name("VALUE")
            .long_help(
                "Fraction of the released reconnection energy going into acceleration of\n\
                 electrons [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("power-law-delta")
            .long("power-law-delta")
            .value_name("VALUE")
            .long_help(
                "Exponent of the inverse power-law describing the non-thermal electron\n\
                 distribution [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-total-power-density")
            .long("min-total-power-density")
            .value_name("VALUE")
            .long_help(
                "Distributions with total power densities smaller than this value are discarded\n\
                 [erg/(cm^3 s)] [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-estimated-depletion-distance")
            .long("min-estimated-depletion-distance")
            .value_name("VALUE")
            .long_help(
                "Distributions with an initial estimated depletion distance smaller than this\n\
                 value are discarded [cm] [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-acceleration-angle")
            .long("max-acceleration-angle")
            .value_name("VALUE")
            .long_help(
                "Distributions with acceleration directions angled more than this away from the\n\
                 magnetic field axis are discarded [deg]",
            )
            .takes_value(true)
            .default_value("70.0"),
    )
    .arg(
        Arg::with_name("initial-cutoff-energy-guess")
            .long("initial-cutoff-energy-guess")
            .value_name("VALUE")
            .long_help("Initial guess to use when estimating lower cut-off energy [keV]")
            .takes_value(true)
            .default_value("4.0"),
    )
    .arg(
        Arg::with_name("acceptable-root-finding-error")
            .long("acceptable-root-finding-error")
            .value_name("VALUE")
            .long_help("Target relative error when estimating lower cut-off energy")
            .takes_value(true)
            .default_value("1e-3"),
    )
    .arg(
        Arg::with_name("max-root-finding-iterations")
            .long("max-root-finding-iterations")
            .value_name("NUMBER")
            .long_help("Maximum number of iterations when estimating lower cut-off energy")
            .takes_value(true)
            .default_value("100"),
    )
    .arg(
        Arg::with_name("ignore-rejection")
            .long("ignore-rejection")
            .help("Generate beams even when they meet a rejection condition"),
    )
}

/// Sets simple power-law distribution accelerator parameters based on present arguments.
pub fn configure_simple_power_law_accelerator_from_options(
    config: &mut SimplePowerLawAccelerationConfig,
    arguments: &ArgMatches,
) {
    cli::assign_value_from_parseable_argument(
        &mut config.acceleration_duration,
        arguments,
        "acceleration-duration",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.particle_energy_fraction,
        arguments,
        "particle-energy-fraction",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.power_law_delta,
        arguments,
        "power-law-delta",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.min_total_power_density,
        arguments,
        "min-total-power-density",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.min_estimated_depletion_distance,
        arguments,
        "min-estimated-depletion-distance",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_acceleration_angle,
        arguments,
        "max-acceleration-angle",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.initial_cutoff_energy_guess,
        arguments,
        "initial-cutoff-energy-guess",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.acceptable_root_finding_error,
        arguments,
        "acceptable-root-finding-error",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_root_finding_iterations,
        arguments,
        "max-root-finding-iterations",
    );
    cli::assign_bool_value_from_flag_presence(
        &mut config.ignore_rejection,
        arguments,
        "ignore-rejection",
    );
}
