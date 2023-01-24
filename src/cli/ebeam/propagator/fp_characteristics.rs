//! Command line interface for the Fokker-Planck characteristics
//! electron distribution propagator.

use crate::{
    add_subcommand_combinations,
    cli::{
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        tracing::stepping::rkf::create_rkf_stepper_subcommand, utils,
    },
    ebeam::{
        feb,
        propagation::fp_characteristics::{
            AnalyticalTransporterConfig, CharacteristicsPropagatorConfig,
        },
    },
    io::snapshot::SnapshotParameters,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the Fokker-Planck characteristics
/// propagator.
pub fn create_characteristics_propagator_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "characteristics_propagator";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the characteristics propagation method")
        .long_about(
            "Use the characteristics propagation method.\n\
             The method solves the equations for the characteristics\n\
             of the frictional part of the Fokker-Planck equation.",
        )
        .arg(
            Arg::new("n-energies")
                .long("n-energies")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Number of energies to use for representing the distribution")
                .takes_value(true)
                .default_value("200"),
        )
        .arg(
            Arg::new("energy-range")
                .long("energy-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Limits for the energy range, relative to the lower cutoff energy")
                .takes_value(true)
                .number_of_values(2)
                .default_value("1e-1,1e2"),
        )
        .arg(
            Arg::new("enable-analytical-solver")
                .long("enable-analytical-solver")
                .help(
                    "Use analytical solutions for electron propagation when applicable",
                ),
        )
        .arg(
            Arg::new("min-depletion-distance")
                .long("min-depletion-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with an estimated depletion distance smaller\n\
                     than this value are discarded [Mm] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("min-residual-factor")
                .long("min-residual-factor")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions are considered depleted when the residual energy factor has\n\
                     decreased below this limit, given that the deposited power per distance is\n\
                     smaller than its lower limit [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("min-deposited-power-per-distance")
                .long("min-deposited-power-per-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions are considered depleted when the deposited power per distance\n\
                     [erg/s/cm] has decreased below this limit, given that the residual energy factor\n\
                     is smaller than its lower limit [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("max-propagation-distance")
                .long("max-propagation-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Maximum distance the distribution can propagate before propagation\n\
                     should be terminated [Mm] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("continue-depleted-beams")
                .long("continue-depleted-beams")
                .help("Keep propagating beams even after they are considered depleted"),
        );

    add_subcommand_combinations!(command, command_name, false; poly_fit_interpolator, rkf_stepper)
}

/// Determines characteristics propagator parameters based on
/// provided options and values in parameter file.
pub fn construct_characteristics_propagator_config_from_options(
    arguments: &ArgMatches,
    parameters: &dyn SnapshotParameters,
) -> CharacteristicsPropagatorConfig {
    let n_energies =
        utils::get_value_from_required_parseable_argument::<usize>(arguments, "n-energies");

    let (min_energy_relative_to_cutoff, max_energy_relative_to_cutoff) = utils::parse_limits(
        arguments,
        "energy-range",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );

    let min_depletion_distance = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "min-depletion-distance",
        "min_stop_dist",
        &|min_stop_dist: feb| min_stop_dist,
        CharacteristicsPropagatorConfig::DEFAULT_MIN_DEPLETION_DISTANCE,
    );

    let enable_analytical_transporter = arguments.is_present("enable-analytical-solver");

    let min_residual_factor = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "min-residual-factor",
        "min_residual",
        &|min_residual: feb| min_residual,
        CharacteristicsPropagatorConfig::DEFAULT_MIN_RESIDUAL_FACTOR,
    );

    let min_deposited_power_per_distance = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "min-deposited-power-per-distance",
        "min_dep_en",
        &|min_dep_en: feb| min_dep_en,
        CharacteristicsPropagatorConfig::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
    );

    let max_propagation_distance = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "max-propagation-distance",
        "max_dist",
        &|max_dist: feb| max_dist,
        CharacteristicsPropagatorConfig::DEFAULT_MAX_PROPAGATION_DISTANCE,
    );

    let continue_depleted_beams = arguments.is_present("continue-depleted-beams");

    let config = CharacteristicsPropagatorConfig {
        analytical_transporter_config: AnalyticalTransporterConfig {
            enabled: enable_analytical_transporter,
            ..AnalyticalTransporterConfig::default()
        },
        n_energies,
        min_energy_relative_to_cutoff,
        max_energy_relative_to_cutoff,
        min_depletion_distance,
        min_residual_factor,
        min_deposited_power_per_distance,
        max_propagation_distance,
        continue_depleted_beams,
    };
    config.validate();
    config
}
