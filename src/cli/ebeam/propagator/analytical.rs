//! Command line interface for the analytical electron distribution
//! propagator.

use crate::{
    add_subcommand_combinations,
    cli::{
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        tracing::stepping::rkf::create_rkf_stepper_subcommand, utils,
    },
    ebeam::{feb, propagation::analytical::AnalyticalPropagatorConfig},
    io::snapshot::SnapshotParameters,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the analytical propagator.
pub fn create_analytical_propagator_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "analytical_propagator";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the analytical propagation method")
        .long_about(
            "Use the analytical propagation method.\n\
             The method follows Hawley & Fisher (1994).",
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
            Arg::new("outside-deposition-threshold")
                .long("outside-deposition-threshold")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Maximum distance outside the initial extended acceleration region the\n\
                     distribution can propagate before energy deposition starts [Mm]\n\
                     [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("continue-depleted-beams")
                .long("continue-depleted-beams")
                .help("Keep propagating beams even after they are considered depleted"),
        )
        .arg(
            Arg::new("max-column-depth-increase")
                .long("max-column-depth-increase")
                .require_equals(true)
                .value_name("VALUE")
                .help("Maximum allowed column depth increase for a single substep")
                .takes_value(true)
                .default_value("2e14"),
        )
        .arg(
            Arg::new("max-substeps")
                .long("max-substeps")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Maximum number of substeps to take for each step")
                .takes_value(true)
                .default_value("10000"),
        )
        .arg(
            Arg::new("n-initial-steps-with-substeps")
                .long("n-initial-steps-with-substeps")
                .require_equals(true)
                .value_name("NUMBER")
                .help("How many of the initial steps that should use substepping")
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::new("keep-initial-ionization-fraction")
                .long("keep-initial-ionization-fraction")
                .help(
                    "Do not update the hydrogen ionization fraction from the inital value\n\
                     while propagating.",
                ),
        );

    add_subcommand_combinations!(command, command_name, false; poly_fit_interpolator, rkf_stepper)
}

/// Determines analytical propagator parameters based on
/// provided options and values in parameter file.
pub fn construct_analytical_propagator_config_from_options(
    arguments: &ArgMatches,
    parameters: &dyn SnapshotParameters,
) -> AnalyticalPropagatorConfig {
    let min_depletion_distance = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "min-depletion-distance",
        "min_stop_dist",
        &|min_stop_dist: feb| min_stop_dist,
        AnalyticalPropagatorConfig::DEFAULT_MIN_DEPLETION_DISTANCE,
    );

    let min_residual_factor = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "min-residual-factor",
        "min_residual",
        &|min_residual: feb| min_residual,
        AnalyticalPropagatorConfig::DEFAULT_MIN_RESIDUAL_FACTOR,
    );
    let min_deposited_power_per_distance = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "min-deposited-power-per-distance",
        "min_dep_en",
        &|min_dep_en: feb| min_dep_en,
        AnalyticalPropagatorConfig::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
    );
    let max_propagation_distance = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "max-propagation-distance",
        "max_dist",
        &|max_dist: feb| max_dist,
        AnalyticalPropagatorConfig::DEFAULT_MAX_PROPAGATION_DISTANCE,
    );
    let outside_deposition_threshold = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "outside-deposition-threshold",
        "out_dep_thresh",
        &|out_dep_thresh: feb| out_dep_thresh,
        AnalyticalPropagatorConfig::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
    );
    let continue_depleted_beams = arguments.is_present("continue-depleted-beams");

    let max_col_depth_increase = utils::get_value_from_required_parseable_argument::<feb>(
        arguments,
        "max-column-depth-increase",
    );

    let max_substeps =
        utils::get_value_from_required_parseable_argument::<usize>(arguments, "max-substeps");

    let n_initial_steps_with_substeps = utils::get_value_from_required_parseable_argument::<usize>(
        arguments,
        "n-initial-steps-with-substeps",
    );

    let keep_initial_ionization_fraction = arguments.is_present("keep-initial-ionization-fraction");

    let config = AnalyticalPropagatorConfig {
        min_depletion_distance,
        min_residual_factor,
        min_deposited_power_per_distance,
        max_propagation_distance,
        outside_deposition_threshold,
        continue_depleted_beams,
        keep_initial_ionization_fraction,
        max_col_depth_increase,
        max_substeps,
        n_initial_steps_with_substeps,
        n_substeps: 1,
    };
    config.validate();
    config
}
