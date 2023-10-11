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
        propagation::fp_characteristics::{CharacteristicsPropagatorConfig, DetailedOutputConfig},
    },
    exit_on_error,
    io::{snapshot::SnapshotParameters, utils::IOContext},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use std::{path::PathBuf, str::FromStr};

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
                .default_value("40"),
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
                .default_value("0.05,120.0"),
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
        )
        .arg(
            Arg::new("assume-ambient-electrons-all-from-hydrogen")
                .long("assume-ambient-electrons-all-from-hydrogen")
                .help(
                    "Assume that the electron-to-hydrogen ratio is the same as the hydrogen\n\
                     ionization factor",
                ),
        )
        .arg(
            Arg::new("disable-helium-collisions")
                .long("disable-helium-collisions")
                .help(
                    "Do not account for collisions with ambient helium when propagating a distribution",
                ),
        )
        .arg(
            Arg::new("disable-ambient-electric-field")
                .long("disable-ambient-electric-field")
                .help(
                    "Do not account for the ambient electric field when propagating a distribution",
                ),
        )
        .arg(
            Arg::new("disable-return-current")
                .long("disable-return-current")
                .help(
                    "Do not account for return current when propagating a distribution",
                ),
        )
        .arg(
            Arg::new("disable-magnetic-mirroring")
                .long("disable-magnetic-mirroring")
                .help(
                    "Do not account for magnetic mirroring when propagating a distribution",
                ),
        )
        .arg(
            Arg::new("enable-warm-target")
                .long("enable-warm-target")
                .help(
                    "Do not assume that the ambient plasma can be treated as a cold target",
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
            Arg::new("min-remaining-flux-fraction")
                .long("min-remaining-flux-fraction")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions are considered depleted when their total flux has\n\
                     decreased by this fraction relative to their initial flux, given that\n\
                     the deposited power per distance is smaller than its lower limit",
                )
                .takes_value(true)
                .default_value("1e-5"),
        )
        .arg(
            Arg::new("min-residual-factor")
                .long("min-residual-factor")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "See `analytical_propagator`. Here, the factor is only used when\n\
                     estimating depletion distance [default: from param file]",
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
        )
        .arg(
            Arg::new("detailed-output-dir")
                .long("detailed-output-dir")
                .require_equals(true)
                .value_name("DIRECTORY")
                .help("Path to a directory in which to write the full distribution data\n\
                       for every beam. The data will only be written if this argument is\n\
                       provided")
                .takes_value(true)
        )
        .arg(
            Arg::new("overwrite-detailed-output")
                .long("overwrite-detailed-output")
                .help("Automatically overwrite any existing files (unless listed as protected)")
        );

    add_subcommand_combinations!(command, command_name, false; poly_fit_interpolator, rkf_stepper)
}

/// Determines characteristics propagator parameters based on
/// provided options and values in parameter file.
pub fn construct_characteristics_propagator_config_from_options(
    arguments: &ArgMatches,
    parameters: &dyn SnapshotParameters,
    io_context: &mut IOContext,
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
    let assume_ambient_electrons_all_from_hydrogen =
        arguments.is_present("assume-ambient-electrons-all-from-hydrogen");
    let include_helium_collisions = !arguments.is_present("disable-helium-collisions");
    let include_ambient_electric_field = !arguments.is_present("disable-ambient-electric-field");
    let include_return_current = !arguments.is_present("disable-return-current");
    let include_magnetic_mirroring = !arguments.is_present("disable-magnetic-mirroring");
    let enable_warm_target = arguments.is_present("enable-warm-target");

    let min_remaining_flux_fraction = utils::get_value_from_required_parseable_argument::<feb>(
        arguments,
        "min-remaining-flux-fraction",
    );

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

    let detailed_output_config = if arguments.is_present("detailed-output-dir") {
        let detailed_output_dir = exit_on_error!(
            PathBuf::from_str(arguments.value_of("detailed-output-dir").unwrap()),
            "Error: Could not interpret path of detailed output directory: {}"
        );

        let overwrite_detailed_output = arguments.is_present("overwrite-detailed-output");

        if !overwrite_detailed_output {
            let dir_has_content = detailed_output_dir
                .read_dir()
                .map(|mut rd| rd.next().is_some())
                .unwrap_or(false);
            if dir_has_content {
                eprintln!(
                    "Warning: Detailed output directory {} not empty",
                    detailed_output_dir.to_string_lossy()
                );
                utils::verify_user_will_continue_or_abort();
            }
        }

        let atomic_output_file_map = io_context.obtain_atomic_file_map_handle();

        Some(DetailedOutputConfig {
            detailed_output_dir,
            atomic_output_file_map,
        })
    } else {
        None
    };

    let config = CharacteristicsPropagatorConfig {
        n_energies,
        min_energy_relative_to_cutoff,
        max_energy_relative_to_cutoff,
        max_col_depth_increase,
        max_substeps,
        n_initial_steps_with_substeps,
        n_substeps: 1,
        keep_initial_ionization_fraction,
        assume_ambient_electrons_all_from_hydrogen,
        include_helium_collisions,
        include_ambient_electric_field,
        include_return_current,
        include_magnetic_mirroring,
        enable_warm_target,
        min_depletion_distance,
        min_remaining_flux_fraction,
        min_residual_factor,
        min_deposited_power_per_distance,
        max_propagation_distance,
        continue_depleted_beams,
        detailed_output_config,
        ..Default::default()
    };
    config.validate();
    config
}
