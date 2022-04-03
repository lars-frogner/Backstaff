//! Command line interface for the DC power-law distribution accelerator.

use crate::{
    cli,
    ebeam::distribution::power_law::acceleration::dc::acceleration_region::{
        AccelerationRegionTracer, AccelerationRegionTracerConfig,
    },
    ebeam::distribution::power_law::acceleration::dc::DCPowerLawAccelerationConfig,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotReader3},
    units::solar::{U_E, U_T},
};
use clap::{App, Arg, ArgMatches};

/// Creates a subcommand for using the DC power-law distribution accelerator.
pub fn create_dc_power_law_accelerator_subcommand(
    parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "dc_power_law_accelerator";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the direct current power-law distribution accelerator model")
        .long_about(
            "Use the direct current power-law distribution accelerator model.\n\
             The total distribution energy is assumed to be a fixed fraction of the\n\
             reconnection energy, and the lower cut-off energy is computed from the\n\
             acceleration region length and average electric field strength parallel\n\
             to the magnetic field direction.",
        )

        .arg(
            Arg::new("acceleration-duration")
                .long("acceleration-duration")
                .require_equals(true)
                .value_name("VALUE")
                .help("Duration of the acceleration events [s] [default: from param file]")
                .takes_value(true),
        )
        .arg(
            Arg::new("particle-energy-fraction")
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
            Arg::new("power-law-delta")
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
            Arg::new("min-total-power-density")
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
            Arg::new("min-parallel-electric-field-strength")
                .long("min-parallel-electric-field-strength")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "The acceleration region ends where the absolute component of the electric field\n\
                     parallel to the magnetic field direction becomes lower than this value [V/m]\n",
                )
                .takes_value(true)
                .default_value("1.0"),
        )
        .arg(
            Arg::new("min-length")
                .long("min-length")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Acceleration regions shorter than this are discarded [Mm]",
                )
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::new("extra-varying-scalars")
                .long("extra-varying-scalars")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of scalar fields to extract along acceleration regions\n \
                     (comma-separated)",
                )
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(
            Arg::new("extra-varying-vectors")
                .long("extra-varying-vectors")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of vector fields to extract along acceleration regions\n \
                     (comma-separated)",
                )
                .takes_value(true)
                .multiple_values(true),
        )
}

/// Determines DC power-law distribution accelerator parameters
/// based on provided options and values in parameter file.
pub fn construct_dc_power_law_accelerator_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &R,
) -> (DCPowerLawAccelerationConfig, AccelerationRegionTracer) {
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

    let min_total_power_density = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-total-power-density",
        "min_beam_en",
        &|min_beam_en| min_beam_en * U_E / U_T,
        DCPowerLawAccelerationConfig::DEFAULT_MIN_TOTAL_POWER_DENSITY,
    );

    let min_depletion_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-depletion-distance",
        "min_stop_dist",
        &|min_stop_dist| min_stop_dist,
        DCPowerLawAccelerationConfig::DEFAULT_MIN_DEPLETION_DISTANCE,
    );

    let min_parallel_electric_field_strength = cli::get_value_from_required_parseable_argument(
        arguments,
        "min-parallel-electric-field-strength",
    );
    let min_length = cli::get_value_from_required_parseable_argument(arguments, "min-length");

    let extra_varying_scalar_names = arguments
        .values_of("extra-varying-scalars")
        .map_or_else(Vec::new, |values| {
            values.map(|name| name.to_string()).collect::<Vec<_>>()
        });

    let extra_varying_vector_names = arguments
        .values_of("extra-varying-vectors")
        .map_or_else(Vec::new, |values| {
            values.map(|name| name.to_string()).collect::<Vec<_>>()
        });

    (
        DCPowerLawAccelerationConfig {
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            min_total_power_density,
            min_depletion_distance,
        },
        AccelerationRegionTracer::new(
            AccelerationRegionTracerConfig {
                min_parallel_electric_field_strength,
                min_length,
            },
            extra_varying_scalar_names,
            extra_varying_vector_names,
        ),
    )
}
