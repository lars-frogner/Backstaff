//! Command line interface for resampling a snapshot to a regular grid.

use super::{
    direct_sampling::create_direct_sampling_subcommand,
    weighted_cell_averaging::create_weighted_cell_averaging_subcommand,
    weighted_sample_averaging::create_weighted_sample_averaging_subcommand,
};
use crate::{
    add_subcommand_combinations,
    cli::{
        snapshot::{
            derive::create_derive_subcommand, write::create_write_subcommand, SnapNumInRange,
        },
        utils as cli_utils,
    },
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D, Vec3,
    },
    grid::{regular::RegularGrid3, Grid3},
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, SnapshotProvider3},
        Verbose,
    },
};
use clap::{Arg, ArgMatches, Command};

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-regular_grid` command line subcommand.
pub fn create_regular_grid_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "regular_grid";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Resample to a regular grid")
        .long_about("Resample to a regular grid of configurable shape and bounds.")
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
                   weighted sample averaging with the default prameters is used.",
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the regular grid to resample to [default: same as original]")
                .takes_value(true),
        )
        .arg(
            Arg::new("x-bounds")
                .short('x')
                .long("x-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the x-coordinates of the regular grid to resample to\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("y-bounds")
                .short('y')
                .long("y-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the y-coordinates of the regular grid to resample to\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("z-bounds")
                .short('z')
                .long("z-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the z-coordinates of the regular grid to resample to\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .subcommand(create_weighted_sample_averaging_subcommand(command_name))
        .subcommand(create_weighted_cell_averaging_subcommand(command_name))
        .subcommand(create_direct_sampling_subcommand(command_name));

    #[cfg(feature = "synthesis")]
    let command =
        add_subcommand_combinations!(command, command_name, true; derive, synthesize, write);
    #[cfg(not(feature = "synthesis"))]
    let command = add_subcommand_combinations!(command, command_name, true; derive, write);

    command
}

pub fn run_resampling_for_regular_grid<G, P, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    verbose: Verbose,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
    I: Interpolator3,
{
    let original_grid = provider.grid();

    let original_shape = original_grid.shape();

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();

    let shape = cli_utils::get_values_from_parseable_argument_with_custom_defaults(
        root_arguments,
        "shape",
        &|| vec![original_shape[X], original_shape[Y], original_shape[Z]],
    );
    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );
    let x_bounds = cli_utils::get_values_from_parseable_argument_with_custom_defaults(
        root_arguments,
        "x-bounds",
        &|| vec![original_lower_bounds[X], original_upper_bounds[X]],
    );
    exit_on_false!(
        x_bounds[1] > x_bounds[0],
        "Error: Upper bound for x must exceed lower bound"
    );
    let y_bounds = cli_utils::get_values_from_parseable_argument_with_custom_defaults(
        root_arguments,
        "y-bounds",
        &|| vec![original_lower_bounds[Y], original_upper_bounds[Y]],
    );
    exit_on_false!(
        y_bounds[1] > y_bounds[0],
        "Error: Upper bound for y must exceed lower bound"
    );
    let z_bounds = cli_utils::get_values_from_parseable_argument_with_custom_defaults(
        root_arguments,
        "z-bounds",
        &|| vec![original_lower_bounds[Z], original_upper_bounds[Z]],
    );
    exit_on_false!(
        z_bounds[1] > z_bounds[0],
        "Error: Upper bound for z must exceed lower bound"
    );
    let new_lower_bounds = Vec3::new(x_bounds[0], y_bounds[0], z_bounds[0]);
    let new_upper_bounds = Vec3::new(x_bounds[1], y_bounds[1], z_bounds[1]);

    let grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        new_lower_bounds,
        new_upper_bounds,
        original_grid.periodicity().clone(),
    );
    super::resample_to_regular_grid(
        grid,
        None,
        arguments,
        provider,
        snap_num_in_range,
        resampled_locations,
        resampling_method,
        continue_on_warnings,
        verbose,
        interpolator,
        protected_file_types,
    );
}
