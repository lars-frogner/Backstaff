//! Command line interface for resampling a snapshot to a regular grid.

use super::{
    cell_averaging::create_cell_averaging_subcommand,
    direct_sampling::create_direct_sampling_subcommand,
    sample_averaging::create_sample_averaging_subcommand,
};
use crate::{
    add_subcommand_combinations,
    cli::{
        snapshot::{
            derive::create_derive_subcommand, inspect::create_inspect_subcommand,
            write::create_write_subcommand, SnapNumInRange,
        },
        utils as cli_utils,
    },
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D, Vec3,
    },
    grid::{fgr, regular::RegularGrid3, Grid3},
    interpolation::Interpolator3,
    io::{snapshot::SnapshotProvider3, Verbose},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-regular_grid` command line subcommand.
pub fn create_regular_grid_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "regular_grid";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Resample to a regular grid")
        .long_about("Resample to a regular grid of configurable shape and bounds.")
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
             sample averaging with the default prameters is used.",
        )
        .arg(
            Arg::new("scales")
                .short('S')
                .long("scales")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(false)
                .value_names(&["SX", "SY", "SZ"])
                .help(
                    "Scale factors for computing shape of the resampled grid based on original shape",
                )
                .takes_value(true)
                .number_of_values(3),
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the resampled grid (`auto` approximates original cell extent)")
                .takes_value(true)
                .number_of_values(3)
                .default_value("auto,auto,auto")
                .conflicts_with("scales"),
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
                .help("Limits for the x-coordinates of the resampled grid")
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
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
                .help("Limits for the y-coordinates of the resampled grid")
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
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
                .help("Limits for the z-coordinates of the resampled grid")
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
        )
        .subcommand(create_sample_averaging_subcommand(command_name))
        .subcommand(create_cell_averaging_subcommand(command_name))
        .subcommand(create_direct_sampling_subcommand(command_name));

    #[cfg(feature = "synthesis")]
    let command = add_subcommand_combinations!(command, command_name, true; derive, synthesize, (write, inspect));
    #[cfg(not(feature = "synthesis"))]
    let command =
        add_subcommand_combinations!(command, command_name, true; derive, (write, inspect));

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
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
    I: Interpolator3,
{
    let original_grid = provider.grid();

    let original_shape = original_grid.shape();

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();

    let original_average_grid_cell_extents = original_grid.average_grid_cell_extents();

    let x_bounds = cli_utils::parse_limits_with_min_max(
        root_arguments,
        "x-bounds",
        cli_utils::AllowSameValue::No,
        cli_utils::AllowInfinity::No,
        original_lower_bounds[X],
        original_upper_bounds[X],
    );
    let y_bounds = cli_utils::parse_limits_with_min_max(
        root_arguments,
        "y-bounds",
        cli_utils::AllowSameValue::No,
        cli_utils::AllowInfinity::No,
        original_lower_bounds[Y],
        original_upper_bounds[Y],
    );
    let z_bounds = cli_utils::parse_limits_with_min_max(
        root_arguments,
        "z-bounds",
        cli_utils::AllowSameValue::No,
        cli_utils::AllowInfinity::No,
        original_lower_bounds[Z],
        original_upper_bounds[Z],
    );
    let new_lower_bounds = Vec3::new(x_bounds.0, y_bounds.0, z_bounds.0);
    let new_upper_bounds = Vec3::new(x_bounds.1, y_bounds.1, z_bounds.1);
    let new_extents = &new_upper_bounds - &new_lower_bounds;

    let scales: Vec<fgr> =
        cli_utils::get_finite_float_values_from_parseable_argument_with_custom_defaults(
            root_arguments,
            "scales",
            &|| vec![1.0, 1.0, 1.0],
        );

    let shape = if scales.iter().all(|&scale| scale == 1.0) {
        cli_utils::parse_3d_values(root_arguments, "shape", Some(1), |dim, value_string| {
            match value_string {
                "same" => Some(original_shape[dim]),
                "auto" => Some(
                    fgr::ceil(new_extents[dim] / original_average_grid_cell_extents[dim]) as usize,
                ),
                _ => None,
            }
        })
    } else {
        super::compute_scaled_grid_shape(original_shape, &scales)
    };

    let grid = RegularGrid3::from_bounds(
        shape,
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
