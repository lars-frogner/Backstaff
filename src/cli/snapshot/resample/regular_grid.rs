//! Command line interface for resampling a snapshot to a regular grid.

use super::{
    cell_averaging::create_cell_averaging_subcommand,
    direct_sampling::create_direct_sampling_subcommand,
    sample_averaging::create_sample_averaging_subcommand,
};
use crate::{
    add_subcommand_combinations,
    cli::{
        snapshot::{inspect::create_inspect_subcommand, write::create_write_subcommand},
        utils as cli_utils,
    },
    field::{DynScalarFieldProvider3, FieldGrid3, ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D, Vec3,
    },
    grid::{fgr, regular::RegularGrid3, Grid3},
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, SnapshotMetadata},
        utils::IOContext,
        Verbosity,
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

#[cfg(feature = "derivation")]
use crate::cli::snapshot::derive::create_derive_subcommand;

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
                .default_value("auto,auto,auto"),
        )
        .arg(
            Arg::new("scales")
                .short('c')
                .long("scales")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(false)
                .value_names(&["SX", "SY", "SZ"])
                .help("Factors for scaling the dimensions specified in the `shape` argument")
                .takes_value(true)
                .number_of_values(3)
                .default_value("1,1,1"),
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

    add_subcommand_combinations!(
        command, command_name, true;
        derive if "derivation",
        synthesize if "synthesis",
        (write, inspect)
    )
}

pub fn run_resampling_for_regular_grid(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    metadata: &dyn SnapshotMetadata,
    provider: DynScalarFieldProvider3<fdt>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    verbosity: Verbosity,
    interpolator: Box<dyn Interpolator3<fdt>>,
    io_context: &mut IOContext,
) {
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

    let unscaled_shape =
        cli_utils::parse_3d_values(root_arguments, "shape", Some(1), |dim, value_string| {
            match value_string {
                "same" => Some(original_shape[dim]),
                "auto" => Some(
                    fgr::ceil(new_extents[dim] / original_average_grid_cell_extents[dim]) as usize,
                ),
                _ => None,
            }
        });

    let scales = cli_utils::parse_3d_float_values(
        root_arguments,
        "scales",
        cli_utils::AllowInfinity::No,
        cli_utils::AllowZero::No,
    );

    let shape = super::compute_scaled_grid_shape(&unscaled_shape, &scales);

    let grid: FieldGrid3 = RegularGrid3::from_bounds(
        shape,
        new_lower_bounds,
        new_upper_bounds,
        original_grid.periodicity().clone(),
    )
    .into();
    super::resample_to_grid(
        grid,
        None,
        arguments,
        metadata,
        provider,
        resampled_locations,
        resampling_method,
        continue_on_warnings,
        verbosity,
        interpolator,
        io_context,
    );
}
