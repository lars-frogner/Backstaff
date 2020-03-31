//! Command line interface for resampling a snapshot to a regular grid.

use super::{
    direct_sampling::create_direct_sampling_subcommand,
    weighted_cell_averaging::create_weighted_cell_averaging_subcommand,
    weighted_sample_averaging::create_weighted_sample_averaging_subcommand,
};
use crate::{
    cli::{snapshot::write::create_write_subcommand, utils as cli_utils},
    create_subcommand,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D, Vec3,
    },
    grid::{regular::RegularGrid3, Grid3},
    interpolation::Interpolator3,
    io::snapshot::{fdt, SnapshotReader3},
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::sync::Arc;

/// Builds a representation of the `snapshot-resample-regular_grid` command line subcommand.
pub fn create_regular_grid_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("regular_grid")
        .about("Resample to a regular grid")
        .long_about("Resample to a regular grid of configurable shape and bounds.")
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
                   weighted sample averaging with the default prameters is used.",
        )
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .arg(
            Arg::with_name("shape")
                .short("s")
                .long("shape")
                .require_equals(true)
                .require_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the regular grid to resample to [default: same as original]")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("x-bounds")
                .short("x")
                .long("x-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the x-coordinates of the regular grid to resample to\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("y-bounds")
                .short("y")
                .long("y-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the y-coordinates of the regular grid to resample to\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("z-bounds")
                .short("z")
                .long("z-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the z-coordinates of the regular grid to resample to\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .subcommand(create_subcommand!(regular_grid, weighted_sample_averaging))
        .subcommand(create_subcommand!(regular_grid, weighted_cell_averaging))
        .subcommand(create_subcommand!(regular_grid, direct_sampling))
        .subcommand(create_subcommand!(regular_grid, write))
}

pub fn run_resampling_for_regular_grid<G, R, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    is_verbose: bool,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    I: Interpolator3,
{
    let original_grid = reader.grid();

    let original_shape = original_grid.shape();

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();

    let original_is_periodic = In3D::new(
        original_grid.is_periodic(X),
        original_grid.is_periodic(Y),
        original_grid.is_periodic(Z),
    );

    let write_arguments = arguments.subcommand_matches("write").unwrap();

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
    let mut new_grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        new_lower_bounds,
        new_upper_bounds,
        original_is_periodic,
    );
    super::correct_periodicity_for_new_grid(
        original_grid,
        &mut new_grid,
        continue_on_warnings,
        is_verbose,
    );
    let new_grid = Arc::new(new_grid);
    super::resample_snapshot_for_grid(
        write_arguments,
        reader,
        snap_num_offset,
        &new_grid,
        resampled_locations,
        resampling_method,
        is_verbose,
        interpolator,
        protected_file_types,
    );
}
