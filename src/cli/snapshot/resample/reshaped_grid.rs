//! Command line interface for resampling a snapshot to a reshaped
//! version of the original grid.

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
        utils,
    },
    exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{fgr, Grid3},
    interpolation::Interpolator3,
    io::{snapshot::SnapshotProvider3, Verbose},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-reshaped_grid` command line subcommand.
pub fn create_reshaped_grid_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "reshaped_grid";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Resample to a reshaped version of the original grid")
        .long_about("Resample to a reshaped version of the original grid.")
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
                .help("Scale factors for computing shape of the new grid based on original shape")
                .takes_value(true)
                .default_value("1,1,1"),
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the grid to resample to [default: same as original]")
                .takes_value(true)
                .conflicts_with("scales"),
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

pub fn run_resampling_for_reshaped_grid<G, P, I>(
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
    let original_shape = provider.grid().shape();

    let scales: Vec<fgr> =
        utils::get_values_from_required_parseable_argument(root_arguments, "scales", Some(3));

    let shape: Vec<usize> = if scales.iter().all(|&scale| scale == 1.0) {
        utils::get_values_from_parseable_argument_with_custom_defaults(
            root_arguments,
            "shape",
            &|| vec![original_shape[X], original_shape[Y], original_shape[Z]],
            Some(3),
        )
    } else {
        super::compute_scaled_grid_shape(original_shape, &scales)
    };

    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );
    let new_shape = Some(In3D::new(shape[0], shape[1], shape[2]));

    super::resample_to_reshaped_grid(
        arguments,
        new_shape,
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
