//! Command line interface for resampling a snapshot to a reshaped
//! version of the original grid.

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
        utils,
    },
    exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::In3D,
    grid::Grid3,
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, SnapshotProvider3},
        Verbose,
    },
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-resample-reshaped_grid` command line subcommand.
pub fn create_reshaped_grid_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "reshaped_grid";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Resample to a reshaped version of the original grid")
        .long_about("Resample to a reshaped version of the original grid.")
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
                .help("Shape of the grid to resample to")
                .takes_value(true)
                .required(true),
        )
        .subcommand(create_weighted_sample_averaging_subcommand(command_name))
        .subcommand(create_weighted_cell_averaging_subcommand(command_name))
        .subcommand(create_direct_sampling_subcommand(command_name));

    add_subcommand_combinations!(command, command_name, true; derive, write)
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
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
    I: Interpolator3,
{
    let shape: Vec<usize> =
        utils::get_values_from_required_parseable_argument(root_arguments, "shape");

    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );
    let new_shape = Some(In3D::new(shape[0], shape[1], shape[2]));

    super::resample_to_same_or_reshaped_grid(
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
