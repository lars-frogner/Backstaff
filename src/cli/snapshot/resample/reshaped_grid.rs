//! Command line interface for resampling a snapshot to a reshaped
//! version of the original grid.

use super::{
    direct_sampling::create_direct_sampling_subcommand,
    weighted_cell_averaging::create_weighted_cell_averaging_subcommand,
    weighted_sample_averaging::create_weighted_sample_averaging_subcommand,
};
use crate::{
    cli::{
        snapshot::{write::create_write_subcommand, SnapNumInRange},
        utils,
    },
    create_subcommand, exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::In3D,
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    interpolation::Interpolator3,
    io::snapshot::{fdt, SnapshotReader3},
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-resample-reshaped_grid` command line subcommand.
pub fn create_reshaped_grid_subcommand() -> Command<'static> {
    Command::new("reshaped_grid")
        .about("Resample to a reshaped version of the original grid")
        .long_about("Resample to a reshaped version of the original grid.")
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
                   weighted sample averaging with the default prameters is used.",
        )
        .subcommand_required(true)
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
        .subcommand(create_subcommand!(reshaped_grid, weighted_sample_averaging))
        .subcommand(create_subcommand!(reshaped_grid, weighted_cell_averaging))
        .subcommand(create_subcommand!(reshaped_grid, direct_sampling))
        .subcommand(create_subcommand!(reshaped_grid, write))
}

pub fn run_resampling_for_reshaped_grid<G, R, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    reader: &R,
    snap_num_in_range: &Option<SnapNumInRange>,
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
    let write_arguments = arguments.subcommand_matches("write").unwrap();

    let shape: Vec<usize> =
        utils::get_values_from_required_parseable_argument(root_arguments, "shape");

    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );
    let new_shape = Some(In3D::new(shape[0], shape[1], shape[2]));

    let grid = reader.grid();

    match grid.grid_type() {
        GridType::Regular => {
            let grid = RegularGrid3::from_coords(
                grid.centers().clone(),
                grid.lower_edges().clone(),
                grid.periodicity().clone(),
                grid.up_derivatives().cloned(),
                grid.down_derivatives().cloned(),
            );
            super::resample_to_regular_grid(
                grid,
                new_shape,
                write_arguments,
                reader,
                snap_num_in_range,
                resampled_locations,
                resampling_method,
                continue_on_warnings,
                is_verbose,
                interpolator,
                protected_file_types,
            );
        }
        GridType::HorRegular => {
            let grid = HorRegularGrid3::from_coords(
                grid.centers().clone(),
                grid.lower_edges().clone(),
                grid.periodicity().clone(),
                grid.up_derivatives().cloned(),
                grid.down_derivatives().cloned(),
            );
            super::resample_to_horizontally_regular_grid(
                grid,
                new_shape,
                write_arguments,
                reader,
                snap_num_in_range,
                resampled_locations,
                resampling_method,
                continue_on_warnings,
                is_verbose,
                interpolator,
                protected_file_types,
            );
        }
    }
}
