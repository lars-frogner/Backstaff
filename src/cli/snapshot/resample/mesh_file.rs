//! Command line interface for resampling a snapshot to a mesh file grid.

use super::{
    direct_sampling::create_direct_sampling_subcommand,
    weighted_cell_averaging::create_weighted_cell_averaging_subcommand,
    weighted_sample_averaging::create_weighted_sample_averaging_subcommand,
};
use crate::{
    cli::snapshot::write::create_write_subcommand,
    create_subcommand, exit_on_error, exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    interpolation::Interpolator3,
    io::snapshot::{fdt, native, SnapshotReader3},
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::{path::PathBuf, str::FromStr, sync::Arc};

/// Builds a representation of the `snapshot-resample-mesh_file` command line subcommand.
pub fn create_mesh_file_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("mesh_file")
        .about("Resample to a grid specified by a mesh file")
        .long_about("Resample to a grid specified by a mesh file.")
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
                   weighted sample averaging with the default prameters is used.",
        )
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .arg(
            Arg::with_name("mesh-file")
                .value_name("MESH_FILE")
                .help("Path to a Bifrost mesh file representing the grid to resample to")
                .required(true)
                .takes_value(true),
        )
        .subcommand(create_subcommand!(mesh_file, weighted_sample_averaging))
        .subcommand(create_subcommand!(mesh_file, weighted_cell_averaging))
        .subcommand(create_subcommand!(mesh_file, direct_sampling))
        .subcommand(create_subcommand!(mesh_file, write))
}

pub fn run_resampling_for_mesh_file<G, R, I>(
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

    let original_is_periodic = In3D::new(
        original_grid.is_periodic(X),
        original_grid.is_periodic(Y),
        original_grid.is_periodic(Z),
    );

    let mesh_file_path = exit_on_error!(
        PathBuf::from_str(
            root_arguments
                .value_of("mesh-file")
                .expect("No value for required argument.")
        ),
        "Error: Could not interpret path to mesh file: {}"
    );

    let write_arguments = arguments.subcommand_matches("write").unwrap();

    let (detected_grid_type, center_coords, lower_edge_coords, up_derivatives, down_derivatives) = exit_on_error!(
        native::parse_mesh_file(mesh_file_path, is_verbose.into()),
        "Error: Could not parse mesh file: {}"
    );
    match detected_grid_type {
        GridType::Regular => {
            let mut new_grid = RegularGrid3::from_coords(
                center_coords,
                lower_edge_coords,
                original_is_periodic,
                Some(up_derivatives),
                Some(down_derivatives),
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
        GridType::HorRegular => {
            let mut new_grid = HorRegularGrid3::from_coords(
                center_coords,
                lower_edge_coords,
                original_is_periodic,
                Some(up_derivatives),
                Some(down_derivatives),
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
    }
}
