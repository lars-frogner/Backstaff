//! Command line interface for resampling a snapshot to a mesh file grid.

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
    exit_on_error, exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, native, SnapshotProvider3},
        Verbose,
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command, ValueHint};
use std::{path::PathBuf, str::FromStr};

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-mesh_file` command line subcommand.
pub fn create_mesh_file_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "mesh_file";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Resample to a grid specified by a mesh file")
        .long_about("Resample to a grid specified by a mesh file.")
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
                   weighted sample averaging with the default prameters is used.",
        )
        .arg(
            Arg::new("mesh-file")
                .value_name("MESH_FILE")
                .help("Path to a Bifrost mesh file representing the grid to resample to")
                .required(true)
                .takes_value(true)
                .value_hint(ValueHint::FilePath),
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the grid to resample to [default: same as in mesh file]")
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

pub fn run_resampling_for_mesh_file<G, P, I>(
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
    let mesh_file_path = exit_on_error!(
        PathBuf::from_str(
            root_arguments
                .value_of("mesh-file")
                .expect("No value for required argument")
        ),
        "Error: Could not interpret path to mesh file: {}"
    );
    let original_shape = provider.grid().shape();
    let shape: Vec<usize> = utils::get_values_from_parseable_argument_with_custom_defaults(
        root_arguments,
        "shape",
        &|| vec![original_shape[X], original_shape[Y], original_shape[Z]],
        Some(3),
    );
    let new_shape = if shape[0] == original_shape[X]
        && shape[1] == original_shape[Y]
        && shape[2] == original_shape[Z]
    {
        None
    } else {
        exit_on_false!(
            shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
            "Error: Grid size must be larger than zero in every dimension"
        );
        Some(In3D::new(shape[0], shape[1], shape[2]))
    };

    let (detected_grid_type, center_coords, lower_edge_coords, up_derivatives, down_derivatives) = exit_on_error!(
        native::parse_mesh_file(mesh_file_path, verbose),
        "Error: Could not parse mesh file: {}"
    );
    match detected_grid_type {
        GridType::Regular => {
            let grid = RegularGrid3::from_coords(
                center_coords,
                lower_edge_coords,
                provider.grid().periodicity().clone(),
                Some(up_derivatives),
                Some(down_derivatives),
            );
            super::resample_to_regular_grid(
                grid,
                new_shape,
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
        GridType::HorRegular => {
            let grid = HorRegularGrid3::from_coords(
                center_coords,
                lower_edge_coords,
                provider.grid().periodicity().clone(),
                Some(up_derivatives),
                Some(down_derivatives),
            );
            super::resample_to_horizontally_regular_grid(
                grid,
                new_shape,
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
    }
}
