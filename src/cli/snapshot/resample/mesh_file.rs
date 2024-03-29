//! Command line interface for resampling a snapshot to a mesh file grid.

use super::{
    cell_averaging::create_cell_averaging_subcommand,
    direct_sampling::create_direct_sampling_subcommand,
    sample_averaging::create_sample_averaging_subcommand,
};
use crate::{
    add_subcommand_combinations,
    cli::{
        snapshot::{inspect::create_inspect_subcommand, write::create_write_subcommand},
        utils,
    },
    exit_on_error,
    field::{DynScalarFieldProvider3, ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::Grid3,
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, native, SnapshotMetadata},
        utils::IOContext,
        Verbosity,
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command, ValueHint};
use std::{path::PathBuf, str::FromStr};

#[cfg(feature = "derivation")]
use crate::cli::snapshot::derive::create_derive_subcommand;

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
             sample averaging with the default parameters is used.",
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
                .help("Shape of the resampled grid (`file` is size from mesh file)")
                .takes_value(true)
                .number_of_values(3)
                .default_value("file,file,file"),
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

pub fn run_resampling_for_mesh_file(
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
    let mesh_file_path = exit_on_error!(
        PathBuf::from_str(
            root_arguments
                .value_of("mesh-file")
                .expect("No value for required argument")
        ),
        "Error: Could not interpret path to mesh file: {}"
    );

    let mesh_file_grid = exit_on_error!(
        native::create_grid_from_mesh_file(
            &mesh_file_path,
            provider.grid().periodicity().clone(),
            &verbosity
        ),
        "Error: Could not parse mesh file: {}"
    );

    let mesh_file_shape = mesh_file_grid.shape();
    let original_shape = provider.grid().shape();

    let unscaled_shape =
        utils::parse_3d_values(root_arguments, "shape", Some(1), |dim, value_string| {
            match value_string {
                "file" => Some(mesh_file_shape[dim]),
                "same" => Some(original_shape[dim]),
                _ => None,
            }
        });

    let scales = utils::parse_3d_float_values(
        root_arguments,
        "scales",
        utils::AllowInfinity::No,
        utils::AllowZero::No,
    );

    let shape = super::compute_scaled_grid_shape(&unscaled_shape, &scales);

    let new_shape = if shape[X] == mesh_file_shape[X]
        && shape[Y] == mesh_file_shape[Y]
        && shape[Z] == mesh_file_shape[Z]
    {
        None
    } else {
        Some(shape)
    };

    super::resample_to_grid(
        mesh_file_grid,
        new_shape,
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
