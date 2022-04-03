//! Command line interface for resampling a snapshot.

use super::SnapNumInRange;
use crate::{
    cli::{
        snapshot::write::{create_write_subcommand, run_write_subcommand},
        utils,
    },
    exit_with_error,
    geometry::{
        Dim3::{X, Y, Z},
        Idx3, Point3,
    },
    grid::Grid3,
    io::snapshot::{fdt, SnapshotProvider3},
};
use clap::{Arg, ArgMatches, Command};
use std::{collections::HashMap, sync::Arc};

/// Builds a representation of the `snapshot-extract` command line subcommand.
pub fn create_extract_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "extract";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Extract a subdomain of the snapshot")
        .long_about(
            "Extract a subdomain of the snapshot.\n\
             Creates new snapshot that is restricted to a specified subgrid of the\n\
             original one, with identical values. The subgrid can be specified\n\
             with coordinate bounds, index ranges or a combination of these.",
        )
        .subcommand_required(true)
        .arg(
            Arg::new("x-bounds")
                .short('x')
                .long("x-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Limits for the x-coordinates of the subgrid to extract")
                .takes_value(true)
                .default_value("-inf,inf"),
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
                .help("Limits for the y-coordinates of the subgrid to extract")
                .takes_value(true)
                .default_value("-inf,inf"),
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
                .help("Limits for the z-coordinates of the subgrid to extract")
                .takes_value(true)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("i-range")
                .short('i')
                .long("i-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["START", "END"])
                .help("Range of indices to extract along the x-axis (inclusive)\n")
                .takes_value(true)
                .default_value("0,max"),
        )
        .arg(
            Arg::new("j-range")
                .short('j')
                .long("j-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["START", "END"])
                .help("Range of indices to extract along the y-axis (inclusive)\n")
                .takes_value(true)
                .default_value("0,max"),
        )
        .arg(
            Arg::new("k-range")
                .short('k')
                .long("k-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["START", "END"])
                .help("Range of indices to extract along the z-axis (inclusive)\n")
                .takes_value(true)
                .default_value("0,max"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to extraction"),
        )
        .subcommand(create_write_subcommand(command_name))
}

/// Runs the actions for the `snapshot-extract` subcommand using the given arguments.
pub fn run_extract_subcommand<G, P>(
    arguments: &ArgMatches,
    provider: &P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    let original_grid = provider.grid();
    let original_shape = original_grid.shape();
    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();

    let x_bounds = utils::parse_limits(arguments, "x-bounds");
    let y_bounds = utils::parse_limits(arguments, "y-bounds");
    let z_bounds = utils::parse_limits(arguments, "z-bounds");

    let i_range = utils::parse_int_limits(arguments, "i-range", 0, original_shape[X] - 1);
    let j_range = utils::parse_int_limits(arguments, "j-range", 0, original_shape[Y] - 1);
    let k_range = utils::parse_int_limits(arguments, "k-range", 0, original_shape[Z] - 1);

    let lower_bounds = Point3::new(x_bounds.0, y_bounds.0, z_bounds.0);
    let upper_bounds = Point3::new(x_bounds.1, y_bounds.1, z_bounds.1);

    exit_on_false!(
        upper_bounds[X] > lower_bounds[X]
            && upper_bounds[Y] > lower_bounds[Y]
            && upper_bounds[Z] > lower_bounds[Z],
        "Error: Lower bounds ({}) must be smaller than upper bounds ({})",
        lower_bounds,
        upper_bounds
    );

    let lower_indices = Idx3::new(i_range.0, j_range.0, k_range.0);
    let upper_indices = Idx3::new(i_range.1, j_range.1, k_range.1);

    exit_on_false!(
        upper_indices[X] >= lower_indices[X]
            && upper_indices[Y] >= lower_indices[Y]
            && upper_indices[Z] >= lower_indices[Z],
        "Error: Lower indices ({}) are too large compared to upper indices ({}) to include any grid cells",
        lower_indices, upper_indices
    );

    let lower_indices_from_bounds = exit_on_none!(
        original_grid.find_fist_grid_cell_inside_lower_bounds(&lower_bounds),
        "Error: Lower coordinate bounds are too high to include any grid cells, upper snapshot bounds are {}",
        original_upper_bounds
    );
    let upper_indices_from_bounds = exit_on_none!(
        original_grid.find_last_grid_cell_inside_upper_bounds(&upper_bounds),
        "Error: Upper coordinate bounds are too low to include any grid cells, lower snapshot bounds are {}",
        original_lower_bounds
    );

    let lower_indices = lower_indices.max_with(&lower_indices_from_bounds);
    let upper_indices = upper_indices.min_with(&upper_indices_from_bounds);

    exit_on_false!(
        upper_indices[X] >= lower_indices[X]
            && upper_indices[Y] >= lower_indices[Y]
            && upper_indices[Z] >= lower_indices[Z],
        "Error: Final lower indices ({}) are too large compared to upper indices ({}) to include any grid cells",
        lower_indices, upper_indices
    );

    let is_verbose = arguments.is_present("verbose");

    let write_arguments = arguments.subcommand_matches("write").unwrap();

    if is_verbose {
        let new_lower_bounds = original_grid.grid_cell_lower_corner(&lower_indices);
        let new_upper_bounds = original_grid.grid_cell_upper_corner(&upper_indices);
        println!(
            "Extracting subgrid\n\
             Index ranges: {} -> {}\n\
             Shape: [{}, {}, {}]\n\
             Bounds: {} -> {}\n\
             Extents: {}",
            lower_indices,
            upper_indices,
            1 + upper_indices[X] - lower_indices[X],
            1 + upper_indices[Y] - lower_indices[Y],
            1 + upper_indices[Z] - lower_indices[Z],
            &new_lower_bounds,
            &new_upper_bounds,
            &new_upper_bounds - &new_lower_bounds
        );
    }
    let new_grid = Arc::new(original_grid.subgrid(&lower_indices, &upper_indices));

    run_write_subcommand(
        write_arguments,
        provider,
        snap_num_in_range,
        Some(Arc::clone(&new_grid)),
        HashMap::new(),
        |field| {
            if is_verbose {
                println!("Extracting {}", field.name());
            }
            Ok(field.subfield(Arc::clone(&new_grid), &lower_indices))
        },
        protected_file_types,
    );
}
