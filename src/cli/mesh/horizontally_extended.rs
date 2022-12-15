//! Command line interface for extending the horizontal dimensions
//! of Bifrost mesh files.

use crate::{
    cli::utils,
    exit_on_error,
    geometry::{
        Coords3, Dim2,
        Dim3::{X, Y, Z},
        In3D, Vec3,
    },
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    io::{snapshot::native, utils::IOContext, Verbosity},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command, ValueHint};
use std::{path::PathBuf, str::FromStr};

/// Builds a representation of the `create_mesh-horizontally_extended` command line subcommand.
pub fn create_horizontally_extended_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "horizontally_extended";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Extend the horizontal dimensions of a mesh file")
        .long_about(
            "Extend the horizontal dimensions of a Bifrost mesh file.\n\
             Specify the number of grid cells to add in the x- and y-directions\n\
             for a given mesh file. The original mesh file will not be modified.",
        )
        .arg(
            Arg::new("mesh-file")
                .value_name("MESH_FILE")
                .help("Path to a Bifrost mesh file to extend")
                .required(true)
                .takes_value(true)
                .value_hint(ValueHint::FilePath),
        )
        .arg(
            Arg::new("added-grid-cells")
                .short('n')
                .long("added-grid-cells")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY"])
                .help("Number of grid cells to add in the x- and y-direction")
                .takes_value(true)
                .number_of_values(2)
                .required(true),
        )
}

/// Runs the actions for the `create_mesh-horizontally_extended` subcommand using the given arguments.
pub fn run_horizontally_extended_subcommand(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    io_context: &mut IOContext,
) {
    let mesh_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("mesh-file")
                .expect("No value for required argument")
        ),
        "Error: Could not interpret path to mesh file: {}"
    );

    let periodicity = In3D::same(false);
    let mesh_file_grid = exit_on_error!(
        native::create_grid_from_mesh_file(&mesh_file_path, periodicity.clone(), &Verbosity::Quiet),
        "Error: Could not parse mesh file: {}"
    );

    let added_grid_cells =
        utils::parse_2d_values_no_special(arguments, "added-grid-cells", Some(0_usize));

    let original_shape = mesh_file_grid.shape();
    let original_upper_bounds = mesh_file_grid.upper_bounds();
    let grid_cell_extents = mesh_file_grid.average_grid_cell_extents();
    let original_centers = mesh_file_grid.centers();
    let original_lower_edges = mesh_file_grid.lower_edges();
    let original_up_derivatives = mesh_file_grid.up_derivatives().unwrap();
    let original_down_derivatives = mesh_file_grid.down_derivatives().unwrap();

    let new_shape = In3D::new(
        original_shape[X] + added_grid_cells[Dim2::X],
        original_shape[Y] + added_grid_cells[Dim2::Y],
        original_shape[Z],
    );
    let new_upper_bounds = Vec3::new(
        original_upper_bounds[X] + added_grid_cells[Dim2::X] as f64 * grid_cell_extents[X],
        original_upper_bounds[Y] + added_grid_cells[Dim2::Y] as f64 * grid_cell_extents[Y],
        original_upper_bounds[Z],
    );

    let extended_grid = RegularGrid3::from_bounds(
        new_shape,
        mesh_file_grid.lower_bounds().clone(),
        new_upper_bounds,
        periodicity.clone(),
    );
    let extended_centers = extended_grid.centers();
    let extended_lower_edges = extended_grid.lower_edges();
    let extended_up_derivatives = extended_grid.up_derivatives().unwrap();
    let extended_down_derivatives = extended_grid.down_derivatives().unwrap();

    let new_grid = HorRegularGrid3::from_coords_unchecked(
        Coords3::new(
            extended_centers[X].clone(),
            extended_centers[Y].clone(),
            original_centers[Z].clone(),
        ),
        Coords3::new(
            extended_lower_edges[X].clone(),
            extended_lower_edges[Y].clone(),
            original_lower_edges[Z].clone(),
        ),
        periodicity.clone(),
        Some(Coords3::new(
            extended_up_derivatives[X].clone(),
            extended_up_derivatives[Y].clone(),
            original_up_derivatives[Z].clone(),
        )),
        Some(Coords3::new(
            extended_down_derivatives[X].clone(),
            extended_down_derivatives[Y].clone(),
            original_down_derivatives[Z].clone(),
        )),
        GridType::HorRegular,
    );

    super::write_mesh_file(root_arguments, new_grid, io_context);
}
