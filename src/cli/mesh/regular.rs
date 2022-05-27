//! Command line interface for creating Bifrost mesh files for regular grids.

use crate::{
    cli::utils,
    exit_on_false,
    geometry::{In3D, Vec3},
    grid::regular::RegularGrid3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `create_mesh-regular` command line subcommand.
pub fn create_regular_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "regular";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Create a regular mesh")
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the mesh")
                .takes_value(true)
                .required(true),
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
                .help("Lower and upper bound for the x-coordinates")
                .takes_value(true)
                .required(true),
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
                .help("Lower and upper bound for the y-coordinates")
                .takes_value(true)
                .required(true),
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
                .help("Lower and upper bound for the z-coordinates")
                .takes_value(true)
                .required(true),
        )
}

/// Runs the actions for the `create_mesh-regular` subcommand using the given arguments.
pub fn run_regular_subcommand(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    protected_file_types: &[&str],
) {
    let shape = utils::get_values_from_required_parseable_argument(arguments, "shape", Some(3));
    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );

    let x_bounds = utils::parse_limits(
        arguments,
        "x-bounds",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );
    let y_bounds = utils::parse_limits(
        arguments,
        "y-bounds",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );
    let z_bounds = utils::parse_limits(
        arguments,
        "z-bounds",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );

    let grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        Vec3::new(x_bounds.0, y_bounds.0, z_bounds.0),
        Vec3::new(x_bounds.1, y_bounds.1, z_bounds.1),
        In3D::same(false),
    );

    super::write_mesh_file(root_arguments, grid, protected_file_types);
}
