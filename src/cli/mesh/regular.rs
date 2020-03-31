//! Command line interface for creating Bifrost mesh files for regular grids.

use crate::{
    cli::utils,
    exit_on_false,
    geometry::{In3D, Vec3},
    grid::regular::RegularGrid3,
};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `create_mesh-regular` command line subcommand.
pub fn create_regular_mesh_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("regular")
        .about("Use a regular grid")
        .help_message("Print help information")
        .arg(
            Arg::with_name("shape")
                .short("s")
                .long("shape")
                .require_equals(true)
                .require_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the grid")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("x-bounds")
                .short("x")
                .long("x-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the x-coordinates")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("y-bounds")
                .short("y")
                .long("y-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the y-coordinates")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("z-bounds")
                .short("z")
                .long("z-bounds")
                .require_equals(true)
                .require_delimiter(true)
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
    let shape = utils::get_values_from_required_parseable_argument(arguments, "shape");

    let x_bounds = utils::get_values_from_required_parseable_argument(arguments, "x-bounds");
    exit_on_false!(
        x_bounds[1] > x_bounds[0],
        "Error: Upper bound on x must be larger than lower bound"
    );
    let y_bounds = utils::get_values_from_required_parseable_argument(arguments, "y-bounds");
    exit_on_false!(
        y_bounds[1] > y_bounds[0],
        "Error: Upper bound on y must be larger than lower bound"
    );
    let z_bounds = utils::get_values_from_required_parseable_argument(arguments, "z-bounds");
    exit_on_false!(
        z_bounds[1] > z_bounds[0],
        "Error: Upper bound on z must be larger than lower bound"
    );

    let grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        Vec3::new(x_bounds[0], y_bounds[0], z_bounds[0]),
        Vec3::new(x_bounds[1], y_bounds[1], z_bounds[1]),
        In3D::new(false, false, false),
    );

    super::write_mesh_file(root_arguments, grid, protected_file_types);
}
