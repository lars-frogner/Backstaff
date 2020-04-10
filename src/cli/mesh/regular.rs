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
        .about("Create a regular grid")
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
    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );

    let bounds_x = utils::get_values_from_required_parseable_argument(arguments, "x-bounds");
    exit_on_false!(
        bounds_x[1] > bounds_x[0],
        "Error: Upper bound on x must be larger than lower bound"
    );
    let bounds_y = utils::get_values_from_required_parseable_argument(arguments, "y-bounds");
    exit_on_false!(
        bounds_y[1] > bounds_y[0],
        "Error: Upper bound on y must be larger than lower bound"
    );
    let bounds_z = utils::get_values_from_required_parseable_argument(arguments, "z-bounds");
    exit_on_false!(
        bounds_z[1] > bounds_z[0],
        "Error: Upper bound on z must be larger than lower bound"
    );

    let grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        Vec3::new(bounds_x[0], bounds_y[0], bounds_z[0]),
        Vec3::new(bounds_x[1], bounds_y[1], bounds_z[1]),
        In3D::same(false),
    );

    super::write_mesh_file(root_arguments, grid, protected_file_types);
}
