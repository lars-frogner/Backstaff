//! Command line interface for producing regularly spaced seed points in a volume of a 3D grid.

use crate::{
    cli::utils,
    geometry::{In3D, Point3, Vec3},
    grid::{fgr, regular::RegularGrid3},
    seeding::volume::VolumeSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the regular volume seeder.
pub fn create_regular_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "regular";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the regular volume seeder")
        .long_about(
            "Use the regular volume seeder.\n\
             Seed points are produced at the cell centers of a regular 3D grid.",
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Number of seed points to generate in each dimension")
                .required(true)
                .takes_value(true)
                .number_of_values(3),
        )
}

/// Creates a regular volume seeder based on the provided arguments.
pub fn create_regular_volume_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    lower_bounds: Vec3<fgr>,
    upper_bounds: Vec3<fgr>,
    satisfies_constraints: &S,
) -> VolumeSeeder3
where
    S: Fn(&Point3<fgr>) -> bool + Sync,
{
    let shape = utils::parse_3d_values_no_special(arguments, "shape", Some(1));

    let grid = RegularGrid3::from_bounds(shape, lower_bounds, upper_bounds, In3D::same(false));

    VolumeSeeder3::regular(&grid, satisfies_constraints)
}
