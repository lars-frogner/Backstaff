//! Command line interface for producing randomly spaced seed points in a volume of a 3D grid.

use crate::{
    cli::utils,
    geometry::{Point3, Vec3},
    io::snapshot::fdt,
    seeding::volume::VolumeSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the random volume seeder.
pub fn create_random_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "random";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the random volume seeder")
        .long_about(
            "Use the random volume seeder.\n\
             Seed points are produced at uniformly random positions within\n\
             a volume of the 3D grid.",
        )
        .arg(
            Arg::new("n-points")
                .short('n')
                .long("n-points")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Number of seed points to generate")
                .required(true)
                .takes_value(true),
        )
}

/// Creates a random volume seeder based on the provided arguments.
pub fn create_random_volume_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    lower_bounds: Vec3<fdt>,
    upper_bounds: Vec3<fdt>,
    satisfies_constraints: &S,
) -> VolumeSeeder3
where
    S: Fn(&Point3<fdt>) -> bool + Sync,
{
    let n_seeds = utils::get_value_from_required_parseable_argument::<usize>(arguments, "n-points");

    VolumeSeeder3::random(&lower_bounds, &upper_bounds, n_seeds, satisfies_constraints)
}
