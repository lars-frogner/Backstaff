//! Command line interface for producing randomly spaced seed points in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::{
    cli::utils, field::FieldGrid3, geometry::Point2, grid::fgr, seeding::slice::SliceSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the random slice seeder.
pub fn create_random_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "random";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the random slice seeder")
        .long_about(
            "Use the random slice seeder.\n\
             Seed points are produced at uniformly random positions within\n\
             a 2D slice of the 3D grid.",
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

/// Creates a random slice seeder based on the provided arguments.
pub fn create_random_slice_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    grid: &FieldGrid3,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    S: Fn(&Point2<fgr>) -> bool + Sync,
{
    let n_seeds = utils::get_value_from_required_parseable_argument::<usize>(arguments, "n-points");

    SliceSeeder3::random(
        grid,
        parameters.axis,
        parameters.coord,
        n_seeds,
        satisfies_constraints,
    )
}
