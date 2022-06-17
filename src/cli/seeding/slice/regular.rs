//! Command line interface for producing regularly spaced seed points in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::{
    cli::utils, field::FieldGrid3, geometry::Point2, grid::fgr, seeding::slice::SliceSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the regular slice seeder.
pub fn create_regular_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "regular";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the regular slice seeder")
        .long_about(
            "Use the regular slice seeder.\n\
             Seed points are produced at the cell centers of a regular 2D grid spanning a\n\
             slice of the 3D grid.",
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["WIDTH", "HEIGHT"])
                .help("Number of seed points to generate")
                .required(true)
                .takes_value(true)
                .number_of_values(2),
        )
}

/// Creates a regular slice seeder based on the provided arguments.
pub fn create_regular_slice_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    grid: &FieldGrid3,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    S: Fn(&Point2<fgr>) -> bool + Sync,
{
    let shape = utils::parse_2d_values_no_special(arguments, "shape", Some(1));

    SliceSeeder3::regular(
        grid,
        parameters.axis,
        parameters.coord,
        shape,
        satisfies_constraints,
    )
}
