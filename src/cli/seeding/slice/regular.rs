//! Command line interface for producing regularly spaced seed points in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::{
    cli::utils,
    geometry::{In2D, Point2},
    grid::Grid3,
    io::snapshot::fdt,
    seeding::slice::SliceSeeder3,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the regular slice seeder.
pub fn create_regular_subcommand() -> Command<'static> {
    Command::new("regular")
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
pub fn create_regular_slice_seeder_from_arguments<G, S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    grid: &G,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    G: Grid3<fdt>,
    S: Fn(&Point2<fdt>) -> bool + Sync,
{
    let shape = utils::get_values_from_required_parseable_argument::<usize>(arguments, "shape");

    SliceSeeder3::regular(
        grid,
        parameters.axis,
        parameters.coord,
        In2D::new(shape[0], shape[1]),
        satisfies_constraints,
    )
}
