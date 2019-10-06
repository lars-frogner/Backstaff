//! Command line interface for producing regularly spaced seed points in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::cli;
use crate::geometry::{In2D, Point2};
use crate::grid::Grid3;
use crate::io::snapshot::fdt;
use crate::tracing::seeding::slice::SliceSeeder3;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the regular slice seeder.
pub fn create_regular_slice_seeder_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("regular")
        .about("Use the regular slice seeder")
        .long_about(
            "Use the regular slice seeder.\n\
             Seed points are produced at the cell centers of a regular 2D grid spanning a\n\
             slice of the 3D grid.",
        );
    add_regular_slice_seeder_options_to_subcommand(app)
}

/// Adds arguments for parameters used by the regular slice seeder.
pub fn add_regular_slice_seeder_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("SHAPE")
            .help("Shape of the regular 2D grid to seed in")
            .value_names(&["WIDTH", "HEIGHT"])
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
    let shape = cli::get_values_from_required_parseable_argument::<usize>(arguments, "SHAPE");

    SliceSeeder3::regular(
        grid,
        parameters.axis,
        parameters.coord,
        In2D::new(shape[0], shape[1]),
        satisfies_constraints,
    )
}
