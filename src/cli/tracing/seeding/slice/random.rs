//! Command line interface for producing randomly spaced seed points in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::cli;
use crate::geometry::Point2;
use crate::grid::Grid3;
use crate::io::snapshot::fdt;
use crate::tracing::seeding::slice::SliceSeeder3;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the random slice seeder.
pub fn create_random_slice_seeder_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("random")
        .about("Use the random slice seeder")
        .long_about(
            "Use the random slice seeder.\n\
             Seed points are produced at uniformly random positions within\n\
             a 2D slice of the 3D grid.",
        );
    add_random_slice_seeder_options_to_subcommand(app)
}

/// Adds arguments for parameters used by the random slice seeder.
pub fn add_random_slice_seeder_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("POINTS")
            .help("Number of seed points to generate")
            .required(true)
            .takes_value(true),
    )
}

/// Creates a random slice seeder based on the provided arguments.
pub fn create_random_slice_seeder_from_arguments<G, S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    grid: &G,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    G: Grid3<fdt>,
    S: Fn(&Point2<fdt>) -> bool + Sync,
{
    let n_seeds = cli::get_value_from_required_parseable_argument::<usize>(arguments, "POINTS");

    SliceSeeder3::random(
        grid,
        parameters.axis,
        parameters.coord,
        n_seeds,
        satisfies_constraints,
    )
}
