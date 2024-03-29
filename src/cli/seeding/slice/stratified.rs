//! Command line interface for producing stratified seed points in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::{
    cli::utils, field::FieldGrid3, geometry::Point2, grid::fgr, seeding::slice::SliceSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the stratified slice seeder.
pub fn create_stratified_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "stratified";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the stratified slice seeder")
        .long_about(
            "Use the stratified slice seeder.\n\
             Seed points are produced at random positions within the cells of a regular 2D\n\
             grid spanning a slice of the 3D grid.",
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["WIDTH", "HEIGHT"])
                .help("Shape of the regular 2D grid to seed in")
                .required(true)
                .takes_value(true)
                .number_of_values(2),
        )
        .arg(
            Arg::new("points-per-cell")
                .long("points-per-cell")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Number of seed points to generate in each grid cell")
                .takes_value(true)
                .default_value("1"),
        )
        .arg(
            Arg::new("randomness")
                .long("randomness")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "How far from the cell centers the seed points can be generated,\n\
                     going from 0 (cell center) to 1 (cell edge)",
                )
                .takes_value(true)
                .default_value("1.0"),
        )
}

/// Creates a stratified slice seeder based on the provided arguments.
pub fn create_stratified_slice_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    grid: &FieldGrid3,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    S: Fn(&Point2<fgr>) -> bool + Sync,
{
    let shape = utils::parse_2d_values_no_special(arguments, "shape", Some(1));
    let n_seeds_per_cell =
        utils::get_value_from_required_parseable_argument::<usize>(arguments, "points-per-cell");
    let randomness = utils::get_finite_float_value_from_required_parseable_argument::<fgr>(
        arguments,
        "randomness",
    );

    SliceSeeder3::stratified(
        grid,
        parameters.axis,
        parameters.coord,
        shape,
        n_seeds_per_cell,
        randomness,
        satisfies_constraints,
    )
}
