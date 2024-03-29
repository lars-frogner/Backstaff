//! Command line interface for producing stratified seed points in a volume of a 3D grid.

use crate::{
    cli::utils,
    geometry::{In3D, Point3, Vec3},
    grid::{fgr, regular::RegularGrid3},
    seeding::volume::VolumeSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the stratified volume seeder.
pub fn create_stratified_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "stratified";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the stratified volume seeder")
        .long_about(
            "Use the stratified volume seeder.\n\
             Seed points are produced at random positions within the cells of a regular 3D grid.",
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the regular 3D grid to seed in")
                .required(true)
                .takes_value(true)
                .number_of_values(3),
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

/// Creates a stratified volume seeder based on the provided arguments.
pub fn create_stratified_volume_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    lower_bounds: Vec3<fgr>,
    upper_bounds: Vec3<fgr>,
    satisfies_constraints: &S,
) -> VolumeSeeder3
where
    S: Fn(&Point3<fgr>) -> bool + Sync,
{
    let shape = utils::parse_3d_values_no_special(arguments, "shape", Some(1));
    let n_seeds_per_cell =
        utils::get_value_from_required_parseable_argument::<usize>(arguments, "points-per-cell");
    let randomness = utils::get_finite_float_value_from_required_parseable_argument::<fgr>(
        arguments,
        "randomness",
    );

    let grid = RegularGrid3::from_bounds(shape, lower_bounds, upper_bounds, In3D::same(false));

    VolumeSeeder3::stratified(&grid, n_seeds_per_cell, randomness, satisfies_constraints)
}
