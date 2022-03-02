//! Command line interface for producing stratified seed points in a volume of a 3D grid.

use crate::{
    cli::utils,
    geometry::{In3D, Point3, Vec3},
    grid::regular::RegularGrid3,
    io::snapshot::fdt,
    seeding::{fsd, volume::VolumeSeeder3},
};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the stratified volume seeder.
pub fn create_stratified_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("stratified")
        .about("Use the stratified volume seeder")
        .long_about(
            "Use the stratified volume seeder.\n\
             Seed points are produced at random positions within the cells of a regular 3D grid.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("shape")
                .short("s")
                .long("shape")
                .require_equals(true)
                .require_delimiter(true)
                .value_names(&["X", "Y", "Z"])
                .help("Shape of the regular 3D grid to seed in")
                .required(true)
                .takes_value(true)
                .number_of_values(3),
        )
        .arg(
            Arg::with_name("points-per-cell")
                .long("points-per-cell")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Number of seed points to generate in each grid cell")
                .takes_value(true)
                .default_value("1"),
        )
        .arg(
            Arg::with_name("randomness")
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
    lower_bounds: Vec3<fdt>,
    upper_bounds: Vec3<fdt>,
    satisfies_constraints: &S,
) -> VolumeSeeder3
where
    S: Fn(&Point3<fdt>) -> bool + Sync,
{
    let shape = utils::get_values_from_required_parseable_argument::<usize>(arguments, "shape");
    let n_seeds_per_cell =
        utils::get_value_from_required_parseable_argument::<usize>(arguments, "points-per-cell");
    let randomness =
        utils::get_value_from_required_parseable_argument::<fsd>(arguments, "randomness");

    let grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        lower_bounds,
        upper_bounds,
        In3D::same(false),
    );

    VolumeSeeder3::stratified(&grid, n_seeds_per_cell, randomness, satisfies_constraints)
}
