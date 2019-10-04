//! Command line interface for producing stratified seed points in a 2D slice of a 3D grid.

use crate::cli;
use crate::geometry::{Dim3, In2D};
use crate::grid::Grid3;
use crate::io::snapshot::fdt;
use crate::tracing::ftr;
use crate::tracing::seeding::slice::SliceSeeder3;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the stratified slice seeder.
pub fn create_stratified_slice_seeder_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app =
        SubCommand::with_name("stratified_slice_seeder").about("Use the stratified slice seeder");
    add_stratified_slice_seeder_options_to_subcommand(app)
}

/// Adds arguments for parameters used by the stratified slice seeder.
pub fn add_stratified_slice_seeder_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("AXIS")
            .help("Which axis to slice across")
            .required(true)
            .takes_value(true)
            .possible_values(&["x", "y", "z"]),
    )
    .arg(
        Arg::with_name("COORD")
            .help("Coordinate along the axis to slice at")
            .required(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("SHAPE")
            .help("Shape of the regular 2D grid to seed in")
            .required(true)
            .takes_value(true)
            .number_of_values(2),
    )
    .arg(
        Arg::with_name("points-per-cell")
            .help("Number of seed points to generate in each grid cell")
            .takes_value(true)
            .default_value("1"),
    )
    .arg(
        Arg::with_name("randomness")
            .long_help(
                "How far from the cell centers the seed points can be generated,\n\
                 going from 0 (cell center) to 1 (cell edge)",
            )
            .takes_value(true)
            .default_value("1.0"),
    )
}

/// Creates a stratified slice seeder based on the provided arguments.
pub fn create_stratified_slice_seeder_from_arguments<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    grid: &G,
) -> SliceSeeder3 {
    let axis = cli::get_value_from_required_constrained_argument(
        arguments,
        "AXIS",
        &["x", "y", "z"],
        &Dim3::slice(),
    );
    let coord = cli::get_value_from_required_parseable_argument::<ftr>(arguments, "COORD");
    let shape = cli::get_values_from_required_parseable_argument::<usize>(arguments, "SHAPE");
    let n_seeds_per_cell =
        cli::get_value_from_required_parseable_argument::<usize>(arguments, "points-per-cell");
    let randomness =
        cli::get_value_from_required_parseable_argument::<ftr>(arguments, "randomness");

    SliceSeeder3::stratified(
        grid,
        axis,
        coord,
        In2D::new(shape[0], shape[1]),
        n_seeds_per_cell,
        randomness,
    )
}
