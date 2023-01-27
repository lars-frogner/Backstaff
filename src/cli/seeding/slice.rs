//! Command line interface for generating of seed points in a slice through a field.

pub mod pdf;
pub mod random;
pub mod regular;
pub mod stratified;

use self::{
    pdf::{create_slice_pdf_seeder_from_arguments, create_value_pdf_subcommand},
    random::{create_random_slice_seeder_from_arguments, create_random_subcommand},
    regular::{create_regular_slice_seeder_from_arguments, create_regular_subcommand},
    stratified::{create_stratified_slice_seeder_from_arguments, create_stratified_subcommand},
};
use crate::{
    cli::utils,
    exit_with_error,
    field::CachingScalarFieldProvider3,
    geometry::{Dim2, Dim3, Point2},
    grid::fgr,
    interpolation::Interpolator3,
    io::snapshot::fdt,
    seeding::slice::SliceSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Holds parameters that are required by all slice seeders.
pub struct CommonSliceSeederParameters {
    axis: Dim3,
    coord: fgr,
}

/// Creates a subcommand for using a slice seeder.
pub fn create_slice_seeder_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "slice_seeder";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use a slice seeder")
        .subcommand_required(true)
        .arg(
            Arg::new("axis")
                .short('a')
                .long("axis")
                .value_name("AXIS")
                .require_equals(true)
                .help("Which axis to slice across")
                .required(true)
                .takes_value(true)
                .possible_values(["x", "y", "z"]),
        )
        .arg(
            Arg::new("coord")
                .short('c')
                .long("coord")
                .require_equals(true)
                .value_name("VALUE")
                .allow_hyphen_values(true)
                .help("Coordinate along the axis to slice at")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("horizontal-bounds")
                .long("horizontal-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Smallest and largest value of the first slice coordinate for which\n\
                     generated seed points will be accepted",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("vertical-bounds")
                .long("vertical-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Smallest and largest value of the second slice coordinate for which\n\
                     generated seed points will be accepted",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("-inf,inf"),
        )
        .subcommand(create_regular_subcommand(command_name))
        .subcommand(create_random_subcommand(command_name))
        .subcommand(create_stratified_subcommand(command_name))
        .subcommand(create_value_pdf_subcommand(command_name))
}

/// Creates a slice seeder based on the provided arguments.
pub fn create_slice_seeder_from_arguments(
    arguments: &ArgMatches,
    snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
) -> SliceSeeder3 {
    let axis = utils::get_value_from_required_constrained_argument(
        arguments,
        "axis",
        &["x", "y", "z"],
        &Dim3::slice(),
    );
    let coord =
        utils::get_finite_float_value_from_required_parseable_argument::<fgr>(arguments, "coord");

    let horizontal_bounds = utils::parse_limits(
        arguments,
        "horizontal-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::Yes,
        None,
    );
    let vertical_bounds = utils::parse_limits(
        arguments,
        "vertical-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::Yes,
        None,
    );

    let parameters = CommonSliceSeederParameters { axis, coord };

    let satisifes_constraints = |point: &Point2<fgr>| {
        point[Dim2::X] >= horizontal_bounds.0
            && point[Dim2::X] < horizontal_bounds.1
            && point[Dim2::Y] >= vertical_bounds.0
            && point[Dim2::Y] < vertical_bounds.1
    };

    if let Some(seeder_arguments) = arguments.subcommand_matches("regular") {
        create_regular_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("random") {
        create_random_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("stratified") {
        create_stratified_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("value_pdf") {
        create_slice_pdf_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot,
            interpolator,
            &satisifes_constraints,
        )
    } else {
        exit_with_error!("Error: No seeder specified")
    }
}
