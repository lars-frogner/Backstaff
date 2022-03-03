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
    create_subcommand, exit_on_false, exit_with_error,
    geometry::{Dim2, Dim3, Point2},
    grid::Grid3,
    interpolation::Interpolator3,
    io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
    seeding::{fsd, slice::SliceSeeder3},
};
use clap::{Arg, ArgMatches, Command};

/// Holds parameters that are required by all slice seeders.
pub struct CommonSliceSeederParameters {
    axis: Dim3,
    coord: fsd,
}

/// Creates a subcommand for using a slice seeder.
pub fn create_slice_seeder_subcommand() -> Command<'static> {
    Command::new("slice_seeder")
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
                .possible_values(&["x", "y", "z"]),
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
            Arg::new("horizontal-limits")
                .long("horizontal-limits")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Smallest and largest value of the first slice coordinate for which\n\
                     generated seed points will be accepted [default: no limits]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::new("vertical-limits")
                .long("vertical-limits")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Smallest and largest value of the second slice coordinate for which\n\
                     generated seed points will be accepted [default: no limits]",
                )
                .takes_value(true),
        )
        .subcommand(create_subcommand!(slice_seeder, regular))
        .subcommand(create_subcommand!(slice_seeder, random))
        .subcommand(create_subcommand!(slice_seeder, stratified))
        .subcommand(create_subcommand!(slice_seeder, value_pdf))
}

/// Creates a slice seeder based on the provided arguments.
pub fn create_slice_seeder_from_arguments<G, R, I>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: &I,
) -> SliceSeeder3
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    I: Interpolator3,
{
    let axis = utils::get_value_from_required_constrained_argument(
        arguments,
        "axis",
        &["x", "y", "z"],
        &Dim3::slice(),
    );
    let coord = utils::get_value_from_required_parseable_argument::<fsd>(arguments, "coord");

    let horizontal_limits = utils::get_values_from_parseable_argument_with_custom_defaults(
        arguments,
        "horizontal-limits",
        &|| vec![std::f32::NEG_INFINITY, std::f32::INFINITY],
    );
    exit_on_false!(
        horizontal_limits[1] >= horizontal_limits[0],
        "Error: Upper horizontal limit must be larger than or equal to lower horizontal limit"
    );
    let vertical_limits = utils::get_values_from_parseable_argument_with_custom_defaults(
        arguments,
        "vertical-limits",
        &|| vec![std::f32::NEG_INFINITY, std::f32::INFINITY],
    );
    exit_on_false!(
        vertical_limits[1] >= vertical_limits[0],
        "Error: Upper vertical limit must be larger than or equal to lower vertical limit"
    );

    let parameters = CommonSliceSeederParameters { axis, coord };

    let satisifes_constraints = |point: &Point2<fdt>| {
        point[Dim2::X] >= horizontal_limits[0]
            && point[Dim2::X] < horizontal_limits[1]
            && point[Dim2::Y] >= vertical_limits[0]
            && point[Dim2::Y] < vertical_limits[1]
    };

    if let Some(seeder_arguments) = arguments.subcommand_matches("regular") {
        create_regular_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.reader().grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("random") {
        create_random_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.reader().grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("stratified") {
        create_stratified_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.reader().grid(),
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
