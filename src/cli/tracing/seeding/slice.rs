//! Command line interface for generating of seed points in a slice through a field.

pub mod pdf;
pub mod random;
pub mod regular;
pub mod stratified;

use crate::cli;
use crate::geometry::{Dim2, Dim3, Point2};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::tracing::ftr;
use crate::tracing::seeding::slice::SliceSeeder3;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};

/// Holds parameters that are required by all slice seeders.
pub struct CommonSliceSeederParameters {
    axis: Dim3,
    coord: ftr,
}

/// Creates a subcommand for using a slice seeder.
pub fn create_slice_seeder_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("slice_seeder")
        .about("Use a slice seeder")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(regular::create_regular_slice_seeder_subcommand())
        .subcommand(random::create_random_slice_seeder_subcommand())
        .subcommand(stratified::create_stratified_slice_seeder_subcommand())
        .subcommand(pdf::create_slice_pdf_seeder_subcommand());

    add_slice_seeder_options_to_subcommand(app)
}

/// Adds arguments for parameters used by a slice seeder.
pub fn add_slice_seeder_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
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
        Arg::with_name("horizontal-limits")
            .long("horizontal-limits")
            .value_names(&["MIN, MAX"])
            .long_help(
                "Smallest and largest value of the first slice coordinate for which generated\n\
                 seed points will be accepted. [default: no limits]",
            )
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("vertical-limits")
            .long("vertical-limits")
            .value_names(&["MIN, MAX"])
            .long_help(
                "Smallest and largest value of the second slice coordinate for which generated\n\
                 seed points will be accepted. [default: no limits]",
            )
            .next_line_help(true)
            .takes_value(true),
    )
}

/// Creates a slice seeder based on the provided arguments.
pub fn create_slice_seeder_from_arguments<G, I>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
    interpolator: &I,
) -> SliceSeeder3
where
    G: Grid3<fdt>,
    I: Interpolator3,
{
    let axis = cli::get_value_from_required_constrained_argument(
        arguments,
        "AXIS",
        &["x", "y", "z"],
        &Dim3::slice(),
    );
    let coord = cli::get_value_from_required_parseable_argument::<ftr>(arguments, "COORD");

    let horizontal_limits = cli::get_values_from_parseable_argument_with_custom_defaults(
        arguments,
        "horizontal-limits",
        &|| vec![std::f32::NEG_INFINITY, std::f32::INFINITY],
    );
    assert!(
        horizontal_limits[1] >= horizontal_limits[0],
        "Upper horizontal limit must be larger than or equal to lower horizontal limit."
    );
    let vertical_limits = cli::get_values_from_parseable_argument_with_custom_defaults(
        arguments,
        "vertical-limits",
        &|| vec![std::f32::NEG_INFINITY, std::f32::INFINITY],
    );
    assert!(
        vertical_limits[1] >= vertical_limits[0],
        "Upper vertical limit must be larger than or equal to lower vertical limit."
    );

    let parameters = CommonSliceSeederParameters { axis, coord };

    let satisifes_constraints = |point: &Point2<fdt>| {
        point[Dim2::X] >= horizontal_limits[0]
            && point[Dim2::X] < horizontal_limits[1]
            && point[Dim2::Y] >= vertical_limits[0]
            && point[Dim2::Y] < vertical_limits[1]
    };

    if let Some(seeder_arguments) = arguments.subcommand_matches("regular") {
        regular::create_regular_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.reader().grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("random") {
        random::create_random_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.reader().grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("stratified") {
        stratified::create_stratified_slice_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot.reader().grid(),
            &satisifes_constraints,
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("value_pdf") {
        pdf::create_slice_pdf_seeder_from_arguments(
            seeder_arguments,
            &parameters,
            snapshot,
            interpolator,
            &satisifes_constraints,
        )
    } else {
        panic!("No seeder specified.")
    }
}
