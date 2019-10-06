//! Command line interface for extracting slices of snapshot quantity fields.

use crate::cli;
use crate::field::ResampledCoordLocations;
use crate::geometry::Dim3;
use crate::grid::{CoordLocation, Grid3};
use crate::interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `snapshot-slice` command line subcommand.
pub fn build_subcommand_slice<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("slice")
        .about("Extract a 2D slice of a quantity field in the snapshot")
        .long_about(
            "Extract a 2D slice of a quantity field in the snapshot.\n\
             Any quantity field in the snapshot can be sliced across any axis at a given\n\
             coordinate.",
        )
        .after_help(
            "You can use a subcommand to configure the interpolator. If left unspecified,\n\
             the default interpolator implementation and parameters are used.",
        )
        .arg(
            Arg::with_name("QUANTITY")
                .help("Quantity whose field to slice")
                .required(true)
                .takes_value(true),
        )
        .arg(
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
            Arg::with_name("OUTPUT_PATH")
                .help("Path where the slice field should be saved in pickle format")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("sample-location")
                .short("l")
                .long("sample-location")
                .value_name("LOCATION")
                .long_help("Location in the grid cell where slice values should be sampled\n")
                .takes_value(true)
                .possible_values(&["center", "lower", "original"])
                .default_value("center"),
        )
        .arg(
            Arg::with_name("allow-non-uniform")
                .short("n")
                .long("allow-non-uniform")
                .long_help(
                    "Make sampled slice values follow the potentially non-uniform underlying grid",
                ),
        )
        .subcommand(cli::interpolation::poly_fit::create_poly_fit_interpolator_subcommand())
}

/// Runs the actions for the `snapshot-slice` subcommand using the given arguments.
pub fn run_subcommand_slice<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
) {
    let quantity = arguments
        .value_of("QUANTITY")
        .expect("No value for required argument");

    let axis = arguments
        .value_of("AXIS")
        .expect("No value for required argument");

    let coord = cli::get_value_from_required_parseable_argument::<fdt>(arguments, "COORD");

    let output_file_path = arguments
        .value_of("OUTPUT_PATH")
        .expect("No value for required argument");

    let sample_location = arguments
        .value_of("sample-location")
        .expect("No value for argument with default");

    let interpolator_config = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        cli::interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options(
            interpolator_arguments,
        )
    } else {
        PolyFitInterpolatorConfig::default()
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    let field = snapshot
        .obtain_scalar_field(quantity)
        .unwrap_or_else(|err| panic!("Could not read {}: {}", quantity, err));

    if arguments.is_present("allow-non-uniform") {
        let resampled_coord_locations = match sample_location {
            "center" => ResampledCoordLocations::centers(),
            "lower" => ResampledCoordLocations::lower_edges(),
            "original" => ResampledCoordLocations::Original,
            invalid => panic!("Invalid sample-location: {}", invalid),
        };

        match axis {
            "x" => {
                field
                    .slice_across_x(&interpolator, coord, resampled_coord_locations)
                    .save_as_pickle(output_file_path)
                    .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
            }
            "y" => {
                field
                    .slice_across_y(&interpolator, coord, resampled_coord_locations)
                    .save_as_pickle(output_file_path)
                    .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
            }
            "z" => {
                field
                    .slice_across_z(&interpolator, coord, resampled_coord_locations)
                    .save_as_pickle(output_file_path)
                    .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
            }
            invalid => panic!("Invalid AXIS: {}", invalid),
        }
    } else {
        let location = match sample_location {
            "center" => CoordLocation::Center,
            "lower" => CoordLocation::LowerEdge,
            "original" => panic!(
                "Invalid sample-location: original (only available with --allow-non-uniform)"
            ),
            invalid => panic!("Invalid sample-location: {}", invalid),
        };

        let axis = match axis {
            "x" => Dim3::X,
            "y" => Dim3::Y,
            "z" => Dim3::Z,
            invalid => panic!("Invalid AXIS: {}", invalid),
        };

        field
            .regular_slice_across_axis(&interpolator, axis, coord, location)
            .save_as_pickle(output_file_path)
            .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
    }
}
