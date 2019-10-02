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
    let app = SubCommand::with_name("slice")
        .about("Extracts a 2D slice of a snapshot quantity field")
        .arg(
            Arg::with_name("QUANTITY")
                .long_help("Quantity whose field to slice")
                .required(true)
                .takes_value(true)
                .index(1),
        )
        .arg(
            Arg::with_name("AXIS")
                .long_help("Which axis to slice across")
                .required(true)
                .takes_value(true)
                .possible_values(&["x", "y", "z"])
                .index(2),
        )
        .arg(
            Arg::with_name("COORD")
                .long_help("Coordinate along the axis to slice at")
                .required(true)
                .takes_value(true)
                .index(3),
        )
        .arg(
            Arg::with_name("OUTPUT_PATH")
                .long_help("Path where the slice field should be saved in pickle format")
                .required(true)
                .takes_value(true)
                .index(4),
        )
        .arg(
            Arg::with_name("sample-location")
                .short("l")
                .long("sample-location")
                .value_name("LOCATION")
                .long_help("Location in the grid cell where slice values should be sampled")
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
        );

    cli::interpolation::poly_fit::add_poly_fit_interpolator_options_to_subcommand(app)
}

/// Runs the actions for the `snapshot-slice` subcommand using the given arguments.
pub fn run_subcommand_slice<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    cacher: &mut SnapshotCacher3<G>,
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

    let mut interpolator_config = PolyFitInterpolatorConfig::default();
    cli::interpolation::poly_fit::configure_poly_fit_interpolator_from_options(
        &mut interpolator_config,
        arguments,
    );
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    let field = cacher
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
