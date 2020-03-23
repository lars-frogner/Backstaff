//! Command line interface for extracting slices of snapshot quantity fields.

use crate::{
    cli::{
        interpolation::poly_fit::{
            construct_poly_fit_interpolator_config_from_options,
            create_poly_fit_interpolator_subcommand,
        },
        utils as cli_utils,
    },
    create_subcommand, exit_on_error, exit_with_error,
    field::ResampledCoordLocation,
    geometry::Dim3,
    grid::{CoordLocation, Grid3},
    interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
    io::{
        snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
        utils as io_utils,
    },
};
use clap::{App, Arg, ArgMatches, SubCommand};
use std::{path::PathBuf, str::FromStr};

/// Builds a representation of the `snapshot-slice` command line subcommand.
pub fn create_slice_subcommand<'a, 'b>() -> App<'a, 'b> {
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
        .help_message("Print help information")
        .arg(
            Arg::with_name("output-file")
                .value_name("OUTPUT_FILE")
                .help("Path where the slice field should be saved in pickle format")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing file"),
        )
        .arg(
            Arg::with_name("quantity")
                .short("q")
                .long("quantity")
                .require_equals(true)
                .value_name("NAME")
                .help("Quantity whose field to slice")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("axis")
                .short("a")
                .long("axis")
                .require_equals(true)
                .value_name("AXIS")
                .help("Which axis to slice across")
                .required(true)
                .takes_value(true)
                .possible_values(&["x", "y", "z"]),
        )
        .arg(
            Arg::with_name("coord")
                .short("c")
                .long("coord")
                .require_equals(true)
                .value_name("VALUE")
                .allow_hyphen_values(true)
                .help("Coordinate along the axis to slice at")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("sample-location")
                .short("l")
                .long("sample-location")
                .require_equals(true)
                .value_name("LOCATION")
                .help("Location in the grid cell where slice values should be sampled\n")
                .takes_value(true)
                .possible_values(&["center", "lower", "original"])
                .default_value("center"),
        )
        .arg(
            Arg::with_name("allow-non-uniform")
                .short("n")
                .long("allow-non-uniform")
                .help(
                    "Make sampled slice values follow the potentially non-uniform underlying grid",
                ),
        )
        .subcommand(create_subcommand!(slice, poly_fit_interpolator))
}

/// Runs the actions for the `snapshot-slice` subcommand using the given arguments.
pub fn run_slice_subcommand<G, R>(arguments: &ArgMatches, snapshot: &mut SnapshotCacher3<G, R>)
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let quantity = arguments
        .value_of("quantity")
        .expect("No value for required argument.");

    let axis = arguments
        .value_of("axis")
        .expect("No value for required argument.");

    let coord = cli_utils::get_value_from_required_parseable_argument::<fdt>(arguments, "coord");

    let mut output_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("output-file")
                .expect("No value for required argument."),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    if output_file_path.extension().is_none() {
        output_file_path.set_extension("pickle");
    }

    let force_overwrite = arguments.is_present("overwrite");

    if !force_overwrite && !io_utils::write_allowed(&output_file_path) {
        exit_with_error!("Aborted");
    }

    let sample_location = arguments
        .value_of("sample-location")
        .expect("No value for argument with default.");

    let interpolator_config = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        construct_poly_fit_interpolator_config_from_options(interpolator_arguments)
    } else {
        PolyFitInterpolatorConfig::default()
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    let field = exit_on_error!(
        snapshot.obtain_scalar_field(quantity),
        "Error: Could not read quantity {0} in snapshot: {1}",
        quantity
    );

    if arguments.is_present("allow-non-uniform") {
        let resampled_coord_locations = match sample_location {
            "center" => ResampledCoordLocation::center(),
            "lower" => ResampledCoordLocation::lower_edge(),
            "original" => ResampledCoordLocation::Original,
            invalid => exit_with_error!("Error: Invalid sample-location: {}", invalid),
        };

        exit_on_error!(
            match axis {
                "x" => {
                    field
                        .slice_across_x(&interpolator, coord, resampled_coord_locations)
                        .save_as_pickle(output_file_path)
                }
                "y" => {
                    field
                        .slice_across_y(&interpolator, coord, resampled_coord_locations)
                        .save_as_pickle(output_file_path)
                }
                "z" => {
                    field
                        .slice_across_z(&interpolator, coord, resampled_coord_locations)
                        .save_as_pickle(output_file_path)
                }
                invalid => exit_with_error!("Error: Invalid axis: {}", invalid),
            },
            "Error: Could not save output data: {}"
        );
    } else {
        let location = match sample_location {
            "center" => CoordLocation::Center,
            "lower" => CoordLocation::LowerEdge,
            "original" => exit_with_error!(
                "Error: Invalid sample-location: original only available with --allow-non-uniform"
            ),
            invalid => exit_with_error!("Error: Invalid sample-location: {}", invalid),
        };

        let axis = match axis {
            "x" => Dim3::X,
            "y" => Dim3::Y,
            "z" => Dim3::Z,
            invalid => exit_with_error!("Error: Invalid axis: {}", invalid),
        };

        exit_on_error!(
            field
                .regular_slice_across_axis(&interpolator, axis, coord, location)
                .save_as_pickle(output_file_path),
            "Error: Could not save output data: {}"
        );
    }
}
