//! Command line interface for extracting slices of snapshot quantity fields.

use super::SnapNumInRange;
use crate::{
    cli::{
        interpolation::poly_fit::{
            construct_poly_fit_interpolator_config_from_options,
            create_poly_fit_interpolator_subcommand,
        },
        utils as cli_utils,
    },
    exit_on_error, exit_with_error,
    field::ResampledCoordLocation,
    geometry::Dim3,
    grid::{CoordLocation, Grid3},
    interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
    io::{
        snapshot::{self, fdt, SnapshotProvider3},
        utils::AtomicOutputPath,
    },
};
use clap::{Arg, ArgMatches, Command};
use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Builds a representation of the `snapshot-slice` command line subcommand.
pub fn create_slice_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "slice";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
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
            Arg::new("output-file")
                .value_name("OUTPUT_FILE")
                .help(
                    "Path where the slice field should be saved\n\
                     Writes in the following format based on the file extension:\
                     \n    *.pickle: Creates a Python pickle file (requires the pickle feature)",
                )
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing files (unless listed as protected)")
                .conflicts_with("no-overwrite"),
        )
        .arg(
            Arg::new("no-overwrite")
                .long("no-overwrite")
                .help("Do not overwrite any existing files")
                .conflicts_with("overwrite"),
        )
        .arg(
            Arg::new("quantity")
                .short('q')
                .long("quantity")
                .require_equals(true)
                .value_name("NAME")
                .help("Quantity whose field to slice")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("axis")
                .short('a')
                .long("axis")
                .require_equals(true)
                .value_name("AXIS")
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
            Arg::new("sample-location")
                .short('l')
                .long("sample-location")
                .require_equals(true)
                .value_name("LOCATION")
                .help("Location in the grid cell where slice values should be sampled\n")
                .takes_value(true)
                .possible_values(&["center", "lower", "original"])
                .default_value("center"),
        )
        .arg(
            Arg::new("allow-non-uniform")
                .short('n')
                .long("allow-non-uniform")
                .help(
                    "Make sampled slice values follow the potentially non-uniform underlying grid",
                ),
        )
        .subcommand(create_poly_fit_interpolator_subcommand(command_name))
}

#[cfg(not(feature = "pickle"))]
pub fn run_slice_subcommand<G, P>(_: &ArgMatches, _: P, _: &Option<SnapNumInRange>, _: &[&str])
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    exit_with_error!(
        "Error: Compile with pickle feature in order to write Pickle files\n\
                      Tip: Use cargo flag --features=pickle"
    );
}

/// Runs the actions for the `snapshot-slice` subcommand using the given arguments.
#[cfg(feature = "pickle")]
pub fn run_slice_subcommand<G, P>(
    arguments: &ArgMatches,
    mut provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    let quantity = arguments
        .value_of("quantity")
        .expect("No value for required argument");

    let axis = arguments
        .value_of("axis")
        .expect("No value for required argument");

    let coord = cli_utils::get_value_from_required_parseable_argument::<fdt>(arguments, "coord");

    let mut output_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("output-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    let output_type = OutputType::from_path(&output_file_path);

    if let Some(snap_num_in_range) = snap_num_in_range {
        output_file_path.set_file_name(snapshot::create_new_snapshot_file_name_from_path(
            &output_file_path,
            snap_num_in_range.offset(),
            &output_type.to_string(),
            true,
        ));
    }

    let overwrite_mode = cli_utils::overwrite_mode_from_arguments(arguments);

    let atomic_output_path = exit_on_error!(
        AtomicOutputPath::new(output_file_path),
        "Error: Could not create temporary output file: {}"
    );

    if !atomic_output_path.check_if_write_allowed(overwrite_mode, protected_file_types) {
        return;
    }

    let sample_location = arguments
        .value_of("sample-location")
        .expect("No value for argument with default");

    let interpolator_config = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        construct_poly_fit_interpolator_config_from_options(interpolator_arguments)
    } else {
        PolyFitInterpolatorConfig::default()
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    let field = exit_on_error!(
        provider.provide_scalar_field(quantity),
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
                        .save_as_pickle(atomic_output_path.temporary_path())
                }
                "y" => {
                    field
                        .slice_across_y(&interpolator, coord, resampled_coord_locations)
                        .save_as_pickle(atomic_output_path.temporary_path())
                }
                "z" => {
                    field
                        .slice_across_z(&interpolator, coord, resampled_coord_locations)
                        .save_as_pickle(atomic_output_path.temporary_path())
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
            match output_type {
                OutputType::Pickle => field
                    .regular_slice_across_axis(&interpolator, axis, coord, location)
                    .save_as_pickle(atomic_output_path.temporary_path()),
            },
            "Error: Could not save output data: {}"
        );
    }

    exit_on_error!(
        atomic_output_path.perform_replace(),
        "Error: Could not move temporary output file to target path: {}"
    );
}

#[derive(Copy, Clone, Debug)]
#[cfg(feature = "pickle")]
enum OutputType {
    #[cfg(feature = "pickle")]
    Pickle,
}

#[cfg(feature = "pickle")]
impl OutputType {
    fn from_path<P: AsRef<Path>>(file_path: P) -> Self {
        Self::from_extension(
            file_path
                .as_ref()
                .extension()
                .unwrap_or_else(|| {
                    exit_with_error!(
                        "Error: Missing extension for output file\n\
                         Valid extensions are: {}",
                        Self::valid_extensions_string()
                    )
                })
                .to_string_lossy()
                .as_ref(),
        )
    }

    fn from_extension(extension: &str) -> Self {
        match extension {
            "pickle" => {
                #[cfg(feature = "pickle")]
                {
                    Self::Pickle
                }
                #[cfg(not(feature = "pickle"))]
                exit_with_error!(
                    "Error: Compile with pickle feature in order to write Pickle files\n\
                                  Tip: Use cargo flag --features=pickle"
                );
            }
            invalid => exit_with_error!(
                "Error: Invalid extension {} for output file\n\
                 Valid extensions are: {}",
                invalid,
                Self::valid_extensions_string()
            ),
        }
    }

    fn valid_extensions_string() -> String {
        "pickle".to_string()
    }
}

#[cfg(feature = "pickle")]
impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                #[cfg(feature = "pickle")]
                Self::Pickle => "pickle",
            }
        )
    }
}
