//! Command line interface for resampling a snapshot.

mod sampling;
mod weighted_cell_averaging;
mod weighted_sample_averaging;

use crate::cli;
use crate::geometry::{Dim3, In3D};
use crate::grid::hor_regular::HorRegularGrid3;
use crate::grid::regular::RegularGrid3;
use crate::grid::{Grid3, GridType};
use crate::interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig};
use crate::interpolation::Interpolator3;
use crate::io::mesh;
use crate::io::snapshot::{self, fdt, SnapshotReader3};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::path;
use std::str::FromStr;
use std::sync::Arc;
use Dim3::{X, Y, Z};

/// Builds a representation of the `snapshot-resample` command line subcommand.
pub fn create_resample_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("resample")
        .about("Creates a resampled version of the snapshot")
        .setting(AppSettings::SubcommandRequired)
        .long_about(
            "Creates a resampled version of the snapshot.\n\
             The snapshot quantity fields are resampled to the grid described by a given mesh\n\
             file.",
        )
        .arg(
            Arg::with_name("MESH_PATH")
                .help("Path to the Bifrost mesh file representing the grid to resample to")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("OUTPUT_PATH")
                .help("Path where the resampled snapshot should be saved")
                .required(true)
                .takes_value(true),
            )
        .arg(
            Arg::with_name("resampled-grid-type")
                .long("resampled-grid-type")
                .require_equals(true)
                .value_name("TYPE")
                .long_help("Type of grid to assume for the resampled snapshot\n[default: same as original]")
                .next_line_help(true)
                .takes_value(true)
                .possible_values(&["horizontally-regular", "regular"]),
        )
        .arg(
            Arg::with_name("include-aux")
                .short("a")
                .long("include-aux")
                .help("Also resample the auxiliary quantity fields associated with the snapshot"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages while resampling the snapshot"),
        )
        .subcommand(weighted_sample_averaging::create_weighted_sample_averaging_subcommand())
        .subcommand(weighted_cell_averaging::create_weighted_cell_averaging_subcommand())
        .subcommand(sampling::create_sampling_subcommand())
}

/// Runs the actions for the `snapshot-resample` subcommand using the given arguments.
pub fn run_resample_subcommand<G: Grid3<fdt>>(arguments: &ArgMatches, reader: &SnapshotReader3<G>) {
    let (resampling_method, method_arguments) = if let Some(method_arguments) =
        arguments.subcommand_matches("weighted_sample_averaging")
    {
        ("weighted_sample_averaging", method_arguments)
    } else if let Some(method_arguments) = arguments.subcommand_matches("weighted_cell_averaging") {
        ("weighted_cell_averaging", method_arguments)
    } else if let Some(method_arguments) = arguments.subcommand_matches("sampling") {
        ("sampling", method_arguments)
    } else {
        panic!("No resampling method specified.")
    };

    run_with_selected_interpolator(arguments, method_arguments, reader, resampling_method)
}

fn run_with_selected_interpolator<G: Grid3<fdt>>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
    resampling_method: &str,
) {
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

    run_resampling(root_arguments, reader, resampling_method, interpolator);
}

fn run_resampling<G, I>(
    root_arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
    resampling_method: &str,
    interpolator: I,
) where
    G: Grid3<fdt>,
    I: Interpolator3,
{
    let mesh_file_path = root_arguments
        .value_of("MESH_PATH")
        .expect("No value for required argument.");

    let output_file_path = path::PathBuf::from_str(
        root_arguments
            .value_of("OUTPUT_PATH")
            .expect("No value for required argument."),
    )
    .unwrap_or_else(|err| panic!("Could not interpret OUTPUT_PATH: {}", err));

    let grid_type = match root_arguments.value_of("resampled-grid-type") {
        None => G::TYPE,
        Some("horizontally-regular") => GridType::HorRegular,
        Some("regular") => GridType::Regular,
        Some(invalid) => panic!("Invalid grid type {}", invalid),
    };

    let include_aux = root_arguments.is_present("include-aux");

    let is_verbose = root_arguments.is_present("verbose");

    let old_grid = reader.grid();
    let is_periodic = In3D::new(
        old_grid.is_periodic(X),
        old_grid.is_periodic(Y),
        old_grid.is_periodic(Z),
    );

    macro_rules! resample_snapshot_for_grid_type {
        ($grid_type:ty) => {{
            if is_verbose {
                println!("Constructing new grid");
            }
            let new_grid = Arc::new(
                mesh::create_grid_from_mesh_file::<_, $grid_type>(mesh_file_path, is_periodic)
                    .unwrap_or_else(|err| panic!("Could not create resampling grid: {}", err)),
            );
            let variable_value_producer = |name: &str| {
                let field = reader.read_scalar_field(name).unwrap_or_else(|err| {
                    panic!("Could not read quantity field {}: {}", name, err)
                });
                if is_verbose {
                    println!("Resampling {}", name);
                }
                let resampled_field =
                    match resampling_method {
                        "weighted_sample_averaging" => field
                            .resampled_to_grid_with_weighted_sample_averaging(
                                Arc::clone(&new_grid),
                                &interpolator,
                            ),
                        "weighted_cell_averaging" => field
                            .resampled_to_grid_with_weighted_cell_averaging(Arc::clone(&new_grid)),
                        "sampling" => field
                            .resampled_to_grid_with_sampling(Arc::clone(&new_grid), &interpolator),
                        invalid => panic!("Invalid mode {}", invalid),
                    };
                resampled_field.into_values()
            };
            snapshot::write_3d_snapfile(
                output_file_path.as_path(),
                reader.primary_variable_names(),
                &variable_value_producer,
                reader.endianness(),
                is_verbose.into(),
            )
            .unwrap_or_else(|err| {
                panic!("Could not write snap file for resampled snapshot: {}", err)
            });

            if include_aux {
                snapshot::write_3d_snapfile(
                    output_file_path.with_extension("aux"),
                    reader.auxiliary_variable_names(),
                    &variable_value_producer,
                    reader.endianness(),
                    is_verbose.into(),
                )
                .unwrap_or_else(|err| {
                    panic!("Could not write aux file for resampled snapshot: {}", err)
                });
            }
        }};
    }

    match grid_type {
        GridType::HorRegular => {
            resample_snapshot_for_grid_type!(HorRegularGrid3<fdt>);
        }
        GridType::Regular => {
            resample_snapshot_for_grid_type!(RegularGrid3<fdt>);
        }
    }
}
