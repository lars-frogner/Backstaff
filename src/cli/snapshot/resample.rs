//! Command line interface for resampling a snapshot.

mod direct_sampling;
mod weighted_cell_averaging;
mod weighted_sample_averaging;

use self::{
    direct_sampling::create_direct_sampling_subcommand,
    weighted_cell_averaging::create_weighted_cell_averaging_subcommand,
    weighted_sample_averaging::create_weighted_sample_averaging_subcommand,
};
use crate::{
    cli::{
        interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options,
        snapshot::write::run_write_subcommand, utils as cli_utils,
    },
    create_subcommand, exit_on_error, exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{fdt, native, SnapshotReader3},
        utils as io_utils,
    },
};
use clap::{App, AppSettings, Arg, ArgGroup, ArgMatches, SubCommand};
use std::{collections::HashMap, path::PathBuf, process, str::FromStr, sync::Arc};

/// Builds a representation of the `snapshot-resample` command line subcommand.
pub fn create_resample_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("resample")
        .about("Creates a resampled version of the snapshot")
        .long_about(
            "Creates a resampled version of the snapshot.\n\
             The snapshot is resampled to a grid specified by a mesh file or to a regular\n\
             grid of a specified shape.",
        )
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .arg(
            Arg::with_name("shape")
                .short("s")
                .long("shape")
                .require_equals(true)
                .require_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the regular grid to resample to")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("mesh-file")
                .short("m")
                .long("mesh-file")
                .require_equals(true)
                .value_name("FILE")
                .help("Path to a Bifrost mesh file representing the grid to resample to")
                .takes_value(true),
        )
        .group(
            ArgGroup::with_name("grid")
                .args(&["shape", "mesh-file"])
                .required(true),
        )
        .arg(
            Arg::with_name("sample-location")
                .short("l")
                .long("sample-location")
                .require_equals(true)
                .value_name("LOCATION")
                .help("Location within the grid cell where resampled values should be specified\n")
                .takes_value(true)
                .possible_values(&["original", "center"])
                .default_value("original"),
        )
        .arg(
            Arg::with_name("yes")
                .short("y")
                .long("yes")
                .help("Automatically continue on warnings"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages related to resampling"),
        )
        .subcommand(create_subcommand!(resample, weighted_sample_averaging))
        .subcommand(create_subcommand!(resample, weighted_cell_averaging))
        .subcommand(create_subcommand!(resample, direct_sampling))
}

/// Runs the actions for the `snapshot-resample` subcommand using the given arguments.
pub fn run_resample_subcommand<G, R>(arguments: &ArgMatches, reader: &R)
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let (resampling_method, method_arguments) = if let Some(method_arguments) =
        arguments.subcommand_matches("weighted_sample_averaging")
    {
        (ResamplingMethod::WeightedSampleAveraging, method_arguments)
    } else if let Some(method_arguments) = arguments.subcommand_matches("weighted_cell_averaging") {
        (ResamplingMethod::WeightedCellAveraging, method_arguments)
    } else if let Some(method_arguments) = arguments.subcommand_matches("direct_sampling") {
        (ResamplingMethod::DirectSampling, method_arguments)
    } else {
        panic!("No resampling method specified.")
    };

    run_with_selected_interpolator(arguments, method_arguments, reader, resampling_method)
}

fn run_with_selected_interpolator<G, R>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    reader: &R,
    resampling_method: ResamplingMethod,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let (interpolator_config, arguments) = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        (
            construct_poly_fit_interpolator_config_from_options(interpolator_arguments),
            interpolator_arguments,
        )
    } else {
        (PolyFitInterpolatorConfig::default(), arguments)
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    run_resampling(
        root_arguments,
        arguments,
        reader,
        resampling_method,
        interpolator,
    );
}

fn run_resampling<G, R, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    reader: &R,
    resampling_method: ResamplingMethod,
    interpolator: I,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    I: Interpolator3,
{
    const BOUNDS_DIFF_THRESHOLD: fdt = 5e-3;

    let resampled_locations = match root_arguments
        .value_of("sample-location")
        .expect("No value for argument with default.")
    {
        "original" => In3D::same(ResampledCoordLocation::Original),
        "center" => In3D::same(ResampledCoordLocation::center()),
        invalid => exit_with_error!("Invalid sample-location: {}", invalid),
    };

    let continue_on_warnings = root_arguments.is_present("yes");
    let is_verbose = root_arguments.is_present("verbose");

    let old_grid = reader.grid();

    let lower_bounds = old_grid.lower_bounds().clone();
    let upper_bounds = old_grid.upper_bounds().clone();

    let is_periodic = In3D::new(
        old_grid.is_periodic(X),
        old_grid.is_periodic(Y),
        old_grid.is_periodic(Z),
    );

    let write_arguments = arguments.subcommand_matches("write").unwrap();

    macro_rules! verify_bounds_for_new_grid {
        ($new_grid:expr) => {{
            let new_lower_bounds = $new_grid.lower_bounds();
            let new_upper_bounds = $new_grid.upper_bounds();
            let different_lower_bounds =
                (new_lower_bounds - &lower_bounds).abs().max() > BOUNDS_DIFF_THRESHOLD;
            let different_upper_bounds =
                (new_upper_bounds - &upper_bounds).abs().max() > BOUNDS_DIFF_THRESHOLD;
            let different_bounds = different_lower_bounds || different_upper_bounds;
            if different_bounds {
                eprintln!("Warning: Bounds of resampling grid differ from original grid bounds");
            }
            if different_lower_bounds {
                eprintln!(
                    "Lower bounds: {:?} (resampling) vs {:?} (original)",
                    new_lower_bounds, &lower_bounds
                );
            }
            if different_upper_bounds {
                eprintln!(
                    "Upper bounds: {:?} (resampling) vs {:?} (original)",
                    new_upper_bounds, &upper_bounds
                );
            }
            if different_bounds
                && !continue_on_warnings
                && !io_utils::user_says_yes("Still continue?", true)
            {
                process::exit(1);
            }
        }};
    }

    macro_rules! resample_snapshot_for_grid {
        ($new_grid:expr) => {
            run_write_subcommand(
                write_arguments,
                reader,
                Some(Arc::clone(&$new_grid)),
                HashMap::new(),
                |field| {
                    if is_verbose {
                        println!("Resampling {}", field.name());
                    }
                    Ok(field.resampled_to_grid(
                        Arc::clone(&$new_grid),
                        resampled_locations.clone(),
                        &interpolator,
                        resampling_method,
                    ))
                },
            )
        };
    }

    match cli_utils::get_values_from_parseable_argument(root_arguments, "shape") {
        Some(shape) => {
            let new_grid = Arc::new(RegularGrid3::from_bounds(
                In3D::new(shape[0], shape[1], shape[2]),
                lower_bounds,
                upper_bounds,
                is_periodic,
            ));
            resample_snapshot_for_grid!(new_grid);
        }
        None => {
            let mesh_file_path = exit_on_error!(
                PathBuf::from_str(
                    root_arguments
                        .value_of("mesh-file")
                        .expect("No value for required argument."),
                ),
                "Error: Could not interpret path to mesh file: {}"
            );
            let (
                detected_grid_type,
                center_coords,
                lower_edge_coords,
                up_derivatives,
                down_derivatives,
            ) = exit_on_error!(
                native::parse_mesh_file(mesh_file_path),
                "Error: Could not parse mesh file: {}"
            );
            match detected_grid_type {
                GridType::Regular => {
                    let new_grid = Arc::new(RegularGrid3::from_coords(
                        center_coords,
                        lower_edge_coords,
                        is_periodic,
                        Some(up_derivatives),
                        Some(down_derivatives),
                    ));
                    verify_bounds_for_new_grid!(new_grid);
                    resample_snapshot_for_grid!(new_grid);
                }
                GridType::HorRegular => {
                    let new_grid = Arc::new(HorRegularGrid3::from_coords(
                        center_coords,
                        lower_edge_coords,
                        is_periodic,
                        Some(up_derivatives),
                        Some(down_derivatives),
                    ));
                    verify_bounds_for_new_grid!(new_grid);
                    resample_snapshot_for_grid!(new_grid);
                }
            }
        }
    }
}
