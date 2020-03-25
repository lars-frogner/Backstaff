//! Command line interface for resampling a snapshot.

mod direct_sampling;
mod mesh_file;
mod regular_grid;
mod weighted_cell_averaging;
mod weighted_sample_averaging;

use self::{
    mesh_file::{create_mesh_file_subcommand, run_resampling_for_mesh_file},
    regular_grid::{create_regular_grid_subcommand, run_resampling_for_regular_grid},
};
use crate::{
    cli::{
        interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options,
        snapshot::write::run_write_subcommand,
    },
    create_subcommand, exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::In3D,
    grid::Grid3,
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{fdt, SnapshotReader3},
        utils as io_utils,
    },
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::{collections::HashMap, process, sync::Arc};

/// Builds a representation of the `snapshot-resample` command line subcommand.
pub fn create_resample_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("resample")
        .about("Creates a resampled version of the snapshot")
        .long_about(
            "Creates a resampled version of the snapshot.\n\
             The snapshot is resampled to a regular grid of configurable shape and bounds,\n\
             or to an arbitrary grid specified by a Bifrost mesh file.",
        )
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
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
            Arg::with_name("ignore-warnings")
                .long("ignore-warnings")
                .help("Automatically continue on warnings"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages related to resampling"),
        )
        .subcommand(create_subcommand!(resample, regular_grid))
        .subcommand(create_subcommand!(resample, mesh_file))
}

enum ResampleGridType {
    Regular,
    MeshFile,
}

/// Runs the actions for the `snapshot-resample` subcommand using the given arguments.
pub fn run_resample_subcommand<G, R>(
    arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let resampled_locations = match arguments
        .value_of("sample-location")
        .expect("No value for argument with default.")
    {
        "original" => In3D::same(ResampledCoordLocation::Original),
        "center" => In3D::same(ResampledCoordLocation::center()),
        invalid => exit_with_error!("Invalid sample-location: {}", invalid),
    };

    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let is_verbose = arguments.is_present("verbose");

    let (resample_grid_type, grid_type_arguments) =
        if let Some(regular_grid_arguments) = arguments.subcommand_matches("regular_grid") {
            (ResampleGridType::Regular, regular_grid_arguments)
        } else if let Some(mesh_file_arguments) = arguments.subcommand_matches("mesh_file") {
            (ResampleGridType::MeshFile, mesh_file_arguments)
        } else {
            unreachable!()
        };

    run_with_selected_method(
        grid_type_arguments,
        reader,
        snap_num_offset,
        resample_grid_type,
        &resampled_locations,
        continue_on_warnings,
        is_verbose,
    );
}

fn run_with_selected_method<G, R>(
    grid_type_arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    resample_grid_type: ResampleGridType,
    resampled_locations: &In3D<ResampledCoordLocation>,
    continue_on_warnings: bool,
    is_verbose: bool,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let (resampling_method, method_arguments) = if let Some(method_arguments) =
        grid_type_arguments.subcommand_matches("weighted_sample_averaging")
    {
        (ResamplingMethod::WeightedSampleAveraging, method_arguments)
    } else if let Some(method_arguments) =
        grid_type_arguments.subcommand_matches("weighted_cell_averaging")
    {
        (ResamplingMethod::WeightedCellAveraging, method_arguments)
    } else if let Some(method_arguments) = grid_type_arguments.subcommand_matches("direct_sampling")
    {
        (ResamplingMethod::DirectSampling, method_arguments)
    } else {
        (
            ResamplingMethod::WeightedSampleAveraging,
            grid_type_arguments,
        )
    };

    run_with_selected_interpolator(
        grid_type_arguments,
        method_arguments,
        reader,
        snap_num_offset,
        resample_grid_type,
        resampled_locations,
        resampling_method,
        continue_on_warnings,
        is_verbose,
    )
}

fn run_with_selected_interpolator<G, R>(
    grid_type_arguments: &ArgMatches,
    arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    resample_grid_type: ResampleGridType,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    is_verbose: bool,
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

    match resample_grid_type {
        ResampleGridType::Regular => run_resampling_for_regular_grid(
            grid_type_arguments,
            arguments,
            reader,
            snap_num_offset,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            is_verbose,
            interpolator,
        ),
        ResampleGridType::MeshFile => run_resampling_for_mesh_file(
            grid_type_arguments,
            arguments,
            reader,
            snap_num_offset,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            is_verbose,
            interpolator,
        ),
    }
}

fn verify_bounds_for_new_grid<GIN: Grid3<fdt>, GOUT: Grid3<fdt>>(
    original_grid: &GIN,
    new_grid: &GOUT,
    continue_on_warnings: bool,
) {
    const BOUNDS_DIFF_THRESHOLD: fdt = 5e-3;

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();
    let new_lower_bounds = new_grid.lower_bounds();
    let new_upper_bounds = new_grid.upper_bounds();

    let different_lower_bounds =
        (new_lower_bounds - original_lower_bounds).abs().max() > BOUNDS_DIFF_THRESHOLD;
    let different_upper_bounds =
        (new_upper_bounds - original_upper_bounds).abs().max() > BOUNDS_DIFF_THRESHOLD;
    let different_bounds = different_lower_bounds || different_upper_bounds;
    if different_bounds {
        eprintln!("Warning: Bounds of resampling grid differ from original grid bounds");
    }
    if different_lower_bounds {
        eprintln!(
            "Lower bounds: {:?} (resampling) vs {:?} (original)",
            new_lower_bounds, original_lower_bounds
        );
    }
    if different_upper_bounds {
        eprintln!(
            "Upper bounds: {:?} (resampling) vs {:?} (original)",
            new_upper_bounds, original_upper_bounds
        );
    }
    if different_bounds
        && !continue_on_warnings
        && !io_utils::user_says_yes("Still continue?", true)
    {
        process::exit(1);
    }
}

fn resample_snapshot_for_grid<GIN, R, GOUT, I>(
    write_arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    new_grid: &Arc<GOUT>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    is_verbose: bool,
    interpolator: I,
) where
    GIN: Grid3<fdt>,
    R: SnapshotReader3<GIN>,
    GOUT: Grid3<fdt>,
    I: Interpolator3,
{
    run_write_subcommand(
        write_arguments,
        reader,
        snap_num_offset,
        Some(Arc::clone(new_grid)),
        HashMap::new(),
        |field| {
            if is_verbose {
                println!("Resampling {}", field.name());
            }
            Ok(field.resampled_to_grid(
                Arc::clone(new_grid),
                resampled_locations.clone(),
                &interpolator,
                resampling_method,
            ))
        },
    );
}
