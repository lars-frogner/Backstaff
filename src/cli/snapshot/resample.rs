//! Command line interface for resampling a snapshot.

mod direct_sampling;
mod mesh_file;
mod regular_grid;
mod reshaped_grid;
mod weighted_cell_averaging;
mod weighted_sample_averaging;

use self::{
    mesh_file::{create_mesh_file_subcommand, run_resampling_for_mesh_file},
    regular_grid::{create_regular_grid_subcommand, run_resampling_for_regular_grid},
    reshaped_grid::{create_reshaped_grid_subcommand, run_resampling_for_reshaped_grid},
};
use crate::{
    cli::{
        interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options,
        snapshot::write::run_write_subcommand,
    },
    create_subcommand, exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Coords3,
        Dim3::{self, X, Y, Z},
        In3D,
    },
    grid::{self, hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3},
    interpolation::{
        poly_fit::{PolyFitInterpolator1, PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{fdt, SnapshotReader3},
        utils,
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
        .subcommand(create_subcommand!(resample, reshaped_grid))
        .subcommand(create_subcommand!(resample, mesh_file))
}

enum ResampleGridType {
    Regular,
    Reshaped,
    MeshFile,
}

/// Runs the actions for the `snapshot-resample` subcommand using the given arguments.
pub fn run_resample_subcommand<G, R>(
    arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let resampled_locations = match arguments
        .value_of("sample-location")
        .expect("No value for argument with default")
    {
        "original" => In3D::same(ResampledCoordLocation::Original),
        "center" => In3D::same(ResampledCoordLocation::center()),
        invalid => exit_with_error!("Invalid sample-location: {}", invalid),
    };

    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let is_verbose = arguments.is_present("verbose");

    let (resample_grid_type, grid_type_arguments) = if let Some(regular_grid_arguments) =
        arguments.subcommand_matches("regular_grid")
    {
        (ResampleGridType::Regular, regular_grid_arguments)
    } else if let Some(reshaped_grid_arguments) = arguments.subcommand_matches("reshaped_grid") {
        (ResampleGridType::Reshaped, reshaped_grid_arguments)
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
        protected_file_types,
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
    protected_file_types: &[&str],
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
        protected_file_types,
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
    protected_file_types: &[&str],
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
            protected_file_types,
        ),
        ResampleGridType::Reshaped => run_resampling_for_reshaped_grid(
            grid_type_arguments,
            arguments,
            reader,
            snap_num_offset,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            is_verbose,
            interpolator,
            protected_file_types,
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
            protected_file_types,
        ),
    }
}

fn resample_to_regular_grid<G, R, I>(
    mut grid: RegularGrid3<fdt>,
    new_shape: Option<In3D<usize>>,
    write_arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    is_verbose: bool,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    I: Interpolator3,
{
    if let Some(new_shape) = new_shape {
        grid = RegularGrid3::from_bounds(
            new_shape,
            grid.lower_bounds().clone(),
            grid.upper_bounds().clone(),
            grid.periodicity().clone(),
        );
    }

    correct_periodicity_for_new_grid(reader.grid(), &mut grid, continue_on_warnings, is_verbose);

    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        write_arguments,
        reader,
        snap_num_offset,
        &new_grid,
        resampled_locations,
        resampling_method,
        is_verbose,
        interpolator,
        protected_file_types,
    );
}

fn resample_to_horizontally_regular_grid<G, R, I>(
    mut grid: HorRegularGrid3<fdt>,
    new_shape: Option<In3D<usize>>,
    write_arguments: &ArgMatches,
    reader: &R,
    snap_num_offset: Option<u32>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    is_verbose: bool,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    I: Interpolator3,
{
    const TARGET_CONTROL_SIZE: usize = 100;

    if let Some(new_shape) = new_shape {
        let horizontal_grid = RegularGrid3::from_bounds(
            new_shape.clone(),
            grid.lower_bounds().clone(),
            grid.upper_bounds().clone(),
            grid.periodicity().clone(),
        );
        let horizontal_centers = horizontal_grid.centers();
        let horizontal_lower_edges = horizontal_grid.lower_edges();

        let lower_bound_z = grid.lower_bounds()[Z];
        let upper_bound_z = grid.upper_bounds()[Z];
        let lower_edges_z = &grid.lower_edges()[Z];

        let size_z = lower_edges_z.len();
        let stride = f32::floor((size_z as f32) / TARGET_CONTROL_SIZE as f32) as usize;
        let mut control_coords: Vec<_> = lower_edges_z.iter().copied().step_by(stride).collect();
        let mut control_grid_cell_extents: Vec<_> = lower_edges_z
            .iter()
            .step_by(stride)
            .zip(lower_edges_z.iter().skip(1).step_by(stride))
            .map(|(lower_edge, upper_edge)| upper_edge - lower_edge)
            .collect();
        *control_coords.last_mut().unwrap() = upper_bound_z;
        *control_grid_cell_extents.last_mut().unwrap() =
            lower_edges_z[size_z - 1] - lower_edges_z[size_z - 2];

        let interpolator = PolyFitInterpolator1::new(PolyFitInterpolatorConfig::default());
        let (new_centers_z, new_lower_edges_z) = exit_on_error!(
            grid::create_new_grid_coords_from_control_extents(
                new_shape[Z],
                lower_bound_z,
                upper_bound_z,
                &control_coords,
                &control_grid_cell_extents,
                &interpolator,
            ),
            "Error: Could not compute new z-coordinates for grid: {}"
        );

        let mut up_derivatives = horizontal_grid.up_derivatives().unwrap().clone();
        let mut down_derivatives = horizontal_grid.down_derivatives().unwrap().clone();
        let (up_derivatives_z, down_derivatives_z) =
            grid::compute_up_and_down_derivatives(&new_centers_z);
        up_derivatives[Z] = up_derivatives_z;
        down_derivatives[Z] = down_derivatives_z;

        grid = HorRegularGrid3::from_coords(
            Coords3::new(
                horizontal_centers[X].clone(),
                horizontal_centers[Y].clone(),
                new_centers_z,
            ),
            Coords3::new(
                horizontal_lower_edges[X].clone(),
                horizontal_lower_edges[Y].clone(),
                new_lower_edges_z,
            ),
            grid.periodicity().clone(),
            Some(up_derivatives),
            Some(down_derivatives),
        );
    }

    correct_periodicity_for_new_grid(reader.grid(), &mut grid, continue_on_warnings, is_verbose);

    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        write_arguments,
        reader,
        snap_num_offset,
        &new_grid,
        resampled_locations,
        resampling_method,
        is_verbose,
        interpolator,
        protected_file_types,
    );
}

fn correct_periodicity_for_new_grid<GIN: Grid3<fdt>, GOUT: Grid3<fdt>>(
    original_grid: &GIN,
    new_grid: &mut GOUT,
    continue_on_warnings: bool,
    is_verbose: bool,
) {
    // A coordinate difference must exceed this fraction of a grid cell
    // extent in order to be detected
    const BOUND_DIFF_THRESHOLD_FACTOR: fdt = 0.1;

    const WARNING_BOUND_DIFF_THRESHOLD_FACTOR: fdt = 10.0;

    let average_grid_cell_extents = original_grid.average_grid_cell_extents();

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();
    let new_lower_bounds = new_grid.lower_bounds();
    let new_upper_bounds = new_grid.upper_bounds();

    let mut is_periodic = In3D::same(false);

    for &dim in &Dim3::slice() {
        let average_grid_cell_extent = average_grid_cell_extents[dim];
        let bound_diff_threshold = average_grid_cell_extent * BOUND_DIFF_THRESHOLD_FACTOR;
        let warning_bound_diff_threshold =
            average_grid_cell_extent * WARNING_BOUND_DIFF_THRESHOLD_FACTOR;

        let original_lower_bound = original_lower_bounds[dim];
        let original_upper_bound = original_upper_bounds[dim];
        let new_lower_bound = new_lower_bounds[dim];
        let new_upper_bound = new_upper_bounds[dim];

        let lower_difference = original_lower_bound - new_lower_bound;
        let upper_difference = new_upper_bound - original_upper_bound;

        let abs_lower_difference = fdt::abs(lower_difference);
        let abs_upper_difference = fdt::abs(upper_difference);

        if original_grid.is_periodic(dim) {
            let original_span = original_upper_bound - original_lower_bound;

            if abs_lower_difference <= bound_diff_threshold
                && abs_upper_difference <= bound_diff_threshold
            {
                is_periodic[dim] = true;
            } else if upper_difference + bound_diff_threshold >= 0.0
                && lower_difference + bound_diff_threshold >= 0.0
            {
                let lower_expansion_ratio = lower_difference / original_span;
                let upper_expansion_ratio = upper_difference / original_span;
                if fdt::abs(
                    (lower_expansion_ratio - fdt::trunc(lower_expansion_ratio)) * original_span,
                ) <= bound_diff_threshold
                    && fdt::abs(
                        (upper_expansion_ratio - fdt::trunc(upper_expansion_ratio)) * original_span,
                    ) <= bound_diff_threshold
                {
                    is_periodic[dim] = true;
                }
            }
            if !is_periodic[dim] {
                let warning_triggered = (abs_lower_difference >= bound_diff_threshold
                    && abs_lower_difference <= warning_bound_diff_threshold)
                    || (abs_upper_difference >= bound_diff_threshold
                        && abs_upper_difference <= warning_bound_diff_threshold);

                if warning_triggered {
                    eprintln!(
                        "Warning: Field is no longer periodic in {}-direction after resampling\n\
                         because new bounds differ slightly from original bounds:\n\
                         Before resampling: [{}, {})\n\
                         After resampling: [{}, {})",
                        dim,
                        original_lower_bound,
                        original_upper_bound,
                        new_lower_bound,
                        new_upper_bound
                    );
                    if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                        process::exit(1);
                    }
                } else if is_verbose {
                    println!(
                        "Field is no longer periodic in {}-direction after resampling\n\
                         because new bounds do not coincide with periodic boundaries:\n\
                         Before resampling: [{}, {})\n\
                         After resampling: [{}, {})",
                        dim,
                        original_lower_bound,
                        original_upper_bound,
                        new_lower_bound,
                        new_upper_bound
                    );
                }
            }
        } else if new_upper_bound - bound_diff_threshold > original_upper_bound
            || new_lower_bound + bound_diff_threshold < original_lower_bound
        {
            eprintln!(
                "Warning: Extrapolating beyond {}-bounds of original field:\n\
                 Before resampling: [{}, {})\n\
                 After resampling: [{}, {})",
                dim, original_lower_bound, original_upper_bound, new_lower_bound, new_upper_bound
            );
            if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                process::exit(1);
            }
        }
    }

    new_grid.set_periodicity(is_periodic);
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
    protected_file_types: &[&str],
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
        protected_file_types,
    );
}
