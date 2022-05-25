//! Command line interface for resampling a snapshot.

mod direct_sampling;
mod mesh_file;
mod regular_grid;
mod reshaped_grid;
mod rotated_regular_grid;
mod weighted_cell_averaging;
mod weighted_sample_averaging;

use self::{
    mesh_file::{create_mesh_file_subcommand, run_resampling_for_mesh_file},
    regular_grid::{create_regular_grid_subcommand, run_resampling_for_regular_grid},
    reshaped_grid::{create_reshaped_grid_subcommand, run_resampling_for_reshaped_grid},
    rotated_regular_grid::{
        create_rotated_regular_grid_subcommand, run_resampling_for_rotated_regular_grid,
    },
};
use super::SnapNumInRange;
use crate::{
    cli::{
        interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options,
        snapshot::{derive::create_derive_provider, write::run_write_subcommand},
    },
    exit_with_error,
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Coords3,
        Dim3::{self, X, Y, Z},
        IdentityTransformation2, In3D, PointTransformation2,
    },
    grid::{self, fgr, hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    interpolation::{
        poly_fit::{PolyFitInterpolator1, PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{CachingSnapshotProvider3, ResampledSnapshotProvider3, SnapshotProvider3},
        utils, Verbose,
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use std::{collections::HashMap, process, sync::Arc};

/// Builds a representation of the `snapshot-resample` command line subcommand.
pub fn create_resample_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "resample";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Create a resampled version of the snapshot")
        .long_about(
            "Create a resampled version of the snapshot.\n\
             The snapshot is resampled to a regular grid of configurable shape and bounds,\n\
             to a potentially reshaped version of the original grid, or to an arbitrary\n\
             grid specified by a Bifrost mesh file.",
        )
        .subcommand_required(true)
        .arg(
            Arg::new("sample-location")
                .short('l')
                .long("sample-location")
                .require_equals(true)
                .value_name("LOCATION")
                .help("Location within the grid cell where resampled values should be specified\n")
                .takes_value(true)
                .possible_values(&["original", "center"])
                .default_value("original"),
        )
        .arg(
            Arg::new("ignore-warnings")
                .long("ignore-warnings")
                .help("Automatically continue on warnings"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to resampling"),
        )
        .subcommand(create_regular_grid_subcommand(command_name))
        .subcommand(create_rotated_regular_grid_subcommand(command_name))
        .subcommand(create_reshaped_grid_subcommand(command_name))
        .subcommand(create_mesh_file_subcommand(command_name))
}

enum ResampleGridType {
    Regular,
    RotatedRegular,
    Reshaped,
    MeshFile,
}

/// Runs the actions for the `snapshot-resample` subcommand using the given arguments.
pub fn run_resample_subcommand<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
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
    let verbose = arguments.is_present("verbose").into();

    let (resample_grid_type, default_method, grid_type_arguments) =
        if let Some(regular_grid_arguments) = arguments.subcommand_matches("regular_grid") {
            (
                ResampleGridType::Regular,
                ResamplingMethod::WeightedSampleAveraging,
                regular_grid_arguments,
            )
        } else if let Some(rotated_regular_grid_arguments) =
            arguments.subcommand_matches("rotated_regular_grid")
        {
            (
                ResampleGridType::RotatedRegular,
                ResamplingMethod::WeightedSampleAveraging,
                rotated_regular_grid_arguments,
            )
        } else if let Some(reshaped_grid_arguments) = arguments.subcommand_matches("reshaped_grid")
        {
            (
                ResampleGridType::Reshaped,
                ResamplingMethod::WeightedSampleAveraging,
                reshaped_grid_arguments,
            )
        } else if let Some(mesh_file_arguments) = arguments.subcommand_matches("mesh_file") {
            (
                ResampleGridType::MeshFile,
                ResamplingMethod::WeightedSampleAveraging,
                mesh_file_arguments,
            )
        } else {
            unreachable!()
        };

    run_with_selected_method(
        grid_type_arguments,
        provider,
        snap_num_in_range,
        resample_grid_type,
        default_method,
        &resampled_locations,
        continue_on_warnings,
        verbose,
        protected_file_types,
    );
}

fn run_with_selected_method<G, P>(
    grid_type_arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resample_grid_type: ResampleGridType,
    default_method: ResamplingMethod,
    resampled_locations: &In3D<ResampledCoordLocation>,
    continue_on_warnings: bool,
    verbose: Verbose,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let (resampling_method, method_arguments, has_interpolator_subcommand) =
        if let Some(method_arguments) =
            grid_type_arguments.subcommand_matches("weighted_sample_averaging")
        {
            (
                ResamplingMethod::WeightedSampleAveraging,
                method_arguments,
                true,
            )
        } else if let Some(method_arguments) =
            grid_type_arguments.subcommand_matches("weighted_cell_averaging")
        {
            (
                ResamplingMethod::WeightedCellAveraging,
                method_arguments,
                false,
            )
        } else if let Some(method_arguments) =
            grid_type_arguments.subcommand_matches("direct_sampling")
        {
            (ResamplingMethod::DirectSampling, method_arguments, true)
        } else {
            (default_method, grid_type_arguments, false)
        };

    run_with_selected_interpolator(
        grid_type_arguments,
        method_arguments,
        provider,
        snap_num_in_range,
        resample_grid_type,
        resampled_locations,
        resampling_method,
        has_interpolator_subcommand,
        continue_on_warnings,
        verbose,
        protected_file_types,
    )
}

fn run_with_selected_interpolator<G, P>(
    grid_type_arguments: &ArgMatches,
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resample_grid_type: ResampleGridType,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    has_interpolator_subcommand: bool,
    continue_on_warnings: bool,
    verbose: Verbose,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let (interpolator_config, arguments) = if has_interpolator_subcommand {
        if let Some(interpolator_arguments) = arguments.subcommand_matches("poly_fit_interpolator")
        {
            (
                construct_poly_fit_interpolator_config_from_options(interpolator_arguments),
                interpolator_arguments,
            )
        } else {
            (PolyFitInterpolatorConfig::default(), arguments)
        }
    } else {
        (PolyFitInterpolatorConfig::default(), arguments)
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    match resample_grid_type {
        ResampleGridType::Regular => run_resampling_for_regular_grid(
            grid_type_arguments,
            arguments,
            provider,
            snap_num_in_range,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbose,
            interpolator,
            protected_file_types,
        ),
        ResampleGridType::RotatedRegular => run_resampling_for_rotated_regular_grid(
            grid_type_arguments,
            arguments,
            provider,
            snap_num_in_range,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbose,
            interpolator,
            protected_file_types,
        ),
        ResampleGridType::Reshaped => run_resampling_for_reshaped_grid(
            grid_type_arguments,
            arguments,
            provider,
            snap_num_in_range,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbose,
            interpolator,
            protected_file_types,
        ),
        ResampleGridType::MeshFile => run_resampling_for_mesh_file(
            grid_type_arguments,
            arguments,
            provider,
            snap_num_in_range,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbose,
            interpolator,
            protected_file_types,
        ),
    }
}

fn resample_to_reshaped_grid<G, P, I>(
    arguments: &ArgMatches,
    new_shape: Option<In3D<usize>>,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    verbose: Verbose,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
    I: Interpolator3,
{
    let grid = provider.grid();

    match grid.grid_type() {
        GridType::Regular => {
            let grid = RegularGrid3::from_coords(
                grid.centers().clone(),
                grid.lower_edges().clone(),
                grid.periodicity().clone(),
                grid.up_derivatives().cloned(),
                grid.down_derivatives().cloned(),
            );
            resample_to_regular_grid(
                grid,
                new_shape,
                arguments,
                provider,
                snap_num_in_range,
                resampled_locations,
                resampling_method,
                continue_on_warnings,
                verbose,
                interpolator,
                protected_file_types,
            );
        }
        GridType::HorRegular => {
            let grid = HorRegularGrid3::from_coords(
                grid.centers().clone(),
                grid.lower_edges().clone(),
                grid.periodicity().clone(),
                grid.up_derivatives().cloned(),
                grid.down_derivatives().cloned(),
            );
            resample_to_horizontally_regular_grid(
                grid,
                new_shape,
                arguments,
                provider,
                snap_num_in_range,
                resampled_locations,
                resampling_method,
                continue_on_warnings,
                verbose,
                interpolator,
                protected_file_types,
            );
        }
    }
}

fn resample_to_regular_grid<G, P, I>(
    mut grid: RegularGrid3<fgr>,
    new_shape: Option<In3D<usize>>,
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    verbose: Verbose,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
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

    correct_periodicity_for_new_grid(provider.grid(), &mut grid, continue_on_warnings);

    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        arguments,
        provider,
        snap_num_in_range,
        &new_grid,
        IdentityTransformation2::new(),
        resampled_locations,
        resampling_method,
        verbose,
        interpolator,
        protected_file_types,
    );
}

fn resample_to_horizontally_regular_grid<G, P, I>(
    mut grid: HorRegularGrid3<fgr>,
    new_shape: Option<In3D<usize>>,
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    verbose: Verbose,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
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

    correct_periodicity_for_new_grid(provider.grid(), &mut grid, continue_on_warnings);

    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        arguments,
        provider,
        snap_num_in_range,
        &new_grid,
        IdentityTransformation2::new(),
        resampled_locations,
        resampling_method,
        verbose,
        interpolator,
        protected_file_types,
    );
}

fn resample_to_transformed_regular_grid<G, P, T, I>(
    grid: RegularGrid3<fgr>,
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    transformation: T,
    _continue_on_warnings: bool,
    verbose: Verbose,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    T: PointTransformation2<fgr>,
    P: SnapshotProvider3<G>,
    I: Interpolator3,
{
    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        arguments,
        provider,
        snap_num_in_range,
        &new_grid,
        transformation,
        resampled_locations,
        resampling_method,
        verbose,
        interpolator,
        protected_file_types,
    );
}

fn correct_periodicity_for_new_grid<GIN: Grid3<fgr>, GOUT: Grid3<fgr>>(
    original_grid: &GIN,
    new_grid: &mut GOUT,
    continue_on_warnings: bool,
) {
    // A coordinate difference must exceed this fraction of a grid cell
    // extent in order to be detected
    const EXTENT_DIFF_THRESHOLD_FACTOR: fgr = 0.1;

    const WARNING_EXTENT_DIFF_THRESHOLD_FACTOR: fgr = 10.0;

    let average_grid_cell_extents = original_grid.average_grid_cell_extents();

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();
    let new_lower_bounds = new_grid.lower_bounds();
    let new_upper_bounds = new_grid.upper_bounds();

    let mut is_periodic = In3D::same(false);

    for &dim in &Dim3::slice() {
        let average_grid_cell_extent = average_grid_cell_extents[dim];
        let extent_diff_threshold = average_grid_cell_extent * EXTENT_DIFF_THRESHOLD_FACTOR;
        let warning_extent_diff_threshold =
            average_grid_cell_extent * WARNING_EXTENT_DIFF_THRESHOLD_FACTOR;

        let original_lower_bound = original_lower_bounds[dim];
        let original_upper_bound = original_upper_bounds[dim];
        let new_lower_bound = new_lower_bounds[dim];
        let new_upper_bound = new_upper_bounds[dim];

        let original_extent = original_upper_bound - original_lower_bound;
        let new_extent = new_upper_bound - new_lower_bound;

        let extent_difference = original_extent - new_extent;
        let abs_extent_difference = fgr::abs(extent_difference);

        if original_grid.is_periodic(dim) {
            if abs_extent_difference <= extent_diff_threshold {
                is_periodic[dim] = true;
            }
            if !is_periodic[dim] {
                let warning_triggered = abs_extent_difference >= extent_diff_threshold
                    && abs_extent_difference <= warning_extent_diff_threshold;

                if warning_triggered {
                    eprintln!(
                        "Warning: Field is no longer periodic in {}-direction after resampling\n\
                         because new extent differs slightly from original extent:\n\
                         Before resampling: {}\n\
                         After resampling: {}",
                        dim, original_extent, new_extent,
                    );
                    if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                        eprintln!("Aborted");
                        process::exit(1);
                    }
                }
            }
        } else if new_upper_bound - extent_diff_threshold > original_upper_bound
            || new_lower_bound + extent_diff_threshold < original_lower_bound
        {
            eprintln!(
                "Warning: Extrapolating beyond {}-bounds of original field:\n\
                 Before resampling: [{}, {})\n\
                 After resampling: [{}, {})",
                dim, original_lower_bound, original_upper_bound, new_lower_bound, new_upper_bound
            );
            if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                eprintln!("Aborted");
                process::exit(1);
            }
        }
    }

    new_grid.set_periodicity(is_periodic);
}

fn compute_scaled_grid_shape(original_shape: &In3D<usize>, scales: &[fgr]) -> Vec<usize> {
    original_shape
        .into_iter()
        .zip(scales.iter())
        .map(|(&n, &scale)| usize::max(1, fgr::round(scale * (n as fgr)) as usize))
        .collect()
}

fn resample_snapshot_for_grid<GIN, P, GOUT, T, I>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    new_grid: &Arc<GOUT>,
    transformation: T,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    verbose: Verbose,
    interpolator: I,
    protected_file_types: &[&str],
) where
    GIN: Grid3<fgr>,
    P: SnapshotProvider3<GIN>,
    GOUT: Grid3<fgr>,
    T: PointTransformation2<fgr>,
    I: Interpolator3,
{
    exit_on_error!(
        interpolator.verify_grid(provider.grid()),
        "Invalid input grid for resampling: {}"
    );

    let provider = ResampledSnapshotProvider3::new(
        provider,
        Arc::clone(new_grid),
        transformation,
        resampled_locations.clone(),
        interpolator,
        resampling_method,
        verbose,
    );

    run_snapshot_resampling_with_derive(
        arguments,
        provider,
        snap_num_in_range,
        protected_file_types,
    );
}

fn run_snapshot_resampling_with_derive<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    if let Some(derive_arguments) = arguments.subcommand_matches("derive") {
        let provider = create_derive_provider(derive_arguments, provider);
        run_snapshot_resampling_with_synthesis(
            derive_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
    } else {
        run_snapshot_resampling_with_synthesis_added_caching(
            arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
    }
}

fn run_snapshot_resampling_with_synthesis<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider =
            super::synthesize::create_synthesize_provider(synthesize_arguments, provider);
        run_snapshot_resampling_for_provider(
            synthesize_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
        return;
    }

    run_snapshot_resampling_for_provider(
        arguments,
        provider,
        snap_num_in_range,
        protected_file_types,
    );
}

fn run_snapshot_resampling_with_synthesis_added_caching<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider = super::synthesize::create_synthesize_provider_added_caching(
            synthesize_arguments,
            provider,
        );
        run_snapshot_resampling_for_provider(
            synthesize_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
        return;
    }

    run_snapshot_resampling_for_provider(
        arguments,
        provider,
        snap_num_in_range,
        protected_file_types,
    );
}

fn run_snapshot_resampling_for_provider<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let write_arguments = arguments.subcommand_matches("write").unwrap();

    run_write_subcommand(
        write_arguments,
        provider,
        snap_num_in_range,
        HashMap::new(),
        protected_file_types,
    );
}
