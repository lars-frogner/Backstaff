//! Command line interface for resampling a snapshot.

mod cell_averaging;
mod direct_sampling;
mod mesh_file;
mod regular_grid;
mod reshaped_grid;
mod rotated_regular_grid;
mod sample_averaging;

use self::{
    mesh_file::{create_mesh_file_subcommand, run_resampling_for_mesh_file},
    regular_grid::{create_regular_grid_subcommand, run_resampling_for_regular_grid},
    reshaped_grid::{create_reshaped_grid_subcommand, run_resampling_for_reshaped_grid},
    rotated_regular_grid::{
        create_rotated_regular_grid_subcommand, run_resampling_for_rotated_regular_grid,
    },
};

use crate::{
    cli::{
        interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options,
        snapshot::{inspect::run_inspect_subcommand, write::run_write_subcommand},
        utils as cli_utils,
    },
    exit_on_error, exit_on_false, exit_with_error,
    field::{FieldGrid3, ResampledCoordLocation, ResamplingMethod},
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
        snapshot::{fdt, CachingSnapshotProvider3, ResampledSnapshotProvider3, SnapshotProvider3},
        utils::IOContext,
        Verbosity,
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use std::{iter, sync::Arc};

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
        .arg(
            Arg::new("progress")
                .short('p')
                .long("progress")
                .help("Show progress bar for resampling (also implies `verbose`)"),
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
pub fn run_resample_subcommand<P>(arguments: &ArgMatches, provider: P, io_context: &mut IOContext)
where
    P: SnapshotProvider3,
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

    let verbosity = cli_utils::parse_verbosity(arguments, true);

    let (resample_grid_type, default_method, grid_type_arguments) =
        if let Some(regular_grid_arguments) = arguments.subcommand_matches("regular_grid") {
            (
                ResampleGridType::Regular,
                ResamplingMethod::SampleAveraging,
                regular_grid_arguments,
            )
        } else if let Some(rotated_regular_grid_arguments) =
            arguments.subcommand_matches("rotated_regular_grid")
        {
            (
                ResampleGridType::RotatedRegular,
                ResamplingMethod::SampleAveraging,
                rotated_regular_grid_arguments,
            )
        } else if let Some(reshaped_grid_arguments) = arguments.subcommand_matches("reshaped_grid")
        {
            (
                ResampleGridType::Reshaped,
                ResamplingMethod::SampleAveraging,
                reshaped_grid_arguments,
            )
        } else if let Some(mesh_file_arguments) = arguments.subcommand_matches("mesh_file") {
            (
                ResampleGridType::MeshFile,
                ResamplingMethod::SampleAveraging,
                mesh_file_arguments,
            )
        } else {
            unreachable!()
        };

    run_with_selected_method(
        grid_type_arguments,
        provider,
        resample_grid_type,
        default_method,
        &resampled_locations,
        continue_on_warnings,
        verbosity,
        io_context,
    );
}

fn run_with_selected_method<P>(
    grid_type_arguments: &ArgMatches,
    provider: P,
    resample_grid_type: ResampleGridType,
    default_method: ResamplingMethod,
    resampled_locations: &In3D<ResampledCoordLocation>,
    continue_on_warnings: bool,
    verbosity: Verbosity,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
{
    let (resampling_method, method_arguments, has_interpolator_subcommand) = if let Some(
        method_arguments,
    ) =
        grid_type_arguments.subcommand_matches("sample_averaging")
    {
        (ResamplingMethod::SampleAveraging, method_arguments, true)
    } else if let Some(method_arguments) = grid_type_arguments.subcommand_matches("cell_averaging")
    {
        (ResamplingMethod::CellAveraging, method_arguments, false)
    } else if let Some(method_arguments) = grid_type_arguments.subcommand_matches("direct_sampling")
    {
        (ResamplingMethod::DirectSampling, method_arguments, true)
    } else {
        (default_method, grid_type_arguments, false)
    };

    run_with_selected_interpolator(
        grid_type_arguments,
        method_arguments,
        provider,
        resample_grid_type,
        resampled_locations,
        resampling_method,
        has_interpolator_subcommand,
        continue_on_warnings,
        verbosity,
        io_context,
    )
}

fn run_with_selected_interpolator<P>(
    grid_type_arguments: &ArgMatches,
    arguments: &ArgMatches,
    provider: P,
    resample_grid_type: ResampleGridType,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    has_interpolator_subcommand: bool,
    continue_on_warnings: bool,
    verbosity: Verbosity,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
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
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbosity,
            interpolator,
            io_context,
        ),
        ResampleGridType::RotatedRegular => run_resampling_for_rotated_regular_grid(
            grid_type_arguments,
            arguments,
            provider,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbosity,
            interpolator,
            io_context,
        ),
        ResampleGridType::Reshaped => run_resampling_for_reshaped_grid(
            grid_type_arguments,
            arguments,
            provider,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbosity,
            interpolator,
            io_context,
        ),
        ResampleGridType::MeshFile => run_resampling_for_mesh_file(
            grid_type_arguments,
            arguments,
            provider,
            resampled_locations,
            resampling_method,
            continue_on_warnings,
            verbosity,
            interpolator,
            io_context,
        ),
    }
}

fn resample_to_grid<P, I>(
    mut grid: FieldGrid3,
    new_shape: Option<In3D<usize>>,
    arguments: &ArgMatches,
    provider: P,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    continue_on_warnings: bool,
    verbosity: Verbosity,
    interpolator: I,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
    I: Interpolator3<fdt>,
{
    if let Some(new_shape) = new_shape {
        grid = match grid.detected_grid_type() {
            GridType::Regular => create_reshaped_regular_version_of_grid(grid, new_shape),
            GridType::HorRegular => create_reshaped_hor_regular_version_of_grid(grid, new_shape),
        };
    }

    verify_new_grid_bounds(provider.grid(), &grid);

    correct_periodicity_for_new_grid(provider.grid(), &mut grid, continue_on_warnings);

    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        arguments,
        provider,
        &new_grid,
        IdentityTransformation2::new(),
        resampled_locations,
        resampling_method,
        verbosity,
        interpolator,
        io_context,
    );
}

fn resample_to_transformed_grid<P, T, I>(
    grid: FieldGrid3,
    arguments: &ArgMatches,
    provider: P,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    transformation: T,
    _continue_on_warnings: bool,
    verbosity: Verbosity,
    interpolator: I,
    io_context: &mut IOContext,
) where
    T: PointTransformation2<fgr>,
    P: SnapshotProvider3,
    I: Interpolator3<fdt>,
{
    verify_new_transformed_grid_bounds(provider.grid(), &grid, &transformation);

    let new_grid = Arc::new(grid);
    resample_snapshot_for_grid(
        arguments,
        provider,
        &new_grid,
        transformation,
        resampled_locations,
        resampling_method,
        verbosity,
        interpolator,
        io_context,
    );
}

fn create_reshaped_regular_version_of_grid(grid: FieldGrid3, new_shape: In3D<usize>) -> FieldGrid3 {
    RegularGrid3::from_bounds(
        new_shape,
        grid.lower_bounds().clone(),
        grid.upper_bounds().clone(),
        grid.periodicity().clone(),
    )
    .into()
}

fn create_reshaped_hor_regular_version_of_grid(
    grid: FieldGrid3,
    new_shape: In3D<usize>,
) -> FieldGrid3 {
    const MIN_NUMBER_OF_VERTICAL_GRID_CELLS: usize = 10; // Grid will be unstable for smaller values
    const TARGET_CONTROL_SIZE: usize = 100;

    exit_on_false!(
        new_shape[Z] >= MIN_NUMBER_OF_VERTICAL_GRID_CELLS,
        "Error: The number of grid cells in the z-direction must be at least {}\n\
         Tip: Resample to a regular grid instead",
        MIN_NUMBER_OF_VERTICAL_GRID_CELLS
    );

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
    let stride = usize::max(
        1,
        f32::floor((size_z as f32) / TARGET_CONTROL_SIZE as f32) as usize,
    );
    let mut control_coords: Vec<_> = lower_edges_z.iter().copied().step_by(stride).collect();
    let mut control_grid_cell_extents: Vec<_> = lower_edges_z
        .iter()
        .step_by(stride)
        .zip(
            lower_edges_z
                .iter()
                .skip(1)
                .chain(iter::once(&upper_bound_z))
                .step_by(stride),
        )
        .map(|(lower_edge, upper_edge)| upper_edge - lower_edge)
        .collect();
    *control_coords.last_mut().unwrap() = upper_bound_z;
    *control_grid_cell_extents.last_mut().unwrap() = upper_bound_z - lower_edges_z[size_z - 1];

    let interpolator = PolyFitInterpolator1::new(PolyFitInterpolatorConfig {
        order: 1,
        ..PolyFitInterpolatorConfig::default()
    });
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

    HorRegularGrid3::from_coords_unchecked(
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
        GridType::HorRegular,
    )
}

fn verify_new_grid_bounds(original_grid: &FieldGrid3, new_grid: &FieldGrid3) {
    exit_on_false!(
        original_grid.contains_grid(new_grid),
        "Error: The resampled grid extends outside a non-periodic boundary of the original grid"
    );
}

fn verify_new_transformed_grid_bounds<T>(
    original_grid: &FieldGrid3,
    new_grid: &FieldGrid3,
    transformation: &T,
) where
    T: PointTransformation2<fgr>,
{
    exit_on_false!(
        original_grid.contains_transformed_grid(new_grid, transformation),
        "Error: The resampled grid extends outside a non-periodic boundary of the original grid"
    );
}

fn correct_periodicity_for_new_grid(
    original_grid: &FieldGrid3,
    new_grid: &mut FieldGrid3,
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
                    if !continue_on_warnings {
                        cli_utils::verify_user_will_continue_or_abort()
                    }
                }
            }
        }
    }

    new_grid.set_periodicity(is_periodic);
}

fn compute_scaled_grid_shape(shape: &In3D<usize>, scales: &In3D<fgr>) -> In3D<usize> {
    In3D::with_each_component(|dim| {
        usize::max(1, fgr::round(scales[dim] * (shape[dim] as fgr)) as usize)
    })
}

fn resample_snapshot_for_grid<P, T, I>(
    arguments: &ArgMatches,
    provider: P,
    new_grid: &Arc<FieldGrid3>,
    transformation: T,
    resampled_locations: &In3D<ResampledCoordLocation>,
    resampling_method: ResamplingMethod,
    verbosity: Verbosity,
    interpolator: I,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
    T: PointTransformation2<fgr>,
    I: Interpolator3<fdt>,
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
        verbosity,
    );

    run_snapshot_resampling_with_derive(arguments, provider, io_context);
}

fn run_snapshot_resampling_with_derive<P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
{
    #[cfg(feature = "derivation")]
    if let Some(derive_arguments) = arguments.subcommand_matches("derive") {
        let provider = super::derive::create_derive_provider(derive_arguments, provider);
        run_snapshot_resampling_with_synthesis(derive_arguments, provider, io_context);
        return;
    }

    run_snapshot_resampling_with_synthesis_added_caching(arguments, provider, io_context);
}

fn run_snapshot_resampling_with_synthesis<P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    P: CachingSnapshotProvider3,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider =
            super::synthesize::create_synthesize_provider(synthesize_arguments, provider);
        run_snapshot_resampling_for_provider(synthesize_arguments, provider, io_context);
        return;
    }

    run_snapshot_resampling_for_provider(arguments, provider, io_context);
}

fn run_snapshot_resampling_with_synthesis_added_caching<P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider = super::synthesize::create_synthesize_provider_added_caching(
            synthesize_arguments,
            provider,
        );
        run_snapshot_resampling_for_provider(synthesize_arguments, provider, io_context);
        return;
    }

    run_snapshot_resampling_for_provider(arguments, provider, io_context);
}

fn run_snapshot_resampling_for_provider<P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    P: SnapshotProvider3,
{
    if let Some(write_arguments) = arguments.subcommand_matches("write") {
        run_write_subcommand(write_arguments, provider, io_context);
    } else if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
        run_inspect_subcommand(inspect_arguments, provider, io_context);
    }
}
