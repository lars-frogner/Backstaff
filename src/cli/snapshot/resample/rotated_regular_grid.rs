//! Command line interface for resampling a snapshot to a rotated regular grid.

use super::{
    cell_averaging::create_cell_averaging_subcommand,
    direct_sampling::create_direct_sampling_subcommand,
    sample_averaging::create_sample_averaging_subcommand,
};
use crate::{
    add_subcommand_combinations,
    cli::{
        snapshot::{
            derive::create_derive_subcommand, inspect::create_inspect_subcommand,
            write::create_write_subcommand, SnapNumInRange,
        },
        utils as cli_utils,
    },
    field::{ResampledCoordLocation, ResamplingMethod},
    geometry::{
        Dim2,
        Dim3::{X, Y, Z},
        In3D, Point2, PointTransformation2, RotationAndTranslationTransformation2,
        RotationTransformation2, TranslationTransformation2, Vec3,
    },
    grid::{fgr, regular::RegularGrid3, Grid3},
    interpolation::Interpolator3,
    io::{snapshot::SnapshotProvider3, Verbose},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-rotated_regular_grid` command line subcommand.
pub fn create_rotated_regular_grid_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "rotated_regular_grid";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Resample to a regular grid rotated around the z-axis")
        .long_about(
            "Resample to a regular grid of configurable shape, bounds\n\
             and rotation around the z-axis.",
        )
        .after_help(
            "You can use a subcommand to configure the resampling method. If left unspecified,\n\
             sample averaging with the default prameters is used.",
        )
        .arg(
            Arg::new("scales")
                .short('S')
                .long("scales")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(false)
                .value_names(&["SX", "SY", "SZ"])
                .help(
                    "Scale factors for computing shape of the resampled grid based on original shape",
                )
                .takes_value(true)
                .default_value("1,1,1"),
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Exact shape of the grid to resample to [default: same as original]")
                .takes_value(true)
                .conflicts_with("scales"),
        )
        .arg(
            Arg::new("x-start")
                .long("x-start")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["X", "Y"])
                .help("Point where the x-axis of the resampled grid should start")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("x-end")
                .long("x-end")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["X", "Y"])
                .help("Point where the x-axis of the resampled grid should end")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("y-extent")
                .long("y-extent")
                .require_equals(true)
                .value_name("EXTENT")
                .allow_hyphen_values(true)
                .help("Extent of the resampled grid along its y-direction")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("z-bounds")
                .short('z')
                .long("z-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the z-coordinates of the resampled grid\n\
                     [default: same as original]",
                )
                .takes_value(true),
        )
        .subcommand(create_sample_averaging_subcommand(command_name))
        .subcommand(create_cell_averaging_subcommand(command_name))
        .subcommand(create_direct_sampling_subcommand(command_name));

    #[cfg(feature = "synthesis")]
    let command = add_subcommand_combinations!(command, command_name, true; derive, synthesize, (write, inspect));
    #[cfg(not(feature = "synthesis"))]
    let command =
        add_subcommand_combinations!(command, command_name, true; derive, (write, inspect));

    command
}

pub fn run_resampling_for_rotated_regular_grid<G, P, I>(
    root_arguments: &ArgMatches,
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
    let original_grid = provider.grid();

    let original_shape = original_grid.shape();

    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();

    let scales: Vec<fgr> =
        cli_utils::get_values_from_required_parseable_argument(root_arguments, "scales", Some(3));

    let shape: Vec<usize> = if scales.iter().all(|&scale| scale == 1.0) {
        cli_utils::get_values_from_parseable_argument_with_custom_defaults(
            root_arguments,
            "shape",
            &|| vec![original_shape[X], original_shape[Y], original_shape[Z]],
            Some(3),
        )
    } else {
        super::compute_scaled_grid_shape(original_shape, &scales)
    };

    exit_on_false!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "Error: Grid size must be larger than zero in every dimension"
    );

    let x_start = cli_utils::get_values_from_required_parseable_argument::<fgr>(
        root_arguments,
        "x-start",
        Some(2),
    );
    let x_end = cli_utils::get_values_from_required_parseable_argument::<fgr>(
        root_arguments,
        "x-end",
        Some(2),
    );

    let x_start = Point2::new(x_start[0], x_start[1]);
    let x_end = Point2::new(x_end[0], x_end[1]);

    let x_axis = x_end - &x_start;
    let x_extent = x_axis.length();
    exit_on_false!(
        x_extent > 0.0,
        "Error: Extent in x-direction must be larger than zero"
    );

    let y_extent =
        cli_utils::get_value_from_required_parseable_argument::<fgr>(root_arguments, "y-extent");
    exit_on_false!(
        y_extent > 0.0,
        "Error: Extent in y-direction must be larger than zero"
    );

    let z_bounds = cli_utils::get_values_from_parseable_argument_with_custom_defaults(
        root_arguments,
        "z-bounds",
        &|| vec![original_lower_bounds[Z], original_upper_bounds[Z]],
        Some(2),
    );
    exit_on_false!(
        z_bounds[1] > z_bounds[0],
        "Error: Upper bound for z must exceed lower bound"
    );

    let transformation = RotationAndTranslationTransformation2::new(
        RotationTransformation2::new_from_rotated_vec(x_axis),
        TranslationTransformation2::new(x_start.to_vec2()),
    );

    let new_lower_bounds = Vec3::new(0.0, 0.0, z_bounds[0]);
    let new_upper_bounds = Vec3::new(x_extent, y_extent, z_bounds[1]);

    if verbose.is_yes() {
        let lower_horizontal_bounds = new_lower_bounds.without_z();
        let upper_horizontal_bounds = new_upper_bounds.without_z();
        let corner_1 = lower_horizontal_bounds.to_point2();
        let corner_2 = Point2::new(
            upper_horizontal_bounds[Dim2::X],
            lower_horizontal_bounds[Dim2::Y],
        );
        let corner_3 = upper_horizontal_bounds.to_point2();
        let corner_4 = Point2::new(
            lower_horizontal_bounds[Dim2::X],
            upper_horizontal_bounds[Dim2::Y],
        );
        println!(
            "Corners of resampled grid:\n\
             {:5.1} -- {:5.1}\n\
             |                              |\n\
             {:5.1} -- {:5.1}",
            transformation.transform(&corner_4),
            transformation.transform(&corner_3),
            transformation.transform(&corner_1),
            transformation.transform(&corner_2)
        );
    }

    let grid = RegularGrid3::from_bounds(
        In3D::new(shape[0], shape[1], shape[2]),
        new_lower_bounds,
        new_upper_bounds,
        In3D::same(false), // Discard periodicity information
    );
    super::resample_to_transformed_regular_grid(
        grid,
        arguments,
        provider,
        snap_num_in_range,
        resampled_locations,
        resampling_method,
        transformation,
        continue_on_warnings,
        verbose,
        interpolator,
        protected_file_types,
    );
}
