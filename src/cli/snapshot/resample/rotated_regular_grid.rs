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
        Dim3::{X, Y, Z},
        In3D, Point2, RotationAndTranslationTransformation2, RotationTransformation2,
        SimplePolygon2, TranslationTransformation2, Vec3,
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
                .number_of_values(3),
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the resampled grid (`auto` approximates original cell extent)")
                .takes_value(true)
                .number_of_values(3)
                .default_value("auto,auto,auto")
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
                .takes_value(true)
                .number_of_values(2),
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
                .takes_value(true)
                .number_of_values(2),
        )
        .arg(
            Arg::new("y-extent")
                .short('Y')
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
                .help("Limits for the z-coordinates of the resampled grid")
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
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

    let mut original_average_grid_cell_extents = original_grid.average_grid_cell_extents();
    let original_average_hor_grid_cell_extent =
        0.5 * (original_average_grid_cell_extents[X] + original_average_grid_cell_extents[Y]);
    original_average_grid_cell_extents[X] = original_average_hor_grid_cell_extent;
    original_average_grid_cell_extents[Y] = original_average_hor_grid_cell_extent;

    let x_start = cli_utils::get_finite_float_values_from_required_parseable_argument::<fgr>(
        root_arguments,
        "x-start",
    );
    let x_end = cli_utils::get_finite_float_values_from_required_parseable_argument::<fgr>(
        root_arguments,
        "x-end",
    );

    let x_start = Point2::with_each_component(|dim| x_start[dim.num()]);
    let x_end = Point2::with_each_component(|dim| x_end[dim.num()]);

    let x_axis = x_end - &x_start;
    let x_extent = x_axis.length();
    exit_on_false!(
        x_extent > 0.0,
        "Error: Extent in x-direction must be larger than zero"
    );

    let y_extent = cli_utils::get_finite_float_value_from_required_parseable_argument::<fgr>(
        root_arguments,
        "y-extent",
    );
    exit_on_false!(
        y_extent > 0.0,
        "Error: Extent in y-direction must be larger than zero"
    );

    let z_bounds = cli_utils::parse_limits_with_min_max(
        root_arguments,
        "z-bounds",
        cli_utils::AllowSameValue::No,
        cli_utils::AllowInfinity::No,
        original_lower_bounds[Z],
        original_upper_bounds[Z],
    );

    let transformation = RotationAndTranslationTransformation2::new(
        RotationTransformation2::new_from_rotated_vec(x_axis),
        TranslationTransformation2::new(x_start.to_vec2()),
    );

    let new_lower_bounds = Vec3::new(0.0, 0.0, z_bounds.0);
    let new_upper_bounds = Vec3::new(x_extent, y_extent, z_bounds.1);
    let new_extents = &new_upper_bounds - &new_lower_bounds;

    let scales: Vec<fgr> =
        cli_utils::get_finite_float_values_from_parseable_argument_with_custom_defaults(
            root_arguments,
            "scales",
            &|| vec![1.0, 1.0, 1.0],
        );

    let shape = if scales.iter().all(|&scale| scale == 1.0) {
        cli_utils::parse_3d_values(root_arguments, "shape", Some(1), |dim, value_string| {
            match value_string {
                "same" => Some(original_shape[dim]),
                "auto" => Some(
                    fgr::ceil(new_extents[dim] / original_average_grid_cell_extents[dim]) as usize,
                ),
                _ => None,
            }
        })
    } else {
        super::compute_scaled_grid_shape(original_shape, &scales)
    };

    if verbose.is_yes() {
        let hor_bound_polygon =
            SimplePolygon2::rectangle_from_horizontal_bounds(&new_lower_bounds, &new_upper_bounds)
                .transformed(&transformation);
        let corners = hor_bound_polygon.vertices();
        println!(
            "Corners of resampled grid:\n\
             {:5.1} -- {:5.1}\n\
             |                              |\n\
             {:5.1} -- {:5.1}",
            corners[0], corners[1], corners[2], corners[3]
        );
    }

    let grid = RegularGrid3::from_bounds(
        shape,
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
