//! Command line interface for creating Bifrost mesh files for horizontally regular grids.

use crate::{
    cli::utils,
    exit_on_error, exit_on_false,
    geometry::{
        Coords3,
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{self, fgr, hor_regular::HorRegularGrid3, Grid3},
    interpolation::cubic_hermite_spline::{
        BoundaryTangents, CubicHermiteSplineInterpolator, CubicHermiteSplineInterpolatorConfig,
        TangentScheme,
    },
    io::utils::IOContext,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `create_mesh-horizontally_regular` command line subcommand.
pub fn create_horizontally_regular_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "horizontally_regular";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Create a horizontally regular mesh")
        .long_about(
            "Create a horizontally regular mesh.\n\
             Cell extents in z-direction can be prescribed at the boundaries and at specified\n\
             interior control points. A cubic Hermite spline will be fitted to the control\n\
             points and used to obtain dz(z). Note that the specified dz scales are not\n\
             absolute, they only describe the relative magnitudes of dz.",
        )
        .arg(
            Arg::new("shape")
                .short('s')
                .long("shape")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the mesh")
                .takes_value(true)
                .number_of_values(3)
                .required(true),
        )
        .arg(
            Arg::new("x-bounds")
                .short('x')
                .long("x-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the x-coordinates")
                .takes_value(true)
                .number_of_values(2)
                .required(true),
        )
        .arg(
            Arg::new("y-bounds")
                .short('y')
                .long("y-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the y-coordinates")
                .takes_value(true)
                .number_of_values(2)
                .required(true),
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
                .help("Lower and upper bound for the z-coordinates")
                .takes_value(true)
                .number_of_values(2)
                .required(true),
        )
        .arg(
            Arg::new("boundary-dz-scales")
                .long("boundary-dz-scales")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Relative magnitudes of dz at the lower and upper z-boundaries\n")
                .takes_value(true)
                .number_of_values(2)
                .default_value("1.0,1.0"),
        )
        .arg(
            Arg::new("interior-z")
                .long("interior-z")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_name("COORDS")
                .help(
                    "Control z-coordinates where relative magnitudes of dz are specified\n \
                     [default: no control points]",
                )
                .takes_value(true)
                .requires("interior-dz-scales"),
        )
        .arg(
            Arg::new("interior-dz-scales")
                .long("interior-dz-scales")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_name("VALUES")
                .help(
                    "Relative magnitudes of dz at the specified control z-coordinates\n \
                     [default: no control points]",
                )
                .takes_value(true)
                .requires("interior-z"),
        )
}

/// Runs the actions for the `create_mesh-horizontally_regular` subcommand using the given arguments.
pub fn run_horizontally_regular_subcommand(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    io_context: &mut IOContext,
) {
    let shape = utils::parse_3d_values_no_special(arguments, "shape", Some(1));

    const MIN_NUMBER_OF_VERTICAL_GRID_CELLS: usize = 10; // Grid will be unstable for smaller values
    exit_on_false!(
        shape[Z] >= MIN_NUMBER_OF_VERTICAL_GRID_CELLS,
        "Error: The number of grid cells in the z-direction must be at least {}\n\
         Tip: Create a regular grid instead",
        MIN_NUMBER_OF_VERTICAL_GRID_CELLS
    );

    let x_bounds = utils::parse_limits(
        arguments,
        "x-bounds",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );
    let y_bounds = utils::parse_limits(
        arguments,
        "y-bounds",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );
    let z_bounds = utils::parse_limits(
        arguments,
        "z-bounds",
        utils::AllowSameValue::No,
        utils::AllowInfinity::No,
        None,
    );
    let boundary_dz_scales = utils::get_finite_float_values_from_required_parseable_argument(
        arguments,
        "boundary-dz-scales",
    );
    exit_on_false!(
        boundary_dz_scales[0] > 0.0 && boundary_dz_scales[1] > 0.0,
        "Error: Boundary dz scales must be larger than zero"
    );

    let interior_z: Vec<fgr> =
        utils::get_finite_float_values_from_parseable_argument_with_custom_defaults(
            arguments,
            "interior-z",
            &Vec::new,
        );

    let interior_dz_scales: Vec<fgr> =
        utils::get_finite_float_values_from_parseable_argument_with_custom_defaults(
            arguments,
            "interior-dz-scales",
            &Vec::new,
        );

    exit_on_false!(
        interior_z.len() == interior_dz_scales.len(),
        "Error: Number of interior control z-coordinates and dz scales must match"
    );

    exit_on_false!(
        interior_z.iter().all(|&z| z > z_bounds.0 && z < z_bounds.1),
        "Error: All interior control z-coordinates must be inside z-boundaries"
    );

    exit_on_false!(
        interior_dz_scales.iter().all(|&dz_scale| dz_scale > 0.0),
        "Error: All interior dz scales must be larger than zero"
    );

    let (centers_x, lower_edges_x) =
        grid::regular_coords_from_bounds(shape[X], x_bounds.0, x_bounds.1);
    let (centers_y, lower_edges_y) =
        grid::regular_coords_from_bounds(shape[Y], y_bounds.0, y_bounds.1);

    let mut coords_and_scales: Vec<_> = interior_z
        .into_iter()
        .zip(interior_dz_scales.into_iter())
        .collect();
    coords_and_scales.sort_by(|&(a, _), &(b, _)| a.partial_cmp(&b).unwrap());
    let (mut interior_z, mut interior_dz_scales): (Vec<_>, Vec<_>) =
        coords_and_scales.into_iter().unzip();

    let mut control_z = Vec::with_capacity(2 + interior_z.len());
    control_z.push(z_bounds.0);
    control_z.append(&mut interior_z);
    control_z.push(z_bounds.1);

    let mut dz_scales = Vec::with_capacity(2 + interior_dz_scales.len());
    dz_scales.push(boundary_dz_scales[0]);
    dz_scales.append(&mut interior_dz_scales);
    dz_scales.push(boundary_dz_scales[1]);

    let interpolator = CubicHermiteSplineInterpolator::new(CubicHermiteSplineInterpolatorConfig {
        tangent_scheme: TangentScheme::ProximityWeightedFiniteDifference,
        boundary_tangents: BoundaryTangents::Zero,
    });

    let (centers_z, lower_edges_z) = exit_on_error!(
        grid::create_new_grid_coords_from_control_extents(
            shape[Z],
            z_bounds.0,
            z_bounds.1,
            &control_z,
            &dz_scales,
            &interpolator,
        ),
        "Error: Could not compute new z-coordinates for grid: {}"
    );

    let derivatives_x = grid::compute_regular_derivatives(
        shape[X],
        grid::cell_extent_from_bounds(shape[X], x_bounds.0, x_bounds.1),
    );
    let derivatives_y = grid::compute_regular_derivatives(
        shape[Y],
        grid::cell_extent_from_bounds(shape[Y], y_bounds.0, y_bounds.1),
    );
    let (up_derivatives_z, down_derivatives_z) = grid::compute_up_and_down_derivatives(&centers_z);

    let grid = HorRegularGrid3::from_coords(
        Coords3::new(centers_x, centers_y, centers_z),
        Coords3::new(lower_edges_x, lower_edges_y, lower_edges_z),
        In3D::same(false),
        Some(Coords3::new(
            derivatives_x.clone(),
            derivatives_y.clone(),
            up_derivatives_z,
        )),
        Some(Coords3::new(
            derivatives_x,
            derivatives_y,
            down_derivatives_z,
        )),
    );

    super::write_mesh_file(root_arguments, grid, io_context);
}
