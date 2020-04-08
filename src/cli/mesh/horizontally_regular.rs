//! Command line interface for creating Bifrost mesh files for horizontally regular grids.

use crate::{
    cli::utils,
    exit_on_false,
    geometry::{Coords3, In3D},
    grid::{self, hor_regular::HorRegularGrid3, Grid3},
    interpolation::cubic_hermite_spline::{
        BoundaryTangents, CubicHermiteSplineInterpolator, CubicHermiteSplineInterpolatorConfig,
        TangentScheme,
    },
    io::snapshot::fdt,
};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `create_mesh-horizontally_regular` command line subcommand.
pub fn create_horizontally_regular_mesh_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("horizontally_regular")
        .about("Create a horizontally regular grid")
        .long_about(
            "Create a horizontally regular grid.\n\
             Cell extents in z-direction can be prescribed at the boundaries and at specified\n\
             interior control points. A cubic Hermite spline will be fitted to the control\n\
             points and used to obtain dz(z). Note that the specified dz scales are not\n\
             absolute, they only describe the relative magnitudes of dz.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("shape")
                .short("s")
                .long("shape")
                .require_equals(true)
                .require_delimiter(true)
                .value_names(&["NX", "NY", "NZ"])
                .help("Shape of the grid")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("x-bounds")
                .short("x")
                .long("x-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the x-coordinates")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("y-bounds")
                .short("y")
                .long("y-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the y-coordinates")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("z-bounds")
                .short("z")
                .long("z-bounds")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Lower and upper bound for the z-coordinates")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("boundary-dz-scales")
                .long("boundary-dz-scales")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help("Relative magnitudes of dz at the lower and upper z-boundaries\n")
                .takes_value(true)
                .default_value("1.0,1.0"),
        )
        .arg(
            Arg::with_name("interior-z")
                .long("interior-z")
                .require_equals(true)
                .require_delimiter(true)
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
            Arg::with_name("interior-dz-scales")
                .long("interior-dz-scales")
                .require_equals(true)
                .require_delimiter(true)
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
    protected_file_types: &[&str],
) {
    let shape = utils::get_values_from_required_parseable_argument(arguments, "shape");
    exit_on_false!(
        shape[0] >= 8 && shape[1] >= 8 && shape[2] >= 8,
        "Error: All grid dimensions must have a size of at least 8"
    );

    let bounds_x = utils::get_values_from_required_parseable_argument(arguments, "x-bounds");
    exit_on_false!(
        bounds_x[1] > bounds_x[0],
        "Error: Upper bound on x must be larger than lower bound"
    );
    let bounds_y = utils::get_values_from_required_parseable_argument(arguments, "y-bounds");
    exit_on_false!(
        bounds_y[1] > bounds_y[0],
        "Error: Upper bound on y must be larger than lower bound"
    );
    let bounds_z = utils::get_values_from_required_parseable_argument(arguments, "z-bounds");
    exit_on_false!(
        bounds_z[1] > bounds_z[0],
        "Error: Upper bound on z must be larger than lower bound"
    );
    let boundary_dz_scales =
        utils::get_values_from_required_parseable_argument(arguments, "boundary-dz-scales");
    exit_on_false!(
        boundary_dz_scales[0] > 0.0 && boundary_dz_scales[1] > 0.0,
        "Error: Boundary dz scales must be larger than zero"
    );

    let interior_z: Vec<fdt> = utils::get_values_from_parseable_argument_with_custom_defaults(
        arguments,
        "interior-z",
        &|| Vec::new(),
    );

    let interior_dz_scales: Vec<fdt> =
        utils::get_values_from_parseable_argument_with_custom_defaults(
            arguments,
            "interior-dz-scales",
            &|| Vec::new(),
        );

    exit_on_false!(
        interior_z.len() == interior_dz_scales.len(),
        "Error: Number of interior control z-coordinates and dz scales must match"
    );

    exit_on_false!(
        interior_z
            .iter()
            .all(|&z| z > bounds_z[0] && z < bounds_z[1]),
        "Error: All interior control z-coordinates must be inside z-boundaries"
    );

    exit_on_false!(
        interior_dz_scales.iter().all(|&dz_scale| dz_scale > 0.0),
        "Error: All interior dz scales must be larger than zero"
    );

    let (centers_x, lower_edges_x) =
        grid::regular_coords_from_bounds(shape[0], bounds_x[0], bounds_x[1]);
    let (centers_y, lower_edges_y) =
        grid::regular_coords_from_bounds(shape[1], bounds_y[0], bounds_y[1]);

    let mut coords_and_scales: Vec<_> = interior_z
        .into_iter()
        .zip(interior_dz_scales.into_iter())
        .collect();
    coords_and_scales.sort_by(|&(a, _), &(b, _)| a.partial_cmp(&b).unwrap());
    let (mut interior_z, mut interior_dz_scales): (Vec<_>, Vec<_>) =
        coords_and_scales.into_iter().unzip();

    let mut control_z = Vec::with_capacity(2 + interior_z.len());
    control_z.push(bounds_z[0]);
    control_z.append(&mut interior_z);
    control_z.push(bounds_z[1]);

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
            shape[2],
            bounds_z[0],
            bounds_z[1],
            &control_z,
            &dz_scales,
            &interpolator,
        ),
        "Error: Could not compute new z-coordinates for grid: {}"
    );

    let (up_derivatives_x, down_derivatives_x) = grid::compute_up_and_down_derivatives(&centers_x);
    let (up_derivatives_y, down_derivatives_y) = grid::compute_up_and_down_derivatives(&centers_y);
    let (up_derivatives_z, down_derivatives_z) = grid::compute_up_and_down_derivatives(&centers_z);

    let grid = HorRegularGrid3::from_coords(
        Coords3::new(centers_x, centers_y, centers_z),
        Coords3::new(lower_edges_x, lower_edges_y, lower_edges_z),
        In3D::same(false),
        Some(Coords3::new(
            up_derivatives_x,
            up_derivatives_y,
            up_derivatives_z,
        )),
        Some(Coords3::new(
            down_derivatives_x,
            down_derivatives_y,
            down_derivatives_z,
        )),
    );

    super::write_mesh_file(root_arguments, grid, protected_file_types);
}
