//! Command line interface for producing seed points distributed
//! according to values in a volume of a 3D grid.

use crate::{
    cli::utils,
    exit_on_error,
    field::CachingScalarFieldProvider3,
    geometry::{In3D, Point3, Vec3},
    grid::{fgr, regular::RegularGrid3, Grid3},
    interpolation::Interpolator3,
    io::snapshot::fdt,
    seeding::volume::VolumeSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the volume PDF seeder.
pub fn create_value_pdf_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "value_pdf";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the value PDF seeder")
        .long_about(
            "Use the value PDF seeder.\n\
             Seed points are drawn from a probability density function computed from the\n\
             values within a volume of a quantity field. If the quantity is a vector,\n\
             the PDF is based on the norm of the vectors.",
        )
        .arg(
            Arg::new("quantity")
                .short('q')
                .long("quantity")
                .require_equals(true)
                .value_name("NAME")
                .help("Quantity to compute the probability density from")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("n-points")
                .short('n')
                .long("n-points")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Number of seed points to generate")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("is-vector-quantity")
                .long("is-vector-quantity")
                .help("Treat the specified quantity as a vector quantity"),
        )
        .arg(
            Arg::new("power")
                .long("power")
                .require_equals(true)
                .value_name("VALUE")
                .help("Power of the quantity value to use for computing the probability density")
                .takes_value(true)
                .default_value("1.0"),
        )
}

/// Creates a volume PDF seeder based on the provided arguments.
pub fn create_volume_pdf_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    lower_bounds: Vec3<fgr>,
    upper_bounds: Vec3<fgr>,
    snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
    satisfies_constraints: &S,
) -> VolumeSeeder3
where
    S: Fn(&Point3<fgr>) -> bool + Sync,
{
    let quantity = arguments
        .value_of("quantity")
        .expect("No value for required argument")
        .to_lowercase();

    let n_seeds = utils::get_value_from_required_parseable_argument::<usize>(arguments, "n-points");
    let power =
        utils::get_finite_float_value_from_required_parseable_argument::<fdt>(arguments, "power");

    let grid_cell_extents = snapshot.grid().average_grid_cell_extents();
    let new_shape = In3D::with_each_component(|dim| {
        ((upper_bounds[dim] - lower_bounds[dim]) / grid_cell_extents[dim]).round() as usize
    });
    let grid = RegularGrid3::from_bounds(new_shape, lower_bounds, upper_bounds, In3D::same(false));

    if arguments.is_present("is-vector-quantity") {
        let field = exit_on_error!(
            snapshot.provide_vector_field(&quantity),
            "Error: Could not read quantity {0} in snapshot: {1}",
            &quantity
        );
        let seeder = VolumeSeeder3::vector_field_pdf(
            &grid,
            field.as_ref(),
            interpolator,
            &|vector| vector.length().powf(power),
            n_seeds,
            satisfies_constraints,
        );
        seeder
    } else {
        let field = exit_on_error!(
            snapshot.provide_scalar_field(&quantity),
            "Error: Could not read quantity {0} in snapshot: {1}",
            &quantity
        );
        let seeder = VolumeSeeder3::scalar_field_pdf(
            &grid,
            field.as_ref(),
            interpolator,
            &|value| value.powf(power),
            n_seeds,
            satisfies_constraints,
        );
        seeder
    }
}
