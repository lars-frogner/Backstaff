//! Command line interface for producing seed points distributed
//! according to values in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::{
    cli::utils, exit_on_error, field::CachingScalarFieldProvider3, geometry::Point2, grid::fgr,
    interpolation::Interpolator3, io::snapshot::fdt, seeding::slice::SliceSeeder3,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the slice PDF seeder.
pub fn create_value_pdf_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "value_pdf";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the slice value PDF seeder")
        .long_about(
            "Use the slice value PDF seeder.\n\
             Seed points are drawn from a probability density function computed from the\n\
             values within a 2D slice of a quantity field. If the quantity is a vector,\n\
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

/// Creates a slice PDF seeder based on the provided arguments.
pub fn create_slice_pdf_seeder_from_arguments<S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    S: Fn(&Point2<fgr>) -> bool + Sync,
{
    let quantity = arguments
        .value_of("quantity")
        .expect("No value for required argument")
        .to_lowercase();

    let n_seeds = utils::get_value_from_required_parseable_argument::<usize>(arguments, "n-points");
    let power =
        utils::get_finite_float_value_from_required_parseable_argument::<fdt>(arguments, "power");

    if arguments.is_present("is-vector-quantity") {
        let field = exit_on_error!(
            snapshot.provide_vector_field(&quantity),
            "Error: Could not read quantity {0} in snapshot: {1}",
            &quantity
        );
        let seeder = SliceSeeder3::vector_field_pdf(
            field.as_ref(),
            interpolator,
            parameters.axis,
            parameters.coord,
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
        let seeder = SliceSeeder3::scalar_field_pdf(
            field.as_ref(),
            interpolator,
            parameters.axis,
            parameters.coord,
            &|value| value.powf(power),
            n_seeds,
            satisfies_constraints,
        );
        seeder
    }
}
