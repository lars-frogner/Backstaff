//! Command line interface for producing seed points distributed
//! according to values in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::{
    cli::utils,
    exit_on_error,
    geometry::Point2,
    grid::Grid3,
    interpolation::Interpolator3,
    io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
    tracing::seeding::slice::SliceSeeder3,
};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the slice PDF seeder.
pub fn create_value_pdf_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("value_pdf")
        .about("Use the slice value PDF seeder")
        .long_about(
            "Use the slice value PDF seeder.\n\
             Seed points are drawn from a probability density function computed from the\n\
             values within a 2D slice of a quantity field. If the quantity is a vector,\n\
             the PDF is based on the norm of the vectors.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("quantity")
                .short("q")
                .long("quantity")
                .require_equals(true)
                .value_name("NAME")
                .help("Quantity to compute the probability density from")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("n-points")
                .short("n")
                .long("n-points")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Number of seed points to generate")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("is-vector-quantity")
                .long("is-vector-quantity")
                .help("Treat the specified quantity as a vector quantity"),
        )
        .arg(
            Arg::with_name("power")
                .long("power")
                .require_equals(true)
                .value_name("VALUE")
                .help("Power of the quantity value to use for computing the probability density")
                .takes_value(true)
                .default_value("1.0"),
        )
}

/// Creates a slice PDF seeder based on the provided arguments.
pub fn create_slice_pdf_seeder_from_arguments<G, R, I, S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: &I,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    I: Interpolator3,
    S: Fn(&Point2<fdt>) -> bool + Sync,
{
    let quantity = arguments
        .value_of("quantity")
        .expect("No value for required argument");
    let n_seeds = utils::get_value_from_required_parseable_argument::<usize>(arguments, "n-points");
    let power = utils::get_value_from_required_parseable_argument::<fdt>(arguments, "power");

    if arguments.is_present("is-vector-quantity") {
        let field = exit_on_error!(
            snapshot.obtain_vector_field(quantity),
            "Error: Could not read quantity {0} in snapshot: {1}",
            quantity
        );
        let seeder = SliceSeeder3::vector_field_pdf(
            field,
            interpolator,
            parameters.axis,
            parameters.coord,
            &|vector| vector.length().powf(power),
            n_seeds,
            satisfies_constraints,
        );
        snapshot.drop_vector_field(quantity);
        seeder
    } else {
        let field = exit_on_error!(
            snapshot.obtain_scalar_field(quantity),
            "Error: Could not read quantity {0} in snapshot: {1}",
            quantity
        );
        let seeder = SliceSeeder3::scalar_field_pdf(
            field,
            interpolator,
            parameters.axis,
            parameters.coord,
            &|value| value.powf(power),
            n_seeds,
            satisfies_constraints,
        );
        snapshot.drop_scalar_field(quantity);
        seeder
    }
}
