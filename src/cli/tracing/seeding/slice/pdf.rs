//! Command line interface for producing seed points distributed
//! according to values in a 2D slice of a 3D grid.

use super::CommonSliceSeederParameters;
use crate::cli;
use crate::geometry::Point2;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::tracing::seeding::slice::SliceSeeder3;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the slice PDF seeder.
pub fn create_slice_pdf_seeder_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("value_pdf")
        .about("Use the slice value PDF seeder")
        .long_about(
            "Use the slice value PDF seeder.\n\
             Seed points are drawn from a probability density function computed from the\n\
             values within a 2D slice of a quantity field. If the quantity is a vector,\n\
             the PDF is based on the norm of the vectors.",
        );
    add_slice_pdf_seeder_options_to_subcommand(app)
}

/// Adds arguments for parameters used by the slice PDF seeder.
pub fn add_slice_pdf_seeder_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("QUANTITY")
            .help("Quantity to compute the probability density from")
            .required(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("POINTS")
            .help("Number of seed points to generate")
            .required(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("vector-quantity")
            .long("vector-quantity")
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
pub fn create_slice_pdf_seeder_from_arguments<G, I, S>(
    arguments: &ArgMatches,
    parameters: &CommonSliceSeederParameters,
    snapshot: &mut SnapshotCacher3<G>,
    interpolator: &I,
    satisfies_constraints: &S,
) -> SliceSeeder3
where
    G: Grid3<fdt>,
    I: Interpolator3,
    S: Fn(&Point2<fdt>) -> bool + Sync,
{
    let quantity = arguments
        .value_of("QUANTITY")
        .expect("No value for required argument.");
    let n_seeds = cli::get_value_from_required_parseable_argument::<usize>(arguments, "POINTS");
    let power = cli::get_value_from_required_parseable_argument::<fdt>(arguments, "power");

    if arguments.is_present("vector-quantity") {
        let field = snapshot
            .obtain_vector_field(quantity)
            .unwrap_or_else(|err| panic!("Could not read {} from snapshot: {}", quantity, err));
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
        let field = snapshot
            .obtain_scalar_field(quantity)
            .unwrap_or_else(|err| panic!("Could not read {} from snapshot: {}", quantity, err));
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
