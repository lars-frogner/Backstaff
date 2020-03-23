//! Command line interface for printing statistics for quantities in a snapshot.

use crate::{
    exit_on_error,
    field::ScalarField3,
    geometry::Dim3,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
};
use clap::{App, Arg, ArgMatches, SubCommand};
use rayon::prelude::*;
use Dim3::{X, Y, Z};

/// Builds a representation of the `snapshot-inspect-statistics` command line subcommand.
pub fn create_statistics_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("statistics")
        .about("Print statistics for quantities in the snapshot")
        .help_message("Print help information")
        .arg(
            Arg::with_name("quantities")
                .short("q")
                .long("quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help("List of quantities to print statistics for (comma-separated)")
                .required(true)
                .takes_value(true)
                .multiple(true)
                .min_values(1),
        )
}

/// Runs the actions for the `snapshot-inspect-statistics` subcommand using the given arguments.
pub fn run_statistics_subcommand<G, R>(arguments: &ArgMatches, snapshot: &mut SnapshotCacher3<G, R>)
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    for name in arguments
        .values_of("quantities")
        .expect("No values for required argument.")
    {
        if !snapshot.reader().has_variable(name) {
            eprintln!("Warning: Quantity {} not present in snapshot", name);
        } else {
            exit_on_error!(
                snapshot
                    .obtain_scalar_field(name)
                    .map(|field| print_statistics_report(&field)),
                "Error: Could not read quantity {0} in snapshot: {1}",
                name
            );
        }
    }
}

fn print_statistics_report<G: Grid3<fdt>>(field: &ScalarField3<fdt, G>) {
    println!(
        "*************** Statistics for {} ***************",
        field.name()
    );

    let coords = field.coords();
    let values = field.values();
    println!("Number of values: {}", values.len());

    let number_of_nans = values.par_iter().filter(|value| value.is_nan()).count();
    println!("Number of NaNs:   {}", number_of_nans);

    match field.find_minimum() {
        Some((min_indices, min_value)) => {
            let min_point = coords.point(&min_indices);
            println!(
                "Minimum value:    {} at [{}, {}, {}] = ({}, {}, {})",
                min_value,
                min_indices[X],
                min_indices[Y],
                min_indices[Z],
                min_point[X],
                min_point[Y],
                min_point[Z]
            );
        }
        None => println!("Minimum value:    N/A"),
    }

    match field.find_maximum() {
        Some((max_indices, max_value)) => {
            let max_point = coords.point(&max_indices);
            println!(
                "Maximum value:    {} at [{}, {}, {}] = ({}, {}, {})",
                max_value,
                max_indices[X],
                max_indices[Y],
                max_indices[Z],
                max_point[X],
                max_point[Y],
                max_point[Z]
            );
        }
        None => println!("Maximum value:    N/A"),
    }

    match values.mean() {
        Some(value) => println!("Average value:    {}", value),
        None => println!("Average value:    N/A"),
    };
}
