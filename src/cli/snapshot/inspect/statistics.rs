//! Command line interface for printing statistics for quantities in a snapshot.

use crate::field::ScalarField3;
use crate::geometry::Dim3;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use clap::{App, Arg, ArgMatches, SubCommand};
use rayon::prelude::*;
use Dim3::{X, Y, Z};

/// Builds a representation of the `snapshot-inspect-statistics` command line subcommand.
pub fn build_subcommand_statistics<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("statistics")
        .about("Print statistics for quantities in the snapshot")
        .arg(
            Arg::with_name("QUANTITIES")
                .help("List of quantities to print statistics for")
                .required(true)
                .takes_value(true)
                .index(1)
                .multiple(true)
                .min_values(1),
        )
}

/// Runs the actions for the `snapshot-inspect-statistics` subcommand using the given arguments.
pub fn run_subcommand_statistics<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
) {
    for quantity in arguments
        .values_of("QUANTITIES")
        .expect("No values for required argument")
    {
        match snapshot.obtain_scalar_field(quantity) {
            Ok(field) => print_statistics_report(&field),
            Err(err) => println!("Could not read {}: {}", quantity, err),
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
