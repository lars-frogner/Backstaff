//! Command line interface for removing auxiliary quantities from a snapshot.

use crate::grid::Grid3;
use crate::io::snapshot::{self, fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};
use std::fs::File;
use std::path::Path;

/// Builds a representation of the `snapshot-modify_aux-remove` command line subcommand.
pub fn create_remove_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("remove")
        .about("Remove auxiliary quantities")
        .long_about("Remove auxiliary quantities.")
        .arg(
            Arg::with_name("all")
                .long("all")
                .help("Remove all auxiliary quantities"),
        )
        .arg(
            Arg::with_name("quantities")
                .long("quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help("List of quantities to remove from the snapshot (comma-separated)")
                .required_unless("all")
                .takes_value(true)
                .multiple(true),
        )
        .help_message("Print help information")
}

/// Generates and writes an aux file with remaining quantities
pub fn generate_auxiliary_variable_file<G: Grid3<fdt>, P: AsRef<Path>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
    output_path: P,
    is_verbose: bool,
) -> Vec<String> {
    let remove_all = arguments.is_present("all");
    let remove_quantity_names = if remove_all {
        reader
            .auxiliary_variable_names()
            .iter()
            .map(|name| name.as_str())
            .collect()
    } else {
        arguments
            .values_of("quantities")
            .expect("No value for required argument.")
            .collect::<Vec<_>>()
    };

    for name in &remove_quantity_names {
        if !reader.has_variable(name) {
            panic!("Quantity {} is not present in the snapshot", name)
        }
    }

    let auxiliary_quantity_names: Vec<_> = reader
        .auxiliary_variable_names()
        .to_vec()
        .into_iter()
        .filter(|name| !remove_quantity_names.contains(&name.as_str()))
        .collect();

    if auxiliary_quantity_names.is_empty() {
        // Write empty file
        File::create(output_path.as_ref())
            .unwrap_or_else(|err| panic!("Could not write auxiliary variable file: {}", err));
    } else {
        let variable_value_producer = |name: &str| {
            reader
                .read_scalar_field(name)
                .unwrap_or_else(|err| panic!("Could not read quantity field {}: {}", name, err))
                .into_values()
        };
        if is_verbose {
            println!(
                "Writing updated auxiliary variable file: {}",
                output_path.as_ref().display()
            );
        }
        snapshot::write_3d_snapfile(
            output_path.as_ref(),
            &auxiliary_quantity_names,
            &variable_value_producer,
            reader.endianness(),
            is_verbose.into(),
        )
        .unwrap_or_else(|err| panic!("Could not write auxiliary variable file: {}", err));
    }

    auxiliary_quantity_names
}
