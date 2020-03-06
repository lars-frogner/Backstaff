//! Command line interface for computing auxiliary quantities for a snapshot.

use crate::field;
use crate::grid::Grid3;
use crate::io::snapshot::{self, fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Builds a representation of the `snapshot-modify_aux-compute` command line subcommand.
pub fn create_compute_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("compute")
        .about("Compute new auxiliary quantities")
        .long_about(
            "Compute new auxiliary quantities.\n\
             Available quantities:\n\
             ubeam - Volume integrated beam heating [energy/time in Bifrost units]",
        )
        .arg(
            Arg::with_name("quantities")
                .long("quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help("List of new quantities to include in the snapshot (comma-separated)")
                .required(true)
                .takes_value(true)
                .multiple(true),
        )
        .help_message("Print help information")
}

/// Generates and writes an aux file with new quantities
pub fn generate_auxiliary_variable_file<G: Grid3<fdt>, P: AsRef<Path>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
    output_path: P,
    is_verbose: bool,
) -> Vec<String> {
    let mut required_field_names = HashMap::new();
    required_field_names.insert("ubeam", vec!["qbeam"]);

    let quantity_names = arguments
        .values_of("quantities")
        .expect("No value for required argument.")
        .collect::<Vec<_>>();

    for &name in &quantity_names {
        if reader.has_variable(name) {
            panic!("Quantity {} is already present in the snapshot", name)
        }
        for &required_field_name in required_field_names
            .get(name)
            .unwrap_or_else(|| panic!("Quantity {} is not supported", name))
        {
            if !reader.has_variable(required_field_name) {
                panic!(
                    "Quantity {} requires missing variable {}",
                    name, required_field_name
                )
            }
        }
    }

    let auxiliary_quantity_names: Vec<_> = reader
        .auxiliary_variable_names()
        .to_vec()
        .into_iter()
        .chain(quantity_names.iter().map(|&name| name.to_string()))
        .collect();

    let variable_value_producer = |name: &str| match name {
        "ubeam" => compute_beam_power_change(reader, is_verbose),
        name => reader
            .read_scalar_field(name)
            .unwrap_or_else(|err| panic!("Could not read quantity field {}: {}", name, err))
            .into_values(),
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

    auxiliary_quantity_names
}

fn compute_beam_power_change<G: Grid3<fdt>>(
    reader: &SnapshotReader3<G>,
    is_verbose: bool,
) -> Array3<fdt> {
    if is_verbose {
        println!("Computing ubeam");
    }
    let grid = reader.grid();
    let grid_shape = grid.shape();

    let mut ubeam_values = reader
        .read_scalar_field("qbeam")
        .unwrap_or_else(|err| panic!("Could not read quantity field qbeam: {}", err))
        .into_values();

    let ubeam_values_buffer = ubeam_values.as_slice_memory_order_mut().unwrap();

    ubeam_values_buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| {
            let indices = field::compute_3d_array_indices_from_flat_idx(&grid_shape, idx);
            *value *= grid.grid_cell_volume(&indices);
        });

    ubeam_values
}
