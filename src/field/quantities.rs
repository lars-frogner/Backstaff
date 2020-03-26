//! Functions for computing various derived physical quantities.

use crate::{
    field::{self, ScalarField3},
    geometry::In3D,
    grid::{CoordLocation, Grid3},
    io::{
        snapshot::{fdt, SnapshotReader3},
        Verbose,
    },
};
use lazy_static::lazy_static;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, io};

lazy_static! {
    static ref DERIVED_QUANTITIES: HashMap<&'static str, (&'static str, Vec<&'static str>)> = vec![
        (
            "ubeam",
            (
                "Volume integrated beam heating\n\
                 [energy/time in Bifrost units]",
                vec!["qbeam"]
            )
        ),
        (
            "log10_pos_qbeam",
            (
                "Log10 of beam heating, with negative and zero values set to NaN\n\
                 [energy/time/volume in Bifrost units]",
                vec!["qbeam"]
            )
        )
    ]
    .into_iter()
    .collect();
}

/// Prints an overview of available quantities and their dependencies.
pub fn print_available_quantities() {
    let mut lines: Vec<_> = DERIVED_QUANTITIES
        .iter()
        .map(|(name, (description, dependencies))| {
            format!(
                "{} - {} (requires: {})",
                name,
                description,
                dependencies.join(", ")
            )
        })
        .collect();
    lines.sort();
    println!(
        "============================= Derivable quantities =============================\n{}{}",
        lines.join(
            "\n--------------------------------------------------------------------------------\n"
        ),
        "\n--------------------------------------------------------------------------------"
    );
}

/// Whether the quantity with the given name can be computed.
pub fn quantity_supported(name: &str) -> bool {
    DERIVED_QUANTITIES.contains_key(name)
}

/// Returns a list of the dependencies for the given derived quantity that
/// are not present in the given snapshot, or `None` if the derived quantity
/// is not known.
pub fn find_missing_quantity_dependencies<G, R>(reader: &R, name: &str) -> Option<Vec<&'static str>>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let all_variable_names = reader.all_variable_names();
    DERIVED_QUANTITIES.get(name).map(|(_, dependencies)| {
        dependencies
            .iter()
            .cloned()
            .filter(|dep| !all_variable_names.contains(dep))
            .collect()
    })
}

/// Computes the derived quantity field with the given name.
pub fn compute_quantity<G, R>(
    reader: &R,
    name: &str,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    if verbose.is_yes() {
        println!("Computing {}", name);
    }
    let (values, locations) = match name {
        "ubeam" => compute_ubeam_values(reader),
        "log10_pos_qbeam" => compute_log10_pos_qbeam_values(reader),
        invalid => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Quantity {} not supported", invalid),
        )),
    }?;
    Ok(ScalarField3::new(
        name.to_string(),
        reader.arc_with_grid(),
        locations,
        values,
    ))
}

fn compute_ubeam_values<G, R>(reader: &R) -> io::Result<(Array3<fdt>, In3D<CoordLocation>)>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let grid = reader.grid();
    let grid_shape = grid.shape();

    let mut values = reader.read_scalar_field("qbeam")?.into_values();
    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| {
            let indices = field::compute_3d_array_indices_from_flat_idx(&grid_shape, idx);
            *value *= grid.grid_cell_volume(&indices);
        });

    Ok((values, In3D::same(CoordLocation::Center)))
}

fn compute_log10_pos_qbeam_values<G, R>(
    reader: &R,
) -> io::Result<(Array3<fdt>, In3D<CoordLocation>)>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let mut values = reader.read_scalar_field("qbeam")?.into_values();
    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer.par_iter_mut().for_each(|value| {
        *value = if *value <= 0.0 {
            std::f32::NAN
        } else {
            fdt::log10(*value)
        };
    });

    Ok((values, In3D::same(CoordLocation::Center)))
}
