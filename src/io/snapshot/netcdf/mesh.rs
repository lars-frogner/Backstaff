//! Utilities for mesh data in NetCDF format.

use super::super::super::{Endianness, Verbose};
use crate::{
    geometry::{
        Coords3,
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{self, Grid3, GridType},
    io::snapshot::fdt,
};
use netcdf_rs::{self, File, GroupMut};
use std::{io, path::PathBuf};

macro_rules! io_result {
    ($result:expr) => {
        $result.map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))
    };
}

/// Tries to construct a grid from the data in the given NetCDF group.
pub fn read_grid<G: Grid3<fdt>>(
    file: &File,
    is_periodic: In3D<bool>,
    verbose: Verbose,
) -> io::Result<(G, Endianness)> {
    let (
        detected_grid_type,
        center_coords,
        lower_edge_coords,
        up_derivatives,
        down_derivatives,
        endianness,
    ) = read_grid_data(file, verbose)?;

    if detected_grid_type != G::TYPE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Wrong reader type for the grid in the specified NetCDF file".to_string(),
        ));
    }

    Ok((
        G::from_coords(
            center_coords,
            lower_edge_coords,
            is_periodic,
            up_derivatives,
            down_derivatives,
        ),
        endianness,
    ))
}

/// Reads the data required to construct a grid from the given NetCDF group.
pub fn read_grid_data(
    file: &File,
    verbose: Verbose,
) -> io::Result<(
    GridType,
    Coords3<fdt>,
    Coords3<fdt>,
    Option<Coords3<fdt>>,
    Option<Coords3<fdt>>,
    Endianness,
)> {
    if verbose.is_yes() {
        println!(
            "Reading grid from {}",
            PathBuf::from(file.path().unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
        );
    }
    let group = &file.root().unwrap();
    let (xm, endianness) = super::read_snapshot_1d_variable::<fdt>(group, "xm")?;
    let ym = match super::read_snapshot_1d_variable::<fdt>(group, "ym")? {
        (ym, e) if e == endianness => Ok(ym),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Inconsistent grid endianness in NetCDF file".to_string(),
        )),
    }?;
    let zm = match super::read_snapshot_1d_variable::<fdt>(group, "zm")? {
        (zm, e) if e == endianness => Ok(zm),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Inconsistent grid endianness in NetCDF file".to_string(),
        )),
    }?;
    let xmdn = match super::read_snapshot_1d_variable::<fdt>(group, "xmdn")? {
        (xmdn, e) if e == endianness => Ok(xmdn),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Inconsistent grid endianness in NetCDF file".to_string(),
        )),
    }?;
    let ymdn = match super::read_snapshot_1d_variable::<fdt>(group, "ymdn")? {
        (ymdn, e) if e == endianness => Ok(ymdn),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Inconsistent grid endianness in NetCDF file".to_string(),
        )),
    }?;
    let zmdn = match super::read_snapshot_1d_variable::<fdt>(group, "zmdn")? {
        (zmdn, e) if e == endianness => Ok(zmdn),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Inconsistent grid endianness in NetCDF file".to_string(),
        )),
    }?;

    let center_coords = Coords3::new(xm, ym, zm);
    let lower_coords = Coords3::new(xmdn, ymdn, zmdn);

    let detected_grid_type =
        grid::verify_coordinate_arrays(&center_coords, &lower_coords, verbose.is_yes())?;

    let derivative_count = group
        .variables()
        .filter(|var| {
            [
                "dxidxup", "dyidyup", "dzidzup", "dxidxdn", "dyidydn", "dzidzdn",
            ]
            .contains(&var.name().as_str())
        })
        .count();

    let (up_derivatives, down_derivatives) = if derivative_count == 6 {
        let dxidxup = match super::read_snapshot_1d_variable::<fdt>(group, "dxidxup")? {
            (dxidxup, e) if e == endianness => Ok(dxidxup),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inconsistent grid endianness in NetCDF file".to_string(),
            )),
        }?;
        let dyidyup = match super::read_snapshot_1d_variable::<fdt>(group, "dyidyup")? {
            (dyidyup, e) if e == endianness => Ok(dyidyup),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inconsistent grid endianness in NetCDF file".to_string(),
            )),
        }?;
        let dzidzup = match super::read_snapshot_1d_variable::<fdt>(group, "dzidzup")? {
            (dzidzup, e) if e == endianness => Ok(dzidzup),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inconsistent grid endianness in NetCDF file".to_string(),
            )),
        }?;
        let dxidxdn = match super::read_snapshot_1d_variable::<fdt>(group, "dxidxdn")? {
            (dxidxdn, e) if e == endianness => Ok(dxidxdn),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inconsistent grid endianness in NetCDF file".to_string(),
            )),
        }?;
        let dyidydn = match super::read_snapshot_1d_variable::<fdt>(group, "dyidydn")? {
            (dyidydn, e) if e == endianness => Ok(dyidydn),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inconsistent grid endianness in NetCDF file".to_string(),
            )),
        }?;
        let dzidzdn = match super::read_snapshot_1d_variable::<fdt>(group, "dzidzdn")? {
            (dzidzdn, e) if e == endianness => Ok(dzidzdn),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inconsistent grid endianness in NetCDF file".to_string(),
            )),
        }?;
        (
            Some(Coords3::new(dxidxup, dyidyup, dzidzup)),
            Some(Coords3::new(dxidxdn, dyidydn, dzidzdn)),
        )
    } else if derivative_count == 0 {
        (None, None)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Inconsistent number of grid derivatives in NetCDF file".to_string(),
        ));
    };

    Ok((
        detected_grid_type,
        center_coords,
        lower_coords,
        up_derivatives,
        down_derivatives,
        endianness,
    ))
}

/// Writes a representation of the given grid to the given NetCDF group.
pub fn write_grid<G: Grid3<fdt>>(
    group: &mut GroupMut,
    grid: &G,
    exclude_derivatives: bool,
) -> io::Result<()> {
    let shape = grid.shape();
    let centers = grid.centers();
    let lower_edges = grid.lower_edges();

    io_result!(group.add_dimension("xm", shape[X]))?;
    io_result!(group.add_dimension("ym", shape[Y]))?;
    io_result!(group.add_dimension("zm", shape[Z]))?;
    io_result!(group.add_dimension("xmdn", shape[X]))?;
    io_result!(group.add_dimension("ymdn", shape[Y]))?;
    io_result!(group.add_dimension("zmdn", shape[Z]))?;

    add_coordinate_variable(
        group,
        "xm",
        "xm",
        "X",
        "x-coordinates of Cartesian grid cell centers",
        Some("Mm"),
        None,
        &centers[X],
    )?;
    add_coordinate_variable(
        group,
        "ym",
        "ym",
        "Y",
        "y-coordinates of Cartesian grid cell centers",
        Some("Mm"),
        None,
        &centers[Y],
    )?;
    add_coordinate_variable(
        group,
        "zm",
        "zm",
        "Z",
        "z-coordinates of Cartesian grid cell centers",
        Some("Mm"),
        None,
        &centers[Z],
    )?;

    add_coordinate_variable(
        group,
        "xmdn",
        "xmdn",
        "X",
        "x-coordinates of Cartesian grid lower edges",
        Some("Mm"),
        None,
        &lower_edges[X],
    )?;
    add_coordinate_variable(
        group,
        "ymdn",
        "ymdn",
        "Y",
        "y-coordinates of Cartesian grid lower edges",
        Some("Mm"),
        None,
        &lower_edges[Y],
    )?;
    add_coordinate_variable(
        group,
        "zmdn",
        "zmdn",
        "Z",
        "z-coordinates of Cartesian grid lower edges",
        Some("Mm"),
        None,
        &lower_edges[Z],
    )?;

    if !exclude_derivatives {
        if let Some(up_derivatives) = grid.up_derivatives() {
            add_coordinate_variable(
                group,
                "dxidxup",
                "xm",
                "X",
                "Upward derivatives of Cartesian grid x-coordinates",
                None,
                None,
                &up_derivatives[X],
            )?;
            add_coordinate_variable(
                group,
                "dyidyup",
                "ym",
                "Y",
                "Upward derivatives of Cartesian grid y-coordinates",
                None,
                None,
                &up_derivatives[Y],
            )?;
            add_coordinate_variable(
                group,
                "dzidzup",
                "zm",
                "Z",
                "Upward derivatives of Cartesian grid z-coordinates",
                None,
                None,
                &up_derivatives[Z],
            )?;
        }

        if let Some(down_derivatives) = grid.down_derivatives() {
            add_coordinate_variable(
                group,
                "dxidxdn",
                "xm",
                "X",
                "Upward derivatives of Cartesian grid x-coordinates",
                None,
                None,
                &down_derivatives[X],
            )?;
            add_coordinate_variable(
                group,
                "dyidydn",
                "ym",
                "Y",
                "Upward derivatives of Cartesian grid y-coordinates",
                None,
                None,
                &down_derivatives[Y],
            )?;
            add_coordinate_variable(
                group,
                "dzidzdn",
                "zm",
                "Z",
                "Upward derivatives of Cartesian grid z-coordinates",
                None,
                None,
                &down_derivatives[Z],
            )?;
        }
    }

    Ok(())
}

fn add_coordinate_variable(
    group: &mut GroupMut,
    name: &str,
    dimension_name: &str,
    axis: &str,
    long_name: &str,
    units: Option<&str>,
    positive: Option<&str>,
    values: &[fdt],
) -> io::Result<()> {
    let mut coord_var = io_result!(group.add_variable::<fdt>(name, &[dimension_name]))?;
    io_result!(coord_var.add_attribute("axis", axis))?;
    io_result!(coord_var.add_attribute("long_name", long_name))?;
    if let Some(units) = units {
        io_result!(coord_var.add_attribute("units", units))?;
    }
    if let Some(positive) = positive {
        io_result!(coord_var.add_attribute("positive", positive))?;
    }
    io_result!(coord_var.put_values(values, None, None))?;
    Ok(())
}
