//! Utilities for mesh files in native format.

use super::super::{super::utils, Verbosity};
use crate::{
    field::FieldGrid3,
    geometry::{
        Coords3,
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{self, fgr, Grid3, GridType},
};
use std::{collections::VecDeque, io, io::BufRead, path::Path};

#[cfg(feature = "for-testing")]
use approx::RelativeEq;

/// Width of formatted coordinate values in native mesh files.
pub const NATIVE_COORD_WIDTH: usize = 17;
/// Precision of formatted coordinate values in native mesh files.
pub const NATIVE_COORD_PRECISION: usize = 10;

/// Constructs a grid from a Bifrost mesh file.
///
/// # Parameters
///
/// - `mesh_path`: Path of the mesh file.
/// - `is_periodic`: Specifies for each dimension whether the grid is periodic.
/// - `verbosity`: Whether and how to pass non-essential information to user while reading grid data.
///
/// # Returns
///
/// A `Result` which is either:
///
/// - `Ok`: Contains the constructed grid.
/// - `Err`: Contains an error encountered while trying to read or interpret the mesh file.
///
/// # Type parameters
///
/// - `P`: A type that can be treated as a reference to a `Path`.
pub fn create_grid_from_mesh_file(
    mesh_path: &Path,
    is_periodic: In3D<bool>,
    verbosity: &Verbosity,
) -> io::Result<FieldGrid3> {
    let NativeGridData {
        detected_grid_type,
        center_coords,
        lower_edge_coords,
        up_derivatives,
        down_derivatives,
    } = parse_mesh_file(mesh_path, verbosity)?;

    Ok(FieldGrid3::from_coords_unchecked(
        center_coords,
        lower_edge_coords,
        is_periodic,
        Some(up_derivatives),
        Some(down_derivatives),
        detected_grid_type,
    ))
}

/// Data for a grid representing a Bifrost mesh file.
#[derive(Debug, Clone)]
struct NativeGridData {
    detected_grid_type: GridType,
    center_coords: Coords3<fgr>,
    lower_edge_coords: Coords3<fgr>,
    up_derivatives: Coords3<fgr>,
    down_derivatives: Coords3<fgr>,
}

/// Creates a Bifrost mesh file from the given grid.
///
/// # Parameters
///
/// - `grid`: Grid to create the mesh file from.
/// - `mesh_path`: Path where the mesh file should be created.
///
/// # Returns
///
/// A `Result` which is either:
///
/// - `Ok`: The mesh file was written successfully.
/// - `Err`: Contains an error encountered while trying to write the mesh file.
///
/// # Type parameters
///
/// - `G`: Type of the grid.
pub fn write_mesh_file_from_grid<G>(grid: &G, mesh_path: &Path) -> io::Result<()>
where
    G: Grid3<fgr>,
{
    let shape = grid.shape();
    let centers = grid.centers();
    let lower_edges = grid.lower_edges();
    let up_derivatives = grid
        .up_derivatives()
        .unwrap_or_else(|| panic!("Upward derivatives were not available."));
    let down_derivatives = grid
        .down_derivatives()
        .unwrap_or_else(|| panic!("Downward derivatives were not available."));

    let format_slice = |slice: &[fgr]| {
        slice
            .iter()
            .map(|&coord| {
                format!(
                    "{:width$.precision$E}",
                    coord,
                    width = NATIVE_COORD_WIDTH,
                    precision = NATIVE_COORD_PRECISION
                )
            })
            .collect::<Vec<_>>()
            .join("")
    };

    let format_for_dim = |dim| {
        [
            format!("{}", shape[dim]),
            format_slice(&centers[dim]),
            format_slice(&lower_edges[dim]),
            format_slice(&up_derivatives[dim]),
            format_slice(&down_derivatives[dim]),
        ]
        .join("\n")
    };

    let text = [format_for_dim(X), format_for_dim(Y), format_for_dim(Z)].join("\n");
    utils::write_text_file(&text, mesh_path)
}

/// Parses the mesh file at the given path and returns relevant data.
fn parse_mesh_file(mesh_path: &Path, verbosity: &Verbosity) -> io::Result<NativeGridData> {
    let file = utils::open_file_and_map_err(mesh_path)?;
    if verbosity.print_messages() {
        println!(
            "Reading grid from {}",
            mesh_path.file_name().unwrap().to_string_lossy()
        );
    }
    let mut lines = io::BufReader::new(file).lines();
    let coord_names = ["x", "y", "z"];
    let mut center_coord_vecs = VecDeque::new();
    let mut lower_coord_vecs = VecDeque::new();
    let mut up_derivative_vecs = VecDeque::new();
    let mut down_derivative_vecs = VecDeque::new();

    for coord_name in coord_names {
        let length = match lines.next() {
            Some(string) => match string {
                Ok(s) => match s.trim().parse::<usize>() {
                    Ok(length) => length,
                    Err(err) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Failed parsing string {} in mesh file: {}", s, err),
                        ))
                    }
                },
                Err(err) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
                }
            },
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Number of {}-coordinates not found in mesh file",
                        coord_name
                    ),
                ))
            }
        };

        let mut center_coords = Vec::with_capacity(length);
        let mut lower_coords = Vec::with_capacity(length);
        let mut up_derivatives = Vec::with_capacity(length);
        let mut down_derivatives = Vec::with_capacity(length);

        for coords in [
            &mut center_coords,
            &mut lower_coords,
            &mut up_derivatives,
            &mut down_derivatives,
        ]
        .iter_mut()
        {
            match lines.next() {
                Some(string) => {
                    for s in string?.split_whitespace() {
                        match s.parse::<fgr>() {
                            Ok(val) => coords.push(val),
                            Err(err) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Failed parsing string {} in mesh file: {}", s, err),
                                ))
                            }
                        };
                    }
                }
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{}-coordinates not found in mesh file", coord_name),
                    ))
                }
            };
        }

        center_coord_vecs.push_back(center_coords);
        lower_coord_vecs.push_back(lower_coords);
        up_derivative_vecs.push_back(up_derivatives);
        down_derivative_vecs.push_back(down_derivatives);
    }

    let center_coords = Coords3::new(
        center_coord_vecs.pop_front().unwrap(),
        center_coord_vecs.pop_front().unwrap(),
        center_coord_vecs.pop_front().unwrap(),
    );

    let lower_edge_coords = Coords3::new(
        lower_coord_vecs.pop_front().unwrap(),
        lower_coord_vecs.pop_front().unwrap(),
        lower_coord_vecs.pop_front().unwrap(),
    );

    let up_derivatives = Coords3::new(
        up_derivative_vecs.pop_front().unwrap(),
        up_derivative_vecs.pop_front().unwrap(),
        up_derivative_vecs.pop_front().unwrap(),
    );

    let down_derivatives = Coords3::new(
        down_derivative_vecs.pop_front().unwrap(),
        down_derivative_vecs.pop_front().unwrap(),
        down_derivative_vecs.pop_front().unwrap(),
    );

    let detected_grid_type = grid::verify_coordinate_arrays(
        &center_coords,
        &lower_edge_coords,
        verbosity.print_messages(),
    )?;

    Ok(NativeGridData {
        detected_grid_type,
        center_coords,
        lower_edge_coords,
        up_derivatives,
        down_derivatives,
    })
}

/// Parses the mesh files at the given paths and compares
/// the resulting grids for approximate equality.
#[cfg(feature = "for-testing")]
pub fn parsed_mesh_files_eq(
    mesh_path_1: &Path,
    mesh_path_2: &Path,
    verbosity: &Verbosity,
    epsilon: fgr,
    max_relative: fgr,
) -> io::Result<bool> {
    match (
        parse_mesh_file(mesh_path_1, verbosity),
        parse_mesh_file(mesh_path_2, verbosity),
    ) {
        (Ok(gd_1), Ok(gd_2)) => {
            Ok(gd_1
                .center_coords
                .relative_eq(&gd_2.center_coords, epsilon, max_relative)
                && gd_1.lower_edge_coords.relative_eq(
                    &gd_2.lower_edge_coords,
                    epsilon,
                    max_relative,
                )
                && gd_1
                    .up_derivatives
                    .relative_eq(&gd_2.up_derivatives, epsilon, max_relative)
                && gd_1
                    .down_derivatives
                    .relative_eq(&gd_2.down_derivatives, epsilon, max_relative))
        }
        (Err(err), _) => Err(err),
        (_, Err(err)) => Err(err),
    }
}
