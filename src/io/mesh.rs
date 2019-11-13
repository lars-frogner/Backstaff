//! Reading and writing of Bifrost meshes.

use super::snapshot::fdt;
use super::utils;
use crate::geometry::{Coords3, Dim3, In3D};
use crate::grid::{Grid3, GridType};
use std::collections::VecDeque;
use std::io;
use std::io::BufRead;
use std::path::Path;
use Dim3::{X, Y, Z};

/// Constructs a grid from a Bifrost mesh file.
///
/// # Parameters
///
/// - `mesh_path`: Path of the mesh file.
/// - `is_periodic`: Specifies for each dimension whether the grid is periodic.
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
/// - `G`: Type of the grid.
pub fn create_grid_from_mesh_file<P, G>(mesh_path: P, is_periodic: In3D<bool>) -> io::Result<G>
where
    P: AsRef<Path>,
    G: Grid3<fdt>,
{
    let file = utils::open_file_and_map_err(mesh_path)?;
    let mut lines = io::BufReader::new(file).lines();
    let coord_names = ["x", "y", "z"];
    let mut center_coord_vecs = VecDeque::new();
    let mut lower_coord_vecs = VecDeque::new();
    let mut up_derivative_vecs = VecDeque::new();
    let mut down_derivative_vecs = VecDeque::new();
    let mut is_uniform = [true; 3];

    for dim in 0..3 {
        let mut center_coords = Vec::new();
        let mut lower_coords = Vec::new();
        let mut up_derivatives = Vec::new();
        let mut down_derivatives = Vec::new();

        let length = match lines.next() {
            Some(string) => match string {
                Ok(s) => match s.trim().parse::<usize>() {
                    Ok(length) => length,
                    Err(err) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Failed parsing string `{}` in mesh file: {}",
                                s,
                                err.to_string()
                            ),
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
                        coord_names[dim]
                    ),
                ))
            }
        };

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
                        match s.parse::<fdt>() {
                            Ok(val) => coords.push(val),
                            Err(err) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!(
                                        "Failed parsing string `{}` in mesh file: {}",
                                        s,
                                        err.to_string()
                                    ),
                                ))
                            }
                        };
                    }
                }
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{}-coordinates not found in mesh file", coord_names[dim]),
                    ))
                }
            };
        }

        if center_coords.len() != length
            || lower_coords.len() != length
            || up_derivatives.len() != length
            || down_derivatives.len() != length
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Inconsistent number of {}-coordinates in mesh file",
                    coord_names[dim]
                ),
            ));
        }

        if length < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Insufficient number of {}-coordinates in mesh file (must be at least 2)",
                    coord_names[dim]
                ),
            ));
        }

        let uniform_up = up_derivatives
            .iter()
            .all(|&element| fdt::abs(element - up_derivatives[0]) < 5e-3);
        let uniform_down = down_derivatives
            .iter()
            .all(|&element| fdt::abs(element - down_derivatives[0]) < 5e-3);

        if uniform_up != uniform_down {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Inconsistent uniformity of {}-coordinates in mesh file",
                    coord_names[dim]
                ),
            ));
        }

        is_uniform[dim] = uniform_up;

        if !center_coords
            .iter()
            .zip(center_coords.iter().skip(1))
            .all(|(&lower, &upper)| upper >= lower)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Center {}-coordinates in mesh file do not increase monotonically",
                    coord_names[dim]
                ),
            ));
        }

        if !lower_coords
            .iter()
            .zip(lower_coords.iter().skip(1))
            .all(|(&lower, &upper)| upper >= lower)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Lower edge {}-coordinates in mesh file do not increase monotonically",
                    coord_names[dim]
                ),
            ));
        }

        if !lower_coords
            .iter()
            .zip(center_coords.iter())
            .all(|(&lower, &upper)| upper > lower)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Found grid cell where center {}-coordinate is not larger than lower edge coordinate",
                    coord_names[dim]
                ),
            ));
        }

        center_coord_vecs.push_back(center_coords);
        lower_coord_vecs.push_back(lower_coords);
        up_derivative_vecs.push_back(up_derivatives);
        down_derivative_vecs.push_back(down_derivatives);
    }

    let detected_grid_type = match is_uniform {
        [true, true, true] => GridType::Regular,
        [true, true, false] => GridType::HorRegular,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Non-uniform x- or y-coordinates not supported",
            ))
        }
    };

    if detected_grid_type != G::TYPE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Wrong reader type for the specified mesh file",
        ));
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

    Ok(G::from_coords(
        center_coords,
        lower_edge_coords,
        is_periodic,
        Some(up_derivatives),
        Some(down_derivatives),
    ))
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
/// - `P`: A type that can be treated as a reference to a `Path`.
/// - `G`: Type of the grid.
pub fn write_mesh_file_from_grid<P, G>(grid: &G, mesh_path: P) -> io::Result<()>
where
    P: AsRef<Path>,
    G: Grid3<fdt>,
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

    let format_slice = |slice: &[fdt]| {
        slice
            .iter()
            .map(|&coord| format!("{:width$.precision$E}", coord, width = 15, precision = 8))
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
