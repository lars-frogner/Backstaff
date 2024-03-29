//! Reading seed points from an input file.

use super::{fgr, Seeder3};
use crate::{
    field::FieldGrid3,
    geometry::{Idx3, Point3},
    io::utils,
};
use std::{
    io::{self, BufRead},
    path::Path,
};

/// Generator for 3D seed points read from an input file.
#[derive(Clone, Debug)]
pub struct ManualSeeder3 {
    seed_points: Vec<Point3<fgr>>,
}

impl ManualSeeder3 {
    /// Creates a new seeder producing 3D seed points read from an input file.
    ///
    /// The input file is assumed to be in CSV format, with each line consisting
    /// of the three comma-separated coordinates of a single seed point.
    ///
    /// # Parameters
    ///
    /// - `input_file_path`: Path to the input file.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a new `ManualSeeder3`.
    /// - `Err`: Contains an error encountered while trying to open or parse the input file.
    pub fn new(input_file_path: &Path) -> io::Result<Self> {
        let file = utils::open_file_and_map_err(input_file_path)?;
        let lines = io::BufReader::new(file).lines();
        let seed_points = lines
            .filter_map(|line_result| {
                match line_result {
                    Ok(line) => {
                        let trimmed_line = line.trim();
                        if trimmed_line.is_empty() || trimmed_line.starts_with('#') {
                            None
                        } else {
                            Some(
                                trimmed_line
                                    .split(',')
                                    .map(|coord_str| {
                                        coord_str.trim().parse::<fgr>().map_err(|err| {
                                            io::Error::new(
                                                io::ErrorKind::InvalidData,
                                                format!(
                                                    "Failed parsing coordinate string {} in input file: {}",
                                                    coord_str,
                                                    err
                                                ),
                                            )
                                        })
                                    })
                                    .collect::<io::Result<Vec<fgr>>>()
                                    .map(|coords| {
                                        if coords.len() >= 3 {
                                            Ok(Point3::with_each_component(|dim| coords[dim.num()]))
                                        } else {
                                            Err(io::Error::new(
                                                io::ErrorKind::InvalidData,
                                                format!(
                                                    "Too few coordinates in input file line: {}",
                                                    line
                                                ),
                                            ))
                                        }
                                    })
                            )
                        }
                    },
                    Err(err) => Some(Err(err))
                }
            })
            .collect::<io::Result<io::Result<Vec<_>>>>()??;
        Ok(Self { seed_points })
    }
}

impl Seeder3 for ManualSeeder3 {
    fn number_of_points(&self) -> usize {
        self.seed_points.len()
    }

    fn points(&self) -> &[Point3<fgr>] {
        &self.seed_points
    }

    fn to_index_seeder(&self, grid: &FieldGrid3) -> Vec<Idx3<usize>> {
        self.seed_points.to_index_seeder(grid)
    }
}
