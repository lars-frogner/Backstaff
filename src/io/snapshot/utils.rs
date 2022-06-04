//! Utilities for reading and writing of Bifrost simulation data.

use std::{borrow::Cow, fmt, io, path::Path};

use crate::{
    grid::{fgr, hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3},
    io::{Endianness, Verbose},
    snapshots_relative_eq,
};

use super::{fdt, SnapshotReader3};

#[cfg(feature = "comparison")]
use approx::RelativeEq;

/// Type of an input snapshot file (or set of files).
#[derive(Clone, Debug)]
pub enum SnapshotInputType {
    Native(NativeSnapshotInputType),
    #[cfg(feature = "netcdf")]
    NetCDF,
}

/// Type of input files for snapshots in native format.
#[derive(Copy, Clone, Debug)]
pub enum NativeSnapshotInputType {
    Snap,
    Scratch,
}

impl SnapshotInputType {
    /// Determines the input type by parsing the given input file.
    pub fn from_path<P: AsRef<Path>>(file_path: P) -> Self {
        let file_name = Path::new(file_path.as_ref().file_name().unwrap_or_else(|| {
            exit_with_error!(
                "Error: Missing extension for input file\n\
                         Valid extensions are: {}",
                Self::valid_extensions_string()
            )
        }));
        let final_extension = file_name.extension().unwrap().to_string_lossy();
        let extension = if final_extension.as_ref() == "scr" {
            match Path::new(file_name.file_stem().unwrap()).extension() {
                Some(extension) => Cow::Owned(format!(
                    "{}.{}",
                    extension.to_string_lossy(),
                    final_extension
                )),
                None => final_extension,
            }
        } else {
            final_extension
        };
        Self::from_extension(extension.as_ref())
    }

    /// Determines the input type from the given input file extension.
    pub fn from_extension(extension: &str) -> Self {
        match extension {
            "idl" => Self::Native(NativeSnapshotInputType::Snap),
            "idl.scr" => Self::Native(NativeSnapshotInputType::Scratch),
            "nc" => {
                #[cfg(feature = "netcdf")]
                {
                    Self::NetCDF
                }
                #[cfg(not(feature = "netcdf"))]
                exit_with_error!("Error: Compile with netcdf feature in order to read NetCDF files\n\
                                  Tip: Use cargo flag --features=netcdf and make sure the NetCDF library is available");
            }
            invalid => exit_with_error!(
                "Error: Invalid extension {} for input file\n\
                 Valid extensions are: {}",
                invalid,
                Self::valid_extensions_string()
            ),
        }
    }

    /// Returns a string listing valid extensions for input files.
    pub fn valid_extensions_string() -> String {
        format!(
            "idl[.scr]{}",
            if cfg!(feature = "netcdf") { ", nc" } else { "" }
        )
    }

    /// Whether this input type represents native snapshot scratch files.
    pub fn is_scratch(&self) -> bool {
        matches!(self, Self::Native(NativeSnapshotInputType::Scratch))
    }
}

impl fmt::Display for SnapshotInputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Native(NativeSnapshotInputType::Snap) => "idl",
                Self::Native(NativeSnapshotInputType::Scratch) => "idl.scr",
                #[cfg(feature = "netcdf")]
                Self::NetCDF => "nc",
            }
        )
    }
}

#[macro_export]
macro_rules! with_new_snapshot_reader {
    ($input_file_path:expr, $endianness:expr, $verbose:expr, |$reader:ident| $action:expr) => {{
        type SnapshotInputType = crate::io::snapshot::utils::SnapshotInputType;
        type NativeSnapshotReaderConfig = crate::io::snapshot::native::NativeSnapshotReaderConfig;
        type NativeSnapshotMetadata = crate::io::snapshot::native::NativeSnapshotMetadata;
        type NativeSnapshotReader3<G> = crate::io::snapshot::native::NativeSnapshotReader3<G>;

        #[cfg(feature = "netcdf")]
        type NetCDFSnapshotReaderConfig = crate::io::snapshot::netcdf::NetCDFSnapshotReaderConfig;
        #[cfg(feature = "netcdf")]
        type NetCDFSnapshotMetadata = crate::io::snapshot::netcdf::NetCDFSnapshotMetadata;
        #[cfg(feature = "netcdf")]
        type NetCDFSnapshotReader3<G> = crate::io::snapshot::netcdf::NetCDFSnapshotReader3<G>;

        type GridType = crate::grid::GridType;
        type RegularGrid3<F> = crate::grid::regular::RegularGrid3<F>;
        type HorRegularGrid3<F> = crate::grid::hor_regular::HorRegularGrid3<F>;

        let input_type = SnapshotInputType::from_path(&$input_file_path);

        match input_type {
            SnapshotInputType::Native(_) => NativeSnapshotMetadata::new(
                NativeSnapshotReaderConfig::new($input_file_path, $endianness, $verbose),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular => metadata
                    .into_reader::<RegularGrid3<_>>()
                    .and_then(|reader| {
                        let action = |$reader: NativeSnapshotReader3<RegularGrid3<_>>| $action;
                        action(reader)
                    }),
                GridType::HorRegular => {
                    metadata
                        .into_reader::<HorRegularGrid3<_>>()
                        .and_then(|reader| {
                            let action =
                                |$reader: NativeSnapshotReader3<HorRegularGrid3<_>>| $action;
                            action(reader)
                        })
                }
            }),
            #[cfg(feature = "netcdf")]
            SnapshotInputType::NetCDF => NetCDFSnapshotMetadata::new(
                NetCDFSnapshotReaderConfig::new($input_file_path, $verbose),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular => {
                    let reader = metadata.into_reader::<RegularGrid3<_>>();
                    let action = |$reader: NetCDFSnapshotReader3<RegularGrid3<_>>| $action;
                    action(reader)
                }
                GridType::HorRegular => {
                    let reader = metadata.into_reader::<HorRegularGrid3<_>>();
                    let action = |$reader: NetCDFSnapshotReader3<HorRegularGrid3<_>>| $action;
                    action(reader)
                }
            }),
        }
    }};
}

#[macro_export]
macro_rules! with_new_snapshot_grid {
    ($input_file_path:expr, $endianness:expr, $verbose:expr, |$grid:ident| $action:expr) => {{
        type SnapshotInputType = crate::io::snapshot::utils::SnapshotInputType;
        type NativeSnapshotReaderConfig = crate::io::snapshot::native::NativeSnapshotReaderConfig;
        type NativeSnapshotMetadata = crate::io::snapshot::native::NativeSnapshotMetadata;

        #[cfg(feature = "netcdf")]
        type NetCDFSnapshotReaderConfig = crate::io::snapshot::netcdf::NetCDFSnapshotReaderConfig;
        #[cfg(feature = "netcdf")]
        type NetCDFSnapshotMetadata = crate::io::snapshot::netcdf::NetCDFSnapshotMetadata;

        type GridType = crate::grid::GridType;
        type RegularGrid3<F> = crate::grid::regular::RegularGrid3<F>;
        type HorRegularGrid3<F> = crate::grid::hor_regular::HorRegularGrid3<F>;

        let input_type = SnapshotInputType::from_path(&$input_file_path);

        match input_type {
            SnapshotInputType::Native(_) => NativeSnapshotMetadata::new(
                NativeSnapshotReaderConfig::new($input_file_path, $endianness, $verbose),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular => {
                    let grid = metadata.into_grid::<RegularGrid3<_>>();
                    let action = |$grid: RegularGrid3<_>| $action;
                    action(grid)
                }
                GridType::HorRegular => {
                    let grid = metadata.into_grid::<HorRegularGrid3<_>>();
                    let action = |$grid: HorRegularGrid3<_>| $action;
                    action(grid)
                }
            }),
            #[cfg(feature = "netcdf")]
            SnapshotInputType::NetCDF => NetCDFSnapshotMetadata::new(
                NetCDFSnapshotReaderConfig::new($input_file_path, $verbose),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular => {
                    let grid = metadata.into_grid::<RegularGrid3<_>>();
                    let action = |$grid: RegularGrid3<_>| $action;
                    action(grid)
                }
                GridType::HorRegular => {
                    let grid = metadata.into_grid::<HorRegularGrid3<_>>();
                    let action = |$grid: HorRegularGrid3<_>| $action;
                    action(grid)
                }
            }),
        }
    }};
}

/// Reads the snapshot at the given path and compares for
/// approximate equality to the given snapshot.
#[cfg(feature = "comparison")]
pub fn read_snapshot_eq_given_snapshot<P, G, R>(
    input_file_path: P,
    endianness: Endianness,
    verbose: Verbose,
    reference_snapshot_reader: &R,
    epsilon: fdt,
    max_relative: fdt,
) -> io::Result<bool>
where
    P: AsRef<Path>,
    G: Grid3<fgr> + RelativeEq<RegularGrid3<fgr>> + RelativeEq<HorRegularGrid3<fgr>>,
    R: SnapshotReader3<G>,
{
    with_new_snapshot_reader!(input_file_path, endianness, verbose, |snapshot_reader| {
        Ok(snapshots_relative_eq!(
            snapshot_reader,
            reference_snapshot_reader,
            epsilon,
            max_relative
        ))
    })
}

/// Reads the snapshots at the given paths and compares them
/// for approximate equality.
#[cfg(feature = "comparison")]
pub fn read_snapshots_eq<P1, P2>(
    input_file_path_1: P1,
    input_file_path_2: P2,
    endianness: Endianness,
    verbose: Verbose,
    epsilon: fdt,
    max_relative: fdt,
) -> io::Result<bool>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    with_new_snapshot_reader!(input_file_path_1, endianness, verbose, |snapshot_reader| {
        read_snapshot_eq_given_snapshot(
            input_file_path_2,
            endianness,
            verbose,
            &snapshot_reader,
            epsilon,
            max_relative,
        )
    })
}

/// Reads the grid of the snapshot at the given path and compares
/// for approximate equality to the given grid.
#[cfg(feature = "comparison")]
pub fn read_snapshot_grid_eq_given_grid<P, G>(
    input_file_path: P,
    endianness: Endianness,
    verbose: Verbose,
    reference_snapshot_grid: &G,
    epsilon: fgr,
    max_relative: fgr,
) -> io::Result<bool>
where
    P: AsRef<Path>,
    G: Grid3<fgr> + RelativeEq<RegularGrid3<fgr>> + RelativeEq<HorRegularGrid3<fgr>>,
{
    with_new_snapshot_grid!(input_file_path, endianness, verbose, |snapshot_grid| {
        Ok(snapshot_grid.relative_eq(reference_snapshot_grid, epsilon, max_relative))
    })
}

/// Reads the grids of the snapshots at the given paths and compares
/// them for approximate equality.
#[cfg(feature = "comparison")]
pub fn read_snapshot_grids_eq<P1, P2>(
    input_file_path_1: P1,
    input_file_path_2: P2,
    endianness: Endianness,
    verbose: Verbose,
    epsilon: fgr,
    max_relative: fgr,
) -> io::Result<bool>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    with_new_snapshot_grid!(input_file_path_1, endianness, verbose, |snapshot_grid| {
        read_snapshot_grid_eq_given_grid(
            input_file_path_2,
            endianness,
            verbose,
            &snapshot_grid,
            epsilon,
            max_relative,
        )
    })
}
