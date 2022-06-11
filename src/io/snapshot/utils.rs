//! Utilities for reading and writing of Bifrost simulation data.

use super::{fdt, SnapshotReader3};
use crate::{
    grid::{fgr, hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3},
    io::{snapshot::SnapshotProvider3, Endianness, Verbosity},
};
use std::{
    borrow::Cow,
    fmt, io,
    path::{Path, PathBuf},
};

#[cfg(feature = "comparison")]
use crate::snapshots_relative_eq;

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
    pub fn from_path(file_path: &Path) -> Self {
        let file_name = Path::new(file_path.file_name().unwrap_or_else(|| {
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

/// Represents a snapshot number as part of a range.
#[derive(Debug, Clone)]
pub struct SnapNumInRange {
    current_offset: u32,
    final_offset: u32,
}

impl SnapNumInRange {
    pub fn new(start_snap_num: u32, end_snap_num: u32, current_snap_num: u32) -> Self {
        assert!(
            end_snap_num >= start_snap_num,
            "End snap number must be larger than or equal to start snap number."
        );
        assert!(
            current_snap_num >= start_snap_num && current_snap_num <= end_snap_num,
            "Current snap number must be between start and end snap number."
        );
        Self {
            current_offset: current_snap_num - start_snap_num,
            final_offset: end_snap_num - current_snap_num,
        }
    }

    pub fn offset(&self) -> u32 {
        self.current_offset
    }

    pub fn is_final(&self) -> bool {
        self.current_offset == self.final_offset
    }
}

#[macro_export]
macro_rules! with_new_snapshot_reader {
    ($input_file_path:expr, $endianness:expr, $verbosity:expr, $force_hor_regular:expr, |$reader:ident| $action:expr) => {{
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
                NativeSnapshotReaderConfig::new($input_file_path, $endianness, $verbosity),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular if !$force_hor_regular => metadata
                    .into_reader::<RegularGrid3<_>>()
                    .and_then(|reader| {
                        #[allow(unused_mut)]
                        let mut action = |$reader: NativeSnapshotReader3<RegularGrid3<_>>| $action;
                        action(reader)
                    }),
                _ => metadata
                    .into_reader::<HorRegularGrid3<_>>()
                    .and_then(|reader| {
                        #[allow(unused_mut)]
                        let mut action =
                            |$reader: NativeSnapshotReader3<HorRegularGrid3<_>>| $action;
                        action(reader)
                    }),
            }),
            #[cfg(feature = "netcdf")]
            SnapshotInputType::NetCDF => NetCDFSnapshotMetadata::new(
                NetCDFSnapshotReaderConfig::new($input_file_path, $verbosity),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular if !$force_hor_regular => {
                    let reader = metadata.into_reader::<RegularGrid3<_>>();
                    #[allow(unused_mut)]
                    let mut action = |$reader: NetCDFSnapshotReader3<RegularGrid3<_>>| $action;
                    action(reader)
                }
                _ => {
                    let reader = metadata.into_reader::<HorRegularGrid3<_>>();
                    #[allow(unused_mut)]
                    let mut action = |$reader: NetCDFSnapshotReader3<HorRegularGrid3<_>>| $action;
                    action(reader)
                }
            }),
        }
    }};
    ($input_file_path:expr, $endianness:expr, $verbosity:expr, |$reader:ident| $action:expr) => {
        with_new_snapshot_reader!(
            $input_file_path,
            $endianness,
            $verbosity,
            false,
            |$reader| { $action }
        )
    };
}

#[macro_export]
macro_rules! with_new_snapshot_grid {
    ($input_file_path:expr, $endianness:expr, $verbosity:expr, $force_hor_regular:expr, |$grid:ident| $action:expr) => {{
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
                NativeSnapshotReaderConfig::new($input_file_path, $endianness, $verbosity),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular if !$force_hor_regular => {
                    let grid = metadata.into_grid::<RegularGrid3<_>>();
                    let action = |$grid: RegularGrid3<_>| $action;
                    action(grid)
                }
                _ => {
                    let grid = metadata.into_grid::<HorRegularGrid3<_>>();
                    let action = |$grid: HorRegularGrid3<_>| $action;
                    action(grid)
                }
            }),
            #[cfg(feature = "netcdf")]
            SnapshotInputType::NetCDF => NetCDFSnapshotMetadata::new(
                NetCDFSnapshotReaderConfig::new($input_file_path, $verbosity),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular if !$force_hor_regular => {
                    let grid = metadata.into_grid::<RegularGrid3<_>>();
                    let action = |$grid: RegularGrid3<_>| $action;
                    action(grid)
                }
                _ => {
                    let grid = metadata.into_grid::<HorRegularGrid3<_>>();
                    let action = |$grid: HorRegularGrid3<_>| $action;
                    action(grid)
                }
            }),
        }
    }};
    ($input_file_path:expr, $endianness:expr, $verbosity:expr, |$grid:ident| $action:expr) => {
        with_new_snapshot_grid!($input_file_path, $endianness, $verbosity, false, |$grid| {
            $action
        })
    };
}

/// Reads the snapshot at the given path and compares for
/// approximate equality to the given snapshot.
#[cfg(feature = "comparison")]
pub fn read_snapshot_eq_given_snapshot<G, R>(
    input_file_path: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    reference_snapshot_reader: &R,
    epsilon: fdt,
    max_relative: fdt,
) -> io::Result<bool>
where
    G: Grid3<fgr> + RelativeEq<RegularGrid3<fgr>> + RelativeEq<HorRegularGrid3<fgr>>,
    R: SnapshotReader3<G>,
{
    with_new_snapshot_reader!(input_file_path, endianness, verbosity, |snapshot_reader| {
        snapshots_relative_eq!(
            snapshot_reader,
            reference_snapshot_reader,
            epsilon,
            max_relative
        )
    })
}

/// Reads the snapshots at the given paths and compares them
/// for approximate equality.
#[cfg(feature = "comparison")]
pub fn read_snapshots_eq(
    input_file_path_1: PathBuf,
    input_file_path_2: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    epsilon: fdt,
    max_relative: fdt,
) -> io::Result<bool> {
    with_new_snapshot_reader!(
        input_file_path_2,
        endianness,
        verbosity.clone(),
        |snapshot_reader| {
            read_snapshot_eq_given_snapshot(
                input_file_path_1,
                endianness,
                verbosity,
                &snapshot_reader,
                epsilon,
                max_relative,
            )
        }
    )
}

/// Reads the field values of the snapshot at the given path and
/// compares for equality to the corresponding given field values
/// using the given closure.
#[cfg(feature = "comparison")]
pub fn read_snapshot_has_given_fields_custom_eq<'a>(
    input_file_path: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    reference_field_values: Vec<(String, &'a [fdt])>,
    are_equal: &dyn Fn(&[fdt], &'a [fdt]) -> bool,
) -> io::Result<bool> {
    with_new_snapshot_reader!(input_file_path, endianness, verbosity, |snapshot_reader| {
        let all_snapshot_variable_names = snapshot_reader.all_variable_names();
        for (name, values) in reference_field_values {
            if !all_snapshot_variable_names.contains(&name) {
                #[cfg(debug_assertions)]
                println!("Field {} not present in other", name);
                return Ok(false);
            } else {
                let read_field = snapshot_reader.read_scalar_field(&name)?;
                if !are_equal(read_field.values().as_slice_memory_order().unwrap(), values) {
                    #[cfg(debug_assertions)]
                    println!("Fields {} not equal", name);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    })
}

/// Reads the grid of the snapshot at the given path and compares
/// for approximate equality to the given grid.
#[cfg(feature = "comparison")]
pub fn read_snapshot_grid_eq_given_grid<G>(
    input_file_path: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    reference_snapshot_grid: &G,
    epsilon: fgr,
    max_relative: fgr,
) -> io::Result<bool>
where
    G: Grid3<fgr> + RelativeEq<RegularGrid3<fgr>> + RelativeEq<HorRegularGrid3<fgr>>,
{
    with_new_snapshot_grid!(input_file_path, endianness, verbosity, |snapshot_grid| {
        Ok(snapshot_grid.relative_eq(reference_snapshot_grid, epsilon, max_relative))
    })
}

/// Reads the grids of the snapshots at the given paths and compares
/// them for approximate equality.
#[cfg(feature = "comparison")]
pub fn read_snapshot_grids_eq(
    input_file_path_1: PathBuf,
    input_file_path_2: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    epsilon: fgr,
    max_relative: fgr,
) -> io::Result<bool> {
    with_new_snapshot_grid!(
        input_file_path_2,
        endianness,
        verbosity.clone(),
        |snapshot_grid| {
            read_snapshot_grid_eq_given_grid(
                input_file_path_1,
                endianness,
                verbosity,
                &snapshot_grid,
                epsilon,
                max_relative,
            )
        }
    )
}
