//! Utilities for reading and writing of Bifrost simulation data.

use super::{
    fdt, fpa,
    native::{NativeSnapshotReader3, NativeSnapshotReaderConfig},
    MapOfSnapshotParameters, SnapshotMetadata, SnapshotParameters,
};
use crate::{
    exit_with_error,
    field::{DynScalarFieldProvider3, FieldGrid3},
    io::{Endianness, Verbosity},
};
use std::{
    borrow::Cow,
    fmt, io,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "netcdf")]
use super::netcdf::{NetCDFSnapshotReader3, NetCDFSnapshotReaderConfig};

/// Dummy metadata type that can be used for writing a snapshot
/// generated in Backstaff rather than read from files.
pub struct OutputSnapshotMetadata {
    parameters: Box<MapOfSnapshotParameters>,
}

impl OutputSnapshotMetadata {
    pub fn new() -> Self {
        Self {
            parameters: Box::new(MapOfSnapshotParameters::empty()),
        }
    }
}

impl Default for OutputSnapshotMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotMetadata for OutputSnapshotMetadata {
    fn parameters(&self) -> &dyn SnapshotParameters {
        &*self.parameters
    }

    fn snap_name(&self) -> &str {
        ""
    }

    fn snap_num(&self) -> Option<u64> {
        None
    }

    fn endianness(&self) -> Endianness {
        Endianness::Native
    }
}

/// Type of an input snapshot file (or set of files).|
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

/// Determines the snapshot input format from the given file
/// path and returns the appropriate reader and associated metadata.
pub fn new_snapshot_reader(
    input_file_path: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
) -> io::Result<(DynScalarFieldProvider3<fdt>, Box<dyn SnapshotMetadata>)> {
    let input_type = SnapshotInputType::from_path(&input_file_path);

    match input_type {
        SnapshotInputType::Native(_) => NativeSnapshotReader3::new(
            NativeSnapshotReaderConfig::new(input_file_path, endianness, verbosity),
        )
        .map(|(reader, metadata)| {
            (
                Box::new(reader) as DynScalarFieldProvider3<fdt>,
                Box::new(metadata) as Box<dyn SnapshotMetadata>,
            )
        }),
        #[cfg(feature = "netcdf")]
        SnapshotInputType::NetCDF => {
            NetCDFSnapshotReader3::new(NetCDFSnapshotReaderConfig::new(input_file_path, verbosity))
                .map(|(reader, metadata)| {
                    (
                        Box::new(reader) as DynScalarFieldProvider3<fdt>,
                        Box::new(metadata) as Box<dyn SnapshotMetadata>,
                    )
                })
        }
    }
}

/// Determines the snapshot input format from the given file
/// path and returns the associated grid.
pub fn read_snapshot_grid(
    input_file_path: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
) -> io::Result<Arc<FieldGrid3>> {
    new_snapshot_reader(input_file_path, endianness, verbosity)
        .map(|(reader, _)| reader.arc_with_grid())
}

/// Reads the snapshots at the given paths and compares them
/// for approximate equality.
#[cfg(feature = "for-testing")]
pub fn read_snapshots_eq(
    input_file_path_1: PathBuf,
    input_file_path_2: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    epsilon: fdt,
    max_relative: fdt,
) -> io::Result<bool> {
    let (mut reader_1, metadata_1) =
        new_snapshot_reader(input_file_path_1, endianness, verbosity.clone())?;
    let (mut reader_2, metadata_2) = new_snapshot_reader(input_file_path_2, endianness, verbosity)?;

    if metadata_1.parameters().relative_eq(
        metadata_2.parameters(),
        epsilon as fpa,
        max_relative as fpa,
    ) {
        reader_1.relative_eq(&mut *reader_2, epsilon, max_relative)
    } else {
        Ok(false)
    }
}

/// Reads the snapshots at the given paths and compares them
/// for approximate equality.
#[cfg(feature = "for-testing")]
pub fn read_snapshot_values_eq(
    input_file_path_1: PathBuf,
    input_file_path_2: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    epsilon: fdt,
    max_relative: fdt,
) -> io::Result<bool> {
    let (mut reader_1, _) = new_snapshot_reader(input_file_path_1, endianness, verbosity.clone())?;
    let (mut reader_2, _) = new_snapshot_reader(input_file_path_2, endianness, verbosity)?;
    reader_1.values_relative_eq(&mut *reader_2, epsilon, max_relative)
}

/// Reads the field values of the snapshot at the given path and
/// compares for equality to the corresponding given field values
/// using the given closure.
#[cfg(feature = "for-testing")]
pub fn read_snapshot_has_given_field_values_custom_eq(
    input_file_path: PathBuf,
    endianness: Endianness,
    verbosity: Verbosity,
    reference_field_values: Vec<(String, &[fdt])>,
    are_equal: &dyn Fn(&[fdt], &[fdt]) -> bool,
) -> io::Result<bool> {
    let (mut reader, _) = new_snapshot_reader(input_file_path, endianness, verbosity)?;

    let all_snapshot_variable_names = reader.all_variable_names().to_vec();
    for (name, values) in reference_field_values {
        if !all_snapshot_variable_names.contains(&name) {
            #[cfg(debug_assertions)]
            println!("Field {} not present in other", name);
            return Ok(false);
        } else {
            let read_field = reader.produce_scalar_field(&name)?;
            if !are_equal(read_field.values().as_slice_memory_order().unwrap(), values) {
                #[cfg(debug_assertions)]
                println!("Fields {} not equal", name);
                return Ok(false);
            }
        }
    }
    Ok(true)
}
