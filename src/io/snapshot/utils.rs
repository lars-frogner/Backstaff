//! Utilities for reading and writing of Bifrost simulation data.

use std::{borrow::Cow, fmt, path::Path};

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
        type NetCDFSnapshotReaderConfig = crate::io::snapshot::netcdf::NetCDFSnapshotReaderConfig;
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
                GridType::Regular => metadata
                    .into_reader::<RegularGrid3<_>>()
                    .and_then(|reader| {
                        let action = |$reader| $action;
                        action(reader)
                    }),
                GridType::HorRegular => {
                    metadata
                        .into_reader::<HorRegularGrid3<_>>()
                        .and_then(|reader| {
                            let action = |$reader| $action;
                            action(reader)
                        })
                }
            }),
            #[cfg(feature = "netcdf")]
            SnapshotInputType::NetCDF => NetCDFSnapshotMetadata::new(
                NetCDFSnapshotReaderConfig::new($input_file_path, $verbose),
            )
            .and_then(|metadata| match metadata.grid_type() {
                GridType::Regular => metadata
                    .into_reader::<RegularGrid3<_>>()
                    .and_then(|reader| {
                        let action = |$reader| $action;
                        action(reader)
                    }),
                GridType::HorRegular => {
                    metadata
                        .into_reader::<HorRegularGrid3<_>>()
                        .and_then(|reader| {
                            let action = |$reader| $action;
                            action(reader)
                        })
                }
            }),
        }
    }};
}
