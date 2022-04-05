//! Command line interface for outputting snapshot data to file.

use super::SnapNumInRange;
use crate::{
    cli::utils as cli_utils,
    exit_on_error, exit_with_error,
    field::ScalarField3,
    grid::Grid3,
    io::snapshot::{self, fdt, native, ParameterValue, SnapshotProvider3},
};
use clap::{Arg, ArgMatches, Command};
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt, io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

#[cfg(feature = "netcdf")]
use crate::io::snapshot::netcdf;

/// Builds a representation of the `snapshot-write` command line subcommand.
pub fn create_write_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "write";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Write snapshot data to file")

        .arg(
            Arg::new("output-file")
                .value_name("OUTPUT_FILE")
                .help(
                    "Path of the output file to produce.\n\
                     Writes in the following format based on the file extension:\
                     \n    *.idl: Creates a parameter file with an associated .snap [and .aux] file\
                     \n    *.nc: Creates a NetCDF file using the CF convention (requires the netcdf feature)\n\
                     If processing multiple snapshots, the output snapshot number will be\n\
                     incremented (or appended if necessary) with basis in this snapshot file name.",
                )
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing files (unless listed as protected)")
                .conflicts_with("no-overwrite"),
        )
        .arg(
            Arg::new("no-overwrite")
                .long("no-overwrite")
                .help("Do not overwrite any existing files")
                .conflicts_with("overwrite"),
        )
        .arg(
            Arg::new("all-quantities")
                .long("all-quantities")
                .short('A')
                .help("Include all original quantities in the output snapshot")
                .conflicts_with_all(&["included-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::new("included-quantities")
                .long("included-quantities")
                .short('I')
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of all the original quantities to include in the output snapshot\n\
                    (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple_values(true)
                .conflicts_with_all(&["all-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::new("excluded-quantities")
                .long("excluded-quantities")
                .short('E')
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of original quantities to leave out of the output snapshot\n\
                    (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple_values(true)
                .conflicts_with_all(&["all-quantities", "included-quantities"]),
        )
        .arg(
            Arg::new("ignore-warnings")
                .long("ignore-warnings")
                .help("Automatically continue on warnings"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to writing"),
        );

    #[cfg(feature = "netcdf")]
    let command = command.arg(
        Arg::new("strip")
            .short('s')
            .long("strip")
            .help("Strip away metadata not required for visualization"),
    );

    command
}

/// Runs the actions for the `snapshot-write` subcommand using the given arguments.
pub fn run_write_subcommand<GIN, PIN, GOUT, FM>(
    arguments: &ArgMatches,
    provider: PIN,
    snap_num_in_range: &Option<SnapNumInRange>,
    new_grid: Option<Arc<GOUT>>,
    modified_parameters: HashMap<&str, ParameterValue>,
    field_modifier: FM,
    protected_file_types: &[&str],
) where
    GIN: Grid3<fdt>,
    PIN: SnapshotProvider3<GIN>,
    GOUT: Grid3<fdt>,
    FM: Fn(ScalarField3<fdt, GIN>) -> io::Result<ScalarField3<fdt, GOUT>>,
{
    let mut output_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("output-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    let output_type = OutputType::from_path(&output_file_path);

    let mut write_mesh_file = true;

    if let Some(snap_num_in_range) = snap_num_in_range {
        exit_on_false!(
            !output_type.is_scratch(),
            "Error: snap-range not supported for scratch files"
        );
        output_file_path.set_file_name(snapshot::create_new_snapshot_file_name_from_path(
            &output_file_path,
            snap_num_in_range.offset(),
            &output_type.to_string(),
            true,
        ));
        if snap_num_in_range.offset() > 0 {
            write_mesh_file = false;
        }
    }

    let overwrite_mode = cli_utils::overwrite_mode_from_arguments(arguments);
    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbose = arguments.is_present("verbose").into();

    let quantity_names =
        super::parse_included_quantity_list(arguments, &provider, continue_on_warnings);

    if quantity_names.is_empty() {
        exit_with_error!("Aborted: No quantities to write");
    }

    exit_on_error!(
        match output_type {
            OutputType::Native(native_type) => native::write_modified_snapshot(
                &provider,
                new_grid,
                &quantity_names,
                modified_parameters,
                |name| provider
                    .provide_scalar_field(name)
                    .and_then(|field| field_modifier(field)),
                &output_file_path,
                native_type == NativeType::Scratch,
                write_mesh_file,
                overwrite_mode,
                protected_file_types,
                verbose,
            ),
            #[cfg(feature = "netcdf")]
            OutputType::NetCDF => {
                let strip_metadata = arguments.is_present("strip");
                netcdf::write_modified_snapshot(
                    &provider,
                    new_grid,
                    &quantity_names,
                    modified_parameters,
                    |name| {
                        provider
                            .provide_scalar_field(name)
                            .and_then(|field| field_modifier(field))
                    },
                    &output_file_path,
                    strip_metadata,
                    overwrite_mode,
                    protected_file_types,
                    verbose,
                )
            }
        },
        "Error: Could not write snapshot: {}"
    );
}

#[derive(Clone, Debug)]
enum OutputType {
    Native(NativeType),
    #[cfg(feature = "netcdf")]
    NetCDF,
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum NativeType {
    Snap,
    Scratch,
}

impl OutputType {
    fn from_path<P: AsRef<Path>>(file_path: P) -> Self {
        let file_name = Path::new(file_path.as_ref().file_name().unwrap_or_else(|| {
            exit_with_error!(
                "Error: Missing extension for output file\n\
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

    fn from_extension(extension: &str) -> Self {
        match extension {
            "idl" => Self::Native(NativeType::Snap),
            "idl.scr" => Self::Native(NativeType::Scratch),
            "nc" => {
                #[cfg(feature = "netcdf")]
                {
                    Self::NetCDF
                }
                #[cfg(not(feature = "netcdf"))]
                exit_with_error!("Error: Compile with netcdf feature in order to write NetCDF files\n\
                                  Tip: Use cargo flag --features=netcdf and make sure the NetCDF library is available");
            }
            invalid => exit_with_error!(
                "Error: Invalid extension {} for output file\n\
                 Valid extensions are: {}",
                invalid,
                Self::valid_extensions_string()
            ),
        }
    }

    fn valid_extensions_string() -> String {
        format!(
            "idl[.scr]{}",
            if cfg!(feature = "netcdf") { ", nc" } else { "" }
        )
    }

    fn is_scratch(&self) -> bool {
        if let Self::Native(NativeType::Scratch) = self {
            true
        } else {
            false
        }
    }
}

impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Native(NativeType::Snap) => "idl",
                Self::Native(NativeType::Scratch) => "idl.scr",
                #[cfg(feature = "netcdf")]
                Self::NetCDF => "nc",
            }
        )
    }
}
