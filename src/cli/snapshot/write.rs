//! Command line interface for outputting snapshot data to file.

use crate::{
    exit_on_error, exit_with_error,
    field::{quantities, ScalarField3},
    grid::Grid3,
    io::snapshot::{self, fdt, native, ParameterValue, SnapshotReader3},
};
use clap::{App, Arg, ArgMatches, SubCommand};
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
pub fn create_write_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("write")
        .about("Write snapshot data to file")
        .help_message("Print help information")
        .arg(
            Arg::with_name("output-file")
                .value_name("OUTPUT_FILE")
                .help(
                    "Path of the output file to produce.\n\
                     Writes in the following format based on the file extension:\
                     \n    *.idl: Creates a parameter file with an associated .snap [and .aux] file\
                     \n    *.nc: Creates a NetCDF file using the CF convention (requires the netcdf feature)\n\
                     If processing multiple snapshots, the output snapshot number will be incremented\n\
                     (or appended if necessary) with basis in this snapshot file name.",
                )
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing files (unless listed as protected)"),
        )
        .arg(
            Arg::with_name("all-quantities")
                .long("all-quantities")
                .help("Include all original quantities in the output snapshot")
                .conflicts_with_all(&["included-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::with_name("included-quantities")
                .long("included-quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of all the original quantities to include in the output snapshot\n\
                    (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple(true)
                .conflicts_with_all(&["all-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::with_name("excluded-quantities")
                .long("excluded-quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of original quantities to leave out of the output snapshot\n\
                    (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple(true)
                .conflicts_with_all(&["included-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::with_name("derived-quantities")
                .long("derived-quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of derived quantities to compute and include in the output snapshot\n\
                    (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("ignore-warnings")
                .long("ignore-warnings")
                .help("Automatically continue on warnings"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages related to writing"),
        );

    #[cfg(feature = "netcdf")]
    let app = app.arg(
        Arg::with_name("strip")
            .short("s")
            .long("strip")
            .help("Strip away metadata not required for visualization"),
    );

    app
}

/// Runs the actions for the `snapshot-write` subcommand using the given arguments.
pub fn run_write_subcommand<GIN, RIN, GOUT, FM>(
    arguments: &ArgMatches,
    reader: &RIN,
    snap_num_offset: Option<u32>,
    new_grid: Option<Arc<GOUT>>,
    modified_parameters: HashMap<&str, ParameterValue>,
    field_modifier: FM,
    protected_file_types: &[&str],
) where
    GIN: Grid3<fdt>,
    RIN: SnapshotReader3<GIN>,
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

    if let Some(snap_num_offset) = snap_num_offset {
        exit_on_false!(
            !output_type.is_scratch(),
            "Error: snap-range not supported for scratch files"
        );
        output_file_path.set_file_name(snapshot::create_new_snapshot_file_name_from_path(
            &output_file_path,
            snap_num_offset,
            &output_type.to_string(),
            true,
        ));
        if snap_num_offset > 0 {
            write_mesh_file = false;
        }
    }

    let automatic_overwrite = arguments.is_present("overwrite");
    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbose = arguments.is_present("verbose").into();

    let (included_quantities, derived_quantities) =
        super::parse_quantity_lists(arguments, reader, continue_on_warnings);

    let quantity_names: Vec<_> = included_quantities
        .iter()
        .cloned()
        .chain(derived_quantities.iter().cloned())
        .collect();

    if quantity_names.is_empty() {
        exit_with_error!("Aborted: No quantities to write");
    }

    macro_rules! modified_field_producer {
        () => {
            |name| match if included_quantities.contains(&name) {
                reader.read_scalar_field(name)
            } else if derived_quantities.contains(&name) {
                quantities::compute_quantity(reader, name, verbose)
            } else {
                unreachable!()
            } {
                Ok(field) => field_modifier(field),
                Err(err) => Err(err),
            }
        };
    }

    exit_on_error!(
        match output_type {
            OutputType::Native(native_type) => native::write_modified_snapshot(
                reader,
                new_grid,
                &quantity_names,
                modified_parameters,
                modified_field_producer!(),
                &output_file_path,
                native_type == NativeType::Scratch,
                write_mesh_file,
                automatic_overwrite,
                protected_file_types,
                verbose,
            ),
            #[cfg(feature = "netcdf")]
            OutputType::NetCDF => {
                let strip_metadata = arguments.is_present("strip");
                netcdf::write_modified_snapshot(
                    reader,
                    new_grid,
                    &quantity_names,
                    modified_parameters,
                    modified_field_producer!(),
                    &output_file_path,
                    strip_metadata,
                    automatic_overwrite,
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
