//! Command line interface for outputting snapshot data to file.

use crate::{
    exit_on_error, exit_with_error,
    field::{quantities, ScalarField3},
    grid::Grid3,
    io::{
        snapshot::{self, fdt, native, ParameterValue, SnapshotReader3},
        utils,
    },
};
use clap::{App, Arg, ArgMatches, SubCommand};
use std::{collections::HashMap, io, path::PathBuf, process, str::FromStr, sync::Arc};

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
                .help("Automatically overwrite any existing files"),
        )
        .arg(
            Arg::with_name("included-quantities")
                .long("included-quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of all the original quantities to include in the output snapshot\n\
                    (comma-separated) [default: all original quantities]",
                )
                .takes_value(true)
                .multiple(true)
                .conflicts_with("excluded-quantities"),
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
                .conflicts_with("included-quantities"),
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
            Arg::with_name("yes")
                .short("y")
                .long("yes")
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
                .expect("Required argument not present."),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    let output_extension = output_file_path
        .extension()
        .unwrap_or_else(|| exit_with_error!("Error: Missing extension for output-file"))
        .to_string_lossy()
        .to_string();

    if let Some(snap_num_offset) = snap_num_offset {
        let (output_snap_name, output_snap_num) =
            snapshot::extract_name_and_num_from_snapshot_path(&output_file_path);
        let output_snap_num = output_snap_num.unwrap_or(snapshot::FALLBACK_SNAP_NUM);
        output_file_path.set_file_name(snapshot::create_snapshot_file_name(
            &output_snap_name,
            output_snap_num + snap_num_offset,
            &output_extension,
        ));
    }

    let force_overwrite = arguments.is_present("overwrite");
    let continue_on_warnings = arguments.is_present("yes");
    let verbose = arguments.is_present("verbose").into();

    let included_quantities = if let Some(included_quantities) = arguments
        .values_of("included-quantities")
        .map(|values| values.collect::<Vec<_>>())
    {
        included_quantities
            .into_iter()
            .filter(|name| {
                let has_variable = reader.has_variable(name);
                if !has_variable {
                    eprintln!("Warning: Quantity {} not present in snapshot", name);
                    if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                        process::exit(1);
                    }
                }
                has_variable
            })
            .collect()
    } else if let Some(excluded_quantities) = arguments
        .values_of("excluded-quantities")
        .map(|values| values.collect::<Vec<_>>())
    {
        reader.all_variable_names_except(&excluded_quantities)
    } else {
        reader.all_variable_names()
    };

    let derived_quantities: Vec<_> = arguments
        .values_of("derived-quantities")
        .map(|values| values.collect::<Vec<_>>())
        .unwrap_or(Vec::new())
        .into_iter()
        .filter(|quantity_name| {
            match quantities::find_missing_quantity_dependencies(reader, quantity_name) {
                Some(missing_dependencies) => {
                    if missing_dependencies.is_empty() {
                        true
                    } else {
                        eprintln!(
                            "Warning: Missing following dependencies for derived quantity {}: {}",
                            quantity_name,
                            missing_dependencies.join(", ")
                        );
                        if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                            process::exit(1);
                        }
                        false
                    }
                }
                None => {
                    eprintln!("Warning: Derived quantity {} not supported", quantity_name);
                    if !continue_on_warnings && !utils::user_says_yes("Still continue?", true) {
                        process::exit(1);
                    }
                    false
                }
            }
        })
        .collect();

    if included_quantities.is_empty() && derived_quantities.is_empty() {
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
        match output_extension.as_ref() {
            "idl" => native::write_modified_snapshot(
                reader,
                new_grid,
                &included_quantities,
                modified_parameters,
                modified_field_producer!(),
                &output_file_path,
                force_overwrite,
                verbose,
            ),
            "nc" => {
                #[cfg(feature = "netcdf")]
                {
                    let strip_metadata = arguments.is_present("strip");
                    netcdf::write_modified_snapshot(
                        reader,
                        new_grid,
                        &included_quantities,
                        modified_parameters,
                        modified_field_producer!(),
                        &output_file_path,
                        strip_metadata,
                        force_overwrite,
                        verbose,
                    )
                }
                #[cfg(not(feature = "netcdf"))]
                exit_with_error!("Error: Compile with netcdf feature in order to write NetCDF files\n\
                                  Tip: Use cargo flag --features=netcdf and make sure libnetcdf is available");
            }
            invalid => exit_with_error!("Error: Invalid extension {} for output-file", invalid),
        },
        "Error: Could not write snapshot: {}"
    );
}
