//! Command line interface for actions related to snapshots.

mod inspect;
mod resample;
mod slice;
mod write;

use self::{
    inspect::create_inspect_subcommand, resample::create_resample_subcommand,
    slice::create_slice_subcommand, write::create_write_subcommand,
};
use crate::{
    cli::utils as cli_utils,
    create_subcommand, exit_on_error, exit_with_error,
    field::quantities,
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    io::{
        snapshot::{
            self, fdt,
            native::{
                self, NativeSnapshotParameters, NativeSnapshotReader3, NativeSnapshotReaderConfig,
            },
            SnapshotCacher3, SnapshotReader3,
        },
        utils as io_utils, Endianness,
    },
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::{collections::HashMap, path::PathBuf, process, str::FromStr};

#[cfg(feature = "tracing")]
use super::tracing::create_trace_subcommand;

#[cfg(feature = "ebeam")]
use super::ebeam::create_ebeam_subcommand;

#[cfg(feature = "netcdf")]
use crate::io::snapshot::netcdf::{
    self, NetCDFSnapshotParameters, NetCDFSnapshotReader3, NetCDFSnapshotReaderConfig,
};

/// Builds a representation of the `snapshot` command line subcommand.
#[allow(clippy::let_and_return)]
pub fn create_snapshot_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("snapshot")
        .about("Specify input snapshot to perform further actions on")
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .arg(
            Arg::with_name("input-file")
                .value_name("INPUT_FILE")
                .help(
                    "Path to the file representing the snapshot.\n\
                     Assumes the following format based on the file extension:\
                     \n    *.idl: Parameter file with associated .snap [and .aux] file\
                     \n    *.nc: NetCDF file using the CF convention (requires the netcdf feature)",
                )
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("snap-range")
                .short("r")
                .long("snap-range")
                .require_equals(true)
                .require_delimiter(true)
                .value_names(&["FIRST", "LAST"])
                .help(
                    "Inclusive range of snapshot numbers associated with the input snapshot to\n\
                     process [default: only process INPUT_FILE]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("endianness")
                .short("e")
                .long("endianness")
                .require_equals(true)
                .value_name("ENDIANNESS")
                .help("Endianness to assume for snapshots in native binary format\n")
                .takes_value(true)
                .possible_values(&["little", "big", "native"])
                .default_value("little"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages related to reading"),
        )
        .subcommand(create_subcommand!(snapshot, inspect))
        .subcommand(create_subcommand!(snapshot, slice))
        .subcommand(create_subcommand!(snapshot, resample))
        .subcommand(create_subcommand!(snapshot, write));

    #[cfg(feature = "tracing")]
    let app = app.subcommand(create_subcommand!(snapshot, trace));

    #[cfg(feature = "ebeam")]
    let app = app.subcommand(create_subcommand!(snapshot, ebeam));

    app
}

macro_rules! create_native_reader_and_run {
    ($config:expr, $snap_num_offset:expr, $run_macro:tt) => {{
        let parameters = exit_on_error!(
            NativeSnapshotParameters::new($config.param_file_path(), $config.verbose()),
            "Error: Could not read parameter file: {}"
        );
        let mesh_path = exit_on_error!(
            parameters.determine_mesh_path(),
            "Error: Could not obtain path to mesh file: {}"
        );
        let (
            detected_grid_type,
            center_coords,
            lower_edge_coords,
            up_derivatives,
            down_derivatives,
        ) = exit_on_error!(
            native::parse_mesh_file(mesh_path, $config.verbose()),
            "Error: Could not parse mesh file: {}"
        );
        let is_periodic = exit_on_error!(
            parameters.determine_grid_periodicity(),
            "Error: Could not determine grid periodicity: {}"
        );
        match detected_grid_type {
            GridType::Regular => {
                let grid = RegularGrid3::from_coords(
                    center_coords,
                    lower_edge_coords,
                    is_periodic,
                    Some(up_derivatives),
                    Some(down_derivatives),
                );
                let reader = exit_on_error!(
                    NativeSnapshotReader3::<RegularGrid3<fdt>>::new_from_parameters_and_grid(
                        $config, parameters, grid,
                    ),
                    "Error: Could not create snapshot reader: {}"
                );
                $run_macro!(reader, $snap_num_offset)
            }
            GridType::HorRegular => {
                let grid = HorRegularGrid3::from_coords(
                    center_coords,
                    lower_edge_coords,
                    is_periodic,
                    Some(up_derivatives),
                    Some(down_derivatives),
                );
                let reader = exit_on_error!(
                    NativeSnapshotReader3::<HorRegularGrid3<fdt>>::new_from_parameters_and_grid(
                        $config, parameters, grid,
                    ),
                    "Error: Could not create snapshot reader: {}"
                );
                $run_macro!(reader, $snap_num_offset)
            }
        }
    }};
}

#[cfg(feature = "netcdf")]
macro_rules! create_netcdf_reader_and_run {
    ($config:expr, $snap_num_offset:expr, $run_macro:tt) => {{
        let file = exit_on_error!(
            netcdf::open_file($config.file_path()),
            "Error: Could not open NetCDF file: {}"
        );
        let parameters = exit_on_error!(
            NetCDFSnapshotParameters::new(&file, $config.verbose()),
            "Error: Could not read snapshot parameters from NetCDF file: {}"
        );
        let (
            detected_grid_type,
            center_coords,
            lower_edge_coords,
            up_derivatives,
            down_derivatives,
            endianness,
        ) = exit_on_error!(
            netcdf::read_grid_data(&file, $config.verbose()),
            "Error: Could not read grid data from NetCDF file: {}"
        );
        let is_periodic = exit_on_error!(
            parameters.determine_grid_periodicity(),
            "Error: Could not determine grid periodicity: {}"
        );
        match detected_grid_type {
            GridType::Regular => {
                let grid = RegularGrid3::from_coords(
                    center_coords,
                    lower_edge_coords,
                    is_periodic,
                    up_derivatives,
                    down_derivatives,
                );
                $run_macro!(
                    NetCDFSnapshotReader3::<RegularGrid3<fdt>>::new_from_parameters_and_grid(
                        $config, file, parameters, grid, endianness,
                    ),
                    $snap_num_offset
                )
            }
            GridType::HorRegular => {
                let grid = HorRegularGrid3::from_coords(
                    center_coords,
                    lower_edge_coords,
                    is_periodic,
                    up_derivatives,
                    down_derivatives,
                );
                $run_macro!(
                    NetCDFSnapshotReader3::<HorRegularGrid3<fdt>>::new_from_parameters_and_grid(
                        $config, file, parameters, grid, endianness,
                    ),
                    $snap_num_offset
                )
            }
        }
    }};
}

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_snapshot_subcommand(arguments: &ArgMatches) {
    macro_rules! run_subcommands_for_reader {
        ($reader:expr, $snap_num_offset:expr) => {{
            let mut snapshot = SnapshotCacher3::new($reader);

            if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
                inspect::run_inspect_subcommand(inspect_arguments, snapshot.reader());
            }
            if let Some(slice_arguments) = arguments.subcommand_matches("slice") {
                slice::run_slice_subcommand(slice_arguments, &mut snapshot, $snap_num_offset);
            }
            if let Some(resample_arguments) = arguments.subcommand_matches("resample") {
                resample::run_resample_subcommand(
                    resample_arguments,
                    snapshot.reader(),
                    $snap_num_offset,
                );
            }
            if let Some(write_arguments) = arguments.subcommand_matches("write") {
                write::run_write_subcommand(
                    write_arguments,
                    snapshot.reader(),
                    $snap_num_offset,
                    None,
                    HashMap::new(),
                    |field| Ok(field),
                );
            }
            #[cfg(feature = "tracing")]
            {
                if let Some(trace_arguments) = arguments.subcommand_matches("trace") {
                    crate::cli::tracing::run_trace_subcommand(
                        trace_arguments,
                        &mut snapshot,
                        $snap_num_offset,
                    );
                }
            }
            #[cfg(feature = "ebeam")]
            {
                if let Some(ebeam_arguments) = arguments.subcommand_matches("ebeam") {
                    crate::cli::ebeam::run_ebeam_subcommand(
                        ebeam_arguments,
                        &mut snapshot,
                        $snap_num_offset,
                    );
                }
            }
        }};
    }

    let input_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("input-file")
                .expect("Required argument not present."),
        ),
        "Error: Could not interpret path to input file: {}"
    );

    let input_extension = input_file_path
        .extension()
        .unwrap_or_else(|| exit_with_error!("Error: Missing extension for input-file"))
        .to_string_lossy()
        .to_string();

    let input_snap_paths_and_num_offsets = match arguments.values_of("snap-range") {
        Some(value_strings) => {
            let snap_num_range: Vec<u32> = value_strings
                .map(|value_string| cli_utils::parse_value_string("snap-range", value_string))
                .collect();
            exit_on_false!(
                snap_num_range.len() == 2,
                "Error: Invalid number of values for snap-range (must be 2)"
            );
            exit_on_false!(
                snap_num_range[1] > snap_num_range[0],
                "Error: Last snapshot number must be larger than first snapshot number"
            );

            let (snap_name, _) =
                snapshot::extract_name_and_num_from_snapshot_path(&input_file_path);

            (snap_num_range[0]..=snap_num_range[1])
                .map(|snap_num| {
                    (
                        input_file_path.with_file_name(snapshot::create_snapshot_file_name(
                            &snap_name,
                            snap_num,
                            &input_extension,
                        )),
                        Some(snap_num - snap_num_range[0]),
                    )
                })
                .collect()
        }
        None => vec![(input_file_path, None)],
    };

    let endianness = match arguments
        .value_of("endianness")
        .expect("No value for argument with default.")
    {
        "little" => Endianness::Little,
        "big" => Endianness::Big,
        "native" => Endianness::Native,
        invalid => exit_with_error!("Error: Invalid endianness {}", invalid),
    };

    let verbose = arguments.is_present("verbose").into();

    match input_extension.as_ref() {
        "idl" => {
            for (file_path, snap_num_offset) in input_snap_paths_and_num_offsets {
                let reader_config = NativeSnapshotReaderConfig::new(file_path, endianness, verbose);
                create_native_reader_and_run!(
                    reader_config,
                    snap_num_offset,
                    run_subcommands_for_reader
                );
            }
        }
        "nc" => {
            #[cfg(feature = "netcdf")]
            {
                for (file_path, snap_num_offset) in input_snap_paths_and_num_offsets {
                    let reader_config = NetCDFSnapshotReaderConfig::new(file_path, verbose);
                    create_netcdf_reader_and_run!(
                        reader_config,
                        snap_num_offset,
                        run_subcommands_for_reader
                    );
                }
            }
            #[cfg(not(feature = "netcdf"))]
            exit_with_error!("Error: Compile with netcdf feature in order to read NetCDF files\n\
                              Tip: Use cargo flag --features=netcdf and make sure the NetCDF library is available");
        }
        invalid => exit_with_error!("Error: Invalid extension {} for input-file", invalid),
    }
}

fn parse_quantity_lists<'a, G, R>(
    arguments: &'a ArgMatches,
    reader: &'a R,
    continue_on_warnings: bool,
) -> (Vec<&'a str>, Vec<&'a str>)
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
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
                    if !continue_on_warnings && !io_utils::user_says_yes("Still continue?", true) {
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
                        if !continue_on_warnings
                            && !io_utils::user_says_yes("Still continue?", true)
                        {
                            process::exit(1);
                        }
                        false
                    }
                }
                None => {
                    eprintln!("Warning: Derived quantity {} not supported", quantity_name);
                    if !continue_on_warnings && !io_utils::user_says_yes("Still continue?", true) {
                        process::exit(1);
                    }
                    false
                }
            }
        })
        .collect();

    (included_quantities, derived_quantities)
}
