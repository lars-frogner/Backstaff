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
    create_subcommand, exit_on_error, exit_with_error,
    grid::{hor_regular::HorRegularGrid3, regular::RegularGrid3, Grid3, GridType},
    io::{
        snapshot::{
            fdt,
            native::{
                self, NativeSnapshotParameters, NativeSnapshotReader3, NativeSnapshotReaderConfig,
            },
            SnapshotCacher3,
        },
        Endianness,
    },
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::{collections::HashMap, path::PathBuf, str::FromStr};

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
                     Assumes the following format based on the file extension:\n\
                     *.idl: Parameter file with associated .snap [and .aux] file\n\
                     *.nc: NetCDF file using the CF convention (requires the netcdf feature)",
                )
                .required(true)
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
    ($config:expr, $run_macro:tt) => {{
        let parameters = exit_on_error!(
            NativeSnapshotParameters::new($config.param_file_path()),
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
            native::parse_mesh_file(mesh_path),
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
                $run_macro!(reader)
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
                $run_macro!(reader)
            }
        }
    }};
}

#[cfg(feature = "netcdf")]
macro_rules! create_netcdf_reader_and_run {
    ($config:expr, $run_macro:tt) => {{
        let file = exit_on_error!(
            netcdf::open_file($config.path()),
            "Error: Could not open NetCDF file: {}"
        );
        let parameters = exit_on_error!(
            NetCDFSnapshotParameters::new(&file),
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
            netcdf::read_grid_data(&file.root().unwrap()),
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
                    )
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
                    )
                )
            }
        }
    }};
}

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_snapshot_subcommand(arguments: &ArgMatches) {
    macro_rules! run_subcommands_for_reader {
        ($reader:expr) => {{
            let mut snapshot = SnapshotCacher3::new($reader);

            if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
                inspect::run_inspect_subcommand(inspect_arguments, &mut snapshot);
            }
            if let Some(slice_arguments) = arguments.subcommand_matches("slice") {
                slice::run_slice_subcommand(slice_arguments, &mut snapshot);
            }
            if let Some(resample_arguments) = arguments.subcommand_matches("resample") {
                resample::run_resample_subcommand(resample_arguments, snapshot.reader());
            }
            if let Some(write_arguments) = arguments.subcommand_matches("write") {
                write::run_write_subcommand(
                    write_arguments,
                    snapshot.reader(),
                    None,
                    HashMap::new(),
                    |field| Ok(field),
                );
            }
            #[cfg(feature = "tracing")]
            {
                if let Some(trace_arguments) = arguments.subcommand_matches("trace") {
                    crate::cli::tracing::run_trace_subcommand(trace_arguments, &mut snapshot);
                }
            }
            #[cfg(feature = "ebeam")]
            {
                if let Some(ebeam_arguments) = arguments.subcommand_matches("ebeam") {
                    crate::cli::ebeam::run_ebeam_subcommand(ebeam_arguments, &mut snapshot);
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

    match input_file_path
        .extension()
        .unwrap_or_else(|| exit_with_error!("Error: Missing extension for input-file"))
        .to_string_lossy()
        .as_ref()
    {
        "idl" => {
            let reader_config =
                NativeSnapshotReaderConfig::new(&input_file_path, endianness, verbose);
            create_native_reader_and_run!(reader_config, run_subcommands_for_reader);
        }
        "nc" => {
            #[cfg(feature = "netcdf")]
            {
                let reader_config = NetCDFSnapshotReaderConfig::new(&input_file_path, verbose);
                create_netcdf_reader_and_run!(reader_config, run_subcommands_for_reader);
            }
            #[cfg(not(feature = "netcdf"))]
            exit_with_error!("Error: Compile with netcdf feature in order to read NetCDF files\n\
                              Tip: Use cargo flag --features=netcdf and make sure libnetcdf is available")
        }
        invalid => exit_with_error!("Error: Invalid extension {} for input-file", invalid),
    }
}
