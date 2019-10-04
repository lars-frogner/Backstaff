//! Command line interface for actions related to snapshots.

pub mod inspect;
pub mod slice;

use crate::cli;
use crate::grid::hor_regular::HorRegularGrid3;
use crate::grid::regular::RegularGrid3;
use crate::grid::GridType;
use crate::io::snapshot::{SnapshotReader3, SnapshotReaderConfig};
use crate::io::Endianness;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};

/// Adds arguments for parameters used by the snapshot reader.
pub fn add_snapshot_reader_arguments_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("PARAM_PATH")
            .help("Path to the parameter (.idl) file for the snapshot")
            .required(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("grid-type")
            .short("g")
            .long("grid-type")
            .value_name("TYPE")
            .long_help("Type of grid to assume for the snapshot\n")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["horizontally-regular", "regular"])
            .default_value("horizontally-regular"),
    )
    .arg(
        Arg::with_name("endianness")
            .short("e")
            .long("endianness")
            .value_name("ENDIANNESS")
            .long_help("Endianness to assume for the snapshot\n")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["little", "big"])
            .default_value("little"),
    )
    .arg(
        Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Print status messages while reading fields"),
    )
}

/// Determines snapshot reader parameters based on provided options.
pub fn construct_snapshot_reader_config_from_arguments(
    arguments: &ArgMatches,
) -> (GridType, SnapshotReaderConfig) {
    let grid_type = match arguments
        .value_of("grid-type")
        .expect("No value for argument with default.")
    {
        "horizontally-regular" => GridType::HorRegular,
        "regular" => GridType::Regular,
        invalid => panic!("Invalid grid type {}", invalid),
    };

    let param_file_path = arguments
        .value_of("PARAM_PATH")
        .expect("Required argument not present.");

    let endianness = match arguments
        .value_of("endianness")
        .expect("No value for argument with default.")
    {
        "little" => Endianness::Little,
        "big" => Endianness::Big,
        invalid => panic!("Invalid endianness {}", invalid),
    };

    let verbose = arguments.is_present("verbose").into();

    (
        grid_type,
        SnapshotReaderConfig::new(param_file_path, endianness, verbose),
    )
}

/// Builds a representation of the `snapshot` command line subcommand.
pub fn build_subcommand_snapshot<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("snapshot")
        .about("Specify input snapshot to perform further actions on")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(inspect::build_subcommand_inspect())
        .subcommand(slice::build_subcommand_slice())
        .subcommand(cli::tracing::build_subcommand_trace())
        .subcommand(cli::ebeam::build_subcommand_ebeam());

    add_snapshot_reader_arguments_to_subcommand(app)
}

macro_rules! run_subcommand_snapshot_for_grid_type {
    ($grid_type:ty, $arguments:expr, $reader_config:expr) => {{
        let reader = SnapshotReader3::<$grid_type>::new($reader_config)
            .unwrap_or_else(|err| panic!("Could not read snapshot: {}", err));
        let mut snapshot = reader.into_cacher();

        if let Some(inspect_arguments) = $arguments.subcommand_matches("inspect") {
            inspect::run_subcommand_inspect(inspect_arguments, &mut snapshot);
        }
        if let Some(slice_arguments) = $arguments.subcommand_matches("slice") {
            slice::run_subcommand_slice(slice_arguments, &mut snapshot);
        }
        if let Some(trace_arguments) = $arguments.subcommand_matches("trace") {
            cli::tracing::run_subcommand_trace(trace_arguments, &mut snapshot);
        }
        if let Some(ebeam_arguments) = $arguments.subcommand_matches("ebeam") {
            cli::ebeam::run_subcommand_ebeam(ebeam_arguments, &mut snapshot);
        }
    }};
}

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_subcommand_snapshot(arguments: &ArgMatches) {
    let (grid_type, reader_config) = construct_snapshot_reader_config_from_arguments(arguments);
    match grid_type {
        GridType::HorRegular => {
            run_subcommand_snapshot_for_grid_type!(HorRegularGrid3<_>, arguments, reader_config)
        }
        GridType::Regular => {
            run_subcommand_snapshot_for_grid_type!(RegularGrid3<_>, arguments, reader_config)
        }
    }
}
