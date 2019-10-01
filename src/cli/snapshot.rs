//! Command line interface for actions related to snapshots.

pub mod inspect;

use crate::grid::hor_regular::HorRegularGrid3;
use crate::grid::regular::RegularGrid3;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use crate::io::Endianness;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `snapshot` command line subcommand.
pub fn build_subcommand_snapshot<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("snapshot")
        .about("Performs actions related to snapshots")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("grid-type")
                .short("g")
                .long("grid-type")
                .value_name("TYPE")
                .help("Type of grid to assume for the snapshot")
                .takes_value(true)
                .possible_values(&["horizontally-regular", "regular"])
                .default_value("horizontally-regular"),
        )
        .arg(
            Arg::with_name("PARAM_PATH")
                .help("Path to the parameter (.idl) file for the snapshot")
                .required(true)
                .takes_value(true)
                .index(1),
        )
        .arg(
            Arg::with_name("endianness")
                .short("e")
                .long("endianness")
                .value_name("ENDIANNESS")
                .help("Endianness to assume for the snapshot")
                .takes_value(true)
                .possible_values(&["little", "big"])
                .default_value("little"),
        )
        .subcommand(inspect::build_subcommand_inspect());

    add_snapshot_reader_arguments_to_subcommand(app)
}

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_subcommand_snapshot(arguments: &ArgMatches) {
    let grid_type = arguments
        .value_of("grid-type")
        .expect("No value for argument with default.");

    let param_file_path = arguments
        .value_of("PARAM_PATH")
        .expect("Required argument not present.");

    let endianness = arguments
        .value_of("endianness")
        .expect("No value for argument with default.");
    let endianness = if endianness == "little" {
        Endianness::Little
    } else if endianness == "big" {
        Endianness::Big
    } else {
        panic!("Invalid endianness {}", endianness)
    };

    if grid_type == "horizontally-regular" {
        let mut reader = SnapshotReader3::<HorRegularGrid3<_>>::new(param_file_path, endianness)
            .unwrap_or_else(|err| panic!("Could not read snapshot: {}", err));
        configure_snapshot_reader_from_arguments(&mut reader, arguments);

        if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
            inspect::run_subcommand_inspect(inspect_arguments, reader);
        }
    } else if grid_type == "regular" {
        let mut reader = SnapshotReader3::<RegularGrid3<_>>::new(param_file_path, endianness)
            .unwrap_or_else(|err| panic!("Could not read snapshot: {}", err));
        configure_snapshot_reader_from_arguments(&mut reader, arguments);

        if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
            inspect::run_subcommand_inspect(inspect_arguments, reader);
        }
    } else {
        panic!("Invalid grid type {}", grid_type)
    }
}

fn add_snapshot_reader_arguments_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Print status messages"),
    )
}

fn configure_snapshot_reader_from_arguments<G: Grid3<fdt>>(
    reader: &mut SnapshotReader3<G>,
    arguments: &ArgMatches,
) {
    reader.set_verbose(arguments.is_present("verbose").into());
}
