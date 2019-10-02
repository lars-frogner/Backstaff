//! Command line interface for actions related to snapshots.

pub mod inspect;
pub mod slice;

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
            Arg::with_name("PARAM_PATH")
                .help("Path to the parameter (.idl) file for the snapshot")
                .required(true)
                .takes_value(true)
                .index(1),
        )
        .arg(
            Arg::with_name("grid-type")
                .short("g")
                .long("grid-type")
                .value_name("TYPE")
                .long_help("Type of grid to assume for the snapshot\n")
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
                .takes_value(true)
                .possible_values(&["little", "big"])
                .default_value("little"),
        )
        .subcommand(inspect::build_subcommand_inspect())
        .subcommand(slice::build_subcommand_slice());

    add_snapshot_reader_options_to_subcommand(app)
}

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_subcommand_snapshot(arguments: &ArgMatches) {
    let param_file_path = arguments
        .value_of("PARAM_PATH")
        .expect("Required argument not present.");

    let grid_type = arguments
        .value_of("grid-type")
        .expect("No value for argument with default.");

    let endianness = match arguments
        .value_of("endianness")
        .expect("No value for argument with default.")
    {
        "little" => Endianness::Little,
        "big" => Endianness::Big,
        invalid => panic!("Invalid endianness {}", invalid),
    };

    if grid_type == "horizontally-regular" {
        let mut reader = SnapshotReader3::<HorRegularGrid3<_>>::new(param_file_path, endianness)
            .unwrap_or_else(|err| panic!("Could not read snapshot: {}", err));
        configure_snapshot_reader_from_options(&mut reader, arguments);
        let mut cacher = reader.into_cacher();

        if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
            inspect::run_subcommand_inspect(inspect_arguments, &mut cacher);
        }
        if let Some(slice_arguments) = arguments.subcommand_matches("slice") {
            slice::run_subcommand_slice(slice_arguments, &mut cacher);
        }
    } else if grid_type == "regular" {
        let mut reader = SnapshotReader3::<RegularGrid3<_>>::new(param_file_path, endianness)
            .unwrap_or_else(|err| panic!("Could not read snapshot: {}", err));
        configure_snapshot_reader_from_options(&mut reader, arguments);
        let mut cacher = reader.into_cacher();

        if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
            inspect::run_subcommand_inspect(inspect_arguments, &mut cacher);
        }
        if let Some(slice_arguments) = arguments.subcommand_matches("slice") {
            slice::run_subcommand_slice(slice_arguments, &mut cacher);
        }
    } else {
        panic!("Invalid grid type {}", grid_type)
    }
}

/// Adds arguments for parameters used by the snapshot reader.
pub fn add_snapshot_reader_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .long_help("Print status messages"),
    )
}

/// Sets snapshot reader parameters based on present arguments.
pub fn configure_snapshot_reader_from_options<G: Grid3<fdt>>(
    reader: &mut SnapshotReader3<G>,
    arguments: &ArgMatches,
) {
    reader.set_verbose(arguments.is_present("verbose").into());
}
