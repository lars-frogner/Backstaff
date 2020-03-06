//! Command line interface for actions related to snapshots.

mod inspect;
mod modify_aux;
mod resample;
mod slice;

use crate::grid::hor_regular::HorRegularGrid3;
use crate::grid::regular::RegularGrid3;
use crate::grid::GridType;
use crate::io::snapshot::{SnapshotReader3, SnapshotReaderConfig};
use crate::io::Endianness;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `snapshot` command line subcommand.
#[allow(clippy::let_and_return)]
pub fn create_snapshot_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("snapshot")
        .about("Specify input snapshot to perform further actions on")
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("param-path")
                .short("p")
                .long("param-path")
                .require_equals(true)
                .value_name("PATH")
                .help("Path to the parameter (.idl) file for the snapshot")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("grid-type")
                .short("g")
                .long("grid-type")
                .require_equals(true)
                .value_name("TYPE")
                .help("Type of grid to assume for the snapshot\n")
                .takes_value(true)
                .possible_values(&["horizontally-regular", "regular"])
                .default_value("horizontally-regular"),
        )
        .arg(
            Arg::with_name("endianness")
                .short("e")
                .long("endianness")
                .require_equals(true)
                .value_name("ENDIANNESS")
                .help("Endianness to assume for the snapshot\n")
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
        .subcommand(inspect::create_inspect_subcommand())
        .subcommand(slice::create_slice_subcommand())
        .subcommand(resample::create_resample_subcommand())
        .subcommand(modify_aux::create_modify_aux_subcommand());

    #[cfg(feature = "tracing")]
    let app = app.subcommand(crate::cli::tracing::create_trace_subcommand());

    #[cfg(feature = "ebeam")]
    let app = app.subcommand(crate::cli::ebeam::create_ebeam_subcommand());

    app
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
        .value_of("param-path")
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

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_snapshot_subcommand(arguments: &ArgMatches) {
    let (grid_type, reader_config) = construct_snapshot_reader_config_from_arguments(arguments);

    macro_rules! run_subcommands_for_grid_type {
        ($grid_type:ty) => {{
            let reader = SnapshotReader3::<$grid_type>::new(reader_config)
                .unwrap_or_else(|err| panic!("Could not read snapshot: {}", err));
            let mut snapshot = reader.into_cacher();

            if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
                inspect::run_inspect_subcommand(inspect_arguments, &mut snapshot);
            }
            if let Some(slice_arguments) = arguments.subcommand_matches("slice") {
                slice::run_slice_subcommand(slice_arguments, &mut snapshot);
            }
            if let Some(resample_arguments) = arguments.subcommand_matches("resample") {
                resample::run_resample_subcommand(resample_arguments, snapshot.reader());
            }
            if let Some(modify_aux_arguments) = arguments.subcommand_matches("modify_aux") {
                modify_aux::run_modify_aux_subcommand(modify_aux_arguments, snapshot.reader_mut());
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

    match grid_type {
        GridType::HorRegular => run_subcommands_for_grid_type!(HorRegularGrid3<_>),
        GridType::Regular => run_subcommands_for_grid_type!(RegularGrid3<_>),
    }
}
