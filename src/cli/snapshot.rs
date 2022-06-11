//! Command line interface for actions related to snapshots.

mod derive;
mod extract;
mod inspect;
mod resample;
mod slice;
mod write;

#[cfg(feature = "corks")]
mod corks;

#[cfg(feature = "synthesis")]
mod synthesize;

use self::{
    derive::create_derive_subcommand, extract::create_extract_subcommand,
    inspect::create_inspect_subcommand, resample::create_resample_subcommand,
    slice::create_slice_subcommand, write::create_write_subcommand,
};
use crate::{
    add_subcommand_combinations,
    cli::utils as cli_utils,
    exit_on_error, exit_on_false, exit_with_error,
    grid::{fgr, Grid3},
    io::{
        snapshot::{
            self,
            utils::{SnapNumInRange, SnapshotInputType},
            CachingSnapshotProvider3, SnapshotProvider3,
        },
        utils::IOContext,
        Endianness, Verbosity,
    },
    update_command_graph, with_new_snapshot_reader,
};
use clap::{Arg, ArgMatches, Command, ValueHint};
use std::{collections::HashSet, path::PathBuf, str::FromStr};

#[cfg(feature = "tracing")]
use super::tracing::create_trace_subcommand;

#[cfg(feature = "corks")]
use self::corks::{create_corks_subcommand, CorksState};

#[cfg(feature = "ebeam")]
use super::ebeam::create_ebeam_subcommand;

#[cfg(feature = "synthesis")]
use self::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot` command line subcommand.
#[allow(clippy::let_and_return)]
pub fn create_snapshot_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "snapshot";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Specify input snapshot to perform further actions on")
        .arg(
            Arg::new("input-file")
                .value_name("INPUT_FILE")
                .help(
                    "Path to the file representing the snapshot.\n\
                     Assumes the following format based on the file extension:\
                     \n    *.idl: Parameter file with associated .snap [and .aux] file\
                     \n    *.nc: NetCDF file using the CF convention (requires the netcdf feature)",
                )
                .required(true)
                .takes_value(true)
                .value_hint(ValueHint::FilePath),
        )
        .arg(
            Arg::new("snap-range")
                .short('r')
                .long("snap-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["FIRST", "LAST"])
                .help(
                    "Inclusive range of snapshot numbers associated with the input\n\
                     snapshot to process [default: only process INPUT_FILE]",
                )
                .takes_value(true)
                .number_of_values(2),
        )
        .arg(
            Arg::new("endianness")
                .short('e')
                .long("endianness")
                .require_equals(true)
                .value_name("ENDIANNESS")
                .help("Endianness to assume for snapshots in native binary format\n")
                .takes_value(true)
                .possible_values(&["little", "big", "native"])
                .default_value("little"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to reading"),
        )
        .arg(
            Arg::new("force-horizontally-regular")
                .long("force-horizontally-regular")
                .help("Treat the grid as horizontally regular even if fully regular")
                .hide(true),
        );

    add_subcommand_combinations!(
        command, command_name, true;
        derive,
        synthesize if "synthesis",
        (inspect, slice, extract, resample, write, corks if "corks", trace if "tracing", ebeam if "ebeam")
    )
}

/// Runs the actions for the `snapshot` subcommand using the given arguments.
pub fn run_snapshot_subcommand(arguments: &ArgMatches, io_context: &mut IOContext) {
    let input_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("input-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to input file: {}"
    );

    let input_type = SnapshotInputType::from_path(&input_file_path);

    let input_snap_paths_and_num_offsets = match arguments.values_of("snap-range") {
        Some(value_strings) => {
            exit_on_false!(
                !input_type.is_scratch(),
                "Error: snap-range not supported for scratch files"
            );

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

            (snap_num_range[0]..=snap_num_range[1])
                .map(|snap_num| {
                    (
                        input_file_path.with_file_name(
                            snapshot::create_new_snapshot_file_name_from_path(
                                &input_file_path,
                                snap_num,
                                &input_type.to_string(),
                                false,
                            ),
                        ),
                        Some(SnapNumInRange::new(
                            snap_num_range[0],
                            snap_num_range[1],
                            snap_num,
                        )),
                    )
                })
                .collect()
        }
        None => vec![(input_file_path, None)],
    };

    let endianness = match arguments
        .value_of("endianness")
        .expect("No value for argument with default")
    {
        "little" => Endianness::Little,
        "big" => Endianness::Big,
        "native" => Endianness::Native,
        invalid => exit_with_error!("Error: Invalid endianness {}", invalid),
    };

    let verbosity: Verbosity = cli_utils::parse_verbosity(arguments, false);

    let force_hor_regular = arguments.is_present("force-horizontally-regular");

    for (file_path, snap_num_in_range) in input_snap_paths_and_num_offsets {
        io_context.set_snap_num_in_range(snap_num_in_range);
        exit_on_error!(
            with_new_snapshot_reader!(
                file_path,
                endianness,
                verbosity.clone(),
                force_hor_regular,
                |reader| {
                    run_snapshot_subcommand_with_derive(arguments, reader, io_context);
                    Ok(())
                }
            ),
            "Error: {}"
        );
    }
}

fn run_snapshot_subcommand_with_derive<G, P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    if let Some(derive_arguments) = arguments.subcommand_matches("derive") {
        let provider = derive::create_derive_provider(derive_arguments, provider);
        run_snapshot_subcommand_with_synthesis(derive_arguments, provider, io_context);
    } else {
        run_snapshot_subcommand_with_synthesis_added_caching(arguments, provider, io_context);
    }
}

fn run_snapshot_subcommand_with_synthesis<G, P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider = synthesize::create_synthesize_provider(synthesize_arguments, provider);
        run_snapshot_subcommand_for_provider(synthesize_arguments, provider, io_context);
        return;
    }

    run_snapshot_subcommand_for_provider(arguments, provider, io_context);
}

fn run_snapshot_subcommand_with_synthesis_added_caching<G, P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider =
            synthesize::create_synthesize_provider_added_caching(synthesize_arguments, provider);
        run_snapshot_subcommand_for_provider(synthesize_arguments, provider, io_context);
        return;
    }

    run_snapshot_subcommand_for_provider(arguments, provider, io_context);
}

fn run_snapshot_subcommand_for_provider<G, P>(
    arguments: &ArgMatches,
    provider: P,
    io_context: &mut IOContext,
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
        inspect::run_inspect_subcommand(inspect_arguments, provider, io_context);
    } else if let Some(slice_arguments) = arguments.subcommand_matches("slice") {
        slice::run_slice_subcommand(slice_arguments, provider, io_context);
    } else if let Some(extract_arguments) = arguments.subcommand_matches("extract") {
        extract::run_extract_subcommand(extract_arguments, provider, io_context);
    } else if let Some(resample_arguments) = arguments.subcommand_matches("resample") {
        resample::run_resample_subcommand(resample_arguments, provider, io_context);
    } else if let Some(write_arguments) = arguments.subcommand_matches("write") {
        write::run_write_subcommand(write_arguments, provider, io_context);
    } else {
        let corks_arguments = if cfg!(feature = "corks") {
            arguments.subcommand_matches("corks")
        } else {
            None
        };
        let trace_arguments = if cfg!(feature = "tracing") {
            arguments.subcommand_matches("trace")
        } else {
            None
        };
        let ebeam_arguments = if cfg!(feature = "ebeam") {
            arguments.subcommand_matches("ebeam")
        } else {
            None
        };
        if let Some(_corks_arguments) = corks_arguments {
            #[cfg(feature = "corks")]
            {
                let mut corks_state: Option<CorksState> = None;
                corks::run_corks_subcommand(
                    _corks_arguments,
                    provider,
                    io_context,
                    &mut corks_state,
                );
            }
        } else if let Some(_trace_arguments) = trace_arguments {
            #[cfg(feature = "tracing")]
            crate::cli::tracing::run_trace_subcommand(_trace_arguments, provider, io_context);
        } else if let Some(_ebeam_arguments) = ebeam_arguments {
            #[cfg(feature = "ebeam")]
            crate::cli::ebeam::run_ebeam_subcommand(_ebeam_arguments, provider, io_context);
        }
    }
}

fn parse_included_quantity_list<G, P>(
    arguments: &ArgMatches,
    provider: &P,
    continue_on_warnings: bool,
) -> Vec<String>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    remove_duplicate_quantities(
        if let Some(excluded_quantities) = arguments.values_of("excluded-quantities") {
            let excluded_quantities: Vec<_> = excluded_quantities
                .into_iter()
                .filter_map(|name| {
                    let name = name.trim().to_lowercase();
                    if name.is_empty() {
                        None
                    } else {
                        Some(name)
                    }
                })
                .collect();
            provider.all_variable_names_except(&excluded_quantities)
        } else {
            let included_quantities = arguments
                .values_of("included-quantities")
                .expect("No default for included-quantities")
                .collect::<Vec<_>>();
            if included_quantities.len() == 1
                && included_quantities[0].trim().to_lowercase() == "all"
            {
                provider.all_variable_names().to_vec()
            } else {
                included_quantities
                    .into_iter()
                    .filter_map(|name| {
                        let name = name.trim().to_lowercase();
                        if name.is_empty() {
                            None
                        } else {
                            let has_variable = provider.has_variable(&name);
                            if has_variable {
                                Some(name)
                            } else {
                                eprintln!("Warning: Quantity {} not available", &name);
                                if !continue_on_warnings {
                                    cli_utils::verify_user_will_continue_or_abort()
                                }
                                None
                            }
                        }
                    })
                    .collect()
            }
        },
    )
}

fn remove_duplicate_quantities(mut quantity_names: Vec<String>) -> Vec<String> {
    let mut uniques = HashSet::new();
    quantity_names.retain(|name| uniques.insert(name.clone()));
    quantity_names
}
