//! Command line interface.

pub mod ebeam;
pub mod snapshot;

use clap::{self, App, AppSettings, Arg, ArgMatches};
use std::time::Instant;

/// Runs the `bifrost` command line program.
pub fn run() {
    let app = App::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("timing")
                .short("t")
                .long("timing")
                .help("Display elapsed time when done"),
        )
        .subcommand(snapshot::build_subcommand_snapshot())
        .subcommand(ebeam::build_subcommand_ebeam());

    let arguments = app.get_matches();

    let start_instant = Instant::now();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        snapshot::run_subcommand_snapshot(snapshot_arguments);
    }

    if let Some(ebeam_arguments) = arguments.subcommand_matches("ebeam") {
        ebeam::run_subcommand_ebeam(ebeam_arguments);
    }

    if arguments.is_present("timing") {
        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());
    }
}

fn assign_bool_value_from_flag_presence(
    value: &mut bool,
    arguments: &ArgMatches,
    argument_name: &str,
) {
    *value = arguments.is_present(argument_name);
}

fn assign_value_from_parseable_argument<T>(
    value: &mut T,
    arguments: &ArgMatches,
    argument_name: &str,
) where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    if let Some(value_str) = arguments.value_of(argument_name) {
        *value = value_str
            .parse()
            .unwrap_or_else(|err| panic!("Could not parse value of {}: {}", argument_name, err));
    }
}

fn assign_value_from_selected_argument<T>(
    value: &mut T,
    arguments: &ArgMatches,
    argument_name: &str,
    possible_value_strs: &[&str],
    possible_values: &[T],
) where
    T: Copy,
{
    if let Some(value_str) = arguments.value_of(argument_name) {
        let mut was_assigned = false;
        for (possible_value_str, possible_value) in possible_value_strs.iter().zip(possible_values)
        {
            if *possible_value_str == value_str {
                *value = *possible_value;
                was_assigned = true;
                break;
            }
        }
        if !was_assigned {
            panic!("Invalid {}: {}", argument_name, value_str)
        }
    }
}
