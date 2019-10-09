//! Command line interface.

pub mod ebeam;
pub mod interpolation;
pub mod mesh;
pub mod snapshot;
pub mod tracing;

use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use clap::{self, App, AppSettings, Arg, ArgMatches};
use num;
use std::time::Instant;
use std::{str, string};

/// Runs the `bifrost` command line program.
pub fn run() {
    let app = App::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .global_setting(AppSettings::VersionlessSubcommands)
        .global_setting(AppSettings::DeriveDisplayOrder)
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("timing")
                .short("t")
                .long("timing")
                .help("Display elapsed time when done"),
        )
        .subcommand(snapshot::create_snapshot_subcommand())
        .subcommand(mesh::create_create_mesh_subcommand());

    let arguments = app.get_matches();

    let start_instant = Instant::now();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        snapshot::run_snapshot_subcommand(snapshot_arguments);
    }
    if let Some(create_mesh_arguments) = arguments.subcommand_matches("create_mesh") {
        mesh::run_create_mesh_subcommand(create_mesh_arguments);
    }

    if arguments.is_present("timing") {
        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());
    }
}

fn parse_value_string<T>(argument_name: &str, value_string: &str) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    value_string
        .parse()
        .unwrap_or_else(|err| panic!("Could not parse value of {}: {}", argument_name, err))
}

fn parse_value_strings<'a, 'b, T, I>(argument_name: &'a str, value_strings: I) -> Vec<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
    I: Iterator<Item = &'b str>,
{
    value_strings
        .map(|value_string| parse_value_string(argument_name, value_string))
        .collect()
}

fn get_value_from_required_parseable_argument<T>(arguments: &ArgMatches, argument_name: &str) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    parse_value_string(
        argument_name,
        arguments
            .value_of(argument_name)
            .expect("No value for required argument."),
    )
}

fn get_values_from_required_parseable_argument<T>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> Vec<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    parse_value_strings(
        argument_name,
        arguments
            .values_of(argument_name)
            .expect("No values for required argument."),
    )
}

fn get_value_from_parseable_argument_with_custom_default<T, D>(
    arguments: &ArgMatches,
    argument_name: &str,
    default_constructor: &D,
) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
    D: Fn() -> T,
{
    if let Some(value_string) = arguments.value_of(argument_name) {
        parse_value_string(argument_name, value_string)
    } else {
        default_constructor()
    }
}

fn get_values_from_parseable_argument_with_custom_defaults<T, D>(
    arguments: &ArgMatches,
    argument_name: &str,
    default_constructor: &D,
) -> Vec<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
    D: Fn() -> Vec<T>,
{
    if let Some(value_strings) = arguments.values_of(argument_name) {
        value_strings
            .map(|value_string| parse_value_string(argument_name, value_string))
            .collect()
    } else {
        default_constructor()
    }
}

#[allow(dead_code)]
fn get_value_from_constrained_argument_with_custom_default<T, D>(
    arguments: &ArgMatches,
    argument_name: &str,
    possible_value_strings: &[&str],
    possible_values: &[T],
    default_constructor: &D,
) -> T
where
    T: Copy,
    D: Fn() -> T,
{
    if let Some(value_string) = arguments.value_of(argument_name) {
        let mut value: Option<T> = None;
        for (possible_value_string, possible_value) in
            possible_value_strings.iter().zip(possible_values)
        {
            if *possible_value_string == value_string {
                value = Some(*possible_value);
                break;
            }
        }
        value.unwrap_or_else(|| panic!("Invalid {}: {}", argument_name, value_string))
    } else {
        default_constructor()
    }
}

fn get_value_from_required_constrained_argument<T>(
    arguments: &ArgMatches,
    argument_name: &str,
    possible_value_strings: &[&str],
    possible_values: &[T],
) -> T
where
    T: Copy,
{
    let value_string = arguments
        .value_of(argument_name)
        .expect("No value for required argument.");
    let mut value: Option<T> = None;
    for (possible_value_string, possible_value) in
        possible_value_strings.iter().zip(possible_values)
    {
        if *possible_value_string == value_string {
            value = Some(*possible_value);
            break;
        }
    }
    value.unwrap_or_else(|| panic!("Invalid {}: {}", argument_name, value_string))
}

#[allow(dead_code)]
fn get_value_from_parseable_argument_with_default<T>(
    arguments: &ArgMatches,
    argument_name: &str,
    default_value: T,
) -> T
where
    T: std::str::FromStr + Copy,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    get_value_from_parseable_argument_with_custom_default(arguments, argument_name, &|| {
        default_value
    })
}

fn get_value_from_param_file_argument_with_default<G, T, C>(
    reader: &SnapshotReader3<G>,
    arguments: &ArgMatches,
    argument_name: &str,
    param_file_argument_name: &str,
    conversion_mapping: &C,
    default_value: T,
) -> T
where
    G: Grid3<fdt>,
    T: num::Num + str::FromStr + std::fmt::Display + Copy,
    T::Err: string::ToString,
    <T as str::FromStr>::Err: std::fmt::Display,
    C: Fn(T) -> T,
{
    get_value_from_parseable_argument_with_custom_default(arguments, argument_name, &|| {
        reader.get_converted_numerical_param_or_fallback_to_default_with_warning(
            argument_name,
            param_file_argument_name,
            conversion_mapping,
            default_value,
        )
    })
}

fn get_values_from_param_file_argument_with_defaults<G, T, C>(
    reader: &SnapshotReader3<G>,
    arguments: &ArgMatches,
    argument_name: &str,
    param_file_argument_names: &[&str],
    conversion_mapping: &C,
    default_values: &[T],
) -> Vec<T>
where
    G: Grid3<fdt>,
    T: num::Num + str::FromStr + std::fmt::Display + Copy,
    T::Err: string::ToString,
    <T as str::FromStr>::Err: std::fmt::Display,
    C: Fn(T) -> T,
{
    get_values_from_parseable_argument_with_custom_defaults(arguments, argument_name, &|| {
        param_file_argument_names
            .iter()
            .zip(default_values)
            .map(|(&param_file_argument_name, &default_value)| {
                reader.get_converted_numerical_param_or_fallback_to_default_with_warning(
                    argument_name,
                    param_file_argument_name,
                    conversion_mapping,
                    default_value,
                )
            })
            .collect()
    })
}
