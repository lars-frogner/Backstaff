//! Utilities for creating the command line interface.

use crate::{
    exit_on_error,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotParameters, SnapshotReader3},
};
use clap::{self, ArgMatches};
use std::str::FromStr;

#[macro_export]
macro_rules! create_subcommand {
    ($parent_command:ident, $child_command:ident) => {{
        let subcommand = paste::expr! { [<create_ $child_command _subcommand>]() };
        if !subcommand.p.is_set(clap::AppSettings::Hidden) {
            crate::cli::command_graph::insert_command_graph_edge(
                stringify!($parent_command),
                stringify!($child_command),
                1.0,
            );
        }
        subcommand
    }};
}

pub fn parse_value_string<T>(argument_name: &str, value_string: &str) -> T
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    exit_on_error!(
        value_string.parse(),
        "Error: Could not parse value of {0}: {1}",
        argument_name
    )
}

fn parse_value_strings<'a, 'b, T, I>(argument_name: &'a str, value_strings: I) -> Vec<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
    I: Iterator<Item = &'b str>,
{
    value_strings
        .map(|value_string| parse_value_string(argument_name, value_string))
        .collect()
}

pub fn get_value_from_required_parseable_argument<T>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> T
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    parse_value_string(
        argument_name,
        arguments
            .value_of(argument_name)
            .expect("No value for required argument."),
    )
}

pub fn get_values_from_parseable_argument<T>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> Option<Vec<T>>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    arguments
        .values_of(argument_name)
        .map(|values| parse_value_strings(argument_name, values))
}

pub fn get_values_from_required_parseable_argument<T>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> Vec<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
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
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
    D: Fn() -> T,
{
    if let Some(value_string) = arguments.value_of(argument_name) {
        parse_value_string(argument_name, value_string)
    } else {
        default_constructor()
    }
}

pub fn get_values_from_parseable_argument_with_custom_defaults<T, D>(
    arguments: &ArgMatches,
    argument_name: &str,
    default_constructor: &D,
) -> Vec<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
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
        value.unwrap_or_else(|| {
            exit_with_error!(
                "Error: Invalid value for {}: {}",
                argument_name,
                value_string
            )
        })
    } else {
        default_constructor()
    }
}

pub fn get_value_from_required_constrained_argument<T>(
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
    value.unwrap_or_else(|| {
        exit_with_error!(
            "Error: Invalid value for {}: {}",
            argument_name,
            value_string
        )
    })
}

#[allow(dead_code)]
fn get_value_from_parseable_argument_with_default<T>(
    arguments: &ArgMatches,
    argument_name: &str,
    default_value: T,
) -> T
where
    T: FromStr + Copy,
    <T as FromStr>::Err: std::fmt::Display,
{
    get_value_from_parseable_argument_with_custom_default(arguments, argument_name, &|| {
        default_value
    })
}

pub fn get_value_from_param_file_argument_with_default<G, R, T, C>(
    reader: &R,
    arguments: &ArgMatches,
    argument_name: &str,
    param_file_argument_name: &str,
    conversion_mapping: &C,
    default_value: T,
) -> T
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    T: From<fdt> + std::fmt::Display + FromStr + Copy,
    <T as FromStr>::Err: std::fmt::Display,
    C: Fn(T) -> T,
{
    get_value_from_parseable_argument_with_custom_default(arguments, argument_name, &|| {
        reader
            .parameters()
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                argument_name,
                param_file_argument_name,
                conversion_mapping,
                default_value,
            )
    })
}

pub fn get_values_from_param_file_argument_with_defaults<G, R, T, C>(
    reader: &R,
    arguments: &ArgMatches,
    argument_name: &str,
    param_file_argument_names: &[&str],
    conversion_mapping: &C,
    default_values: &[T],
) -> Vec<T>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
    T: From<fdt> + std::fmt::Display + FromStr + Copy,
    <T as FromStr>::Err: std::fmt::Display,
    C: Fn(T) -> T,
{
    get_values_from_parseable_argument_with_custom_defaults(arguments, argument_name, &|| {
        param_file_argument_names
            .iter()
            .zip(default_values)
            .map(|(&param_file_argument_name, &default_value)| {
                reader
                    .parameters()
                    .get_converted_numerical_param_or_fallback_to_default_with_warning(
                        argument_name,
                        param_file_argument_name,
                        conversion_mapping,
                        default_value,
                    )
            })
            .collect()
    })
}
