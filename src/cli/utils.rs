//! Utilities for creating the command line interface.

use crate::{
    exit_on_error,
    grid::Grid3,
    io::{
        snapshot::{fdt, SnapshotParameters, SnapshotProvider3},
        OverwriteMode,
    },
};
use clap::{self, ArgMatches, Command};
use num;
use std::{collections::HashMap, str::FromStr};

pub type CommandCreator = fn(&'static str) -> Command<'static>;

#[macro_export]
macro_rules! add_subcommand_combinations {
    ($command:expr, $command_name:expr, $subcommand_required:expr; $($child_subcommand_names:tt),+) => {{
        let mut nested_subcommand_names = Vec::new();
        let mut subcommand_creators = Vec::<crate::cli::utils::CommandCreator>::new();

        $(
            add_subcommand_combinations!(
                @extend nested_subcommand_names,
                subcommand_creators;
                $child_subcommand_names
            );
        )+

        let subcommand_creator_map = nested_subcommand_names
            .clone()
            .into_iter()
            .flatten()
            .zip(subcommand_creators.into_iter())
            .collect::<std::collections::HashMap<_, _>>();

        crate::cli::utils::add_subcommand_combinations_with_map(
            $command,
            $command_name,
            &nested_subcommand_names,
            &subcommand_creator_map,
            $subcommand_required,
        )
    }};
    (@extend $nested_subcommand_names:expr, $subcommand_creators:expr; ($($subcommand_name:ident),+)) => {{
        let mut same_level_subcommand_names = Vec::new();
        $(
            same_level_subcommand_names.push(stringify!($subcommand_name));
            $subcommand_creators.push(paste::paste! {[<create_ $subcommand_name _subcommand>]});
        )+
        $nested_subcommand_names.push(same_level_subcommand_names);
    }};
    (@extend $nested_subcommand_names:expr, $subcommand_creators:expr; $subcommand_name:ident) => {{
        $nested_subcommand_names.push(vec![stringify!($subcommand_name)]);
        $subcommand_creators.push(paste::paste! {[<create_ $subcommand_name _subcommand>]});
    }};
}

pub fn add_subcommand_combinations_with_map(
    mut command: Command<'static>,
    command_name: &'static str,
    nested_subcommand_names: &[Vec<&'static str>],
    subcommand_creators: &HashMap<&'static str, CommandCreator>,
    subcommand_required: bool,
) -> Command<'static> {
    command = command.subcommand_required(subcommand_required);

    for (idx, same_level_subcommand_names) in nested_subcommand_names.iter().enumerate() {
        for subcommand_name in same_level_subcommand_names {
            let mut subcommand =
                subcommand_creators.get(subcommand_name).unwrap_or_else(|| {
                    panic!("Missing creator for subcommand {}", subcommand_name)
                })(command_name);
            if idx + 1 < nested_subcommand_names.len() {
                subcommand = add_subcommand_combinations_with_map(
                    subcommand,
                    subcommand_name,
                    &nested_subcommand_names[idx + 1..],
                    subcommand_creators,
                    subcommand_required,
                );
            }
            command = command.subcommand(subcommand);
        }
    }
    command
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
        .filter_map(|value_string| {
            if value_string.is_empty() {
                None
            } else {
                Some(parse_value_string(argument_name, value_string))
            }
        })
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
            .expect("No value for required argument"),
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
            .expect("No values for required argument"),
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
        .expect("No value for required argument");
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

pub fn get_value_from_param_file_argument_with_default<G, P, T, C>(
    reader: &P,
    arguments: &ArgMatches,
    argument_name: &str,
    param_file_argument_name: &str,
    conversion_mapping: &C,
    default_value: T,
) -> T
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
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

pub fn get_values_from_param_file_argument_with_defaults<G, P, T, C>(
    reader: &P,
    arguments: &ArgMatches,
    argument_name: &str,
    param_file_argument_names: &[&str],
    conversion_mapping: &C,
    default_values: &[T],
) -> Vec<T>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
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

pub fn parse_limits(
    arguments: &ArgMatches,
    argument_name: &str,
    allow_infinity: bool,
) -> (fdt, fdt) {
    let limits: Vec<_> = arguments
        .values_of(argument_name)
        .expect("No value for argument with default")
        .into_iter()
        .map(|string| match string {
            "-inf" if allow_infinity => std::f32::NEG_INFINITY,
            "inf" if allow_infinity => std::f32::INFINITY,
            values_str => exit_on_error!(
                values_str.parse::<fdt>(),
                "Error: Could not parse value in {0}: {1}",
                argument_name
            ),
        })
        .collect();
    exit_on_false!(
        limits[1] >= limits[0],
        "Error: Second value in {} ({}) must be larger than or equal to first value ({})",
        argument_name,
        limits[1],
        limits[0]
    );
    (limits[0], limits[1])
}

pub fn parse_int_limits<I>(
    arguments: &ArgMatches,
    argument_name: &str,
    min_value: I,
    max_value: I,
) -> (I, I)
where
    I: num::Integer + Copy + FromStr + std::fmt::Display,
    <I as FromStr>::Err: std::fmt::Display,
{
    assert!(
        max_value >= min_value,
        "Max int value ({}) not larger than or equal to min value ({})",
        max_value,
        min_value
    );
    let limits: Vec<_> = arguments
        .values_of(argument_name)
        .expect("No value for argument with default")
        .into_iter()
        .map(|string| match string {
            "min" => min_value,
            "max" => max_value,
            values_str => exit_on_error!(
                values_str.parse::<I>(),
                "Error: Could not parse value in {0}: {1}",
                argument_name
            ),
        })
        .collect();
    exit_on_false!(
        limits[1] >= limits[0],
        "Error: Second value in {} ({}) must be larger than or equal to first value ({})",
        argument_name,
        limits[1],
        limits[0]
    );
    (limits[0], limits[1])
}

pub fn overwrite_mode_from_arguments(arguments: &ArgMatches) -> OverwriteMode {
    if arguments.is_present("overwrite") {
        OverwriteMode::Always
    } else if arguments.is_present("no-overwrite") {
        OverwriteMode::Never
    } else {
        OverwriteMode::Ask
    }
}
