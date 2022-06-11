//! Utilities for creating the command line interface.

use crate::{
    exit_on_error, exit_on_false, exit_with_error,
    geometry::{Dim2, Dim3, In2D, In3D},
    grid::{fgr, Grid3},
    io::{
        snapshot::{fpa, SnapshotParameters, SnapshotProvider3},
        utils as io_utils, OverwriteMode, Verbosity,
    },
    num::BFloat,
};
use clap::{self, ArgMatches, Command};
use indicatif::ProgressStyle;
use lazy_static::lazy_static;
use num;
use std::{collections::HashMap, process, str::FromStr};

lazy_static! {
    static ref DEFAULT_PROGRESS_STYLE: ProgressStyle =
        ProgressStyle::default_bar().template("Progress: {bar:40}  {percent}% | ETA: {eta}");
}

pub type CommandCreator = fn(&'static str) -> Command<'static>;

#[macro_export]
macro_rules! add_subcommand_combinations {
    ($command:expr, $command_name:expr, $subcommand_required:expr; $($child_subcommand_names:tt $(if $cfg_condition:expr)?),+) => {{
        let mut nested_subcommand_names = Vec::new();
        let mut subcommand_creators = Vec::<$crate::cli::utils::CommandCreator>::new();

        $(
            $( #[cfg(feature = $cfg_condition)] )*
            add_subcommand_combinations!(
                @extend nested_subcommand_names,
                subcommand_creators;
                $child_subcommand_names
            );
        )*

        let subcommand_creator_map = nested_subcommand_names
            .clone()
            .into_iter()
            .flatten()
            .zip(subcommand_creators.into_iter())
            .collect::<::std::collections::HashMap<_, _>>();

        $crate::cli::utils::add_subcommand_combinations_with_map(
            $command,
            $command_name,
            &nested_subcommand_names,
            &subcommand_creator_map,
            $subcommand_required,
        )
    }};
    (@extend $nested_subcommand_names:expr, $subcommand_creators:expr; ($($subcommand_name:ident $(if $cfg_condition:expr)?),+)) => {{
        #[allow(clippy::vec_init_then_push)]
        {
            let mut same_level_subcommand_names = Vec::new();
            $(
                $( #[cfg(feature = $cfg_condition)] )*
                {
                    same_level_subcommand_names.push(stringify!($subcommand_name));
                    $subcommand_creators.push(paste::paste! {[<create_ $subcommand_name _subcommand>]});
                }
            )+
            $nested_subcommand_names.push(same_level_subcommand_names);
        }
    }};
    (@extend $nested_subcommand_names:expr, $subcommand_creators:expr; $subcommand_name:ident $(if $cfg_condition:expr)?) => {{
        $( #[cfg(feature = $cfg_condition)] )*
        {
            $nested_subcommand_names.push(vec![stringify!($subcommand_name)]);
            $subcommand_creators.push(paste::paste! {[<create_ $subcommand_name _subcommand>]});
        }
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

pub fn verify_user_will_continue_or_abort() {
    let abort = !io_utils::user_says_yes("Still continue?", true).unwrap_or_else(|err| {
        eprintln!("Warning: Not continuing due to error: {}", err);
        false
    });
    if abort {
        eprintln!("Aborted");
        process::exit(1);
    }
}

pub fn parse_value_string<T>(argument_name: &str, value_string: &str) -> T
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    exit_on_error!(
        value_string.parse(),
        "Error: Could not parse value for {0}: {1}",
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

fn verify_finite_float_value<F: BFloat>(argument_name: &str, value: F) {
    exit_on_false!(value.is_finite(), "Error: {} must be finite", argument_name);
}

fn verify_argument_value_count<T>(argument_name: &str, values: &[T], required_count: usize) {
    let count = values.len();
    exit_on_false!(
        count == required_count,
        "Error: {} must have {} values, got {}",
        argument_name,
        required_count,
        count
    );
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

pub fn get_finite_float_value_from_required_parseable_argument<F>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> F
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    let value: F = get_value_from_required_parseable_argument(arguments, argument_name);
    verify_finite_float_value(argument_name, value);
    value
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

pub fn get_finite_float_values_from_parseable_argument<F>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> Option<Vec<F>>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    let values = get_values_from_parseable_argument(arguments, argument_name);
    if let Some(values) = values.as_ref() {
        values
            .iter()
            .for_each(|&value| verify_finite_float_value(argument_name, value))
    };
    values
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

pub fn get_finite_float_values_from_required_parseable_argument<F>(
    arguments: &ArgMatches,
    argument_name: &str,
) -> Vec<F>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    let values = get_values_from_required_parseable_argument(arguments, argument_name);
    values
        .iter()
        .for_each(|&value| verify_finite_float_value(argument_name, value));
    values
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

pub fn get_finite_float_values_from_parseable_argument_with_custom_defaults<F, D>(
    arguments: &ArgMatches,
    argument_name: &str,
    default_constructor: &D,
) -> Vec<F>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
    D: Fn() -> Vec<F>,
{
    let values = get_values_from_parseable_argument_with_custom_defaults(
        arguments,
        argument_name,
        default_constructor,
    );
    values
        .iter()
        .for_each(|&value| verify_finite_float_value(argument_name, value));
    values
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
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
    T: From<fpa> + std::fmt::Display + FromStr + Copy,
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
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
    T: From<fpa> + std::fmt::Display + FromStr + Copy,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllowSameValue {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllowInfinity {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllowZero {
    Yes,
    No,
}

pub fn parse_limits<F>(
    arguments: &ArgMatches,
    argument_name: &str,
    allow_same_value: AllowSameValue,
    allow_infinity: AllowInfinity,
    special_values: Option<&HashMap<&str, F>>,
) -> (F, F)
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    let limits: Vec<_> = arguments
        .values_of(argument_name)
        .expect("No value for argument with default")
        .into_iter()
        .collect();

    verify_argument_value_count(argument_name, &limits, 2);

    let parse = |string: &str| {
        special_values
            .and_then(|special_values| special_values.get(string).cloned())
            .unwrap_or_else(|| {
                exit_on_error!(
                    string.parse::<F>(),
                    "Error: Could not parse value in {0}: {1}",
                    argument_name
                )
            })
    };

    let lower_limit = parse(limits[0]);
    let upper_limit = parse(limits[1]);

    exit_on_false!(
        !(lower_limit.is_nan() || upper_limit.is_nan()),
        "Error: {} contains a NaN value",
        argument_name
    );

    exit_on_false!(
        allow_infinity == AllowInfinity::Yes
            || (lower_limit.is_finite() && upper_limit.is_finite()),
        "Error: {} must be finite",
        argument_name
    );

    match allow_same_value {
        AllowSameValue::Yes => exit_on_false!(
            upper_limit >= lower_limit,
            "Error: Second value in {} ({}) must be larger than or equal to first value ({})",
            argument_name,
            upper_limit,
            lower_limit
        ),
        AllowSameValue::No => exit_on_false!(
            upper_limit > lower_limit,
            "Error: Second value in {} ({}) must be larger than first value ({})",
            argument_name,
            upper_limit,
            lower_limit
        ),
    };
    (lower_limit, upper_limit)
}

pub fn parse_limits_with_min_max<F>(
    arguments: &ArgMatches,
    argument_name: &str,
    allow_same_value: AllowSameValue,
    allow_infinity: AllowInfinity,
    min_value: F,
    max_value: F,
) -> (F, F)
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    match allow_same_value {
        AllowSameValue::Yes => assert!(
            max_value >= min_value,
            "Max value ({}) not larger than or equal to min value ({})",
            max_value,
            min_value
        ),
        AllowSameValue::No => assert!(
            max_value > min_value,
            "Max value ({}) not larger than min value ({})",
            max_value,
            min_value
        ),
    };
    let special_values = HashMap::from_iter([("min", min_value), ("max", max_value)].into_iter());
    parse_limits(
        arguments,
        argument_name,
        allow_same_value,
        allow_infinity,
        Some(&special_values),
    )
}

pub fn parse_int_limits_with_min_max<I>(
    arguments: &ArgMatches,
    argument_name: &str,
    allow_same_value: AllowSameValue,
    min_value: I,
    max_value: I,
) -> (I, I)
where
    I: num::Integer + Copy + FromStr + std::fmt::Display,
    <I as FromStr>::Err: std::fmt::Display,
{
    match allow_same_value {
        AllowSameValue::Yes => assert!(
            max_value >= min_value,
            "Max value ({}) not larger than or equal to min value ({})",
            max_value,
            min_value
        ),
        AllowSameValue::No => assert!(
            max_value > min_value,
            "Max value ({}) not larger than min value ({})",
            max_value,
            min_value
        ),
    };
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

    match allow_same_value {
        AllowSameValue::Yes => exit_on_false!(
            limits[1] >= limits[0],
            "Error: Second value in {} ({}) must be larger than or equal to first value ({})",
            argument_name,
            limits[1],
            limits[0]
        ),
        AllowSameValue::No => exit_on_false!(
            limits[1] > limits[0],
            "Error: Second value in {} ({}) must be larger than first value ({})",
            argument_name,
            limits[1],
            limits[0]
        ),
    };

    (limits[0], limits[1])
}

pub fn parse_3d_values<T, P>(
    arguments: &ArgMatches,
    argument_name: &str,
    min_value: Option<T>,
    convert_special_value_for_dim: P,
) -> In3D<T>
where
    T: FromStr + PartialOrd + std::fmt::Display,
    <T as FromStr>::Err: std::fmt::Display,
    P: Fn(Dim3, &str) -> Option<T>,
{
    let value_strings: Vec<_> = arguments
        .values_of(argument_name)
        .expect("No values for required argument")
        .collect();

    verify_argument_value_count(argument_name, &value_strings, 3);

    let values = In3D::with_each_component(|dim| {
        let value_string = value_strings[dim.num()];
        convert_special_value_for_dim(dim, value_string)
            .unwrap_or_else(|| parse_value_string(argument_name, value_string))
    });

    if let Some(min_value) = min_value {
        exit_on_false!(
            Dim3::slice().iter().all(|&dim| values[dim] >= min_value),
            "Error: All values in {} must be at least {}",
            argument_name,
            min_value
        );
    }

    values
}

pub fn parse_3d_values_no_special<T>(
    arguments: &ArgMatches,
    argument_name: &str,
    min_value: Option<T>,
) -> In3D<T>
where
    T: FromStr + PartialOrd + std::fmt::Display,
    <T as FromStr>::Err: std::fmt::Display,
{
    parse_3d_values(arguments, argument_name, min_value, |_, _| None)
}

pub fn parse_2d_values<T, P>(
    arguments: &ArgMatches,
    argument_name: &str,
    min_value: Option<T>,
    convert_special_value_for_dim: P,
) -> In2D<T>
where
    T: FromStr + PartialOrd + std::fmt::Display,
    <T as FromStr>::Err: std::fmt::Display,
    P: Fn(Dim2, &str) -> Option<T>,
{
    let value_strings: Vec<_> = arguments
        .values_of(argument_name)
        .expect("No values for required argument")
        .collect();

    verify_argument_value_count(argument_name, &value_strings, 2);

    let values = In2D::with_each_component(|dim| {
        let value_string = value_strings[dim.num()];
        convert_special_value_for_dim(dim, value_string)
            .unwrap_or_else(|| parse_value_string(argument_name, value_string))
    });

    if let Some(min_value) = min_value {
        exit_on_false!(
            Dim2::slice().iter().all(|&dim| values[dim] >= min_value),
            "Error: All values in {} must be at least {}",
            argument_name,
            min_value
        );
    }

    values
}

pub fn parse_3d_float_values<F>(
    arguments: &ArgMatches,
    argument_name: &str,
    allow_infinity: AllowInfinity,
    allow_zero: AllowZero,
) -> In3D<F>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    let values: In3D<F> = parse_3d_values_no_special(arguments, argument_name, None);

    exit_on_false!(
        Dim3::slice().into_iter().all(|dim| !values[dim].is_nan()),
        "Error: {} contains a NaN value",
        argument_name
    );

    exit_on_false!(
        allow_infinity == AllowInfinity::Yes
            || Dim3::slice().into_iter().all(|dim| values[dim].is_finite()),
        "Error: {} must be finite",
        argument_name
    );

    exit_on_false!(
        allow_zero == AllowZero::Yes
            || Dim3::slice()
                .into_iter()
                .all(|dim| values[dim] != F::zero()),
        "Error: {} must be non-zero",
        argument_name
    );

    values
}

pub fn parse_2d_values_no_special<T>(
    arguments: &ArgMatches,
    argument_name: &str,
    min_value: Option<T>,
) -> In2D<T>
where
    T: FromStr + PartialOrd + std::fmt::Display,
    <T as FromStr>::Err: std::fmt::Display,
{
    parse_2d_values(arguments, argument_name, min_value, |_, _| None)
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

pub fn parse_verbosity(arguments: &ArgMatches, support_progress: bool) -> Verbosity {
    if support_progress && arguments.is_present("progress") {
        Verbosity::Progress(DEFAULT_PROGRESS_STYLE.clone())
    } else if arguments.is_present("verbose") {
        Verbosity::Messages
    } else {
        Verbosity::Quiet
    }
}
