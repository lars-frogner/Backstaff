//! Command line interface for printing statistics for quantities in a snapshot.

use crate::{
    cli::{
        interpolation::poly_fit::{
            construct_poly_fit_interpolator_config_from_options,
            create_poly_fit_interpolator_subcommand,
        },
        utils,
    },
    exit_on_error,
    field::{self, ResampledCoordLocation, ScalarField3},
    geometry::{
        CoordRefs3,
        Dim2::{X as X2, Y as Y2},
        Dim3::{X, Y, Z},
        Point3,
    },
    grid::Grid3,
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::snapshot::{fdt, SnapshotProvider3},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use float_pretty_print::PrettyPrintFloat;
use ndarray::prelude::*;
use ndarray_stats::{interpolate::Linear, QuantileExt};
use noisy_float::types::n64;
use pad::{Alignment, PadStr};
use rayon::prelude::*;
use std::fmt::Display;

const TABLE_WIDTH: usize = 80;
const NAME_WIDTH: usize = 20;
const VALUE_WIDTH: usize = 12;
const COORD_PRECISION: usize = 3;
const COORD_WIDTH: usize = 7;
const IDX_WIDTH: usize = 3;

/// Builds a representation of the `snapshot-inspect-statistics` command line subcommand.
pub fn create_statistics_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "statistics";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Print statistics for quantities in the snapshot")

        .arg(
            Arg::new("slice-depths")
                .short('s')
                .long("slice-depths")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_name("Z_COORDS")
                .help("List of z-coordinates for which horizontal slice statistics should be printed\n\
                       (comma-separated)")
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(
            Arg::new("percentages")
                .short('p')
                .long("percentages")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .value_name("PERCENTAGES")
                .help("List of percentages for which to compute percentiles of the quantity values\n\
                       (comma-separated)")
                .takes_value(true)
                .multiple_values(true)
                .default_value("5,30,50,70,95"),
        )
        .arg(
            Arg::new("value-range")
                .short('v')
                .long("value-range")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Range of quantity values that will be included when computing statistics\n",
                )
                .takes_value(true)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("x-range")
                .short('x')
                .long("x-range")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Range of x-coordinate values that will be included when computing statistics\n",
                )
                .takes_value(true)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("y-range")
                .short('y')
                .long("y-range")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Range of y-coordinate values that will be included when computing statistics\n",
                )
                .takes_value(true)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("z-range")
                .short('z')
                .long("z-range")
                .require_equals(true)
                .use_value_delimiter(true).require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Range of z-coordinate values that will be included when computing statistics\n",
                )
                .takes_value(true)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("no-global")
                .long("no-global")
                .help("Skip computation of global statistics"),
        )
        .subcommand(create_poly_fit_interpolator_subcommand(command_name))
}

/// Runs the actions for the `snapshot-inspect-statistics` subcommand using the given arguments.
pub fn run_statistics_subcommand<G, P>(
    arguments: &ArgMatches,
    mut provider: P,
    quantity_names: Vec<String>,
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    let value_range = utils::parse_limits(arguments, "value-range", true);
    let x_range = utils::parse_limits(arguments, "x-range", true);
    let y_range = utils::parse_limits(arguments, "y-range", true);
    let z_range = utils::parse_limits(arguments, "z-range", true);

    let slice_depths =
        utils::get_values_from_parseable_argument::<fdt>(arguments, "slice-depths", None);

    let percentages =
        utils::get_values_from_parseable_argument::<f64>(arguments, "percentages", None);

    let no_global = arguments.is_present("no-global");

    let quantile_p_values = percentages.map(|p| {
        p.into_iter()
            .map(|percentage| {
                let quantile_p_value = 0.01 * percentage;
                if !(0.0..=1.0).contains(&quantile_p_value) {
                    exit_with_error!("Percentage not between 0 and 100: {}", percentage);
                }
                quantile_p_value
            })
            .collect::<Vec<_>>()
    });

    let interpolator_config = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        construct_poly_fit_interpolator_config_from_options(interpolator_arguments)
    } else {
        PolyFitInterpolatorConfig::default()
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    for name in &quantity_names {
        let field = exit_on_error!(
            provider.produce_scalar_field(name),
            "Error: Could not obtain quantity {0}: {1}",
            name
        );
        print_statistics_report(
            field,
            provider.obtain_snap_name_and_num(),
            value_range,
            x_range,
            y_range,
            z_range,
            slice_depths.as_deref(),
            quantile_p_values.as_deref(),
            no_global,
            &interpolator,
        );
    }
    print_whole_line('-');
}

fn print_padded_headline(text: &str, pad_char: char) {
    println!(
        "{}",
        format!(" {} ", text).pad(TABLE_WIDTH, pad_char, Alignment::Middle, false)
    );
}

fn print_whole_line(line_type: char) {
    println!("{}", "".pad_to_width_with_char(TABLE_WIDTH, line_type));
}

fn print_name_value_pair<S: Display>(name: &str, value: S) {
    println!(
        "{} {}",
        format!("{}:", name).pad_to_width(NAME_WIDTH),
        value
    );
}

fn format_value(value: fdt) -> String {
    format!(
        "{:min_width$.max_width$}",
        PrettyPrintFloat(value as f64),
        min_width = VALUE_WIDTH,
        max_width = VALUE_WIDTH
    )
    .pad_to_width(VALUE_WIDTH)
}

fn format_coord(value: fdt) -> String {
    format!("{:cw$.cp$}", value, cw = COORD_WIDTH, cp = COORD_PRECISION)
}

fn format_idx(idx: usize) -> String {
    format!("{:iw$}", idx, iw = IDX_WIDTH)
}

fn format_range<S, M>(name: &str, range: &(fdt, fdt), precision: usize, mapper: M) -> String
where
    S: Display,
    M: Fn(fdt) -> S,
{
    if range.0 == fdt::NEG_INFINITY && range.1 == fdt::INFINITY {
        format!("all {}", name)
    } else if range.1 == fdt::INFINITY {
        format!("{} \u{2265} {:.p$}", name, mapper(range.0), p = precision)
    } else if range.0 == fdt::NEG_INFINITY {
        format!("{} \u{2264} {:.p$}", name, mapper(range.1), p = precision)
    } else {
        format!(
            "{} \u{2208} [{:.p$}, {:.p$}]",
            name,
            mapper(range.0),
            mapper(range.1),
            p = precision
        )
    }
}

fn print_statistics_report<G, I>(
    field: ScalarField3<fdt, G>,
    snap_name_and_num: (String, Option<u32>),
    value_range: (fdt, fdt),
    x_range: (fdt, fdt),
    y_range: (fdt, fdt),
    z_range: (fdt, fdt),
    slice_depths: Option<&[fdt]>,
    quantile_p_values: Option<&[f64]>,
    no_global: bool,
    interpolator: &I,
) where
    G: Grid3<fdt>,
    I: Interpolator3,
{
    let quantity_name = field.name().to_string();
    let locations = field.locations().clone();
    let grid = field.arc_with_grid();
    let mut values = field.into_values();

    print_whole_line('=');
    print_padded_headline(
        &format!(
            "Statistics for {} from {}",
            &quantity_name,
            match snap_name_and_num {
                (snap_name, Some(snap_num)) => format!("snapshot {} of {}", snap_num, snap_name),
                (snap_name, None) => snap_name,
            }
        ),
        ' ',
    );
    print_whole_line('-');
    print_padded_headline(
        &format!(
            "For {}, {}, {}, {}",
            format_range(&quantity_name, &value_range, VALUE_WIDTH, |v| {
                PrettyPrintFloat(v as f64)
            }),
            format_range("x", &x_range, COORD_PRECISION, |c| c),
            format_range("y", &y_range, COORD_PRECISION, |c| c),
            format_range("z", &z_range, COORD_PRECISION, |c| c),
        ),
        ' ',
    );
    print_whole_line('=');

    let number_of_nans = values.par_iter().filter(|value| value.is_nan()).count();
    if number_of_nans > 0 {
        eprintln!("Warning: NaN values detected (will be ignored in statistics)");
    }

    // Create local scope to ensure borrowed variables are dropped before move
    {
        let grid_shape = grid.shape();
        let coords = CoordRefs3::new(
            &grid.coords_by_type(locations[X])[X],
            &grid.coords_by_type(locations[Y])[Y],
            &grid.coords_by_type(locations[Z])[Z],
        );

        // Set all values outside the ranges to NaN, so that they will be ignored
        // in all statistics.
        let values_buffer = values.as_slice_memory_order_mut().unwrap();
        values_buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let indices = field::compute_3d_array_indices_from_flat_idx(grid_shape, idx);
                let point = coords.point(&indices);
                if *value < value_range.0
                    || *value > value_range.1
                    || point[X] < x_range.0
                    || point[X] > x_range.1
                    || point[Y] < y_range.0
                    || point[Y] > y_range.1
                    || point[Z] < z_range.0
                    || point[Z] > z_range.1
                {
                    *value = fdt::NAN;
                }
            });
    }

    let filtered_field = ScalarField3::new(quantity_name, grid, locations, values);
    let coords = filtered_field.coords();

    if !no_global {
        let values_slice = filtered_field.values().as_slice_memory_order().unwrap();

        let number_of_values = values_slice
            .par_iter()
            .filter(|value| !value.is_nan())
            .count();
        print_name_value_pair("Number of values", number_of_values);

        let min_value = match filtered_field.find_minimum() {
            Some((min_indices, min_value)) => {
                let min_point = coords.point(&min_indices);
                print_name_value_pair(
                    "Minimum value",
                    format!(
                        "{} at ({}, {}, {}) [{}, {}, {}]",
                        format_value(min_value),
                        format_coord(min_point[X]),
                        format_coord(min_point[Y]),
                        format_coord(min_point[Z]),
                        format_idx(min_indices[X]),
                        format_idx(min_indices[Y]),
                        format_idx(min_indices[Z]),
                    ),
                );
                min_value
            }
            None => {
                print_name_value_pair("Minimum value", "N/A");
                fdt::NAN
            }
        };

        let max_value = match filtered_field.find_maximum() {
            Some((max_indices, max_value)) => {
                let max_point = coords.point(&max_indices);
                print_name_value_pair(
                    "Maximum value",
                    format!(
                        "{} at ({}, {}, {}) [{}, {}, {}]",
                        format_value(max_value),
                        format_coord(max_point[X]),
                        format_coord(max_point[Y]),
                        format_coord(max_point[Z]),
                        format_idx(max_indices[X]),
                        format_idx(max_indices[Y]),
                        format_idx(max_indices[Z]),
                    ),
                );
                max_value
            }
            None => {
                print_name_value_pair("Maximum value", "N/A");
                fdt::NAN
            }
        };

        if number_of_values > 0 {
            let sum: fdt = values_slice
                .par_iter()
                .filter(|value| !value.is_nan())
                .sum();
            let mean = sum / (number_of_values as fdt);
            print_name_value_pair("Average value", format_value(mean));
        } else {
            print_name_value_pair("Average value", "N/A");
        }

        match quantile_p_values {
            Some(quantile_p_values) if !quantile_p_values.is_empty() => {
                if number_of_values > 0 && max_value > min_value {
                    let percentiles: Vec<_> = quantile_p_values
                        .par_iter()
                        .map(|&p| {
                            exit_on_error!(
                                Array1::from(values_slice.to_vec()).quantile_axis_skipnan_mut(
                                    Axis(0),
                                    n64(p),
                                    &Linear
                                ),
                                "Could not compute percentile: {}"
                            )
                            .into_scalar()
                        })
                        .collect();
                    for (p, percentile) in quantile_p_values.iter().zip(percentiles) {
                        print_name_value_pair(
                            &format!("{}th percentile", p * 100.0),
                            format_value(percentile),
                        );
                    }
                } else {
                    print_name_value_pair("Percentiles", "N/A");
                }
            }
            _ => {}
        }
    }

    match slice_depths {
        Some(slice_depths) if !slice_depths.is_empty() => {
            let lower_bound_z = filtered_field.grid().lower_bounds()[Z];
            let upper_bound_z = filtered_field.grid().upper_bounds()[Z];
            for &z_coord in slice_depths {
                if z_coord >= fdt::max(z_range.0, lower_bound_z)
                    && z_coord <= z_range.1
                    && z_coord < upper_bound_z
                {
                    print_slice_statistics_report(
                        &filtered_field,
                        z_coord,
                        quantile_p_values,
                        interpolator,
                    );
                }
            }
        }
        _ => {}
    }
}

fn print_slice_statistics_report<G: Grid3<fdt>, I: Interpolator3>(
    field: &ScalarField3<fdt, G>,
    z_coord: fdt,
    quantile_p_values: Option<&[f64]>,
    interpolator: &I,
) {
    print_whole_line('-');
    print_padded_headline(&format!("In slice at z = {}", z_coord), ' ');
    print_whole_line('-');

    let sliced_field =
        field.slice_across_z(interpolator, z_coord, ResampledCoordLocation::Original);

    let coords = sliced_field.coords();
    let values = sliced_field.values();

    let values_slice = values.as_slice_memory_order().unwrap();

    let number_of_values = values_slice
        .par_iter()
        .filter(|value| !value.is_nan())
        .count();
    print_name_value_pair("Number of values", number_of_values);

    let min_value = match sliced_field.find_minimum() {
        Some((min_indices, min_value)) => {
            let min_point = coords.point(&min_indices);
            let z_idx = field
                .grid()
                .find_grid_cell(&Point3::new(min_point[X2], min_point[Y2], z_coord))
                .expect_inside()[Z];
            print_name_value_pair(
                "Minimum value",
                format!(
                    "{} at ({}, {}, {}) [{}, {}, {}]",
                    format_value(min_value),
                    format_coord(min_point[X2]),
                    format_coord(min_point[Y2]),
                    format_coord(z_coord),
                    format_idx(min_indices[X2]),
                    format_idx(min_indices[Y2]),
                    format_idx(z_idx),
                ),
            );
            min_value
        }
        None => {
            print_name_value_pair("Minimum value", "N/A");
            fdt::NAN
        }
    };

    let max_value = match sliced_field.find_maximum() {
        Some((max_indices, max_value)) => {
            let max_point = coords.point(&max_indices);
            let z_idx = field
                .grid()
                .find_grid_cell(&Point3::new(max_point[X2], max_point[Y2], z_coord))
                .expect_inside()[Z];
            print_name_value_pair(
                "Maximum value",
                format!(
                    "{} at ({}, {}, {}) [{}, {}, {}]",
                    format_value(max_value),
                    format_coord(max_point[X2]),
                    format_coord(max_point[Y2]),
                    format_coord(z_coord),
                    format_idx(max_indices[X2]),
                    format_idx(max_indices[Y2]),
                    format_idx(z_idx),
                ),
            );
            max_value
        }
        None => {
            print_name_value_pair("Maximum value", "N/A");
            fdt::NAN
        }
    };

    if number_of_values > 0 {
        let sum: fdt = values_slice
            .par_iter()
            .filter(|value| !value.is_nan())
            .sum();
        let mean = sum / (number_of_values as fdt);
        print_name_value_pair("Average value", format_value(mean));
    } else {
        print_name_value_pair("Average value", "N/A");
    }

    match quantile_p_values {
        Some(quantile_p_values) if !quantile_p_values.is_empty() => {
            if !quantile_p_values.is_empty() && number_of_values > 0 && max_value > min_value {
                let percentiles: Vec<_> = quantile_p_values
                    .par_iter()
                    .map(|&p| {
                        exit_on_error!(
                            Array1::from(values_slice.to_vec()).quantile_axis_skipnan_mut(
                                Axis(0),
                                n64(p),
                                &Linear
                            ),
                            "Could not compute percentile: {}"
                        )
                        .into_scalar()
                    })
                    .collect();
                for (p, percentile) in quantile_p_values.iter().zip(percentiles) {
                    print_name_value_pair(
                        &format!("{}th percentile", p * 100.0),
                        format_value(percentile),
                    );
                }
            } else {
                print_name_value_pair("Percentiles", "N/A");
            }
        }
        _ => {}
    }
}
