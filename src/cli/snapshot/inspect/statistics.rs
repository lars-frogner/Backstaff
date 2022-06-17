//! Command line interface for printing statistics for quantities in a snapshot.

use crate::{
    cli::{
        interpolation::poly_fit::{
            construct_poly_fit_interpolator_config_from_options,
            create_poly_fit_interpolator_subcommand,
        },
        utils,
    },
    exit_on_error, exit_with_error,
    field::{self, ResampledCoordLocation, ScalarField3},
    geometry::{
        CoordRefs3,
        Dim2::{X as X2, Y as Y2},
        Dim3::{X, Y, Z},
        Point3,
    },
    grid::{fgr, Grid3},
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{fdt, SnapshotProvider3},
        utils::IOContext,
        Verbosity,
    },
    num::BFloat,
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use float_pretty_print::PrettyPrintFloat;
use ndarray::prelude::*;
use ndarray_stats::{interpolate::Linear, QuantileExt};
use noisy_float::types::n64;
use pad::{Alignment, PadStr};
use rayon::prelude::*;
use std::{
    fmt::Display,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    str::FromStr,
};

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
            Arg::new("percentages")
                .short('p')
                .long("percentages")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("PERCENTAGES")
                .help("List of percentages for which to compute percentiles of the quantity values\n\
                       (comma-separated) [default: none]")
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(
            Arg::new("slice-depths")
                .short('s')
                .long("slice-depths")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_name("Z_COORDS")
                .help("List of z-coordinates for which horizontal slice statistics should be printed\n\
                       (comma-separated) [default: none]")
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(
            Arg::new("value-range")
                .short('v')
                .long("value-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Range of quantity values that will be included when computing statistics\n",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("-inf,inf"),
        )
        .arg(
            Arg::new("x-bounds")
                .short('x')
                .long("x-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the x-coordinates that will be included when computing statistics\n",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("y-bounds")
                .short('y')
                .long("y-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the y-coordinates that will be included when computing statistics\n",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("z-bounds")
                .short('z')
                .long("z-bounds")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for the x-coordinates that will be included when computing statistics\n",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("no-global")
                .long("no-global")
                .help("Skip computation of global statistics"),
        )
        .arg(
            Arg::new("output-file")
                .short('o')
                .long("output-file")
                .require_equals(true)
                .value_name("FILE")
                .help("Optional file path to write the statistics report to instead of stdout")
                .takes_value(true),
        )
        .arg(
            Arg::new("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing files (unless listed as protected)")
                .conflicts_with("no-overwrite"),
        )
        .arg(
            Arg::new("no-overwrite")
                .long("no-overwrite")
                .help("Do not overwrite any existing files")
                .conflicts_with("overwrite"),
        )
        .subcommand(create_poly_fit_interpolator_subcommand(command_name))
}

/// Runs the actions for the `snapshot-inspect-statistics` subcommand using the given arguments.
pub fn run_statistics_subcommand<P>(
    arguments: &ArgMatches,
    provider: P,
    quantity_names: Vec<String>,
    io_context: &mut IOContext,
    verbosity: &Verbosity,
) where
    P: SnapshotProvider3,
{
    let grid = provider.grid();
    let lower_bounds = grid.lower_bounds();
    let upper_bounds = grid.upper_bounds();

    let value_range = utils::parse_limits(
        arguments,
        "value-range",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::Yes,
        None,
    );

    let x_range = utils::parse_limits_with_min_max(
        arguments,
        "x-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::No,
        lower_bounds[X],
        upper_bounds[X],
    );
    let y_range = utils::parse_limits_with_min_max(
        arguments,
        "y-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::No,
        lower_bounds[Y],
        upper_bounds[Y],
    );
    let z_range = utils::parse_limits_with_min_max(
        arguments,
        "z-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::No,
        lower_bounds[Z],
        upper_bounds[Z],
    );

    let slice_depths =
        utils::get_finite_float_values_from_parseable_argument::<fgr>(arguments, "slice-depths");

    let percentages =
        utils::get_finite_float_values_from_parseable_argument::<f64>(arguments, "percentages");

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

    match arguments.value_of("output-file") {
        Some(output_file_path) => {
            let output_file_path = exit_on_error!(
                PathBuf::from_str(output_file_path),
                "Error: Could not interpret path to input file: {}"
            );

            let overwrite_mode = utils::overwrite_mode_from_arguments(arguments);
            io_context.set_overwrite_mode(overwrite_mode);

            let atomic_output_file = exit_on_error!(
                io_context.create_atomic_output_file(output_file_path),
                "Error: Could not create temporary output file: {}"
            );

            if !atomic_output_file.check_if_write_allowed(io_context, verbosity) {
                return;
            }

            let mut writer = exit_on_error!(
                File::create(atomic_output_file.temporary_path()),
                "Error: Could not create output file: {}"
            );

            exit_on_error!(
                print_statistics_reports(
                    provider,
                    &quantity_names,
                    value_range,
                    x_range,
                    y_range,
                    z_range,
                    slice_depths.as_deref(),
                    quantile_p_values.as_deref(),
                    no_global,
                    &interpolator,
                    &mut writer,
                ),
                "Error: Could not write statistics report: {}"
            );

            exit_on_error!(
                io_context.close_atomic_output_file(atomic_output_file),
                "Error: Could not move temporary output file to target path: {}"
            );
        }
        None => {
            let mut writer = io::stdout();

            exit_on_error!(
                print_statistics_reports(
                    provider,
                    &quantity_names,
                    value_range,
                    x_range,
                    y_range,
                    z_range,
                    slice_depths.as_deref(),
                    quantile_p_values.as_deref(),
                    no_global,
                    &interpolator,
                    &mut writer,
                ),
                "Error: Could not write statistics report: {}"
            );
        }
    }
}

fn print_statistics_reports<P, I, W>(
    mut provider: P,
    quantity_names: &[String],
    value_range: (fdt, fdt),
    x_range: (fgr, fgr),
    y_range: (fgr, fgr),
    z_range: (fgr, fgr),
    slice_depths: Option<&[fgr]>,
    quantile_p_values: Option<&[f64]>,
    no_global: bool,
    interpolator: &I,
    writer: &mut W,
) -> io::Result<()>
where
    P: SnapshotProvider3,
    I: Interpolator3<fdt>,
    W: Write,
{
    for name in quantity_names {
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
            slice_depths,
            quantile_p_values,
            no_global,
            interpolator,
            writer,
        )?;
    }
    print_whole_line(writer, '-')
}

fn print_padded_headline<W: Write>(writer: &mut W, text: &str, pad_char: char) -> io::Result<()> {
    writeln!(
        writer,
        "{}",
        format!(" {} ", text).pad(TABLE_WIDTH, pad_char, Alignment::Middle, false)
    )
}

fn print_whole_line<W: Write>(writer: &mut W, line_type: char) -> io::Result<()> {
    writeln!(
        writer,
        "{}",
        "".pad_to_width_with_char(TABLE_WIDTH, line_type)
    )
}

fn print_name_value_pair<W, S>(writer: &mut W, name: &str, value: S) -> io::Result<()>
where
    W: Write,
    S: Display,
{
    writeln!(
        writer,
        "{} {}",
        format!("{}:", name).pad_to_width(NAME_WIDTH),
        value
    )
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

fn format_coord(value: fgr) -> String {
    format!("{:cw$.cp$}", value, cw = COORD_WIDTH, cp = COORD_PRECISION)
}

fn format_idx(idx: usize) -> String {
    format!("{:iw$}", idx, iw = IDX_WIDTH)
}

fn format_range<F, S, M>(
    name: &str,
    range: &(F, F),
    min: F,
    max: F,
    precision: usize,
    mapper: M,
) -> String
where
    F: BFloat,
    S: Display,
    M: Fn(F) -> S,
{
    if range.0 <= min && range.1 >= max {
        format!("all {}", name)
    } else if range.1 >= max {
        format!("{} \u{2265} {:.p$}", name, mapper(range.0), p = precision)
    } else if range.0 <= min {
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

fn print_statistics_report<I, W>(
    field: ScalarField3<fdt>,
    snap_name_and_num: (String, Option<u64>),
    value_range: (fdt, fdt),
    x_range: (fgr, fgr),
    y_range: (fgr, fgr),
    z_range: (fgr, fgr),
    slice_depths: Option<&[fgr]>,
    quantile_p_values: Option<&[f64]>,
    no_global: bool,
    interpolator: &I,
    writer: &mut W,
) -> io::Result<()>
where
    I: Interpolator3<fdt>,
    W: Write,
{
    let quantity_name = field.name().to_string();
    let locations = field.locations().clone();
    let grid = field.arc_with_grid();
    let mut values = field.into_values();
    let lower_bounds = grid.lower_bounds();
    let upper_bounds = grid.upper_bounds();

    print_whole_line(writer, '=')?;
    print_padded_headline(
        writer,
        &format!(
            "Statistics for {} from {}",
            &quantity_name,
            match snap_name_and_num {
                (snap_name, Some(snap_num)) => format!("snapshot {} of {}", snap_num, snap_name),
                (snap_name, None) => snap_name,
            }
        ),
        ' ',
    )?;
    print_whole_line(writer, '-')?;
    print_padded_headline(
        writer,
        &format!(
            "For {}, {}, {}, {}",
            format_range(
                &quantity_name,
                &value_range,
                fdt::NEG_INFINITY,
                fdt::INFINITY,
                VALUE_WIDTH,
                |v| { PrettyPrintFloat(v as f64) }
            ),
            format_range(
                "x",
                &x_range,
                lower_bounds[X],
                upper_bounds[X],
                COORD_PRECISION,
                |c| c
            ),
            format_range(
                "y",
                &y_range,
                lower_bounds[Y],
                upper_bounds[Y],
                COORD_PRECISION,
                |c| c
            ),
            format_range(
                "z",
                &z_range,
                lower_bounds[Z],
                upper_bounds[Z],
                COORD_PRECISION,
                |c| c
            ),
        ),
        ' ',
    )?;
    print_whole_line(writer, '=')?;

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
        print_name_value_pair(writer, "Number of values", number_of_values)?;

        let min_value = match filtered_field.find_minimum() {
            Some((min_indices, min_value)) => {
                let min_point = coords.point(&min_indices);
                print_name_value_pair(
                    writer,
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
                )?;
                min_value
            }
            None => {
                print_name_value_pair(writer, "Minimum value", "N/A")?;
                fdt::NAN
            }
        };

        let max_value = match filtered_field.find_maximum() {
            Some((max_indices, max_value)) => {
                let max_point = coords.point(&max_indices);
                print_name_value_pair(
                    writer,
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
                )?;
                max_value
            }
            None => {
                print_name_value_pair(writer, "Maximum value", "N/A")?;
                fdt::NAN
            }
        };

        if number_of_values > 0 {
            let sum: fdt = values_slice
                .par_iter()
                .filter(|value| !value.is_nan())
                .sum();
            let mean = sum / (number_of_values as fdt);
            print_name_value_pair(writer, "Average value", format_value(mean))?;
        } else {
            print_name_value_pair(writer, "Average value", "N/A")?;
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
                            writer,
                            &format!("{}th percentile", p * 100.0),
                            format_value(percentile),
                        )?;
                    }
                } else {
                    print_name_value_pair(writer, "Percentiles", "N/A")?;
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
                if z_coord >= fgr::max(z_range.0, lower_bound_z)
                    && z_coord <= z_range.1
                    && z_coord < upper_bound_z
                {
                    print_slice_statistics_report(
                        &filtered_field,
                        z_coord,
                        quantile_p_values,
                        interpolator,
                        writer,
                    )?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn print_slice_statistics_report<I, W>(
    field: &ScalarField3<fdt>,
    z_coord: fgr,
    quantile_p_values: Option<&[f64]>,
    interpolator: &I,
    writer: &mut W,
) -> io::Result<()>
where
    I: Interpolator3<fdt>,
    W: Write,
{
    print_whole_line(writer, '-')?;
    print_padded_headline(writer, &format!("In slice at z = {}", z_coord), ' ')?;
    print_whole_line(writer, '-')?;

    let sliced_field =
        field.slice_across_z(interpolator, z_coord, ResampledCoordLocation::Original);

    let coords = sliced_field.coords();
    let values = sliced_field.values();

    let values_slice = values.as_slice_memory_order().unwrap();

    let number_of_values = values_slice
        .par_iter()
        .filter(|value| !value.is_nan())
        .count();
    print_name_value_pair(writer, "Number of values", number_of_values)?;

    let min_value = match sliced_field.find_minimum() {
        Some((min_indices, min_value)) => {
            let min_point = coords.point(&min_indices);
            let z_idx = field
                .grid()
                .find_grid_cell(&Point3::new(min_point[X2], min_point[Y2], z_coord))
                .expect_inside()[Z];
            print_name_value_pair(
                writer,
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
            )?;
            min_value
        }
        None => {
            print_name_value_pair(writer, "Minimum value", "N/A")?;
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
                writer,
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
            )?;
            max_value
        }
        None => {
            print_name_value_pair(writer, "Maximum value", "N/A")?;
            fdt::NAN
        }
    };

    if number_of_values > 0 {
        let sum: fdt = values_slice
            .par_iter()
            .filter(|value| !value.is_nan())
            .sum();
        let mean = sum / (number_of_values as fdt);
        print_name_value_pair(writer, "Average value", format_value(mean))?;
    } else {
        print_name_value_pair(writer, "Average value", "N/A")?;
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
                        writer,
                        &format!("{}th percentile", p * 100.0),
                        format_value(percentile),
                    )?;
                }
            } else {
                print_name_value_pair(writer, "Percentiles", "N/A")?;
            }
        }
        _ => {}
    }
    Ok(())
}
