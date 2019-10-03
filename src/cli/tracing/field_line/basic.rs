//! Command line interface for basic field line tracing.

use crate::cli;
use crate::tracing::field_line::basic::{
    BasicFieldLineTracerConfig, FieldLinePointSpacing, FieldLineTracingSense,
};
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by the basic field line tracer.
pub fn add_basic_field_line_tracer_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("field-line-tracing-sense")
            .long("field-line-tracing-sense")
            .value_name("SENSE")
            .long_help("Direction(s) to trace the field line relative to the field direction\n")
            .takes_value(true)
            .possible_values(&["both", "same", "opposite"])
            .default_value("both"),
    )
    .arg(
        Arg::with_name("field-line-point-spacing")
            .long("field-line-point-spacing")
            .value_name("SPACING")
            .long_help("Form of spacing between field line points\n")
            .takes_value(true)
            .possible_values(&["regular", "natural"])
            .default_value("regular"),
    )
    .arg(
        Arg::with_name("max-field-line-length")
            .long("max-field-line-length")
            .value_name("VALUE")
            .long_help("Field lines reaching lengths larger than this will be terminated\n")
            .takes_value(true)
            .default_value("inf"),
    )
}

/// Sets basic field line tracer parameters based on present arguments.
pub fn configure_basic_field_line_tracer_from_options(
    config: &mut BasicFieldLineTracerConfig,
    arguments: &ArgMatches,
) {
    cli::assign_value_from_selected_argument(
        &mut config.tracing_sense,
        arguments,
        "field-line-tracing-sense",
        &["both", "same", "opposite"],
        &[
            FieldLineTracingSense::Both,
            FieldLineTracingSense::same(),
            FieldLineTracingSense::opposite(),
        ],
    );
    cli::assign_value_from_selected_argument(
        &mut config.tracing_sense,
        arguments,
        "field-line-point-spacing",
        &["regular", "natural"],
        &[
            FieldLinePointSpacing::Regular,
            FieldLinePointSpacing::Natural,
        ],
    );
    config.max_length = match arguments
        .value_of("max-field-line-length")
        .expect("No value for argument with default")
    {
        "inf" => None,
        length_str => Some(length_str.parse()).unwrap_or_else(|err| {
            panic!("Could not parse value of max-field-line-length: {}", err)
        }),
    };
}
