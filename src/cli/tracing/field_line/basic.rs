//! Command line interface for basic field line tracer.

use crate::cli;
use crate::tracing::field_line::basic::{
    BasicFieldLineTracerConfig, FieldLinePointSpacing, FieldLineTracingSense,
};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the basic field line tracer.
pub fn create_basic_field_line_tracer_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("basic_tracer")
        .about("Use the basic field line tracer")
        .long_about(
            "Use the basic field line tracer.\n\
             A field line can be traced in both or one direction relative to the vector field\n\
             direction, and output points can be produced at regular intervals or at the\n\
             natural positions provided by the stepper. The field line can have a maximum\n\
             length.",
        );
    add_basic_field_line_tracer_options_to_subcommand(app)
}

/// Adds arguments for parameters used by the basic field line tracer.
pub fn add_basic_field_line_tracer_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("field-line-tracing-sense")
            .long("field-line-tracing-sense")
            .require_equals(true)
            .value_name("SENSE")
            .long_help("Direction(s) to trace the field line relative to the field direction\n")
            .takes_value(true)
            .possible_values(&["both", "same", "opposite"])
            .default_value("both"),
    )
    .arg(
        Arg::with_name("field-line-point-spacing")
            .long("field-line-point-spacing")
            .require_equals(true)
            .value_name("SPACING")
            .long_help("Form of spacing between field line points\n")
            .takes_value(true)
            .possible_values(&["regular", "natural"])
            .default_value("regular"),
    )
    .arg(
        Arg::with_name("max-field-line-length")
            .long("max-field-line-length")
            .require_equals(true)
            .value_name("VALUE")
            .long_help("Field lines reaching lengths larger than this will be terminated\n")
            .takes_value(true)
            .default_value("inf"),
    )
}

/// Determines basic field line tracer parameters based on
/// provided options.
pub fn construct_basic_field_line_tracer_config_from_options(
    arguments: &ArgMatches,
) -> BasicFieldLineTracerConfig {
    let tracing_sense = cli::get_value_from_required_constrained_argument(
        arguments,
        "field-line-tracing-sense",
        &["both", "same", "opposite"],
        &[
            FieldLineTracingSense::Both,
            FieldLineTracingSense::same(),
            FieldLineTracingSense::opposite(),
        ],
    );
    let point_spacing = cli::get_value_from_required_constrained_argument(
        arguments,
        "field-line-point-spacing",
        &["regular", "natural"],
        &[
            FieldLinePointSpacing::Regular,
            FieldLinePointSpacing::Natural,
        ],
    );
    let max_length = match arguments
        .value_of("max-field-line-length")
        .expect("No value for argument with default")
    {
        "inf" => None,
        length_str => Some(length_str.parse().unwrap_or_else(|err| {
            panic!("Could not parse value of max-field-line-length: {}", err)
        })),
    };
    BasicFieldLineTracerConfig {
        tracing_sense,
        point_spacing,
        max_length,
    }
}
