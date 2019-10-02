//! Command line interface for Runge-Kutta-Fehlberg steppers.

use crate::cli;
use crate::tracing::stepping::rkf::RKFStepperConfig;
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by Runge-Kutta-Fehlberg steppers.
pub fn add_rkf_stepper_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("dense-step-length")
            .long("dense-step-length")
            .value_name("VALUE")
            .long_help(
                "Step length to use for dense (uniform) output positions [Mm]\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-step-attempts")
            .long("max-step-attempts")
            .value_name("NUMBER")
            .long_help("Maximum number of step attempts before terminating")
            .takes_value(true)
            .default_value("16"),
    )
    .arg(
        Arg::with_name("stepping-absolute-tolerance")
            .long("stepping-absolute-tolerance")
            .value_name("VALUE")
            .long_help("Absolute error tolerance for stepping")
            .takes_value(true)
            .default_value("1e-6"),
    )
    .arg(
        Arg::with_name("stepping-relative-tolerance")
            .long("stepping-relative-tolerance")
            .value_name("VALUE")
            .long_help("Relative error tolerance for stepping")
            .takes_value(true)
            .default_value("1e-6"),
    )
    .arg(
        Arg::with_name("stepping-safety-factor")
            .long("stepping-safety-factor")
            .value_name("VALUE")
            .long_help("Scaling factor for the error to reduce step length oscillations")
            .takes_value(true)
            .default_value("0.9"),
    )
    .arg(
        Arg::with_name("min-step-scale")
            .long("min-step-scale")
            .value_name("VALUE")
            .long_help("Smallest allowed scaling of the step size in one step")
            .takes_value(true)
            .default_value("0.2"),
    )
    .arg(
        Arg::with_name("max-step-scale")
            .long("max-step-scale")
            .value_name("VALUE")
            .long_help("Largest allowed scaling of the step size in one step")
            .takes_value(true)
            .default_value("10.0"),
    )
    .arg(
        Arg::with_name("stepping-initial-error")
            .long("stepping-initial-error")
            .value_name("VALUE")
            .long_help("Start value for stepping error")
            .takes_value(true)
            .default_value("1e-4"),
    )
    .arg(
        Arg::with_name("initial-step-length")
            .long("stepping-initial-step-length")
            .value_name("VALUE")
            .long_help("Initial step size")
            .takes_value(true)
            .default_value("1e-4"),
    )
    .arg(
        Arg::with_name("sudden-reversals-for-sink")
            .long("sudden-reversals-for-sink")
            .value_name("NUMBER")
            .long_help("Number of sudden direction reversals before the area is considered a sink")
            .takes_value(true)
            .default_value("3"),
    )
    .arg(
        Arg::with_name("pi-control")
            .long("pi-control")
            .value_name("STATE")
            .long_help(
                "Whether to use Proportional Integral (PI) control for stabilizing the stepping\n\
                 [default: from param file]",
            )
            .takes_value(true)
            .possible_values(&["off", "on"]),
    )
}

/// Sets Runge-Kutta-Fehlberg stepper parameters based on present arguments.
pub fn configure_rkf_stepper_from_options(config: &mut RKFStepperConfig, arguments: &ArgMatches) {
    cli::assign_value_from_parseable_argument(
        &mut config.dense_step_length,
        arguments,
        "dense-step-length",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_step_attempts,
        arguments,
        "max-step-attempts",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.absolute_tolerance,
        arguments,
        "stepping-absolute-tolerance",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.relative_tolerance,
        arguments,
        "stepping-relative-tolerance",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.safety_factor,
        arguments,
        "stepping-safety-factor",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.min_step_scale,
        arguments,
        "min-step-scale",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_step_scale,
        arguments,
        "max-step-scale",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.initial_error,
        arguments,
        "stepping-initial-error",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.initial_step_length,
        arguments,
        "initial-step-length",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.sudden_reversals_for_sink,
        arguments,
        "sudden-reversals-for-sink",
    );
    cli::assign_value_from_selected_argument(
        &mut config.use_pi_control,
        arguments,
        "pi-control",
        &["off", "on"],
        &[false, true],
    );
}
