//! Command line interface for Runge-Kutta-Fehlberg steppers.

use crate::{
    cli::utils,
    tracing::stepping::rkf::{RKFStepperConfig, RKFStepperType},
};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using a Runge-Kutta-Fehlberg stepper.
pub fn create_rkf_stepper_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("rkf_stepper")
        .about("Use a Runge-Kutta-Fehlberg stepper")
        .long_about(
            "Use a Runge-Kutta-Fehlberg stepper.\n\
             The next position is computed with a Runge-Kutta scheme, and tbe resulting error\n\
             is estimated using an embedded lower-order step. The errors are used to adjust\n\
             the step length, and steps are re-attempted until the error is below a certain\n\
             tolerance.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("dense-step-length")
                .long("dense-step-length")
                .require_equals(true)
                .value_name("VALUE")
                .help("Step length to use for dense (uniform) output positions [Mm]\n")
                .takes_value(true)
                .default_value("0.01"),
        )
        .arg(
            Arg::with_name("max-step-attempts")
                .long("max-step-attempts")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Maximum number of step attempts before terminating")
                .takes_value(true)
                .default_value("16"),
        )
        .arg(
            Arg::with_name("absolute-tolerance")
                .long("absolute-tolerance")
                .require_equals(true)
                .value_name("VALUE")
                .help("Absolute error tolerance for stepping")
                .takes_value(true)
                .default_value("1e-6"),
        )
        .arg(
            Arg::with_name("relative-tolerance")
                .long("relative-tolerance")
                .require_equals(true)
                .value_name("VALUE")
                .help("Relative error tolerance for stepping")
                .takes_value(true)
                .default_value("1e-6"),
        )
        .arg(
            Arg::with_name("safety-factor")
                .long("safety-factor")
                .require_equals(true)
                .value_name("VALUE")
                .help("Scaling factor for the error to reduce step length oscillations\n")
                .takes_value(true)
                .default_value("0.9"),
        )
        .arg(
            Arg::with_name("min-step-scale")
                .long("min-step-scale")
                .require_equals(true)
                .value_name("VALUE")
                .help("Smallest allowed scaling of the step size in one step")
                .takes_value(true)
                .default_value("0.2"),
        )
        .arg(
            Arg::with_name("max-step-scale")
                .long("max-step-scale")
                .require_equals(true)
                .value_name("VALUE")
                .help("Largest allowed scaling of the step size in one step")
                .takes_value(true)
                .default_value("10.0"),
        )
        .arg(
            Arg::with_name("initial-error")
                .long("initial-error")
                .require_equals(true)
                .value_name("VALUE")
                .help("Start value for stepping error")
                .takes_value(true)
                .default_value("1e-4"),
        )
        .arg(
            Arg::with_name("initial-step-length")
                .long("initial-step-length")
                .require_equals(true)
                .value_name("VALUE")
                .help("Initial step size")
                .takes_value(true)
                .default_value("1e-4"),
        )
        .arg(
            Arg::with_name("sudden-reversals-for-sink")
                .long("sudden-reversals-for-sink")
                .require_equals(true)
                .value_name("NUMBER")
                .help(
                    "Number of sudden direction reversals before the area is considered\n\
                     a sink",
                )
                .takes_value(true)
                .default_value("3"),
        )
        .arg(
            Arg::with_name("disable-pi-control")
                .long("disable-pi-control")
                .help(
                    "Disable Proportional Integral (PI) control used for stabilizing the stepping",
                ),
        )
        .arg(
            Arg::with_name("stepping-scheme")
                .long("stepping-scheme")
                .require_equals(true)
                .value_name("NAME")
                .help("Which Runge-Kutta-Fehlberg stepping scheme to use\n")
                .takes_value(true)
                .possible_values(&["rkf23", "rkf45"])
                .default_value("rkf45"),
        )
}

/// Determines Runge-Kutta-Fehlberg stepper parameters based on
/// provided options.
pub fn construct_rkf_stepper_config_from_options(
    arguments: &ArgMatches,
) -> (RKFStepperType, RKFStepperConfig) {
    let dense_step_length =
        utils::get_value_from_required_parseable_argument(arguments, "dense-step-length");
    let max_step_attempts =
        utils::get_value_from_required_parseable_argument(arguments, "max-step-attempts");
    let absolute_tolerance =
        utils::get_value_from_required_parseable_argument(arguments, "absolute-tolerance");
    let relative_tolerance =
        utils::get_value_from_required_parseable_argument(arguments, "relative-tolerance");
    let safety_factor =
        utils::get_value_from_required_parseable_argument(arguments, "safety-factor");
    let min_step_scale =
        utils::get_value_from_required_parseable_argument(arguments, "min-step-scale");
    let max_step_scale =
        utils::get_value_from_required_parseable_argument(arguments, "max-step-scale");
    let initial_error =
        utils::get_value_from_required_parseable_argument(arguments, "initial-error");
    let initial_step_length =
        utils::get_value_from_required_parseable_argument(arguments, "initial-step-length");
    let sudden_reversals_for_sink =
        utils::get_value_from_required_parseable_argument(arguments, "sudden-reversals-for-sink");
    let use_pi_control = !arguments.is_present("disable-pi-control");

    let stepper_type = utils::get_value_from_required_constrained_argument(
        arguments,
        "stepping-scheme",
        &["rkf23", "rkf45"],
        &[RKFStepperType::RKF23, RKFStepperType::RKF45],
    );

    (
        stepper_type,
        RKFStepperConfig {
            dense_step_length,
            max_step_attempts,
            absolute_tolerance,
            relative_tolerance,
            safety_factor,
            min_step_scale,
            max_step_scale,
            initial_error,
            initial_step_length,
            sudden_reversals_for_sink,
            use_pi_control,
        },
    )
}
