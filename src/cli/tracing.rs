//! Command line interface for field line tracing.

pub mod field_line;
pub mod seeding;
pub mod stepping;

use crate::cli;
use crate::grid::Grid3;
use crate::interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig};
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::tracing::field_line::basic::{BasicFieldLineTracer3, BasicFieldLineTracerConfig};
use crate::tracing::field_line::{FieldLineSet3, FieldLineTracer3};
use crate::tracing::seeding::Seeder3;
use crate::tracing::stepping::rkf::rkf23::RKF23StepperFactory3;
use crate::tracing::stepping::rkf::rkf45::RKF45StepperFactory3;
use crate::tracing::stepping::rkf::{RKFStepperConfig, RKFStepperType};
use crate::tracing::stepping::StepperFactory3;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `trace` command line subcommand.
pub fn build_subcommand_trace<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("trace")
        .about("Trace field lines of a vector field in the snapshot")
        .after_help(
            "You can use subcommands to configure each action. The subcommands must be specified\n\
             in the order tracer -> stepper -> interpolator -> seeder, with options for each\n\
             action directly following the subcommand. Any action(s) except a seeder can be left\n\
             unspecified, in which case the default implementation and parameters are used for\n\
             that action.",
        )
        .arg(
            Arg::with_name("OUTPUT_PATH")
                .help("Path where the field line data should be saved")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("vector-quantity")
                .short("q")
                .long("vector-quantity")
                .value_name("NAME")
                .long_help("Which vector field from the snapshot to trace")
                .next_line_help(true)
                .takes_value(true)
                .default_value("b"),
        )
        .arg(
            Arg::with_name("output-format")
                .short("f")
                .long("output-format")
                .value_name("FORMAT")
                .long_help("Format to use for saving field line data")
                .next_line_help(true)
                .takes_value(true)
                .possible_values(&["pickle", "json"])
                .default_value("pickle"),
        )
        .arg(
            Arg::with_name("extra-fixed-scalars")
                .long("extra-fixed-scalars")
                .value_name("NAMES")
                .long_help("List of scalar fields to extract at seed positions")
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("extra-varying-scalars")
                .long("extra-varying-scalars")
                .value_name("NAMES")
                .long_help("List of scalar fields to extract along field line paths")
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages while tracing field lines"),
        );

    let basic_field_line_tracer_subcommand =
        cli::tracing::field_line::basic::create_basic_field_line_tracer_subcommand();
    let rkf_stepper_subcommand = cli::tracing::stepping::rkf::create_rkf_stepper_subcommand();
    let poly_fit_interpolator_subcommand =
        cli::interpolation::poly_fit::create_poly_fit_interpolator_subcommand();
    let regular_slice_seeder_subcommand =
        cli::tracing::seeding::slice::regular::create_regular_slice_seeder_subcommand();
    let random_slice_seeder_subcommand =
        cli::tracing::seeding::slice::random::create_random_slice_seeder_subcommand();
    let stratified_slice_seeder_subcommand =
        cli::tracing::seeding::slice::stratified::create_stratified_slice_seeder_subcommand();

    let poly_fit_interpolator_subcommand = poly_fit_interpolator_subcommand
        .setting(AppSettings::SubcommandRequired)
        .subcommand(regular_slice_seeder_subcommand.clone())
        .subcommand(random_slice_seeder_subcommand.clone())
        .subcommand(stratified_slice_seeder_subcommand.clone());

    let rkf_stepper_subcommand = rkf_stepper_subcommand
        .setting(AppSettings::SubcommandRequired)
        .subcommand(poly_fit_interpolator_subcommand.clone())
        .subcommand(regular_slice_seeder_subcommand.clone())
        .subcommand(random_slice_seeder_subcommand.clone())
        .subcommand(stratified_slice_seeder_subcommand.clone());

    let basic_field_line_tracer_subcommand = basic_field_line_tracer_subcommand
        .setting(AppSettings::SubcommandRequired)
        .subcommand(rkf_stepper_subcommand.clone())
        .subcommand(poly_fit_interpolator_subcommand.clone())
        .subcommand(regular_slice_seeder_subcommand.clone())
        .subcommand(random_slice_seeder_subcommand.clone())
        .subcommand(stratified_slice_seeder_subcommand.clone());

    app.setting(AppSettings::SubcommandRequired)
        .subcommand(basic_field_line_tracer_subcommand)
        .subcommand(rkf_stepper_subcommand)
        .subcommand(poly_fit_interpolator_subcommand)
        .subcommand(regular_slice_seeder_subcommand)
        .subcommand(random_slice_seeder_subcommand)
        .subcommand(stratified_slice_seeder_subcommand)
}

/// Runs the actions for the `trace` subcommand using the given arguments.
pub fn run_subcommand_trace<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
) {
    run_with_selected_tracer(arguments, snapshot);
}

fn run_with_selected_tracer<G>(arguments: &ArgMatches, snapshot: &mut SnapshotCacher3<G>)
where
    G: Grid3<fdt>,
{
    let (tracer_config, tracer_arguments) =
        if let Some(tracer_arguments) = arguments.subcommand_matches("basic_tracer") {
            (cli::tracing::field_line::basic::construct_basic_field_line_tracer_config_from_options(
            tracer_arguments,
        ), tracer_arguments)
        } else {
            (BasicFieldLineTracerConfig::default(), arguments)
        };
    let tracer = BasicFieldLineTracer3::new(tracer_config);

    run_with_selected_stepper_factory(arguments, tracer_arguments, snapshot, tracer);
}

fn run_with_selected_stepper_factory<G, Tr>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
    tracer: Tr,
) where
    G: Grid3<fdt>,
    Tr: FieldLineTracer3 + Sync,
{
    let ((stepper_type, stepper_config), stepper_arguments) =
        if let Some(stepper_arguments) = arguments.subcommand_matches("rkf_stepper") {
            (
                cli::tracing::stepping::rkf::construct_rkf_stepper_config_from_options(
                    stepper_arguments,
                ),
                stepper_arguments,
            )
        } else {
            (
                (RKFStepperType::RKF45, RKFStepperConfig::default()),
                arguments,
            )
        };

    match stepper_type {
        RKFStepperType::RKF23 => run_with_selected_interpolator(
            root_arguments,
            stepper_arguments,
            snapshot,
            tracer,
            RKF23StepperFactory3::new(stepper_config),
        ),
        RKFStepperType::RKF45 => run_with_selected_interpolator(
            root_arguments,
            stepper_arguments,
            snapshot,
            tracer,
            RKF45StepperFactory3::new(stepper_config),
        ),
    }
}

fn run_with_selected_interpolator<G, Tr, StF>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
    tracer: Tr,
    stepper_factory: StF,
) where
    G: Grid3<fdt>,
    Tr: FieldLineTracer3 + Sync,
    StF: StepperFactory3 + Sync,
{
    let (interpolator_config, interpolator_arguments) = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        (
            cli::interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options(
                interpolator_arguments,
            ),
            interpolator_arguments,
        )
    } else {
        (PolyFitInterpolatorConfig::default(), arguments)
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    run_with_selected_seeder(
        root_arguments,
        interpolator_arguments,
        snapshot,
        tracer,
        stepper_factory,
        interpolator,
    );
}

fn run_with_selected_seeder<G, Tr, StF, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
    tracer: Tr,
    stepper_factory: StF,
    interpolator: I,
) where
    G: Grid3<fdt>,
    Tr: FieldLineTracer3 + Sync,
    StF: StepperFactory3 + Sync,
    I: Interpolator3,
{
    let seeder = if let Some(seeder_arguments) =
        arguments.subcommand_matches("regular_slice_seeder")
    {
        cli::tracing::seeding::slice::regular::create_regular_slice_seeder_from_arguments(
            seeder_arguments,
            snapshot.reader().grid(),
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("random_slice_seeder") {
        cli::tracing::seeding::slice::random::create_random_slice_seeder_from_arguments(
            seeder_arguments,
            snapshot.reader().grid(),
        )
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("stratified_slice_seeder") {
        cli::tracing::seeding::slice::stratified::create_stratified_slice_seeder_from_arguments(
            seeder_arguments,
            snapshot.reader().grid(),
        )
    } else {
        panic!("No seeder specified.")
    };

    run_tracing(
        root_arguments,
        snapshot,
        tracer,
        stepper_factory,
        interpolator,
        seeder,
    );
}

fn run_tracing<G, Tr, StF, I, Sd>(
    root_arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
    tracer: Tr,
    stepper_factory: StF,
    interpolator: I,
    seeder: Sd,
) where
    G: Grid3<fdt>,
    Tr: FieldLineTracer3 + Sync,
    StF: StepperFactory3 + Sync,
    I: Interpolator3,
    Sd: Seeder3,
{
    let quantity = root_arguments
        .value_of("vector-quantity")
        .expect("No value for argument with default.");
    let field = snapshot
        .obtain_vector_field(quantity)
        .unwrap_or_else(|err| panic!("Could not read {}: {}", quantity, err));

    let field_lines = FieldLineSet3::trace(
        seeder,
        &tracer,
        &field,
        &interpolator,
        stepper_factory,
        root_arguments.is_present("verbose").into(),
    );
    snapshot.drop_all_fields();
    perform_post_tracing_actions(root_arguments, snapshot, interpolator, field_lines);
}

fn perform_post_tracing_actions<G, I>(
    root_arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
    interpolator: I,
    mut field_lines: FieldLineSet3,
) where
    G: Grid3<fdt>,
    I: Interpolator3,
{
    let output_path = root_arguments
        .value_of("OUTPUT_PATH")
        .expect("No value for required argument.");

    if let Some(extra_fixed_scalars) = root_arguments
        .values_of("extra-fixed-scalars")
        .map(|values| values.collect::<Vec<_>>())
    {
        for name in extra_fixed_scalars {
            field_lines.extract_fixed_scalars(
                snapshot
                    .obtain_scalar_field(name)
                    .unwrap_or_else(|err| panic!("Could not read {} from snapshot: {}", name, err)),
                &interpolator,
            );
            snapshot.drop_scalar_field(name);
        }
    }
    if let Some(extra_varying_scalars) = root_arguments
        .values_of("extra-varying-scalars")
        .map(|values| values.collect::<Vec<_>>())
    {
        for name in extra_varying_scalars {
            field_lines.extract_varying_scalars(
                snapshot
                    .obtain_scalar_field(name)
                    .unwrap_or_else(|err| panic!("Could not read {} from snapshot: {}", name, err)),
                &interpolator,
            );
            snapshot.drop_scalar_field(name);
        }
    }

    match root_arguments
        .value_of("output-format")
        .expect("No value for argument with default.")
    {
        "pickle" => {
            field_lines
                .save_as_combined_pickles(output_path)
                .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
        }
        "json" => {
            field_lines
                .save_as_json(output_path)
                .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
        }
        invalid => panic!("Invalid output format {}.", invalid),
    }
}
