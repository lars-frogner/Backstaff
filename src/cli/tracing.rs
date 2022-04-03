//! Command line interface for field line tracing.

pub mod field_line;
pub mod stepping;

use self::{
    field_line::basic::{
        construct_basic_field_line_tracer_config_from_options,
        create_basic_field_line_tracer_subcommand,
    },
    stepping::rkf::{construct_rkf_stepper_config_from_options, create_rkf_stepper_subcommand},
};
use crate::{
    add_subcommand_combinations,
    cli::{
        interpolation::poly_fit::{
            construct_poly_fit_interpolator_config_from_options,
            create_poly_fit_interpolator_subcommand,
        },
        seeding::{
            manual::{create_manual_seeder_from_arguments, create_manual_seeder_subcommand},
            slice::{create_slice_seeder_from_arguments, create_slice_seeder_subcommand},
            volume::{create_volume_seeder_from_arguments, create_volume_seeder_subcommand},
        },
        snapshot::SnapNumInRange,
        utils as cli_utils,
    },
    exit_on_error, exit_with_error,
    grid::Grid3,
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{self, fdt, SnapshotCacher3, SnapshotProvider3},
        utils::AtomicOutputPath,
    },
    seeding::Seeder3,
    tracing::{
        field_line::{
            basic::{BasicFieldLineTracer3, BasicFieldLineTracerConfig},
            FieldLineSet3, FieldLineSetProperties3, FieldLineTracer3,
        },
        stepping::{
            rkf::{
                rkf23::RKF23StepperFactory3, rkf45::RKF45StepperFactory3, RKFStepperConfig,
                RKFStepperType,
            },
            StepperFactory3,
        },
    },
};
use clap::{Arg, ArgMatches, Command};
use rayon::prelude::*;
use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Builds a representation of the `trace` command line subcommand.
pub fn create_trace_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "trace";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Trace field lines of a vector field in the snapshot")
        .after_help(
            "You can use subcommands to configure each action. The subcommands must be\n\
             specified in the order tracer -> stepper -> interpolator -> seeder, with options\n\
             for each action directly following the subcommand. Any action(s) except seeding\n\
             can be left unspecified, in which case the default implementation and parameters\n\
             are used for that action.",
        )
        .arg(
            Arg::new("output-file")
                .value_name("OUTPUT_FILE")
                .help(
                    "Path of the file where the field line data should be saved\n\
                       Writes in the following format based on the file extension:\
                       \n    *.fl: Creates a binary file readable by the backstaff Python package\
                       \n    *.pickle: Creates a Python pickle file\
                       \n    *.json: Creates a JSON file\
                       \n    *.h5part: Creates a H5Part file (requires the hdf5 feature)",
                )
                .required(true)
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
        .arg(
            Arg::new("vector-quantity")
                .short('q')
                .long("vector-quantity")
                .require_equals(true)
                .value_name("NAME")
                .help("Vector field from the snapshot to trace")
                .takes_value(true)
                .default_value("b"),
        )
        .arg(
            Arg::new("extracted-quantities")
                .long("extracted-quantities")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help("List of quantities to extract along field line paths (comma-separated)")
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(
            Arg::new("extracted-seed-quantities")
                .long("extracted-seed-quantities")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help("List of quantities to extract at seed positions (comma-separated)")
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(Arg::new("drop-h5part-id").long("drop-h5part-id").help(
            "Reduce H5Part file size by excluding particle IDs required by some tools\n\
                     (e.g. VisIt)",
        ))
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages while tracing field lines"),
        )
        .arg(
            Arg::new("print-parameter-values")
                .short('p')
                .long("print-parameter-values")
                .help("Prints the values of all the parameters that will be used")
                .hide(true),
        )
        .subcommand_required(true)
        .subcommand(create_basic_field_line_tracer_subcommand(command_name));

    add_subcommand_combinations!(command, command_name, true; rkf_stepper, poly_fit_interpolator, (slice_seeder, volume_seeder, manual_seeder))
}

/// Runs the actions for the `trace` subcommand using the given arguments.
pub fn run_trace_subcommand<G, P>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, P>,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G> + Sync,
{
    run_with_selected_tracer(arguments, snapshot, snap_num_in_range, protected_file_types);
}

#[derive(Copy, Clone, Debug)]
enum OutputType {
    Fl,
    Pickle,
    JSON,
    #[cfg(feature = "hdf5")]
    H5Part,
}

impl OutputType {
    fn from_path<P: AsRef<Path>>(file_path: P) -> Self {
        Self::from_extension(
            file_path
                .as_ref()
                .extension()
                .unwrap_or_else(|| {
                    exit_with_error!(
                        "Error: Missing extension for output file\n\
                         Valid extensions are: {}",
                        Self::valid_extensions_string()
                    )
                })
                .to_string_lossy()
                .as_ref(),
        )
    }

    fn from_extension(extension: &str) -> Self {
        match extension {
            "fl" => Self::Fl,
            "pickle" => Self::Pickle,
            "json" => Self::JSON,
            "h5part" => {
                #[cfg(feature = "hdf5")]
                {
                    Self::H5Part
                }
                #[cfg(not(feature = "hdf5"))]
                exit_with_error!("Error: Compile with hdf5 feature in order to write H5Part files\n\
                                  Tip: Use cargo flag --features=hdf5 and make sure the HDF5 library is available");
            }
            invalid => exit_with_error!(
                "Error: Invalid extension {} for output file\n\
                 Valid extensions are: {}",
                invalid,
                Self::valid_extensions_string()
            ),
        }
    }

    fn valid_extensions_string() -> String {
        format!(
            "fl, pickle, json{}",
            if cfg!(feature = "hdf5") {
                ", h5part"
            } else {
                ""
            }
        )
    }
}

impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Fl => "fl",
                Self::Pickle => "pickle",
                Self::JSON => "json",
                #[cfg(feature = "hdf5")]
                Self::H5Part => "h5part",
            }
        )
    }
}

fn run_with_selected_tracer<G, P>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, P>,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G> + Sync,
{
    let (tracer_config, tracer_arguments) =
        if let Some(tracer_arguments) = arguments.subcommand_matches("basic_tracer") {
            (
                construct_basic_field_line_tracer_config_from_options(tracer_arguments),
                tracer_arguments,
            )
        } else {
            (BasicFieldLineTracerConfig::default(), arguments)
        };

    if arguments.is_present("print-parameter-values") {
        println!("{:#?}", tracer_config);
    }

    let tracer = BasicFieldLineTracer3::new(tracer_config);

    run_with_selected_stepper_factory(
        arguments,
        tracer_arguments,
        snapshot,
        snap_num_in_range,
        tracer,
        protected_file_types,
    );
}

fn run_with_selected_stepper_factory<G, P, Tr>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, P>,
    snap_num_in_range: &Option<SnapNumInRange>,
    tracer: Tr,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G> + Sync,
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
{
    let ((stepper_type, stepper_config), stepper_arguments) =
        if let Some(stepper_arguments) = arguments.subcommand_matches("rkf_stepper") {
            (
                construct_rkf_stepper_config_from_options(stepper_arguments),
                stepper_arguments,
            )
        } else {
            (
                (RKFStepperType::RKF45, RKFStepperConfig::default()),
                arguments,
            )
        };

    if root_arguments.is_present("print-parameter-values") {
        println!("{:#?}\nstepper_type: {:?}", stepper_config, stepper_type);
    }

    match stepper_type {
        RKFStepperType::RKF23 => run_with_selected_interpolator(
            root_arguments,
            stepper_arguments,
            snapshot,
            snap_num_in_range,
            tracer,
            RKF23StepperFactory3::new(stepper_config),
            protected_file_types,
        ),
        RKFStepperType::RKF45 => run_with_selected_interpolator(
            root_arguments,
            stepper_arguments,
            snapshot,
            snap_num_in_range,
            tracer,
            RKF45StepperFactory3::new(stepper_config),
            protected_file_types,
        ),
    }
}

fn run_with_selected_interpolator<G, P, Tr, StF>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, P>,
    snap_num_in_range: &Option<SnapNumInRange>,
    tracer: Tr,
    stepper_factory: StF,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G> + Sync,
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
    StF: StepperFactory3 + Sync,
{
    let (interpolator_config, interpolator_arguments) = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        (
            construct_poly_fit_interpolator_config_from_options(interpolator_arguments),
            interpolator_arguments,
        )
    } else {
        (PolyFitInterpolatorConfig::default(), arguments)
    };

    if root_arguments.is_present("print-parameter-values") {
        println!("{:#?}", interpolator_config);
    }

    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    run_with_selected_seeder(
        root_arguments,
        interpolator_arguments,
        snapshot,
        snap_num_in_range,
        tracer,
        stepper_factory,
        interpolator,
        protected_file_types,
    );
}

fn run_with_selected_seeder<G, P, Tr, StF, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, P>,
    snap_num_in_range: &Option<SnapNumInRange>,
    tracer: Tr,
    stepper_factory: StF,
    interpolator: I,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G> + Sync,
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
    StF: StepperFactory3 + Sync,
    I: Interpolator3,
{
    if let Some(seeder_arguments) = arguments.subcommand_matches("slice_seeder") {
        let seeder = create_slice_seeder_from_arguments(seeder_arguments, snapshot, &interpolator);
        run_tracing(
            root_arguments,
            snapshot,
            snap_num_in_range,
            tracer,
            stepper_factory,
            interpolator,
            seeder,
            protected_file_types,
        );
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("volume_seeder") {
        let seeder = create_volume_seeder_from_arguments(seeder_arguments, snapshot, &interpolator);
        run_tracing(
            root_arguments,
            snapshot,
            snap_num_in_range,
            tracer,
            stepper_factory,
            interpolator,
            seeder,
            protected_file_types,
        );
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("manual_seeder") {
        let seeder = create_manual_seeder_from_arguments(seeder_arguments);
        run_tracing(
            root_arguments,
            snapshot,
            snap_num_in_range,
            tracer,
            stepper_factory,
            interpolator,
            seeder,
            protected_file_types,
        );
    } else {
        exit_with_error!("Error: No seeder specified")
    };
}

fn run_tracing<G, P, Tr, StF, I, Sd>(
    root_arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, P>,
    snap_num_in_range: &Option<SnapNumInRange>,
    tracer: Tr,
    stepper_factory: StF,
    interpolator: I,
    seeder: Sd,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G> + Sync,
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
    StF: StepperFactory3 + Sync,
    I: Interpolator3,
    Sd: Seeder3,
{
    let mut output_file_path = exit_on_error!(
        PathBuf::from_str(
            root_arguments
                .value_of("output-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    let output_type = OutputType::from_path(&output_file_path);

    if let Some(snap_num_in_range) = snap_num_in_range {
        output_file_path.set_file_name(snapshot::create_new_snapshot_file_name_from_path(
            &output_file_path,
            snap_num_in_range.offset(),
            &output_type.to_string(),
            true,
        ));
    }

    let overwrite_mode = cli_utils::overwrite_mode_from_arguments(root_arguments);

    let atomic_output_path = exit_on_error!(
        AtomicOutputPath::new(output_file_path),
        "Error: Could not create temporary output file: {}"
    );

    if !atomic_output_path.check_if_write_allowed(overwrite_mode, protected_file_types) {
        return;
    }

    let extra_atomic_output_path = match output_type {
        #[cfg(feature = "hdf5")]
        OutputType::H5Part => {
            let extra_atomic_output_path = exit_on_error!(
                AtomicOutputPath::new(
                    atomic_output_path
                        .target_path()
                        .with_extension("seeds.h5part")
                ),
                "Error: Could not create temporary output file: {}"
            );
            if !extra_atomic_output_path
                .check_if_write_allowed(overwrite_mode, protected_file_types)
            {
                return;
            }
            Some(extra_atomic_output_path)
        }
        _ => None,
    };

    let quantity = root_arguments
        .value_of("vector-quantity")
        .expect("No value for argument with default");
    exit_on_error!(
        snapshot.cache_vector_field(quantity),
        "Error: Could not read quantity {0} in snapshot: {1}",
        quantity
    );

    let field_lines = FieldLineSet3::trace(
        quantity,
        snapshot,
        seeder,
        &tracer,
        &interpolator,
        &stepper_factory,
        root_arguments.is_present("verbose").into(),
    );
    snapshot.drop_all_fields();
    perform_post_tracing_actions(
        root_arguments,
        output_type,
        atomic_output_path,
        extra_atomic_output_path,
        snapshot,
        interpolator,
        field_lines,
    );
}

fn perform_post_tracing_actions<G, P, I>(
    root_arguments: &ArgMatches,
    output_type: OutputType,
    atomic_output_path: AtomicOutputPath,
    extra_atomic_output_path: Option<AtomicOutputPath>,
    snapshot: &mut SnapshotCacher3<G, P>,
    interpolator: I,
    mut field_lines: FieldLineSet3,
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
    I: Interpolator3,
{
    if let Some(extra_fixed_scalars) = root_arguments
        .values_of("extracted-seed-quantities")
        .map(|values| values.collect::<Vec<_>>())
    {
        for name in extra_fixed_scalars {
            field_lines.extract_fixed_scalars(
                exit_on_error!(
                    snapshot.obtain_scalar_field(name),
                    "Error: Could not read quantity {0} in snapshot: {1}",
                    name
                ),
                &interpolator,
            );
            snapshot.drop_scalar_field(name);
        }
    }
    if let Some(extra_varying_scalars) = root_arguments
        .values_of("extracted-quantities")
        .map(|values| values.collect::<Vec<_>>())
    {
        for name in extra_varying_scalars {
            if let Some(name) = snapshot::extract_magnitude_name(name) {
                field_lines.extract_varying_vector_magnitudes(
                    exit_on_error!(
                        snapshot.obtain_vector_field(name),
                        "Error: Could not read quantity {0} from snapshot: {1}",
                        name
                    ),
                    &interpolator,
                );
                snapshot.drop_vector_field(name);
            } else {
                field_lines.extract_varying_scalars(
                    exit_on_error!(
                        snapshot.obtain_scalar_field(name),
                        "Error: Could not read quantity {0} from snapshot: {1}",
                        name
                    ),
                    &interpolator,
                );
                snapshot.drop_scalar_field(name);
            }
        }
    }

    if field_lines.verbose().is_yes() {
        println!(
            "Saving field lines in {}",
            atomic_output_path
                .target_path()
                .file_name()
                .unwrap()
                .to_string_lossy()
        );
    }

    exit_on_error!(
        match output_type {
            OutputType::Fl =>
                field_lines.save_into_custom_binary(atomic_output_path.temporary_path()),
            OutputType::Pickle =>
                field_lines.save_as_combined_pickles(atomic_output_path.temporary_path()),
            OutputType::JSON => field_lines.save_as_json(atomic_output_path.temporary_path()),
            #[cfg(feature = "hdf5")]
            OutputType::H5Part => field_lines.save_as_h5part(
                atomic_output_path.temporary_path(),
                extra_atomic_output_path.as_ref().unwrap().temporary_path(),
                root_arguments.is_present("drop-h5part-id"),
            ),
        },
        "Error: Could not save output data: {}"
    );

    exit_on_error!(
        atomic_output_path.perform_replace(),
        "Error: Could not move temporary output file to target path: {}"
    );
    if let Some(extra_atomic_output_path) = extra_atomic_output_path {
        exit_on_error!(
            extra_atomic_output_path.perform_replace(),
            "Error: Could not move temporary output file to target path: {}"
        );
    }
}
