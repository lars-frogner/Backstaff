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
        utils as cli_utils,
    },
    exit_on_error, exit_with_error,
    field::{DynCachingScalarFieldProvider3, DynScalarFieldProvider3, ScalarFieldCacher3},
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        InterpGridVerifier3, Interpolator3,
    },
    io::{
        snapshot::{self, fdt},
        utils::{AtomicOutputFile, IOContext},
    },
    seeding::Seeder3,
    tracing::{
        field_line::{
            basic::{BasicFieldLineTracer3, BasicFieldLineTracerConfig},
            FieldLineSet3, FieldLineSetProperties3, FieldLineTracer3,
        },
        stepping::{
            rkf::{rkf23::RKF23Stepper3, rkf45::RKF45Stepper3, RKFStepperConfig, RKFStepperType},
            DynStepper3,
        },
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use rayon::prelude::*;
use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Builds a representation of the `trace` command line subcommand.
pub fn create_trace_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "trace";

    update_command_graph!(_parent_command_name, command_name);

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
                       \n    *.pickle: Creates a Python pickle file (requires the pickle feature)\
                       \n    *.json: Creates a JSON file (requires the json feature)\
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
            Arg::new("progress")
                .short('p')
                .long("progress")
                .help("Show progress bar for tracing (also implies `verbose`)"),
        )
        .arg(
            Arg::new("print-parameter-values")
                .long("print-parameter-values")
                .help("Prints the values of all the parameters that will be used")
                .hide(true),
        )
        .subcommand_required(true)
        .subcommand(create_basic_field_line_tracer_subcommand(command_name));

    add_subcommand_combinations!(command, command_name, true; rkf_stepper, poly_fit_interpolator, (slice_seeder, volume_seeder, manual_seeder))
}

/// Runs the actions for the `trace` subcommand using the given arguments.
pub fn run_trace_subcommand(
    arguments: &ArgMatches,
    provider: DynScalarFieldProvider3<fdt>,
    io_context: &mut IOContext,
) {
    let verbosity = cli_utils::parse_verbosity(arguments, false);
    let snapshot = Box::new(ScalarFieldCacher3::new_manual_cacher(provider, verbosity));
    run_with_selected_tracer(arguments, snapshot, io_context);
}

#[derive(Copy, Clone, Debug)]
enum OutputType {
    Fl,
    #[cfg(feature = "pickle")]
    Pickle,
    #[cfg(feature = "json")]
    Json,
    #[cfg(feature = "hdf5")]
    H5Part,
}

impl OutputType {
    fn from_path(file_path: &Path) -> Self {
        Self::from_extension(
            file_path
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
            "pickle" => {
                #[cfg(feature = "pickle")]
                {
                    Self::Pickle
                }
                #[cfg(not(feature = "pickle"))]
                exit_with_error!(
                    "Error: Compile with pickle feature in order to write Pickle files\n\
                     Tip: Use cargo flag --features=pickle"
                );
            }
            "json" => {
                #[cfg(feature = "json")]
                {
                    Self::Json
                }
                #[cfg(not(feature = "json"))]
                exit_with_error!(
                    "Error: Compile with json feature in order to write JSON files\n\
                     Tip: Use cargo flag --features=json"
                );
            }
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
                #[cfg(feature = "pickle")]
                Self::Pickle => "pickle",
                #[cfg(feature = "json")]
                Self::Json => "json",
                #[cfg(feature = "hdf5")]
                Self::H5Part => "h5part",
            }
        )
    }
}

fn run_with_selected_tracer(
    arguments: &ArgMatches,
    snapshot: DynCachingScalarFieldProvider3<fdt>,
    io_context: &mut IOContext,
) {
    let (tracer_config, tracer_arguments) =
        if let Some(tracer_arguments) = arguments.subcommand_matches("basic_field_line_tracer") {
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

    run_with_selected_stepper(arguments, tracer_arguments, snapshot, tracer, io_context);
}

fn run_with_selected_stepper<Tr>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: DynCachingScalarFieldProvider3<fdt>,
    tracer: Tr,
    io_context: &mut IOContext,
) where
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
            tracer,
            Box::new(RKF23Stepper3::new(stepper_config)),
            io_context,
        ),
        RKFStepperType::RKF45 => run_with_selected_interpolator(
            root_arguments,
            stepper_arguments,
            snapshot,
            tracer,
            Box::new(RKF45Stepper3::new(stepper_config)),
            io_context,
        ),
    }
}

fn run_with_selected_interpolator<Tr>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: DynCachingScalarFieldProvider3<fdt>,
    tracer: Tr,
    stepper: DynStepper3<fdt>,
    io_context: &mut IOContext,
) where
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
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

    let interpolator = Box::new(PolyFitInterpolator3::new(interpolator_config));

    exit_on_error!(
        interpolator.verify_grid(snapshot.grid()),
        "Invalid input grid for tracing field lines: {}"
    );

    run_with_selected_seeder(
        root_arguments,
        interpolator_arguments,
        snapshot,
        tracer,
        stepper,
        interpolator.as_ref(),
        io_context,
    );
}

fn run_with_selected_seeder<Tr>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    mut snapshot: DynCachingScalarFieldProvider3<fdt>,
    tracer: Tr,
    stepper: DynStepper3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
    io_context: &mut IOContext,
) where
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
{
    if let Some(seeder_arguments) = arguments.subcommand_matches("slice_seeder") {
        let seeder =
            create_slice_seeder_from_arguments(seeder_arguments, &mut *snapshot, interpolator);
        run_tracing(
            root_arguments,
            snapshot,
            tracer,
            stepper,
            interpolator,
            seeder,
            io_context,
        );
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("volume_seeder") {
        let seeder =
            create_volume_seeder_from_arguments(seeder_arguments, &mut *snapshot, interpolator);
        run_tracing(
            root_arguments,
            snapshot,
            tracer,
            stepper,
            interpolator,
            seeder,
            io_context,
        );
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("manual_seeder") {
        let seeder = create_manual_seeder_from_arguments(seeder_arguments);
        run_tracing(
            root_arguments,
            snapshot,
            tracer,
            stepper,
            interpolator,
            seeder,
            io_context,
        );
    } else {
        exit_with_error!("Error: No seeder specified")
    };
}

fn run_tracing<Tr, Sd>(
    root_arguments: &ArgMatches,
    mut snapshot: DynCachingScalarFieldProvider3<fdt>,
    tracer: Tr,
    stepper: DynStepper3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
    seeder: Sd,
    io_context: &mut IOContext,
) where
    Tr: FieldLineTracer3 + Sync,
    <Tr as FieldLineTracer3>::Data: Send,
    FieldLineSetProperties3: FromParallelIterator<<Tr as FieldLineTracer3>::Data>,
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

    if let Some(snap_num_in_range) = io_context.get_snap_num_in_range() {
        output_file_path.set_file_name(snapshot::create_new_snapshot_file_name_from_path(
            &output_file_path,
            snap_num_in_range.offset(),
            &output_type.to_string(),
            true,
        ));
    }

    let overwrite_mode = cli_utils::overwrite_mode_from_arguments(root_arguments);
    io_context.set_overwrite_mode(overwrite_mode);

    let atomic_output_file = exit_on_error!(
        io_context.create_atomic_output_file(output_file_path),
        "Error: Could not create temporary output file: {}"
    );

    let verbosity = cli_utils::parse_verbosity(root_arguments, true);

    if !atomic_output_file.check_if_write_allowed(io_context, &verbosity) {
        return;
    }

    let extra_atomic_output_file = match output_type {
        #[cfg(feature = "hdf5")]
        OutputType::H5Part => {
            let extra_atomic_output_file = exit_on_error!(
                io_context.create_atomic_output_file(
                    atomic_output_file
                        .target_path()
                        .with_extension("seeds.h5part")
                ),
                "Error: Could not create temporary output file: {}"
            );
            if !extra_atomic_output_file.check_if_write_allowed(io_context, &verbosity) {
                return;
            }
            Some(extra_atomic_output_file)
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
        &*snapshot,
        seeder,
        &tracer,
        interpolator,
        stepper,
        verbosity,
    );
    perform_post_tracing_actions(
        root_arguments,
        output_type,
        atomic_output_file,
        extra_atomic_output_file,
        io_context,
        snapshot,
        interpolator,
        field_lines,
    );
}

fn perform_post_tracing_actions(
    root_arguments: &ArgMatches,
    output_type: OutputType,
    atomic_output_file: AtomicOutputFile,
    extra_atomic_output_file: Option<AtomicOutputFile>,
    io_context: &IOContext,
    mut snapshot: DynCachingScalarFieldProvider3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
    mut field_lines: FieldLineSet3,
) {
    if let Some(extra_fixed_scalars) = root_arguments
        .values_of("extracted-seed-quantities")
        .map(|values| values.collect::<Vec<_>>())
    {
        for name in extra_fixed_scalars {
            let name = name.to_lowercase();
            field_lines.extract_fixed_scalars(
                exit_on_error!(
                    snapshot.provide_scalar_field(&name).as_ref(),
                    "Error: Could not read quantity {0} in snapshot: {1}",
                    &name
                ),
                interpolator,
            );
        }
    }
    if let Some(extra_varying_scalars) = root_arguments
        .values_of("extracted-quantities")
        .map(|values| values.collect::<Vec<_>>())
    {
        for name in extra_varying_scalars {
            let name = name.to_lowercase();
            field_lines.extract_varying_scalars(
                exit_on_error!(
                    snapshot.provide_scalar_field(&name).as_ref(),
                    "Error: Could not read quantity {0} from snapshot: {1}",
                    &name
                ),
                interpolator,
            );
        }
    }

    if field_lines.verbosity().print_messages() {
        println!(
            "Saving field lines in {}",
            atomic_output_file
                .target_path()
                .file_name()
                .unwrap()
                .to_string_lossy()
        );
    }

    exit_on_error!(
        match output_type {
            OutputType::Fl =>
                field_lines.save_into_custom_binary(atomic_output_file.temporary_path()),
            #[cfg(feature = "pickle")]
            OutputType::Pickle =>
                field_lines.save_as_combined_pickles(atomic_output_file.temporary_path()),
            #[cfg(feature = "json")]
            OutputType::Json => field_lines.save_as_json(atomic_output_file.temporary_path()),
            #[cfg(feature = "hdf5")]
            OutputType::H5Part => field_lines.save_as_h5part(
                atomic_output_file.temporary_path(),
                extra_atomic_output_file.as_ref().unwrap().temporary_path(),
                root_arguments.is_present("drop-h5part-id"),
            ),
        },
        "Error: Could not save output data: {}"
    );

    exit_on_error!(
        io_context.close_atomic_output_file(atomic_output_file),
        "Error: Could not move temporary output file to target path: {}"
    );
    if let Some(extra_atomic_output_file) = extra_atomic_output_file {
        exit_on_error!(
            io_context.close_atomic_output_file(extra_atomic_output_file),
            "Error: Could not move temporary output file to target path: {}"
        );
    }
}
