//! Command line interface for tracing corks in a set of snapshots.

use super::SnapNumInRange;
use crate::{
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
    corks::{ConstantCorkAdvector, CorkAdvector, CorkSet, CorkStepper, HeunCorkStepper},
    create_subcommand,
    grid::Grid3,
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
        utils::AtomicOutputPath,
    },
    seeding::Seeder3,
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

pub type CorksState = CorkSet;

/// Builds a representation of the `snapshot-corks` command line subcommand.
pub fn create_corks_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("corks")
        .about("Trace corks in the velocity field of a set of snapshots")
        .after_help(
            "You can use subcommands to configure each action. The subcommands must be\n\
             specified in the order interpolator -> seeder, with options\n\
             for each action directly following the subcommand. Any action(s) except seeding\n\
             can be left unspecified, in which case the default implementation and parameters\n\
             are used for that action.",
        )
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .arg(
            Arg::with_name("output-file")
                .value_name("OUTPUT_FILE")
                .help(
                    "Path of the file where the cork data should be saved\n\
                     Writes in the following format based on the file extension:\
                     \n    *.corks: Creates a binary file readable by the backstaff Python package\
                     \n    *.pickle: Creates a Python pickle file\
                     \n    *.json: Creates a JSON file\
                     \n    *.h5part: Creates a H5Part file (requires the hdf5 feature)",
                )
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing file (unless listed as protected)"),
        )
        .arg(
            Arg::with_name("sampled-scalar-quantities")
                .long("sampled-scalar-quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help("List of scalar quantities to sample along cork paths (comma-separated)")
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("sampled-vector-quantities")
                .long("sampled-vector-quantities")
                .require_equals(true)
                .require_delimiter(true)
                .value_name("NAMES")
                .help("List of vector quantities to sample along cork paths (comma-separated)")
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("drop-h5part-id")
                .long("drop-h5part-id")
                .help(
                    "Reduce H5Part file size by excluding particle IDs required by some tools\n\
                     (e.g. VisIt)",
                ),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages while tracing corks"),
        )
        .subcommand(
            create_subcommand!(corks, poly_fit_interpolator)
                .setting(AppSettings::SubcommandRequired)
                .subcommand(create_subcommand!(poly_fit_interpolator, slice_seeder))
                .subcommand(create_subcommand!(poly_fit_interpolator, volume_seeder))
                .subcommand(create_subcommand!(poly_fit_interpolator, manual_seeder)),
        )
        .subcommand(create_subcommand!(corks, slice_seeder))
        .subcommand(create_subcommand!(corks, volume_seeder))
        .subcommand(create_subcommand!(corks, manual_seeder))
}

/// Runs the actions for the `snapshot-corks` subcommand using the given arguments.
pub fn run_corks_subcommand<G, R>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
    corks_state: &mut Option<CorksState>,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
{
    run_with_selected_interpolator(
        arguments,
        snapshot,
        snap_num_in_range,
        protected_file_types,
        corks_state,
    );
}

fn run_with_selected_interpolator<G, R>(
    root_arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
    corks_state: &mut Option<CorksState>,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
{
    let (interpolator_config, interpolator_arguments) = if let Some(interpolator_arguments) =
        root_arguments.subcommand_matches("poly_fit_interpolator")
    {
        (
            construct_poly_fit_interpolator_config_from_options(interpolator_arguments),
            interpolator_arguments,
        )
    } else {
        (PolyFitInterpolatorConfig::default(), root_arguments)
    };

    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    run_tracing(
        root_arguments,
        interpolator_arguments,
        snapshot,
        snap_num_in_range,
        interpolator,
        protected_file_types,
        corks_state,
    );
}

fn run_tracing<G, R, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    snap_num_in_range: &Option<SnapNumInRange>,
    interpolator: I,
    protected_file_types: &[&str],
    corks_state: &mut Option<CorksState>,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
    I: Interpolator3,
{
    if is_first_iteration(corks_state) {
        initialize_with_selected_seeder(
            root_arguments,
            arguments,
            snapshot,
            interpolator,
            corks_state,
        );
    } else {
        let corks = corks_state.as_mut().expect("Corks state not initialized");
        advect_with_selected_advector(snapshot, interpolator, corks);
    }
    write_output(
        root_arguments,
        snap_num_in_range,
        protected_file_types,
        corks_state,
    );
}

fn is_first_iteration(corks_state: &Option<CorksState>) -> bool {
    corks_state.is_none()
}

fn initialize_with_selected_seeder<G, R, I>(
    root_arguments: &ArgMatches,
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: I,
    corks_state: &mut Option<CorksState>,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
    I: Interpolator3,
{
    if let Some(seeder_arguments) = arguments.subcommand_matches("slice_seeder") {
        let seeder = create_slice_seeder_from_arguments(seeder_arguments, snapshot, &interpolator);
        initialize_corks(root_arguments, snapshot, interpolator, seeder, corks_state);
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("volume_seeder") {
        let seeder = create_volume_seeder_from_arguments(seeder_arguments, snapshot, &interpolator);
        initialize_corks(root_arguments, snapshot, interpolator, seeder, corks_state);
    } else if let Some(seeder_arguments) = arguments.subcommand_matches("manual_seeder") {
        let seeder = create_manual_seeder_from_arguments(seeder_arguments);
        initialize_corks(root_arguments, snapshot, interpolator, seeder, corks_state);
    } else {
        exit_with_error!("Error: No seeder specified")
    };
}

fn obtain_sampled_quantity_names(
    root_arguments: &ArgMatches,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let (scalar_quantity_names, vector_magnitude_names) = if let Some(sampled_scalar_values) =
        root_arguments.values_of("sampled-scalar-quantities")
    {
        let (scalar_quantity_names, vector_magnitude_names): (Vec<_>, Vec<_>) =
            sampled_scalar_values
                .into_iter()
                .map(|name| match cli_utils::extract_magnitude_name(name) {
                    Some(magnitude_name) => (None, Some(magnitude_name.to_string())),
                    None => (Some(name.to_string()), None),
                })
                .unzip();
        (
            scalar_quantity_names
                .into_iter()
                .filter_map(|name| name)
                .collect(),
            vector_magnitude_names
                .into_iter()
                .filter_map(|name| name)
                .collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let vector_quantity_names = if let Some(sampled_scalar_values) =
        root_arguments.values_of("sampled-vector-quantities")
    {
        sampled_scalar_values
            .into_iter()
            .map(|name| name.to_string())
            .collect()
    } else {
        Vec::new()
    };
    (
        scalar_quantity_names,
        vector_quantity_names,
        vector_magnitude_names,
    )
}

fn initialize_corks<G, R, I, Sd>(
    root_arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: I,
    seeder: Sd,
    corks_state: &mut Option<CorksState>,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
    I: Interpolator3,
    Sd: Seeder3,
{
    let (scalar_quantity_names, vector_quantity_names, vector_magnitude_names) =
        obtain_sampled_quantity_names(root_arguments);

    *corks_state = Some(exit_on_error!(
        CorkSet::new(
            seeder,
            snapshot,
            &interpolator,
            scalar_quantity_names,
            vector_quantity_names,
            vector_magnitude_names,
            root_arguments.is_present("verbose").into(),
        ),
        "Error: Could not initialize corks: {}"
    ));
    snapshot.drop_all_fields();
}

fn advect_with_selected_advector<G, R, I>(
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: I,
    corks: &mut CorkSet,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
    I: Interpolator3,
{
    let advector = ConstantCorkAdvector;

    advect_with_selected_stepper(snapshot, interpolator, advector, corks);
}

fn advect_with_selected_stepper<G, R, I, A>(
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: I,
    advector: A,
    corks: &mut CorkSet,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
    I: Interpolator3,
    A: CorkAdvector,
{
    let stepper = HeunCorkStepper;

    advect_corks(snapshot, interpolator, advector, stepper, corks);
}

fn advect_corks<G, R, I, A, St>(
    snapshot: &mut SnapshotCacher3<G, R>,
    interpolator: I,
    advector: A,
    stepper: St,
    corks: &mut CorkSet,
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
    I: Interpolator3,
    A: CorkAdvector,
    St: CorkStepper,
{
    exit_on_error!(
        advector.advect_corks(corks, snapshot, &interpolator, &stepper),
        "Error: Could not advect corks: {}"
    );
    snapshot.drop_all_fields();
}

fn should_write_output(snap_num_in_range: &Option<SnapNumInRange>) -> bool {
    match snap_num_in_range {
        Some(snap_num_in_range) => {
            snap_num_in_range.current_offset == snap_num_in_range.final_offset
        }
        None => {
            eprintln!(
                "Warning: No snap range specified for cork tracing, using single snapshot only\
                 \n(add --snap-range=<FIRST,LAST> flag after snapshot command to fix)"
            );
            true
        }
    }
}

fn write_output(
    root_arguments: &ArgMatches,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
    corks_state: &Option<CorksState>,
) {
    let write_output = should_write_output(snap_num_in_range);

    if write_output || is_first_iteration(corks_state) {
        let output_file_path = exit_on_error!(
            PathBuf::from_str(
                root_arguments
                    .value_of("output-file")
                    .expect("No value for required argument"),
            ),
            "Error: Could not interpret path to output file: {}"
        );

        let output_type = OutputType::from_path(&output_file_path);

        let automatic_overwrite = root_arguments.is_present("overwrite");

        let atomic_output_path = exit_on_error!(
            AtomicOutputPath::new(output_file_path),
            "Error: Could not create temporary output file: {}"
        );

        if atomic_output_path.write_should_be_skipped(automatic_overwrite, protected_file_types) {
            return;
        }

        if !write_output {
            return;
        }

        let corks = corks_state.as_ref().expect("Corks state not initialized");

        if corks.verbose().is_yes() {
            println!(
                "Saving corks in {}",
                atomic_output_path
                    .target_path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
            );
        }

        exit_on_error!(
            match output_type {
                OutputType::Cork => unimplemented!(),
                OutputType::Pickle => corks.save_as_pickle(atomic_output_path.temporary_path()),
                OutputType::JSON => corks.save_as_json(atomic_output_path.temporary_path()),
                #[cfg(feature = "hdf5")]
                OutputType::H5Part => unimplemented!(),
            },
            "Error: Could not save output data: {}"
        );

        exit_on_error!(
            atomic_output_path.perform_replace(),
            "Error: Could not move temporary output file to target path: {}"
        );
    }
}

#[derive(Copy, Clone, Debug)]
enum OutputType {
    Cork,
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
            "cork" => Self::Cork,
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
            "cork, pickle, json{}",
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
                Self::Cork => "cork",
                Self::Pickle => "pickle",
                Self::JSON => "json",
                #[cfg(feature = "hdf5")]
                Self::H5Part => "h5part",
            }
        )
    }
}
