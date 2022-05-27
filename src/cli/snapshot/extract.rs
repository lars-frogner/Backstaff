//! Command line interface for extracting a subdomain of a snapshot.

use crate::{
    add_subcommand_combinations,
    cli::{
        snapshot::{
            derive::{create_derive_provider, create_derive_subcommand},
            inspect::{create_inspect_subcommand, run_inspect_subcommand},
            resample::{create_resample_subcommand, run_resample_subcommand},
            write::{create_write_subcommand, run_write_subcommand},
            SnapNumInRange,
        },
        utils,
    },
    exit_with_error,
    geometry::{
        Dim3::{X, Y, Z},
        Idx3, Point3,
    },
    grid::{fgr, Grid3},
    io::{
        snapshot::{CachingSnapshotProvider3, ExtractedSnapshotProvider3, SnapshotProvider3},
        Verbose,
    },
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use std::collections::HashMap;

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-extract` command line subcommand.
pub fn create_extract_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "extract";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Extract a subdomain of the snapshot")
        .long_about(
            "Extract a subdomain of the snapshot.\n\
             Creates new snapshot that is restricted to a specified subgrid of the\n\
             original one, with identical values. The subgrid can be specified\n\
             with coordinate bounds, index ranges or a combination of these.",
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
                .help("Limits for the x-coordinates of the subgrid to extract")
                .takes_value(true)
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
                .help("Limits for the y-coordinates of the subgrid to extract")
                .takes_value(true)
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
                .help("Limits for the z-coordinates of the subgrid to extract")
                .takes_value(true)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("i-range")
                .short('i')
                .long("i-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["START", "END"])
                .help("Range of indices to extract along the x-axis (inclusive)\n")
                .takes_value(true)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("j-range")
                .short('j')
                .long("j-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["START", "END"])
                .help("Range of indices to extract along the y-axis (inclusive)\n")
                .takes_value(true)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("k-range")
                .short('k')
                .long("k-range")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_names(&["START", "END"])
                .help("Range of indices to extract along the z-axis (inclusive)\n")
                .takes_value(true)
                .default_value("min,max"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to extraction"),
        );

    #[cfg(feature = "synthesis")]
    let command = add_subcommand_combinations!(command, command_name, true; derive, synthesize, (resample, write, inspect));
    #[cfg(not(feature = "synthesis"))]
    let command = add_subcommand_combinations!(command, command_name, true; derive, (resample, write, inspect));

    command
}

/// Runs the actions for the `snapshot-extract` subcommand using the given arguments.
pub fn run_extract_subcommand<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let original_grid = provider.grid();
    let original_shape = original_grid.shape();
    let original_lower_bounds = original_grid.lower_bounds();
    let original_upper_bounds = original_grid.upper_bounds();

    let x_bounds = utils::parse_limits_with_min_max(
        arguments,
        "x-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::No,
        original_lower_bounds[X],
        original_upper_bounds[X],
    );
    let y_bounds = utils::parse_limits_with_min_max(
        arguments,
        "y-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::No,
        original_lower_bounds[Y],
        original_upper_bounds[Y],
    );
    let z_bounds = utils::parse_limits_with_min_max(
        arguments,
        "z-bounds",
        utils::AllowSameValue::Yes,
        utils::AllowInfinity::No,
        original_lower_bounds[Z],
        original_upper_bounds[Z],
    );

    let i_range = utils::parse_int_limits_with_min_max(
        arguments,
        "i-range",
        utils::AllowSameValue::Yes,
        0,
        original_shape[X] - 1,
    );
    let j_range = utils::parse_int_limits_with_min_max(
        arguments,
        "j-range",
        utils::AllowSameValue::Yes,
        0,
        original_shape[Y] - 1,
    );
    let k_range = utils::parse_int_limits_with_min_max(
        arguments,
        "k-range",
        utils::AllowSameValue::Yes,
        0,
        original_shape[Z] - 1,
    );

    let lower_bounds = Point3::new(x_bounds.0, y_bounds.0, z_bounds.0);
    let upper_bounds = Point3::new(x_bounds.1, y_bounds.1, z_bounds.1);

    exit_on_false!(
        upper_bounds[X] > lower_bounds[X]
            && upper_bounds[Y] > lower_bounds[Y]
            && upper_bounds[Z] > lower_bounds[Z],
        "Error: Lower bounds ({}) must be smaller than upper bounds ({})",
        lower_bounds,
        upper_bounds
    );

    let lower_indices = Idx3::new(i_range.0, j_range.0, k_range.0);
    let upper_indices = Idx3::new(i_range.1, j_range.1, k_range.1);

    exit_on_false!(
        upper_indices[X] >= lower_indices[X]
            && upper_indices[Y] >= lower_indices[Y]
            && upper_indices[Z] >= lower_indices[Z],
        "Error: Lower indices ({}) are too large compared to upper indices ({}) to include any grid cells",
        lower_indices, upper_indices
    );

    let lower_indices_from_bounds = exit_on_none!(
        original_grid.find_fist_grid_cell_inside_lower_bounds(&lower_bounds),
        "Error: Lower coordinate bounds are too high to include any grid cells, upper snapshot bounds are {}",
        original_upper_bounds
    );
    let upper_indices_from_bounds = exit_on_none!(
        original_grid.find_last_grid_cell_inside_upper_bounds(&upper_bounds),
        "Error: Upper coordinate bounds are too low to include any grid cells, lower snapshot bounds are {}",
        original_lower_bounds
    );

    let lower_indices = lower_indices.max_with(&lower_indices_from_bounds);
    let upper_indices = upper_indices.min_with(&upper_indices_from_bounds);

    exit_on_false!(
        upper_indices[X] >= lower_indices[X]
            && upper_indices[Y] >= lower_indices[Y]
            && upper_indices[Z] >= lower_indices[Z],
        "Error: Final lower indices ({}) are too large compared to upper indices ({}) to include any grid cells",
        lower_indices, upper_indices
    );

    let verbose: Verbose = arguments.is_present("verbose").into();

    if verbose.is_yes() {
        let new_lower_bounds = original_grid.grid_cell_lower_bounds(&lower_indices);
        let new_upper_bounds = original_grid.grid_cell_upper_bounds(&upper_indices);
        println!(
            "Creating subgrid\n\
             Index ranges: {} -> {}\n\
             Shape: [{}, {}, {}]\n\
             Bounds: {} -> {}\n\
             Extents: {}",
            lower_indices,
            upper_indices,
            1 + upper_indices[X] - lower_indices[X],
            1 + upper_indices[Y] - lower_indices[Y],
            1 + upper_indices[Z] - lower_indices[Z],
            &new_lower_bounds,
            &new_upper_bounds,
            &new_upper_bounds - &new_lower_bounds
        );
    }

    let provider = ExtractedSnapshotProvider3::new(provider, lower_indices, upper_indices, verbose);

    run_extract_subcommand_with_derive(
        arguments,
        provider,
        snap_num_in_range,
        protected_file_types,
    );
}

fn run_extract_subcommand_with_derive<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G> + Sync,
{
    if let Some(derive_arguments) = arguments.subcommand_matches("derive") {
        let provider = create_derive_provider(derive_arguments, provider);
        run_extract_subcommand_with_synthesis(
            derive_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
    } else {
        run_extract_subcommand_with_synthesis_added_caching(
            arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
    }
}

fn run_extract_subcommand_with_synthesis<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G> + Sync,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider =
            super::synthesize::create_synthesize_provider(synthesize_arguments, provider);
        run_extract_subcommand_for_provider(
            synthesize_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
        return;
    }

    run_extract_subcommand_for_provider(
        arguments,
        provider,
        snap_num_in_range,
        protected_file_types,
    );
}

fn run_extract_subcommand_with_synthesis_added_caching<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G> + Sync,
{
    #[cfg(feature = "synthesis")]
    if let Some(synthesize_arguments) = arguments.subcommand_matches("synthesize") {
        let provider = super::synthesize::create_synthesize_provider_added_caching(
            synthesize_arguments,
            provider,
        );
        run_extract_subcommand_for_provider(
            synthesize_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
        return;
    }

    run_extract_subcommand_for_provider(
        arguments,
        provider,
        snap_num_in_range,
        protected_file_types,
    );
}

fn run_extract_subcommand_for_provider<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G> + Sync,
{
    if let Some(resample_arguments) = arguments.subcommand_matches("resample") {
        run_resample_subcommand(
            resample_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
    } else if let Some(write_arguments) = arguments.subcommand_matches("write") {
        run_write_subcommand(
            write_arguments,
            provider,
            snap_num_in_range,
            HashMap::new(),
            protected_file_types,
        );
    } else if let Some(inspect_arguments) = arguments.subcommand_matches("inspect") {
        run_inspect_subcommand(inspect_arguments, provider);
    }
}
