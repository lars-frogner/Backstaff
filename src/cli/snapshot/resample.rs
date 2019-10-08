//! Command line interface for resampling a snapshot.

use crate::cli;
use crate::geometry::{Dim3, In3D};
use crate::grid::hor_regular::HorRegularGrid3;
use crate::grid::regular::RegularGrid3;
use crate::grid::{Grid3, GridType};
use crate::interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig};
use crate::io::snapshot::{self, fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};
use std::sync::Arc;
use Dim3::{X, Y, Z};

/// Builds a representation of the `snapshot-resample` command line subcommand.
pub fn build_subcommand_resample<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("resample")
        .about("Creates a resampled version of the snapshot")
        .long_about(
            "Creates a resampled version of the snapshot.\n\
             The snapshot quantity fields are resampled to the grid described by a given mesh\n\
             file. ",
        )
        .after_help(
            "You can use a subcommand to configure the interpolator. If left unspecified,\n\
             the default interpolator implementation and parameters are used.",
        )
        .arg(
            Arg::with_name("MESH_PATH")
                .help("Path to the Bifrost mesh file representing the grid to resample to.")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("OUTPUT_PATH")
                .help("Path where the resample field should be saved in pickle format")
                .required(true)
                .takes_value(true),
        )
    .arg(
        Arg::with_name("resampled-grid-type")
            .long("resampled-grid-type")
            .require_equals(true)
            .value_name("TYPE")
            .long_help("Type of grid to assume for the resampled snapshot\n[default: same as original]")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["horizontally-regular", "regular"]),
    )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages while resampling the snapshot"),
        )
        .subcommand(cli::interpolation::poly_fit::create_poly_fit_interpolator_subcommand())
}

/// Runs the actions for the `snapshot-resample` subcommand using the given arguments.
pub fn run_subcommand_resample<G: Grid3<fdt>>(arguments: &ArgMatches, reader: &SnapshotReader3<G>) {
    let mesh_file_path = arguments
        .value_of("MESH_PATH")
        .expect("No value for required argument");

    let output_file_path = arguments
        .value_of("OUTPUT_PATH")
        .expect("No value for required argument");

    let is_verbose = arguments.is_present("verbose");

    let interpolator_config = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        cli::interpolation::poly_fit::construct_poly_fit_interpolator_config_from_options(
            interpolator_arguments,
        )
    } else {
        PolyFitInterpolatorConfig::default()
    };
    let interpolator = PolyFitInterpolator3::new(interpolator_config);

    let grid_type = match arguments.value_of("resampled-grid-type") {
        None => G::TYPE,
        Some("horizontally-regular") => GridType::HorRegular,
        Some("regular") => GridType::Regular,
        Some(invalid) => panic!("Invalid grid type {}", invalid),
    };

    let old_grid = reader.grid();
    let is_periodic = In3D::new(
        old_grid.is_periodic(X),
        old_grid.is_periodic(Y),
        old_grid.is_periodic(Z),
    );

    macro_rules! resample_snapshot_for_grid_type {
        ($grid_type:ty) => {{
            if is_verbose {
                println!("Constructing new grid");
            }
            let new_grid = Arc::new(
                snapshot::create_grid_from_mesh_file::<_, $grid_type>(mesh_file_path, is_periodic)
                    .unwrap_or_else(|err| panic!("Could not create resampling grid: {}", err)),
            );
            let variable_value_producer = |name: &str| {
                let field = reader.read_scalar_field(name).unwrap_or_else(|err| {
                    panic!("Could not read quantity field {}: {}", name, err)
                });
                if is_verbose {
                    println!("Resampling {}", name);
                }
                field
                    .resampled_to_grid(Arc::clone(&new_grid), &interpolator)
                    .into_values()
            };
            let variable_names = if reader.is_mhd() {
                vec!["r", "px", "py", "pz", "e", "bx", "by", "bz"]
            } else {
                vec!["r", "px", "py", "pz", "e"]
            };
            snapshot::write_3d_snapfile(
                output_file_path,
                &variable_names,
                &variable_value_producer,
                reader.endianness(),
            )
            .unwrap_or_else(|err| panic!("Could not write resampled snapshot: {}", err));
        }};
    }

    match grid_type {
        GridType::HorRegular => {
            resample_snapshot_for_grid_type!(HorRegularGrid3<fdt>);
        }
        GridType::Regular => {
            resample_snapshot_for_grid_type!(RegularGrid3<fdt>);
        }
    }
}
