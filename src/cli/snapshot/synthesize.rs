//! Command line interface for computing synthesized quantities for a snapshot.

use crate::{
    field::{quantities::DerivedSnapshotProvider3, synthesis::EmissivityTables},
    grid::Grid3,
    io::snapshot::{fdt, SnapshotProvider3},
    io::Verbose,
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-synthesize` command line subcommand.
pub fn create_synthesize_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "synthesize";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Computes synthetic quantities for the snapshot")
        .long_about("Computes synthetic quantities for the snapshot.")
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to computation of synthetic quantities"),
        )
}

#[cfg(feature = "synthesis")]
pub fn create_synthesize_provider<G, P>(arguments: &ArgMatches, provider: P) -> P
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    use rand::Rng;

    use crate::interpolation::poly_fit::{PolyFitInterpolator2, PolyFitInterpolatorConfig};

    let ion_lines = [("si_4".to_owned(), vec![1393.755])].into();
    let n_temperature_points = 100;
    let n_electron_density_points = 100;
    let log_temperature_limits = (3.0, 7.0);
    let log_electron_density_limits = (8.0, 13.0);
    let verbose = Verbose::Yes;

    let interpolator = PolyFitInterpolator2::new(PolyFitInterpolatorConfig {
        order: 1,
        ..PolyFitInterpolatorConfig::default()
    });

    let emissivity_tables = EmissivityTables::<fdt>::new(
        &ion_lines,
        n_temperature_points,
        n_electron_density_points,
        log_temperature_limits,
        log_electron_density_limits,
        verbose,
    );

    let mut rng = rand::thread_rng();

    let ten: fdt = 10.0;
    let min_tg = ten.powf(log_temperature_limits.0);
    let max_tg = ten.powf(log_temperature_limits.1);
    let min_nel = ten.powf(log_electron_density_limits.0);
    let max_nel = ten.powf(log_electron_density_limits.1);

    const N_POINTS: usize = 768 * 768 * 768;
    let temperatures: Vec<fdt> = (0..N_POINTS)
        .map(|_| min_tg + rng.gen::<fdt>() * (max_tg - min_tg))
        .collect();
    let electron_densities: Vec<fdt> = (0..N_POINTS)
        .map(|_| min_nel + rng.gen::<fdt>() * (max_nel - min_nel))
        .collect();

    let result = emissivity_tables.evaluate(
        &interpolator,
        "si_4_1393.755",
        &temperatures,
        &electron_densities,
    );
    println!("{}", result[0]);

    provider
}
