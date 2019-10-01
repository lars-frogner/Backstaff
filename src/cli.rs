//! Command line interface.

pub mod ebeam;
pub mod snapshot;

use clap::{self, App, AppSettings, Arg};
use std::time::Instant;

/// Runs the `bifrost` command line program.
pub fn run() {
    let app = App::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("timing")
                .short("t")
                .long("timing")
                .help("Display elapsed time when done"),
        )
        .subcommand(snapshot::build_subcommand_snapshot())
        .subcommand(ebeam::build_subcommand_ebeam());

    let arguments = app.get_matches();

    let start_instant = Instant::now();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        snapshot::run_subcommand_snapshot(snapshot_arguments);
    }

    if let Some(ebeam_arguments) = arguments.subcommand_matches("ebeam") {
        ebeam::run_subcommand_ebeam(ebeam_arguments);
    }

    if arguments.is_present("timing") {
        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());
    }
}
