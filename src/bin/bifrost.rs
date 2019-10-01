//! Command line runner for the `bifrost` library.

use bifrost::cli::{ebeam, snapshot};
use clap::{self, App, AppSettings};

fn main() {
    let app = App::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(snapshot::build_subcommand_snapshot())
        .subcommand(ebeam::build_subcommand_ebeam());

    let arguments = app.get_matches();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        snapshot::run_subcommand_snapshot(snapshot_arguments);
    }

    if let Some(ebeam_arguments) = arguments.subcommand_matches("ebeam") {
        ebeam::run_subcommand_ebeam(ebeam_arguments);
    }
}
