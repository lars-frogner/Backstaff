//! Command line interface for the `bifrost` crate.

use bifrost::ebeam::execution::cli as ebeam;
use clap::{self, App};

fn main() {
    let app = App::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .subcommand(ebeam::build_subcommand());

    let arguments = app.get_matches();

    if let Some(ebeam_arguments) = arguments.subcommand_matches("ebeam") {
        ebeam::run(ebeam_arguments);
    }
}
