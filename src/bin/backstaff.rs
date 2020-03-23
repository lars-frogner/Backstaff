//! Command line runner for the `backstaff` library.

fn main() {
    #[cfg(feature = "cli")]
    backstaff::cli::run::run();
}
