//! Command line runner for the `bifrost` library.

fn main() {
    #[cfg(feature = "cli")]
    bifrost::cli::run();
}
