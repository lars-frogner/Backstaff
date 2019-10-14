//! Command line runner for the `bifrust` library.

fn main() {
    #[cfg(feature = "cli")]
    bifrust::cli::run();
}
