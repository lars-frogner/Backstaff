//! Command line runner for the `backstaff` library.

#[cfg(not(feature = "for-testing"))]
#[quit::main]
fn main() {
    #[cfg(feature = "cli")]
    backstaff::cli::run::run();
}

#[cfg(feature = "for-testing")]
fn main() {
    #[cfg(feature = "cli")]
    {
        eprintln!(
            "Warning: The `for-testing` feature is enabled, which will clutter error messages\n\
             Tip: Use cargo flag --features=all-non-testing to include all features except `for-testing`"
        );
        backstaff::cli::run::run();
    }
}
