//! File input/output.

pub mod snapshot;
pub mod utils;

use indicatif::ProgressStyle;

/// Little- or big-endian byte order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Endianness {
    Native,
    Little,
    Big,
}

/// Whether or not to print non-critical status messages
/// and showing progress.
#[derive(Clone, Debug)]
pub enum Verbose {
    Yes,
    No,
    Progress(ProgressStyle),
}

impl Verbose {
    /// Creates a new verbosity indicator for showing progress with
    /// the given progress bar style.
    pub fn with_progress(style: ProgressStyle) -> Self {
        Self::Progress(style)
    }

    /// Whether verbosity is activated.
    pub fn is_yes(&self) -> bool {
        match *self {
            Verbose::Yes => true,
            Verbose::No => false,
            Verbose::Progress(_) => true,
        }
    }
}

impl From<bool> for Verbose {
    fn from(is_verbose: bool) -> Self {
        if is_verbose {
            Verbose::Yes
        } else {
            Verbose::No
        }
    }
}

impl From<Verbose> for bool {
    fn from(verbose: Verbose) -> Self {
        verbose.is_yes()
    }
}

/// How to handle existing files.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverwriteMode {
    Ask,
    Always,
    Never,
}
