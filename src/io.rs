//! File input/output.

pub mod utils;
pub mod snapshot;

/// Little- or big-endian byte order.
#[derive(Clone, Copy, Debug)]
pub enum Endianness {
    Little,
    Big
}

/// Whether or not to print non-critical status messages.
#[derive(Clone, Copy, Debug)]
pub enum Verbose {
    Yes,
    No
}

impl Verbose {
    pub fn is_yes(&self) -> bool {
        match self {
            Verbose::Yes => true,
            Verbose::No => false
        }
    }
}
