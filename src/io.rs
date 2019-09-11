//! File input/output.

pub mod utils;
pub mod snapshot;

/// Little- or big-endian byte order.
#[derive(Clone, Copy, Debug)]
pub enum Endianness {
    Little,
    Big
}
