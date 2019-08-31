//! Utilities for input/output.

use std::{io, path, fs};
use std::io::Read;
use serde::Serialize;
use serde_pickle;

/// Reads and returns the content of the specified text file.
pub fn read_text_file(file_path: &path::Path) -> io::Result<String> {
    let file = fs::File::open(file_path)?;
    let mut text = String::new();
    let _ = io::BufReader::new(file).read_to_string(&mut text)?;
    Ok(text)
}

/// Serializes the given data into a protocol 3 pickle file saved at the given path.
pub fn save_data_as_pickle<T: Serialize>(data: &T, file_path: &path::Path) -> io::Result<()> {
    let mut file = fs::File::create(file_path)?;
    match serde_pickle::to_writer(&mut file, data, true) {
        Ok(_) => Ok(()),
        Err(serde_pickle::Error::Io(err)) => Err(err),
        Err(_) => Err(io::Error::new(io::ErrorKind::Other, "Unexpected error while serializing data to pickle file"))
    }
}