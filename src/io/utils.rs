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

/// Serializes the given data into protocol 3 pickle format and save at the given path.
pub fn save_data_as_pickle<T: Serialize>(file_path: &path::Path, data: &T) -> io::Result<()> {
    let mut file = fs::File::create(file_path)?;
    write_data_as_pickle_to_file(&mut file, data)
}

/// Serializes the given data into protocol 3 pickle format and write to the given file.
pub fn write_data_as_pickle_to_file<T: Serialize>(file: &mut fs::File, data: &T) -> io::Result<()> {
    match serde_pickle::to_writer(file, data, true) {
        Ok(_) => Ok(()),
        Err(serde_pickle::Error::Io(err)) => Err(err),
        Err(_) => Err(io::Error::new(io::ErrorKind::Other, "Unexpected error while serializing data to pickle file"))
    }
}