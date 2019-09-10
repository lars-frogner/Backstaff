//! Utilities for input/output.

use std::{io, path, fs, mem};
use std::io::{Read, Seek, SeekFrom};
use byteorder;
use byteorder::{ByteOrder, ReadBytesExt};
use serde::Serialize;
use serde_pickle;
use super::Endianness;

/// Reads and returns the content of the specified text file.
pub fn read_text_file<P: AsRef<path::Path>>(file_path: P) -> io::Result<String> {
    let file = fs::File::open(file_path)?;
    let mut text = String::new();
    let _ = io::BufReader::new(file).read_to_string(&mut text)?;
    Ok(text)
}

/// Reads and returns a buffer of f32 values from the specified binary file.
pub fn read_f32_from_binary_file<P: AsRef<path::Path>>(file_path: P, length: usize, offset: usize, endianness: Endianness) -> io::Result<Vec<f32>> {
    let mut file = fs::File::open(file_path)?;
    file.seek(SeekFrom::Start((offset*mem::size_of::<f32>()) as u64))?;
    let mut buffer = vec![0.0; length];
    match endianness {
        Endianness::Little => file.read_f32_into::<byteorder::LittleEndian>(&mut buffer)?,
        Endianness::Big => file.read_f32_into::<byteorder::BigEndian>(&mut buffer)?
    };
    Ok(buffer)
}

/// Reads and returns a buffer of f64 values from the specified binary file.
pub fn read_f64_from_binary_file<P: AsRef<path::Path>>(file_path: P, length: usize, offset: usize, endianness: Endianness) -> io::Result<Vec<f64>> {
    let mut file = fs::File::open(file_path)?;
    file.seek(SeekFrom::Start((offset*mem::size_of::<f64>()) as u64))?;
    let mut buffer = vec![0.0; length];
    match endianness {
        Endianness::Little => file.read_f64_into::<byteorder::LittleEndian>(&mut buffer)?,
        Endianness::Big => file.read_f64_into::<byteorder::BigEndian>(&mut buffer)?
    };
    Ok(buffer)
}

/// Writes the given source buffer of f32 values into the given byte buffer,
/// starting at the specified offset.
pub fn write_f32_into_byte_buffer(source: &[f32], dest: &mut [u8], float_offset: usize, endianness: Endianness) {
    let float_size = mem::size_of::<f32>();
    let byte_offset = float_offset*float_size;
    let number_of_bytes = source.len()*float_size;
    let dest_slice = &mut dest[byte_offset..byte_offset+number_of_bytes];
    match endianness {
        Endianness::Little => byteorder::LittleEndian::write_f32_into(source, dest_slice),
        Endianness::Big => byteorder::BigEndian::write_f32_into(source, dest_slice)
    };
}

/// Writes the given source buffer of f64 values into the given byte buffer,
/// starting at the specified offset.
pub fn write_f64_into_byte_buffer(source: &[f64], dest: &mut [u8], float_offset: usize, endianness: Endianness) {
    let float_size: usize = mem::size_of::<f64>();
    let byte_offset = float_offset*float_size;
    let number_of_bytes = source.len()*float_size;
    let dest_slice = &mut dest[byte_offset..byte_offset+number_of_bytes];
    match endianness {
        Endianness::Little => byteorder::LittleEndian::write_f64_into(source, dest_slice),
        Endianness::Big => byteorder::BigEndian::write_f64_into(source, dest_slice)
    };
}

/// Serializes the given data into protocol 3 pickle format and save at the given path.
pub fn save_data_as_pickle<P, T>(file_path: P, data: &T) -> io::Result<()>
where P: AsRef<path::Path>,
      T: Serialize
{
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