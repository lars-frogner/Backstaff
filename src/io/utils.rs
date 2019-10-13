//! Utilities for input/output.

use super::Endianness;
use byteorder;
use byteorder::{ByteOrder, ReadBytesExt};
use serde::Serialize;
use serde_json;
use serde_pickle;
use std::io::{Read, Seek, SeekFrom, Write};
use std::{fs, io, mem, path};

/// Describes the properties of a type that can be translated to and from bytes
/// by the `byteorder` crate.
pub trait ByteorderData
where
    Self: std::marker::Sized + Copy + Default,
{
    /// Writes the given slice of data elements as bytes with the given endianness
    /// into the given byte buffer.
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness);

    /// Reads bytes with the given endianness from the given file and stores
    /// in the given data buffer.
    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()>;
}

/// Opens file with the given path and returns it, or returns an error with the
/// file path included in the error message.
pub fn open_file_and_map_err<P: AsRef<path::Path>>(file_path: P) -> io::Result<fs::File> {
    let file_path = file_path.as_ref();
    fs::File::open(file_path).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("Could not open {}: {}", file_path.to_string_lossy(), err),
        )
    })
}

/// Reads and returns the content of the specified text file.
pub fn read_text_file<P: AsRef<path::Path>>(file_path: P) -> io::Result<String> {
    let file = open_file_and_map_err(file_path)?;
    let mut text = String::new();
    let _ = io::BufReader::new(file).read_to_string(&mut text)?;
    Ok(text)
}

/// Writes the given string as a text file with the specified path.
pub fn write_text_file<P: AsRef<path::Path>>(text: &str, file_path: P) -> io::Result<()> {
    let mut file = fs::File::create(file_path)?;
    write!(&mut file, "{}", text)
}

/// Reads and returns a buffer of values from the specified binary file.
pub fn read_from_binary_file<P: AsRef<path::Path>, T: ByteorderData>(
    file_path: P,
    number_of_values: usize,
    byte_offset: usize,
    endianness: Endianness,
) -> io::Result<Vec<T>> {
    let mut file = open_file_and_map_err(file_path)?;
    file.seek(SeekFrom::Start(byte_offset as u64))?;
    let mut buffer = vec![T::default(); number_of_values];
    T::read_from_binary_file(&mut file, &mut buffer, endianness)?;
    Ok(buffer)
}

/// Writes the given source buffer of values into the given byte buffer,
/// starting at the specified offset. Returns the number of bytes written.
pub fn write_into_byte_buffer<T: ByteorderData>(
    source: &[T],
    dest: &mut [u8],
    byte_offset: usize,
    endianness: Endianness,
) -> usize {
    let type_size = mem::size_of::<T>();
    let number_of_bytes = source.len() * type_size;
    let dest_slice = &mut dest[byte_offset..byte_offset + number_of_bytes];
    T::write_into_byte_buffer(source, dest_slice, endianness);
    number_of_bytes
}

/// Saves the given byte buffer directly as a binary file at the given path.
pub fn save_data_as_binary<P>(file_path: P, byte_buffer: &[u8]) -> io::Result<()>
where
    P: AsRef<path::Path>,
{
    let mut file = fs::File::create(file_path)?;
    file.write_all(&byte_buffer)
}

/// Serializes the given data into JSON format and saves at the given path.
pub fn save_data_as_json<P, T>(file_path: P, data: &T) -> io::Result<()>
where
    P: AsRef<path::Path>,
    T: Serialize,
{
    let mut file = fs::File::create(file_path)?;
    write_data_as_json(&mut file, data)
}

/// Serializes the given data into protocol 3 pickle format and saves at the given path.
pub fn save_data_as_pickle<P, T>(file_path: P, data: &T) -> io::Result<()>
where
    P: AsRef<path::Path>,
    T: Serialize,
{
    let mut file = fs::File::create(file_path)?;
    write_data_as_pickle(&mut file, data)
}

/// Serializes the given data into JSON format and writes to the given file.
pub fn write_data_as_json<T: Serialize, W: io::Write>(writer: &mut W, data: &T) -> io::Result<()> {
    serde_json::to_writer(writer, data).map_err(|err| err.into())
}

/// Serializes the given data into protocol 3 pickle format and writes to the given file.
pub fn write_data_as_pickle<T: Serialize, W: io::Write>(
    writer: &mut W,
    data: &T,
) -> io::Result<()> {
    match serde_pickle::to_writer(writer, data, true) {
        Ok(_) => Ok(()),
        Err(serde_pickle::Error::Io(err)) => Err(err),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::Other,
            "Unexpected error while serializing data to pickle format",
        )),
    }
}

impl ByteorderData for f32 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Little => byteorder::LittleEndian::write_f32_into(source, dest),
            Endianness::Big => byteorder::BigEndian::write_f32_into(source, dest),
        };
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()> {
        match endianness {
            Endianness::Little => file.read_f32_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_f32_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for f64 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Little => byteorder::LittleEndian::write_f64_into(source, dest),
            Endianness::Big => byteorder::BigEndian::write_f64_into(source, dest),
        };
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()> {
        match endianness {
            Endianness::Little => file.read_f64_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_f64_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for i32 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Little => byteorder::LittleEndian::write_i32_into(source, dest),
            Endianness::Big => byteorder::BigEndian::write_i32_into(source, dest),
        };
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()> {
        match endianness {
            Endianness::Little => file.read_i32_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_i32_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for i64 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Little => byteorder::LittleEndian::write_i64_into(source, dest),
            Endianness::Big => byteorder::BigEndian::write_i64_into(source, dest),
        };
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()> {
        match endianness {
            Endianness::Little => file.read_i64_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_i64_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for u8 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], _endianness: Endianness) {
        dest.copy_from_slice(source);
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        _endianness: Endianness,
    ) -> io::Result<()> {
        file.read_to_end(buffer).map(|_| ())
    }
}

impl ByteorderData for u32 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Little => byteorder::LittleEndian::write_u32_into(source, dest),
            Endianness::Big => byteorder::BigEndian::write_u32_into(source, dest),
        };
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()> {
        match endianness {
            Endianness::Little => file.read_u32_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_u32_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for u64 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Little => byteorder::LittleEndian::write_u64_into(source, dest),
            Endianness::Big => byteorder::BigEndian::write_u64_into(source, dest),
        };
    }

    fn read_from_binary_file(
        file: &mut fs::File,
        buffer: &mut Vec<Self>,
        endianness: Endianness,
    ) -> io::Result<()> {
        match endianness {
            Endianness::Little => file.read_u64_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_u64_into::<byteorder::BigEndian>(buffer),
        }
    }
}
