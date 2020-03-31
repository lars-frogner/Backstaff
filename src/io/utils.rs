//! Utilities for input/output.

use super::Endianness;
use crate::exit_with_error;
use byteorder::{self, ByteOrder, ReadBytesExt};
use serde::Serialize;
use serde_json;
use serde_pickle;
use std::{
    fs, io,
    io::{Read, Seek, SeekFrom, Write},
    mem,
    path::{Path, PathBuf},
};
use tempfile::{NamedTempFile, TempPath};

#[macro_export]
macro_rules! io_result {
    ($result:expr) => {
        $result.map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
    };
}

/// Path to a target output file, with an associated temporary path for writing data to.
/// The temporary file can take the place of the target output file in a single operation.
pub struct AtomicOutputPath {
    target_output_file_path: PathBuf,
    temp_output_file_path: TempPath,
}

impl AtomicOutputPath {
    /// Creates a new atomic output path for the given target file path.
    pub fn new<P: AsRef<Path>>(output_file_path: P) -> io::Result<Self> {
        let target_output_file_path = output_file_path.as_ref().to_path_buf();
        let output_dir = target_output_file_path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "No extension for output file")
        })?;
        let temp_output_file_path = io_result!(NamedTempFile::new_in(output_dir))?.into_temp_path();
        Ok(Self {
            target_output_file_path,
            temp_output_file_path,
        })
    }

    /// Returns the path to the target output file.
    pub fn target_path(&self) -> &Path {
        &self.target_output_file_path
    }

    /// Returns the path to the temporary output file.
    pub fn temporary_path(&self) -> &Path {
        &self.temp_output_file_path
    }

    /// Makes sure that the target output file can be overwritten, either automatically
    /// or with user's consent. If not, aborts the program.
    pub fn ensure_write_allowed(&self, automatic_overwrite: bool, protected_file_types: &[&str]) {
        ensure_write_allowed(
            self.target_path(),
            automatic_overwrite,
            protected_file_types,
        );
    }

    /// Moves the temporary file to the target output file path.
    pub fn perform_replace(self) -> io::Result<()> {
        let Self {
            target_output_file_path,
            temp_output_file_path,
        } = self;
        io_result!(temp_output_file_path.persist(target_output_file_path))
    }
}

/// Prompts the user with a question and returns whether the answer was yes.
///
/// The given default answer is assumed if the user simply presses `return`.
pub fn user_says_yes(question: &str, default_is_no: bool) -> bool {
    let full_question = format!(
        "{} [{}]",
        question,
        if default_is_no { "y/N" } else { "Y/n" }
    );
    let accepted_answers = ["y", "n"];
    let mut answer = String::with_capacity(2);
    let final_answer;
    println!("{}", &full_question);
    loop {
        let _ = io::stdout().flush();
        match io::stdin().read_line(&mut answer) {
            Ok(_) => {
                if let Some('\n') = answer.chars().next_back() {
                    answer.pop();
                }
                if let Some('\r') = answer.chars().next_back() {
                    answer.pop();
                }
                if answer.is_empty() {
                    final_answer = if default_is_no { "n" } else { "y" };
                    break;
                } else {
                    answer = answer.to_ascii_lowercase();
                    if accepted_answers.contains(&answer.as_str()) {
                        final_answer = answer.as_str();
                        break;
                    }
                }
            }
            Err(err) => println!("{}", err.to_string()),
        }
        answer.clear();
        println!("{}", &full_question);
    }
    final_answer == "y"
}

/// Check if the file at the given path exists, and if so, ask user whether it
/// can be overwritten.
pub fn write_allowed<P: AsRef<Path>>(file_path: P) -> bool {
    let file_path = file_path.as_ref();
    if file_path.exists() {
        user_says_yes(
            &format!(
                "File {} already exists, overwrite?",
                file_path.file_name().unwrap().to_string_lossy()
            ),
            true,
        )
    } else {
        true
    }
}

/// Makes sure that the file at the given path can be overwritten, either automatically
/// or with user's consent. If not, aborts the program.
pub fn ensure_write_allowed<P: AsRef<Path>>(
    file_path: P,
    automatic_overwrite: bool,
    protected_file_types: &[&str],
) {
    let file_path = file_path.as_ref();
    let is_protected = match file_path.extension() {
        Some(extension) => protected_file_types.contains(&extension.to_string_lossy().as_ref()),
        None => false,
    };
    if (!automatic_overwrite || is_protected) && !write_allowed(file_path) {
        exit_with_error!("Aborted");
    }
}

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
pub fn open_file_and_map_err<P: AsRef<Path>>(file_path: P) -> io::Result<fs::File> {
    let file_path = file_path.as_ref();
    fs::File::open(file_path).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("Could not open {}: {}", file_path.to_string_lossy(), err),
        )
    })
}

/// Reads and returns the content of the specified text file.
pub fn read_text_file<P: AsRef<Path>>(file_path: P) -> io::Result<String> {
    let file = open_file_and_map_err(file_path)?;
    let mut text = String::new();
    let _ = io::BufReader::new(file).read_to_string(&mut text)?;
    Ok(text)
}

/// Creates any directories missing in order for the given path to
/// be valid.
pub fn create_directory_if_missing<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let path = path.as_ref();
    if path.file_name().is_some() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
        } else {
            Ok(())
        }
    } else {
        fs::create_dir_all(path)
    }
}

/// Creates the file at the given path, as well as any missing parent
/// directories.
pub fn create_file_and_required_directories<P: AsRef<Path>>(file_path: P) -> io::Result<fs::File> {
    create_directory_if_missing(&file_path)?;
    fs::File::create(file_path)
}

/// Writes the given string as a text file with the specified path,
/// regardless of whether the file already exists.
pub fn write_text_file<P: AsRef<Path>>(text: &str, output_file_path: P) -> io::Result<()> {
    let mut file = create_file_and_required_directories(output_file_path)?;
    write!(&mut file, "{}", text)
}

/// Reads and returns a buffer of values from the specified binary file.
pub fn read_from_binary_file<P: AsRef<Path>, T: ByteorderData>(
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
pub fn save_data_as_binary<P>(output_file_path: P, byte_buffer: &[u8]) -> io::Result<()>
where
    P: AsRef<Path>,
{
    let mut file = create_file_and_required_directories(output_file_path)?;
    file.write_all(&byte_buffer)
}

/// Serializes the given data into JSON format and saves at the given path.
pub fn save_data_as_json<P, T>(output_file_path: P, data: &T) -> io::Result<()>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let mut file = create_file_and_required_directories(output_file_path)?;
    write_data_as_json(&mut file, data)
}

/// Serializes the given data into protocol 3 pickle format and saves at the given path.
pub fn save_data_as_pickle<P, T>(output_file_path: P, data: &T) -> io::Result<()>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let mut file = create_file_and_required_directories(output_file_path)?;
    write_data_as_pickle(&mut file, data)
}

/// Serializes the given data into JSON format and writes to the given writer.
pub fn write_data_as_json<T: Serialize, W: io::Write>(writer: &mut W, data: &T) -> io::Result<()> {
    serde_json::to_writer(writer, data).map_err(|err| err.into())
}

/// Serializes the given data into protocol 3 pickle format and writes to the given writer.
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
            Endianness::Native => byteorder::NativeEndian::write_f32_into(source, dest),
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
            Endianness::Native => file.read_f32_into::<byteorder::NativeEndian>(buffer),
            Endianness::Little => file.read_f32_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_f32_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for f64 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Native => byteorder::NativeEndian::write_f64_into(source, dest),
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
            Endianness::Native => file.read_f64_into::<byteorder::NativeEndian>(buffer),
            Endianness::Little => file.read_f64_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_f64_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for i32 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Native => byteorder::NativeEndian::write_i32_into(source, dest),
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
            Endianness::Native => file.read_i32_into::<byteorder::NativeEndian>(buffer),
            Endianness::Little => file.read_i32_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_i32_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for i64 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Native => byteorder::NativeEndian::write_i64_into(source, dest),
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
            Endianness::Native => file.read_i64_into::<byteorder::NativeEndian>(buffer),
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
            Endianness::Native => byteorder::NativeEndian::write_u32_into(source, dest),
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
            Endianness::Native => file.read_u32_into::<byteorder::NativeEndian>(buffer),
            Endianness::Little => file.read_u32_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_u32_into::<byteorder::BigEndian>(buffer),
        }
    }
}

impl ByteorderData for u64 {
    fn write_into_byte_buffer(source: &[Self], dest: &mut [u8], endianness: Endianness) {
        match endianness {
            Endianness::Native => byteorder::NativeEndian::write_u64_into(source, dest),
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
            Endianness::Native => file.read_u64_into::<byteorder::NativeEndian>(buffer),
            Endianness::Little => file.read_u64_into::<byteorder::LittleEndian>(buffer),
            Endianness::Big => file.read_u64_into::<byteorder::BigEndian>(buffer),
        }
    }
}
