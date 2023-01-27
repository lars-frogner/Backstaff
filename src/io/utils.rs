//! Utilities for input/output.

use super::{snapshot::utils::SnapNumInRange, Endianness, OverwriteMode, Verbosity};
use byteorder::{self, ByteOrder, ReadBytesExt};
use std::{
    collections::HashMap,
    fs, io,
    io::{Read, Seek, SeekFrom, Write},
    mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tempfile::{Builder, TempPath};

#[cfg(feature = "serialization")]
use serde::Serialize;

#[cfg(feature = "json")]
use serde_json;

#[cfg(feature = "pickle")]
use serde_pickle;

#[macro_export]
macro_rules! io_result {
    ($result:expr) => {
        $result.map_err(|err| ::std::io::Error::new(::std::io::ErrorKind::Other, err.to_string()))
    };
}

#[macro_export]
macro_rules! with_io_err_msg {
    ($result:expr, $($print_arg:tt)+) => {
        $result.map_err(|err| {
            ::std::io::Error::new(
            err.kind(),
            format!($($print_arg)*, err),
            )
        })
    };
}

/// Holds state and information relevant for I/O.
#[derive(Debug)]
pub struct IOContext {
    atomic_output_file_map: Arc<Mutex<AtomicOutputFileMap>>,
    protected_file_types: Vec<String>,
    snap_num_in_range: Option<SnapNumInRange>,
    overwrite_mode: OverwriteMode,
}

impl IOContext {
    pub fn new() -> Self {
        Self {
            atomic_output_file_map: Arc::new(Mutex::new(AtomicOutputFileMap::new())),
            protected_file_types: Vec::new(),
            snap_num_in_range: None,
            overwrite_mode: OverwriteMode::Ask,
        }
    }

    /// Returns a reference counted pointer to the atomic output file map,
    /// to be used for cleaning temporary files on shutdown.
    pub fn obtain_atomic_file_map_handle(&self) -> Arc<Mutex<AtomicOutputFileMap>> {
        self.atomic_output_file_map.clone()
    }

    pub fn set_protected_file_types(&mut self, protected_file_types: Vec<String>) {
        self.protected_file_types = protected_file_types
    }

    pub fn set_snap_num_in_range(&mut self, snap_num_in_range: Option<SnapNumInRange>) {
        self.snap_num_in_range = snap_num_in_range
    }

    pub fn get_snap_num_in_range(&self) -> Option<&SnapNumInRange> {
        self.snap_num_in_range.as_ref()
    }

    pub fn set_overwrite_mode(&mut self, overwrite_mode: OverwriteMode) {
        self.overwrite_mode = overwrite_mode
    }

    pub fn get_overwrite_mode(&self) -> OverwriteMode {
        self.overwrite_mode
    }

    /// Whether the given file path is protected from automatic
    /// overwriting.
    pub fn file_path_is_protected(&self, file_path: &Path) -> bool {
        match file_path.extension() {
            Some(extension) => self
                .protected_file_types
                .contains(&extension.to_string_lossy().to_string()),
            None => false,
        }
    }

    /// Creates an "atomic" file that can be used for atomic
    /// writing to the given output path.
    pub fn create_atomic_output_file(
        &self,
        output_file_path: PathBuf,
    ) -> io::Result<AtomicOutputFile> {
        self.atomic_output_file_map
            .lock()
            .unwrap()
            .register_output_path(output_file_path)
    }

    /// Moves the data writted to the given atomic file to its
    /// associated target output path.
    pub fn close_atomic_output_file(&self, atomic_output_file: AtomicOutputFile) -> io::Result<()> {
        self.atomic_output_file_map
            .lock()
            .unwrap()
            .move_to_target(atomic_output_file)
    }
}

impl Default for IOContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IOContext {
    fn drop(&mut self) {
        self.atomic_output_file_map.lock().unwrap().clear();
    }
}

/// Manages temporary files that can be persisted by moving
/// to an associated target file path when all writing to
/// the file is completed.
#[derive(Debug)]
pub struct AtomicOutputFileMap {
    temporary_paths: HashMap<PathBuf, TempPath>,
}

impl AtomicOutputFileMap {
    /// Creates a new map for managing atomic output files.
    fn new() -> Self {
        Self {
            temporary_paths: HashMap::new(),
        }
    }

    /// Clears the map and deletes all temporary files.
    pub fn clear(&mut self) {
        self.temporary_paths.clear()
    }

    /// Registers the given target output file path and returns an associated
    /// atomic output file.
    fn register_output_path(
        &mut self,
        target_output_file_path: PathBuf,
    ) -> io::Result<AtomicOutputFile> {
        if self.path_is_registered(&target_output_file_path) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "Output path {} already has an associated temporary file",
                    target_output_file_path.to_string_lossy()
                ),
            ));
        }

        let temp_output_file_path = Self::create_temporary_output_file(&target_output_file_path)?;

        self.temporary_paths
            .insert(target_output_file_path.clone(), temp_output_file_path);

        let temp_output_file_path = self.temporary_paths[&target_output_file_path].to_path_buf();
        Ok(AtomicOutputFile::new(
            target_output_file_path,
            temp_output_file_path,
        ))
    }

    /// Moves the temporary file associated with the given atomic output file
    /// to its target output path.
    fn move_to_target(&mut self, atomic_output_file: AtomicOutputFile) -> io::Result<()> {
        let target_output_file_path = atomic_output_file.target_path();

        let temp_output_file_path = self
            .temporary_paths
            .remove(target_output_file_path)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "Output path {} has no an associated temporary file",
                        target_output_file_path.to_string_lossy()
                    ),
                )
            })?;

        if target_output_file_path.exists() {
            fs::remove_file(target_output_file_path)?;
        }
        io_result!(temp_output_file_path.persist(target_output_file_path))
    }

    fn path_is_registered(&self, target_output_file_path: &Path) -> bool {
        self.temporary_paths.contains_key(target_output_file_path)
    }

    fn create_temporary_output_file(target_output_file_path: &Path) -> io::Result<TempPath> {
        let output_dir = target_output_file_path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "No extension for output file")
        })?;

        create_directory_if_missing(output_dir)?;

        Ok(io_result!(Builder::new()
            .prefix(".backstaff_tmp")
            .suffix(&format!(
                "_{}",
                target_output_file_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
            ))
            .tempfile_in(output_dir))?
        .into_temp_path())
    }
}

/// Holds a target output path with an associated temporary path for writing data to.
/// When finished, the temporary file can be moved to the target output path in a
/// single operation.
#[derive(Debug)]
pub struct AtomicOutputFile {
    target_output_file_path: PathBuf,
    temp_output_file_path: PathBuf,
}

impl AtomicOutputFile {
    /// Creates a new atomic output path for the given target file path.
    fn new(output_file_path: PathBuf, temp_output_file_path: PathBuf) -> Self {
        let target_output_file_path = output_file_path;
        Self {
            target_output_file_path,
            temp_output_file_path,
        }
    }

    /// Returns the path to the target output file.
    pub fn target_path(&self) -> &Path {
        &self.target_output_file_path
    }

    /// Returns the path to the temporary output file.
    pub fn temporary_path(&self) -> &Path {
        &self.temp_output_file_path
    }

    /// Checks whether the target path is available to write to (by moving the temporary file),
    /// either automatically or with user's consent.
    pub fn check_if_write_allowed(&self, io_context: &IOContext, verbosity: &Verbosity) -> bool {
        check_if_write_allowed(self.target_path(), io_context, verbosity)
    }
}

/// Prompts the user with a question and returns whether the answer was yes.
///
/// The given default answer is assumed if the user simply presses `return`.
pub fn user_says_yes(question: &str, default_is_no: bool) -> io::Result<bool> {
    if atty::isnt(atty::Stream::Stdin) {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Can not prompt user without a terminal",
        ));
    }
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
            Err(err) => println!("{}", err),
        }
        answer.clear();
        println!("{}", &full_question);
    }
    Ok(final_answer == "y")
}

/// Checks whether the current path can be written to, either automatically
/// or with user's consent.
pub fn check_if_write_allowed<P: AsRef<Path>>(
    file_path: P,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> bool {
    let file_path = file_path.as_ref();
    let is_protected = io_context.file_path_is_protected(file_path);
    let file_path_string = file_path.to_string_lossy();
    let file_name = file_path.file_name().unwrap().to_string_lossy();
    if file_path.exists() {
        let can_overwrite = match io_context.get_overwrite_mode() {
            OverwriteMode::Ask | OverwriteMode::Always if is_protected => user_says_yes(
                &format!(
                    "File {} already exists and is a protected file type, overwrite?",
                    file_path_string
                ),
                true,
            )
            .unwrap_or_else(|err| {
                eprintln!(
                    "Warning: Not overwriting {} due to error: {}",
                    file_path_string, err
                );
                false
            }),
            OverwriteMode::Ask => user_says_yes(
                &format!("File {} already exists, overwrite?", file_path_string),
                true,
            )
            .unwrap_or_else(|err| {
                eprintln!(
                    "Warning: Not overwriting {} due to error: {}",
                    file_path_string, err
                );
                false
            }),
            OverwriteMode::Always => true,
            OverwriteMode::Never => false,
        };
        if !can_overwrite
            && (verbosity.print_messages()
                || io_context.get_overwrite_mode() != OverwriteMode::Never)
        {
            println!("Skipping {}", file_name);
        }
        can_overwrite
    } else {
        true
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
    if path.extension().is_some() {
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
    file.write_all(byte_buffer)
}

/// Serializes the given data into JSON format and saves at the given path.
#[cfg(feature = "json")]
pub fn save_data_as_json<P, T>(output_file_path: P, data: &T) -> io::Result<()>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let mut file = create_file_and_required_directories(output_file_path)?;
    write_data_as_json(&mut file, data)
}

/// Serializes the given data into protocol 3 pickle format and saves at the given path.
#[cfg(feature = "pickle")]
pub fn save_data_as_pickle<P, T>(output_file_path: P, data: &T) -> io::Result<()>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let mut file = create_file_and_required_directories(output_file_path)?;
    write_data_as_pickle(&mut file, data)
}

/// Serializes the given data into JSON format and writes to the given writer.
#[cfg(feature = "json")]
pub fn write_data_as_json<T: Serialize, W: io::Write>(writer: &mut W, data: &T) -> io::Result<()> {
    serde_json::to_writer(writer, data).map_err(|err| err.into())
}

/// Serializes the given data into protocol 3 pickle format and writes to the given writer.
#[cfg(feature = "pickle")]
pub fn write_data_as_pickle<T: Serialize, W: io::Write>(
    writer: &mut W,
    data: &T,
) -> io::Result<()> {
    match serde_pickle::to_writer(writer, data, serde_pickle::SerOptions::new()) {
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
