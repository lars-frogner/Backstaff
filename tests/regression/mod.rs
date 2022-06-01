use backstaff::{cli, exit_on_error, exit_with_error, io::utils as io_utils};
use clap::Command;
use lazy_static::lazy_static;
use std::{
    env,
    ffi::OsString,
    io::{self, Read},
    path::{Path, PathBuf},
};

const GENERATE_ENV_VAR: &str = "BACKSTAFF_TEST_GENERATE";

lazy_static! {
    static ref EXPECTED_OUTPUT_DIR: PathBuf = ["tests", "data", "expected_output"].iter().collect();
    static ref ACTUAL_OUTPUT_DIR: PathBuf = ["tests", "data", "actual_output"].iter().collect();
    static ref GENERATE_EXPECTED: bool = shoud_generate_expected();
    static ref OUTPUT_DIR: PathBuf = determine_output_dir();
    static ref COMMAND: Command<'static> = cli::build::build().no_binary_name(true);
}

pub fn in_output_dir<S: AsRef<str>>(file_name: S) -> PathBuf {
    OUTPUT_DIR.join(file_name.as_ref())
}

pub fn run<I, T>(args: I)
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    cli::run::run_with_args(COMMAND.clone().get_matches_from(args));
}

pub fn assert_file_identical<P: AsRef<Path>>(output_file: P) {
    if !*GENERATE_EXPECTED {
        let actual_output_file = output_file.as_ref();
        let expected_output_file = expected_output_file_from_actual(actual_output_file);
        let identical = exit_on_error!(
            file_content_is_identical(actual_output_file, expected_output_file),
            "Error: Could not read files for comparison: {}"
        );
        assert!(identical);
    }
}

fn shoud_generate_expected() -> bool {
    matches!(env::var_os(GENERATE_ENV_VAR), Some(value) if (value.to_string_lossy() == "1"))
}

fn determine_output_dir() -> PathBuf {
    if *GENERATE_EXPECTED {
        EXPECTED_OUTPUT_DIR.clone()
    } else {
        ACTUAL_OUTPUT_DIR.clone()
    }
}

fn in_expected_output_dir<P: AsRef<Path>>(file_name: P) -> PathBuf {
    EXPECTED_OUTPUT_DIR.join(file_name.as_ref())
}

fn in_actual_output_dir<P: AsRef<Path>>(file_name: P) -> PathBuf {
    ACTUAL_OUTPUT_DIR.join(file_name.as_ref())
}

fn expected_output_file_from_actual<P: AsRef<Path>>(actual_output_file: P) -> PathBuf {
    let file_name = actual_output_file
        .as_ref()
        .file_name()
        .expect("Missing file name");
    in_expected_output_dir(file_name)
}

fn file_content_is_identical<P1, P2>(file_path_1: P1, file_path_2: P2) -> io::Result<bool>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let file_1 = io_utils::open_file_and_map_err(file_path_1)?;
    let file_2 = io_utils::open_file_and_map_err(file_path_2)?;

    let mut reader_1 = io::BufReader::new(file_1);
    let mut reader_2 = io::BufReader::new(file_2);

    const BUFFER_SIZE: usize = 10000; // Bytes
    let mut buffer_1 = [0; BUFFER_SIZE];
    let mut buffer_2 = [0; BUFFER_SIZE];

    loop {
        let n_bytes_read_1 = reader_1.read(&mut buffer_1)?;
        let n_bytes_read_2 = reader_2.read(&mut buffer_2)?;

        if n_bytes_read_1 == 0 && n_bytes_read_2 == 0 {
            break Ok(true);
        } else if n_bytes_read_2 != n_bytes_read_1 || buffer_1 != buffer_2 {
            break Ok(false);
        }
    }
}
