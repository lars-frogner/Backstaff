use backstaff::{cli, exit_on_error, exit_with_error, io::utils as io_utils};
use clap::Command;
use lazy_static::lazy_static;
use std::{
    env,
    ffi::OsString,
    io::{self, Read},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct RegressionTest {
    expected_output_path: PathBuf,
    actual_output_path: PathBuf,
    output_path: String,
}

pub struct Actual<S>(S);
pub struct Expected<S>(S);

impl RegressionTest {
    pub fn for_output_file<S: AsRef<str>>(output_file_name: S) -> Self {
        let output_file_name = output_file_name.as_ref();
        Self::for_output_files(Actual(output_file_name), Expected(output_file_name))
    }

    pub fn for_output_files<S1, S2>(
        actual_output_file_name: Actual<S1>,
        expected_output_file_name: Expected<S2>,
    ) -> Self
    where
        S1: AsRef<str>,
        S2: AsRef<str>,
    {
        let expected_output_path = CONTEXT
            .expected_output_dir()
            .join(expected_output_file_name.0.as_ref());
        let actual_output_path = CONTEXT
            .actual_output_dir()
            .join(actual_output_file_name.0.as_ref());

        let output_path = match CONTEXT.action {
            RegressionTestAction::GenerateExpected => &expected_output_path,
            RegressionTestAction::Compare => &actual_output_path,
        }
        .to_string_lossy()
        .to_string();

        Self {
            expected_output_path,
            actual_output_path,
            output_path,
        }
    }

    pub fn output_path(&self) -> &str {
        &self.output_path
    }

    pub fn assert_files_identical(&self) {
        if let RegressionTestAction::Compare = CONTEXT.action {
            let identical = exit_on_error!(
                file_content_is_identical(self.actual_output_path(), self.expected_output_path()),
                "Error: Could not read files for comparison: {}"
            );
            assert!(identical);
        }
    }

    fn expected_output_path(&self) -> &Path {
        &self.expected_output_path
    }

    fn actual_output_path(&self) -> &Path {
        &self.actual_output_path
    }
}

pub fn run<I, T>(args: I)
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    cli::run::run_with_args(COMMAND.clone().get_matches_from(args));
}

lazy_static! {
    static ref CONTEXT: RegressionTestContext = RegressionTestContext::new();
    static ref COMMAND: Command<'static> = cli::build::build().no_binary_name(true);
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RegressionTestAction {
    GenerateExpected,
    Compare,
}

impl RegressionTestAction {
    /// If this environment variable is `1`, the output files generated
    /// by the tests are written into the `expected_output` folder and
    /// the regression comparisons are skipped. Otherwise, the files
    /// are written into the `actual_output` folder and are compared
    /// with the corresponding data in the `expected_output` folder.
    const ENV_VAR: &'static str = "BACKSTAFF_REGRESSION_GENERATE";

    fn from_env() -> Self {
        match env::var_os(Self::ENV_VAR) {
            Some(value) if (value.to_string_lossy() == "1") => Self::GenerateExpected,
            _ => Self::Compare,
        }
    }
}

#[derive(Debug, Clone)]
struct RegressionTestContext {
    action: RegressionTestAction,
    expected_output_dir: PathBuf,
    actual_output_dir: PathBuf,
}

impl RegressionTestContext {
    const EXPECTED_DIR_PATH_COMPONENTS: [&'static str; 3] = ["tests", "data", "expected_output"];
    const ACTUAL_DIR_PATH_COMPONENTS: [&'static str; 3] = ["tests", "data", "actual_output"];

    fn new() -> Self {
        let action = RegressionTestAction::from_env();

        let expected_output_dir: PathBuf = Self::EXPECTED_DIR_PATH_COMPONENTS.iter().collect();
        let actual_output_dir: PathBuf = Self::ACTUAL_DIR_PATH_COMPONENTS.iter().collect();

        Self {
            action,
            expected_output_dir,
            actual_output_dir,
        }
    }

    fn expected_output_dir(&self) -> &Path {
        &self.expected_output_dir
    }

    fn actual_output_dir(&self) -> &Path {
        &self.actual_output_dir
    }
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
