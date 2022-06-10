use crate::common;
use lazy_static::lazy_static;
use std::{
    env,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct RegressionTest {
    expected_output_path: PathBuf,
    actual_output_path: PathBuf,
    output_path: String,
}

pub struct Actual<S>(pub S);
pub struct Expected<S>(pub S);

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
            common::assert_files_identical(self.actual_output_path(), self.expected_output_path());
        }
    }

    pub fn assert_mesh_files_equal(&self) {
        if let RegressionTestAction::Compare = CONTEXT.action {
            common::assert_mesh_files_equal(self.actual_output_path(), self.expected_output_path());
        }
    }

    fn expected_output_path(&self) -> &Path {
        &self.expected_output_path
    }

    fn actual_output_path(&self) -> &Path {
        &self.actual_output_path
    }
}

lazy_static! {
    static ref CONTEXT: RegressionTestContext = RegressionTestContext::new();
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
    const EXPECTED_DIR_PATH_COMPONENTS: [&'static str; 4] =
        ["tests", "data", "regression", "expected_output"];
    const ACTUAL_DIR_PATH_COMPONENTS: [&'static str; 4] =
        ["tests", "data", "regression", "actual_output"];

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
