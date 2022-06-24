#![allow(dead_code)]

use backstaff::{
    exit_on_error,
    field::{CustomScalarFieldGenerator3, FieldGrid3, FieldValueComputer},
    grid::fgr,
    io::{
        snapshot::{
            fdt, native,
            utils::{self as snapshot_utils, OutputSnapshotMetadata},
        },
        utils::{self as io_utils, IOContext},
        Endianness, Verbosity,
    },
};
use lazy_static::lazy_static;
use std::{
    ffi::OsString,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "cli")]
use backstaff::cli;

#[macro_export]
macro_rules! def_test {
    (
        IN[$($in_ident:ident = $in_str:expr),*]
        OUT[$($out_ident:ident = $out_str:expr),*]
        fn $name:ident() $test_body:expr
    ) => {
        #[test]
        fn $name() {
            let test = common::Test::new(stringify!($name));

            $( let $in_ident = test.input_path(::std::path::PathBuf::from($in_str)); )*
            $( let $out_ident = test.output_path(::std::path::PathBuf::from($out_str)); )*

            let test_body = |$( $in_ident, )* $( $out_ident, )*| $test_body;

            test_body(
                $( path_str!($in_ident), )* $( path_str!($out_ident), )*
            );
        }
    };
}

#[macro_export]
macro_rules! def_bench {
    (
        IN[$($in_ident:ident = $in_str:expr),*]
        OUT[$($out_ident:ident = $out_str:expr),*]
        fn $name:ident ($bencher:ident: &mut Bencher) $bench_body:expr
    ) => {
        #[bench]
        fn $name($bencher: &mut test::Bencher) {
            let test = common::Test::new(stringify!($name));

            $( let $in_ident = test.input_path(::std::path::PathBuf::from($in_str)); )*
            $( let $out_ident = test.output_path(::std::path::PathBuf::from($out_str)); )*

            let bench_body = | $( $in_ident, )* $( $out_ident, )* $bencher: &mut Bencher| $bench_body;

            bench_body(
                $( path_str!($in_ident), )* $( path_str!($out_ident), )* $bencher
            );
        }
    };
}

#[macro_export]
macro_rules! path_str {
    ($path:expr) => {
        $path.to_string_lossy().as_ref()
    };
}

#[cfg(feature = "cli")]
pub fn run<I, T>(args: I)
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    cli::run::run_with_args(COMMAND.clone().get_matches_from(args), IOContext::new());
}

pub fn assert_file_exists<P: AsRef<Path>>(file_path: P) {
    let file_path = file_path.as_ref();
    let exists = file_path.exists();
    assert!(
        exists,
        "File {} does not exist",
        file_path.to_string_lossy()
    );
}

pub fn assert_files_identical<P1, P2>(file_path_1: P1, file_path_2: P2)
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let file_path_1 = file_path_1.as_ref();
    let file_path_2 = file_path_2.as_ref();

    let identical = exit_on_error!(
        file_content_is_identical(file_path_1, file_path_2),
        "Error: Could not read files for comparison: {}"
    );
    assert!(
        identical,
        "Files {} and {} not identical",
        file_path_1.to_string_lossy(),
        file_path_2.to_string_lossy()
    );
}

#[cfg(feature = "for-testing")]
pub fn assert_mesh_files_equal<P1, P2>(mesh_path_1: P1, mesh_path_2: P2, max_relative_diff: fgr)
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let mesh_path_1 = mesh_path_1.as_ref();
    let mesh_path_2 = mesh_path_2.as_ref();

    let equal = exit_on_error!(
        native::parsed_mesh_files_eq(
            mesh_path_1,
            mesh_path_2,
            &Verbosity::Quiet,
            fgr::EPSILON,
            max_relative_diff
        ),
        "Error: Could not parse mesh files for comparison: {}"
    );
    assert!(
        equal,
        "Grids of mesh files {} and {} not equal",
        mesh_path_1.to_string_lossy(),
        mesh_path_2.to_string_lossy()
    );
}

#[cfg(feature = "for-testing")]
pub fn assert_snapshot_files_equal<P1, P2>(file_path_1: P1, file_path_2: P2, max_relative_diff: fdt)
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let file_path_1 = file_path_1.as_ref();
    let file_path_2 = file_path_2.as_ref();

    let equal = exit_on_error!(
        snapshot_utils::read_snapshots_eq(
            file_path_1.to_path_buf(),
            file_path_2.to_path_buf(),
            ENDIANNESS,
            Verbosity::Quiet,
            fdt::EPSILON,
            max_relative_diff,
        ),
        "Error: Could not read snapshot files for comparison: {}"
    );
    assert!(
        equal,
        "Data for snapshot files {} and {} not equal",
        file_path_1.to_string_lossy(),
        file_path_2.to_string_lossy()
    );
}

#[cfg(feature = "for-testing")]
pub fn assert_snapshot_field_values_equal<P1, P2>(
    file_path_1: P1,
    file_path_2: P2,
    max_relative_diff: fdt,
) where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let file_path_1 = file_path_1.as_ref();
    let file_path_2 = file_path_2.as_ref();

    let equal = exit_on_error!(
        snapshot_utils::read_snapshot_values_eq(
            file_path_1.to_path_buf(),
            file_path_2.to_path_buf(),
            ENDIANNESS,
            Verbosity::Quiet,
            fdt::EPSILON,
            max_relative_diff,
        ),
        "Error: Could not read snapshot files for comparison: {}"
    );
    assert!(
        equal,
        "Values for snapshot files {} and {} not equal",
        file_path_1.to_string_lossy(),
        file_path_2.to_string_lossy()
    );
}

#[cfg(feature = "for-testing")]
pub fn assert_snapshot_field_values_custom_equal<P1, P2>(
    input_file_path: P1,
    reference_file_path: P2,
    are_equal: &dyn Fn(&[fdt], &[fdt]) -> bool,
) where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let input_file_path = input_file_path.as_ref();
    let reference_file_path = reference_file_path.as_ref();

    let (mut reference_reader, _) = exit_on_error!(
        snapshot_utils::new_snapshot_reader(
            reference_file_path.to_path_buf(),
            ENDIANNESS,
            Verbosity::Quiet
        ),
        "Error: Could not read reference snapshot file for custom comparison: {}"
    );
    let all_variable_names = reference_reader.all_variable_names().to_vec();
    let owned_reference_field_values: Vec<_> = all_variable_names
        .iter()
        .map(|name| {
            exit_on_error!(
                reference_reader.produce_scalar_field(name.as_str()),
                "Error: Could not read field {} in reference snapshot: {}",
                name
            )
        })
        .collect();
    let reference_field_values = all_variable_names
        .into_iter()
        .zip(owned_reference_field_values.iter())
        .map(|(name, values)| (name, values.values().as_slice_memory_order().unwrap()))
        .collect();

    let equal = exit_on_error!(
        snapshot_utils::read_snapshot_has_given_field_values_custom_eq(
            input_file_path.to_path_buf(),
            ENDIANNESS,
            Verbosity::Quiet,
            reference_field_values,
            are_equal,
        ),
        "Error: Could not read snapshot file for custom comparison: {}"
    );
    assert!(
        equal,
        "Values for snapshot file {} not custom equal to reference values in {}",
        input_file_path.to_string_lossy(),
        reference_file_path.to_string_lossy()
    );
}

pub fn write_snapshot_with_computed_variable<P: AsRef<Path>>(
    grid: Arc<FieldGrid3>,
    variable_name: &str,
    variable_computer: FieldValueComputer<fdt>,
    output_path: P,
) {
    let metadata = Box::new(OutputSnapshotMetadata::new());

    let mut generator = Box::new(
        CustomScalarFieldGenerator3::new(grid, Verbosity::Quiet)
            .with_variable(variable_name.to_string(), variable_computer),
    );

    exit_on_error!(
        native::write_new_snapshot(
            &*metadata,
            &mut *generator,
            output_path.as_ref(),
            &IOContext::new(),
            &Verbosity::Quiet,
        ),
        "Error: {}"
    );
}

#[derive(Debug, Clone)]
pub struct Test {
    output_dir: PathBuf,
}

impl Test {
    pub fn new<S: AsRef<str>>(name: S) -> Self {
        let name = name.as_ref();
        let output_dir = exit_on_error!(
            CONTEXT.prepared_output_dir(name),
            "Error: Could not prepare output directory for test {}: {}",
            name
        );
        Self { output_dir }
    }

    pub fn input_path<P: AsRef<Path>>(&self, file_path: P) -> PathBuf {
        CONTEXT.input_path(file_path)
    }

    pub fn output_path<P: AsRef<Path>>(&self, file_path: P) -> PathBuf {
        self.output_dir().join(file_path)
    }

    fn output_dir(&self) -> &Path {
        self.output_dir.as_path()
    }
}

#[derive(Debug, Clone)]
pub struct TestContext {
    base_input_dir: PathBuf,
    base_output_dir: PathBuf,
}

impl TestContext {
    const BASE_INPUT_DIR_PATH_COMPONENTS: [&'static str; 3] = ["tests", "data", "input"];
    const BASE_OUTPUT_DIR_PATH_COMPONENTS: [&'static str; 3] = ["tests", "data", "output"];

    fn new() -> Self {
        let base_input_dir: PathBuf = Self::BASE_INPUT_DIR_PATH_COMPONENTS.iter().collect();
        let base_output_dir: PathBuf = Self::BASE_OUTPUT_DIR_PATH_COMPONENTS.iter().collect();
        Self {
            base_input_dir,
            base_output_dir,
        }
    }

    pub fn input_path<P: AsRef<Path>>(&self, file_path: P) -> PathBuf {
        self.base_input_dir().join(file_path)
    }

    pub fn output_dir<S: AsRef<str>>(&self, test_name: S) -> PathBuf {
        self.base_output_dir().join(test_name.as_ref())
    }

    pub fn output_path<S, P>(&self, test_name: S, file_path: P) -> PathBuf
    where
        S: AsRef<str>,
        P: AsRef<Path>,
    {
        self.output_dir(test_name).join(file_path)
    }

    pub fn prepared_output_dir<S: AsRef<str>>(&self, test_name: S) -> io::Result<PathBuf> {
        let output_dir = self.output_dir(test_name);
        Self::clear_output_dir(&output_dir)
            .and_then(|_| Self::create_output_dir(&output_dir).map(|_| output_dir))
    }

    fn clear_output_dir<P: AsRef<Path>>(output_dir: P) -> io::Result<()> {
        let output_dir = output_dir.as_ref();
        if output_dir.exists() {
            fs::remove_dir_all(output_dir)
        } else {
            Ok(())
        }
    }

    fn create_output_dir<P: AsRef<Path>>(output_dir: P) -> io::Result<()> {
        fs::create_dir_all(output_dir)
    }

    fn base_input_dir(&self) -> &Path {
        self.base_input_dir.as_path()
    }

    fn base_output_dir(&self) -> &Path {
        self.base_output_dir.as_path()
    }
}

lazy_static! {
    pub static ref CONTEXT: TestContext = TestContext::new();
}
#[cfg(feature = "cli")]
lazy_static! {
    static ref COMMAND: clap::Command<'static> = cli::build::build().no_binary_name(true);
}

pub const ENDIANNESS: Endianness = Endianness::Little;

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
