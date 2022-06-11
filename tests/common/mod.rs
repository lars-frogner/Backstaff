use backstaff::{
    cli, exit_on_error,
    grid::fgr,
    io::{
        snapshot::{fdt, native, utils as snapshot_utils},
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
};

#[macro_export]
macro_rules! def_test {
    (
        IN[$($in_ident:ident = $in_str:expr),*]
        OUT[$($out_ident:ident = $out_str:expr),*]
        fn $name:ident $test_body:expr
    ) => {
        #[test]
        fn $name() {
            let test = common::Test::new(stringify!($name));

            $( let $in_ident = test.input_path($in_str); )*
            $( let $out_ident = test.output_path($out_str); )*

            let test_body = |$( $in_ident, )* $( $out_ident, )*| $test_body;

            test_body(
                $( path_str!($in_ident), )* $( path_str!($out_ident), )*
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

pub fn assert_mesh_files_equal<P1, P2>(mesh_path_1: P1, mesh_path_2: P2)
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
            <fgr as approx::RelativeEq>::default_max_relative()
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

pub fn assert_snapshot_files_equal<P1, P2>(file_path_1: P1, file_path_2: P2)
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
            <fdt as approx::RelativeEq>::default_max_relative(),
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

    pub fn input_path<S: AsRef<str>>(&self, file_name: S) -> PathBuf {
        CONTEXT.input_path(file_name)
    }

    pub fn output_path<S: AsRef<str>>(&self, file_name: S) -> PathBuf {
        self.output_dir().join(file_name.as_ref())
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

    pub fn input_path<S: AsRef<str>>(&self, file_name: S) -> PathBuf {
        self.base_input_dir().join(file_name.as_ref())
    }

    pub fn output_dir<S: AsRef<str>>(&self, test_name: S) -> PathBuf {
        self.base_output_dir().join(test_name.as_ref())
    }

    pub fn output_path<S1, S2>(&self, test_name: S1, file_name: S2) -> PathBuf
    where
        S1: AsRef<str>,
        S2: AsRef<str>,
    {
        self.output_dir(test_name).join(file_name.as_ref())
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
    static ref COMMAND: clap::Command<'static> = cli::build::build().no_binary_name(true);
}

const ENDIANNESS: Endianness = Endianness::Little;

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
