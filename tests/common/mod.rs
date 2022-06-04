use backstaff::{
    cli, exit_on_error, exit_with_error,
    grid::fgr,
    io::{snapshot::native, utils as io_utils, Verbose},
};
use lazy_static::lazy_static;
use std::{
    ffi::OsString,
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
            const NAME: &str = stringify!($name);

            $( let $in_ident = common::CONTEXT.input_path($in_str); )*
            $( let $out_ident = common::CONTEXT.output_path(&format!("{}_{}", NAME, $out_str)); )*

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
    cli::run::run_with_args(COMMAND.clone().get_matches_from(args));
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

pub fn assert_mesh_files_similar<P1, P2>(mesh_path_1: P1, mesh_path_2: P2)
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let mesh_path_1 = mesh_path_1.as_ref();
    let mesh_path_2 = mesh_path_2.as_ref();

    let similar = exit_on_error!(
        native::parsed_mesh_files_eq(
            mesh_path_1,
            mesh_path_2,
            Verbose::No,
            fgr::EPSILON,
            <fgr as approx::RelativeEq>::default_max_relative()
        ),
        "Error: Could not parse mesh files for comparison: {}"
    );
    assert!(
        similar,
        "Grids of mesh files {} and {} not similar",
        mesh_path_1.to_string_lossy(),
        mesh_path_2.to_string_lossy()
    );
}

#[derive(Debug, Clone)]
pub struct TestContext {
    input_dir: PathBuf,
    output_dir: PathBuf,
}

impl TestContext {
    const INPUT_DIR_PATH_COMPONENTS: [&'static str; 3] = ["tests", "data", "input"];
    const OUTPUT_DIR_PATH_COMPONENTS: [&'static str; 3] = ["tests", "data", "output"];

    fn new() -> Self {
        let input_dir: PathBuf = Self::INPUT_DIR_PATH_COMPONENTS.iter().collect();
        let output_dir: PathBuf = Self::OUTPUT_DIR_PATH_COMPONENTS.iter().collect();
        Self {
            input_dir,
            output_dir,
        }
    }

    pub fn input_path<S: AsRef<str>>(&self, file_name: S) -> PathBuf {
        self.input_dir().join(file_name.as_ref())
    }

    pub fn output_path<S: AsRef<str>>(&self, file_name: S) -> PathBuf {
        self.output_dir().join(file_name.as_ref())
    }

    fn input_dir(&self) -> &Path {
        self.input_dir.as_path()
    }

    fn output_dir(&self) -> &Path {
        self.output_dir.as_path()
    }
}

lazy_static! {
    pub static ref CONTEXT: TestContext = TestContext::new();
    static ref COMMAND: clap::Command<'static> = cli::build::build().no_binary_name(true);
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
