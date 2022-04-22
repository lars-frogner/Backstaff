use std::{
    env,
    path::PathBuf,
    process::{self, Command},
};

#[cfg(feature = "hdf5")]
use regex::Regex;

macro_rules! exit_with_error {
    ($($print_arg:tt)*) => {{
        eprintln!($($print_arg)*);
        process::exit(1)
    }};
}

macro_rules! exit_on_error {
    ($result:expr, $($fmt_arg:tt)*) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                exit_with_error!($($fmt_arg)*, err.to_string())
            }
        }
    };
}

fn trim_newline(s: &mut String) {
    if s.ends_with('\n') {
        s.pop();
        if s.ends_with('\r') {
            s.pop();
        }
    }
}

#[cfg(not(feature = "hdf5"))]
fn setup_hdf5() {}
#[cfg(feature = "hdf5")]
fn setup_hdf5() {
    if env::var("HDF5_DIR").is_ok() {
        return;
    }

    let config_text = String::from_utf8(
        exit_on_error!(
            Command::new("h5cc").arg("-showconfig").output(),
            "Error: Could run h5cc: {}\n\
             Make sure HDF5 is installed and that h5cc is in $PATH"
        )
        .stdout,
    )
    .unwrap();

    let hdf5_root_path = Regex::new(r"(?m)^\s*Installation point:\s*(.+)\s*$")
        .unwrap()
        .captures(&config_text)
        .expect("Could not find installation point in h5cc -showconfig output")
        .get(1)
        .unwrap()
        .as_str();

    println!("cargo:rustc-env=HDF5_DIR={}", hdf5_root_path);
}

#[cfg(not(feature = "python"))]
fn setup_python() {}
#[cfg(feature = "python")]
fn setup_python() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let project_path = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set or invalid"),
    );

    let project_path = project_path
        .canonicalize()
        .unwrap_or_else(|err| exit_with_error!("Error: Could not resolve project path: {}", err));

    let mut python_exec_path = String::from_utf8(
        exit_on_error!(
            Command::new("which")
                .arg(env::var("PYO3_PYTHON").unwrap_or("python3".to_string()))
                .output(),
            "Error: Could not determine location of python binary: {}"
        )
        .stdout,
    )
    .unwrap();
    trim_newline(&mut python_exec_path);

    let python_exec_path = PathBuf::from(python_exec_path);
    if !python_exec_path.exists() {
        exit_with_error!(
            "Error: Could not find Python executable at {}",
            python_exec_path.to_string_lossy()
        );
    }

    let python_exec_path = python_exec_path.canonicalize().unwrap_or_else(|err| {
        exit_with_error!(
            "Error: Could not resolve path to Python executable: {}",
            err
        )
    });

    let python_binary_name = python_exec_path.file_name().unwrap_or_else(|| {
        exit_with_error!(
            "Error: Could not extract final component of Python executable path {}",
            python_exec_path.to_string_lossy()
        )
    });

    let python_root_path = python_exec_path
        .parent()
        .unwrap()
        .parent()
        .unwrap_or_else(|| {
            exit_with_error!(
                "Error: Could not extract root path from Python executable path {}",
                python_exec_path.to_string_lossy()
            )
        });

    println!(
        "cargo:rustc-env=PYTHONHOME={}",
        python_root_path.to_string_lossy()
    );
    println!(
        "cargo:rustc-env=PYTHONPATH={}:{}",
        python_root_path
            .join("lib")
            .join(python_binary_name)
            .join("site-packages")
            .to_string_lossy(),
        project_path.to_string_lossy()
    );
}

fn main() {
    setup_hdf5();
    setup_python();
}
