use std::process::{self, Command};

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

#[cfg(not(feature = "hdf5"))]
fn setup_hdf5() {}
#[cfg(feature = "hdf5")]
fn setup_hdf5() {
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

#[cfg(not(feature = "netcdf"))]
fn setup_netcdf() {}
#[cfg(feature = "netcdf")]
fn setup_netcdf() {
    let netcdf_lib_path = String::from_utf8(
        exit_on_error!(
            Command::new("nc-config").arg("--libdir").output(),
            "Error: Could run nc-config: {}\n\
             Make sure NetCDF is installed and that nc-config is in $PATH"
        )
        .stdout,
    )
    .unwrap();

    print!("cargo:rustc-link-search={}", netcdf_lib_path);
}

fn main() {
    setup_hdf5();
    setup_netcdf();
}
