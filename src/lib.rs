//! Useful tools for working with Bifrost in Rust.

#[macro_export]
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

#[macro_export]
macro_rules! exit_on_false {
    ($logic:expr, $($print_arg:tt)*) => {
        if !$logic {
            exit_with_error!($($print_arg)*)
        }
    };
}

#[macro_export]
macro_rules! exit_on_none {
    ($option:expr, $($print_arg:tt)*) => {
        $option.unwrap_or_else(|| exit_with_error!($($print_arg)*))
    };
}

#[macro_export]
macro_rules! exit_with_error {
    ($($print_arg:tt)*) => {{
        eprintln!($($print_arg)*);
        ::std::process::exit(1)
    }};
}

pub mod constants;
pub mod corks;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod interpolation;
pub mod io;
pub mod math;
pub mod num;
pub mod plasma;
pub mod random;
pub mod seeding;
pub mod units;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "tracing")]
pub mod tracing;

#[cfg(feature = "ebeam")]
pub mod ebeam;
