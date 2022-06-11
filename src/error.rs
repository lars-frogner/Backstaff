#[macro_export]
macro_rules! exit_with_error {
    ($($print_arg:tt)*) => {{
        eprintln!($($print_arg)*);
        ::std::process::exit(1);
    }};
}

#[macro_export]
macro_rules! exit_on_error {
    ($result:expr, $($print_arg:tt)*) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                $crate::exit_with_error!($($print_arg)*, err)
            }
        }
    };
}

#[macro_export]
macro_rules! exit_on_false {
    ($logic:expr, $($print_arg:tt)*) => {
        if $logic {
            true
        } else {
            $crate::exit_with_error!($($print_arg)*)
        }
    };
}

#[macro_export]
macro_rules! exit_on_none {
    ($option:expr, $($print_arg:tt)*) => {
        $option.unwrap_or_else(|| $crate::exit_with_error!($($print_arg)*))
    };
}
