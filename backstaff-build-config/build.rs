use std::{
    env, fs,
    path::PathBuf,
    process::{self, Command},
};

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

fn create_cargo_config_for_python() {
    println!("cargo:rerun-if-env-changed=BACKSTAFF_PYTHON_PATH");

    let project_path = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set or invalid"),
    );

    let root_project_path = project_path.parent().expect("No parent folder");

    let root_project_path = root_project_path
        .canonicalize()
        .unwrap_or_else(|err| exit_with_error!("Error: Could not resolve project path: {}", err));

    let cargo_config_path = root_project_path.join(".cargo").join("config.toml");

    if cargo_config_path.exists() {
        process::exit(0);
    }

    let mut python_exec_path = env::var("BACKSTAFF_PYTHON_PATH").unwrap_or_else(|_| {
        String::from_utf8(
            exit_on_error!(
                Command::new("which").arg("python").output(),
                "Error: Could not determine location of python binary: {}"
            )
            .stdout,
        )
        .unwrap()
    });
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

    fs::create_dir_all(cargo_config_path.parent().unwrap()).unwrap_or_else(|err| {
        exit_with_error!(
            "Error: Could not create directories for config file: {}",
            err
        )
    });

    let cargo_python_config_content = format!(
        "[build]\n\
     rustflags = [\"-C\", \"link-args=-Wl,-rpath,{}\"]\n\
     \n\
     [env]\n\
     PYTHONHOME = \"{}\"\n\
     PYTHONPATH = \"{}:{}\"",
        python_root_path.join("lib").to_string_lossy(),
        python_root_path.to_string_lossy(),
        python_root_path
            .join("lib")
            .join(python_binary_name)
            .join("site-packages")
            .to_string_lossy(),
        root_project_path.to_string_lossy()
    );

    fs::write(&cargo_config_path, cargo_python_config_content)
        .unwrap_or_else(|err| exit_with_error!("Error: Could not write config file: {}", err));
}

fn main() {
    #[cfg(feature = "python")]
    create_cargo_config_for_python();
}
