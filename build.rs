use std::{env, fs, path::PathBuf, process};

macro_rules! exit_with_error {
    ($($print_arg:tt)*) => {{
        eprintln!($($print_arg)*);
        process::exit(1)
    }};
}

fn main() {
    #[cfg(feature = "python")]
    create_cargo_config_for_python();
}

fn create_cargo_config_for_python() {
    let project_path = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set or invalid"),
    );

    let project_path = project_path
        .canonicalize()
        .unwrap_or_else(|err| exit_with_error!("Error: Could not resolve project path: {}", err));

    let cargo_config_path = project_path.join(".cargo").join("config.toml");

    if cargo_config_path.exists() {
        process::exit(0);
    }

    let python_exec_path = env::var_os("BACKSTAFF_PYTHON_PATH").unwrap_or_else(|| {
        exit_with_error!(
            "Please specify the path to the Python executable using\n\
             the BACKSTAFF_PYTHON_PATH environment variable\n\
             (e.g. BACKSTAFF_PYTHON_PATH=\"$(which python)\")"
        );
    });

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
        project_path.to_string_lossy()
    );

    fs::write(&cargo_config_path, cargo_python_config_content)
        .unwrap_or_else(|err| exit_with_error!("Error: Could not write config file: {}", err));

    exit_with_error!(
        "Warning: Aborted build due to missing configuration for linking with Python.\n\
         Configuration has now been generated in {},\n\
         please rerun the build command.",
        cargo_config_path.to_string_lossy()
    );
}
