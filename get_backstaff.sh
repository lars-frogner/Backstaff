#!/bin/bash
set -e

MIN_PYTHON_MAJOR_VERSION=3
MIN_PYTHON_MINOR_VERSION=7
MIN_PYTHON_VERSION=$MIN_PYTHON_MAJOR_VERSION.$MIN_PYTHON_MINOR_VERSION
REQUIRED_PYTHON_PACKAGES='numpy scipy numba ChiantiPy'

FEATURES=cli
CONFIGURED_ENV_VARS=''

command_succeeded() {
    local command="$1"
    eval "$command >/dev/null 2>&1"
    local return_code=$?
    if [[ $return_code = 0 ]]; then
        echo 1
    else
        echo 0
    fi
}

command_exists() {
    local command="$1"
    command_succeeded "which $command"
}

get_user_input() {
    local question="$1"
    read -p "$question"$'\n' -r
    echo "$REPLY"
}

user_says_yes() {
    local question="$1"
    while true; do
        local reply="$(get_user_input "$question [y/N]")"
        if [[ "$reply" =~ ^[Yy]$ ]]; then
            echo 1
            break
        elif [[ -z "$reply" || "$reply" =~ ^[Nn]$ ]]; then
            echo 0
            break
        fi
    done
}

user_says_yes_default_yes() {
    local question="$1"
    while true; do
        local reply="$(get_user_input "$question [Y/n]")"
        if [[ -z "$reply" || "$reply" =~ ^[Yy]$ ]]; then
            echo 1
            break
        elif [[ "$reply" =~ ^[Nn]$ ]]; then
            echo 0
            break
        fi
    done
}

abort() {
    local message="$1"
    echo "$message" >&2
    exit 1
}

abort_if_user_says_no() {
    if [[ $(user_says_yes "$@") = 0 ]]; then
        abort Aborted
    fi
}

abort_if_user_says_no_default_yes() {
    if [[ $(user_says_yes_default_yes "$@") = 0 ]]; then
        abort Aborted
    fi
}

ask_feature() {
    local feature_name="$1"
    user_says_yes "Include $feature_name feature?"
}

ask_feature_default_yes() {
    local feature_name="$1"
    user_says_yes_default_yes "Include $feature_name feature?"
}

add_feature() {
    local feature_name="$1"
    FEATURES="$FEATURES,$feature_name"
}

ask_and_add_feature() {
    local feature_name="$1"
    if [[ $(ask_feature "$feature_name") = 1 ]]; then
        add_feature "$feature_name"
    fi
}

ask_and_add_feature_default_yes() {
    local feature_name="$1"
    if [[ $(ask_feature_default_yes "$feature_name") = 1 ]]; then
        add_feature "$feature_name"
    fi
}

set_env_var() {
    local var_name="$1"
    local var_value="$2"
    eval "$var_name='$var_value'"
    CONFIGURED_ENV_VARS="$CONFIGURED_ENV_VARS $var_name"
}

setup_rust() {
    if [[ $(command_exists rustup) = 0 ]]; then
        abort_if_user_says_no 'No Rust toolchain found. Install Rust?'
        echo 'Installing Rust'
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source $HOME/.cargo/env
    fi
}

setup_python() {
    if [[ $(command_exists python3) = 1 ]]; then
        local python="$(which python3)"
    elif [[ $(command_exists python) = 1 ]]; then
        local python="$(which python)"
    else
        echo 'No Python executable found' >&2
        while true; do
            local python="$(get_user_input "Please enter path to Python (>= $MIN_PYTHON_VERSION) executable:")"
            if [[ $(command_exists "$python") = 1 ]]; then
                break
            else
                echo "No executable at $python" >&2
            fi
        done
    fi

    while true; do
        local version_string=$(eval "$python" --version)
        local python_major_version="$(echo "$version_string" | sed -n 's/^[[:space:]]*Python[[:space:]]*\([0-9]*\)\..*$/\1/p')"
        local python_minor_version="$(echo "$version_string" | sed -n 's/^[[:space:]]*Python[[:space:]]*[0-9]*\.\([0-9]*\).*$/\1/p')"
        local python_version=$python_major_version.$python_minor_version
        if [[ "$python_major_version" -eq "$MIN_PYTHON_MAJOR_VERSION" && "$python_minor_version" -ge "$MIN_PYTHON_MINOR_VERSION" ]] || [[ "$python_major_version" -gt "$MIN_PYTHON_MAJOR_VERSION" ]]; then
            break
        else
            echo "Executable $python has version $python_version, minimum Python version is $MIN_PYTHON_VERSION" >&2
            local python="$(get_user_input "Please enter path to Python (>= $MIN_PYTHON_VERSION) executable:")"
        fi
    done

    for package in $REQUIRED_PYTHON_PACKAGES; do
        if [[ $(command_succeeded "$python -c 'import $package'") = 0 ]]; then
            if [[ $(user_says_yes "Missing required Python package $package. Install it now?") = 1 ]]; then
                if [[ $(user_says_yes "Install $package only for this user (using pip install --user)?") = 1 ]]; then
                    local user_flag=--user
                else
                    local user_flag=''
                fi
                eval "$python -m pip install $user_flag $package"
            fi
        fi
    done

    if [[ $(command_exists realpath) = 1 ]]; then
        local python="$(realpath "$python")"
    fi
    set_env_var PYO3_PYTHON "$python"
    set_env_var RUSTFLAGS "$RUSTFLAGS -C link-args=-Wl,-rpath,""$(dirname "$(dirname "$PYO3_PYTHON")")/lib"""
}

verify_chianti() {
    if [[ ! -z "$XUVTOP" ]]; then
        echo
        echo 'Warning: The XUVTOP environment variable is currently not set' >&2
        echo 'Make sure it is set to the root CHIANTI directory when running backstaff' >&2
        echo 'CHIANTI can be obtained from here:' >&2
        echo 'https://www.chiantidatabase.org/chianti_download.html' >&2
        echo
    fi
}

setup_hdf5() {
    if [[ $(command_exists h5cc) = 1 ]]; then
        set_env_var HDF5_DIR "$(h5cc -showconfig | sed -n 's/^[[:space:]]*Installation point:[[:space:]]*\(.*\)[[:space:]]*$/\1/p')"
    else
        echo 'No HDF5 library found' >&2
        echo 'The HDF5 library can be obtained from here:' >&2
        echo 'https://www.hdfgroup.org/downloads/hdf5/' >&2
        echo
        while true; do
            local hdf5_dir="$(get_user_input 'Please enter path to HDF5 root directory:')"
            if [[ -d "$hdf5_dir" ]]; then
                set_env_var HDF5_DIR "$hdf5_dir"
                break
            else
                echo "No directory at $hdf5_dir" >&2
            fi
        done
    fi
}

setup_netcdf() {
    if [[ $(command_exists nc-config) = 1 ]]; then
        set_env_var NETCDF_DIR "$(nc-config --prefix)"
    else
        echo 'No NetCDF library found' >&2
        echo 'Installation instructions for NetCDF can be found here:' >&2
        echo 'https://docs.unidata.ucar.edu/nug/current/getting_and_building_netcdf.html' >&2
        echo
        while true; do
            local netcdf_dir="$(get_user_input 'Please enter path to NetCDF root directory:')"
            if [[ -d "$netcdf_dir" ]]; then
                set_env_var NETCDF_DIR "$netcdf_dir"
                break
            else
                echo "No directory at $netcdf_dir" >&2
            fi
        done
    fi
}

setup_tab_completion() {
    while true; do
        local shell="$(get_user_input 'Please enter shell to configure (must be bash, zsh or fish):')"
        if [[ "$shell" = bash ]]; then
            if [[ $(command_exists brew) = 1 ]]; then
                local completion_dir="$(brew --prefix)/etc/bash_completion.d"
                local completion_file="$completion_dir/backstaff.bash-completion"
            else
                local completion_dir="$HOME/.local/share/bash-completion/completions"
                local completion_file="$completion_dir/backstaff"
            fi
            mkdir -p "$completion_dir"
            backstaff completions bash >"$completion_file"
            echo
            echo "Generated completion file at $completion_file"
            echo 'You may have to log out and log back in to your shell session for the changes to take affect.'
            echo "If completion still doesn't work, try sourcing the completion script in your .bashrc file:"
            echo "source $completion_file"
            echo
            break
        elif [[ "$shell" = zsh ]]; then
            local completion_dir="$HOME/.zfunc"
            local completion_file="$completion_dir/_backstaff"
            mkdir -p "$completion_dir"
            backstaff completions zsh >"$completion_file"
            echo
            echo "Generated completion file at $completion_file"
            echo "Please add the following to the beginning of your .zshrc file if not already present:"
            echo "fpath+=$completion_dir"
            echo
            echo 'You will have to log out and log back in to your shell session for the changes to take affect.'
            echo
            break
        elif [[ "$shell" = fish ]]; then
            local completion_dir="$HOME/.config/fish/completions"
            local completion_file="$completion_dir/backstaff.fish"
            mkdir -p "$completion_dir"
            backstaff completions fish >"$completion_file"
            echo
            echo "Generated completion file at $completion_file"
            echo 'You may have to log out and log back in to your shell session for the changes to take affect.'
            echo
            break
        else
            echo "Invalid shell $shell" >&2
        fi
    done
}

configure_cargo() {
    set_env_var CARGO_NET_GIT_FETCH_WITH_CLI true
}

print_configured_env_vars() {
    if [[ ! -z "$CONFIGURED_ENV_VARS" ]]; then
        echo 'Configured environment variables:'
        for var_name in $CONFIGURED_ENV_VARS; do
            local var_value="$(eval "echo \$$var_name")"
            echo "$var_name='$var_value'"
        done
        echo
    fi
}

echo 'This script will install the Backstaff program'

setup_rust

echo
echo 'Please select which features to include'
echo 'Feature descriptions are found here:'
echo 'https://github.com/lars-frogner/Backstaff/blob/main/README.md#features'
echo

ask_and_add_feature_default_yes statistics
ask_and_add_feature tracing
ask_and_add_feature corks
if [[ $(ask_feature synthesis) = 1 ]]; then
    add_feature synthesis
    setup_python
    verify_chianti
fi
ask_and_add_feature ebeam

ask_and_add_feature json
ask_and_add_feature pickle
if [[ $(ask_feature hdf5) = 1 ]]; then
    add_feature hdf5
    setup_hdf5
fi
if [[ $(ask_feature netcdf) = 1 ]]; then
    add_feature netcdf
    setup_netcdf
fi

echo
echo 'Enabling link time optimization may improve performance at the expense of compile time'
if [[ $(user_says_yes 'Enable link time optimization?') = 1 ]]; then
    PROFILE_ARGS='--profile release-lto'
else
    PROFILE_ARGS=''
fi
echo

echo "Default installation directory is $HOME/.cargo/bin"
if [[ $(user_says_yes 'Use a different directory?') = 1 ]]; then
    target_dir="$(get_user_input 'Enter new installation directory:')"
    mkdir -p "$target_dir"
    ROOT_ARGS="--root $target_dir"
else
    ROOT_ARGS=''
fi

configure_cargo

print_configured_env_vars

install_command="cargo install --locked --no-default-features --features=$FEATURES $(echo $PROFILE_ARGS) $(echo $ROOT_ARGS) --git=https://github.com/lars-frogner/Backstaff.git"
echo 'Will now install Backstaff using the following command:'
echo "$install_command"
echo
abort_if_user_says_no_default_yes 'Continue?'
eval "$install_command"

echo
if [[ $(user_says_yes 'Setup tab-completion (available for bash, zsh and fish)?') = 1 ]]; then
    setup_tab_completion
fi

echo 'Done'
