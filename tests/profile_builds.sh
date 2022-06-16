#!/bin/zsh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"

FEATURE_SETS=(
    'NONE'
    'cli'
    'cli,statistics'
    'cli,statistics,json'
    'cli,statistics,json,pickle'
    'cli,statistics,json,pickle,hdf5'
    'cli,statistics,json,pickle,hdf5,netcdf'
    'cli,statistics,json,pickle,hdf5,netcdf,corks'
    'cli,statistics,json,pickle,hdf5,netcdf,corks,tracing'
    'cli,statistics,json,pickle,hdf5,netcdf,corks,tracing,ebeam'
    'cli,statistics,json,pickle,hdf5,netcdf,corks,tracing,ebeam,synthesis'
    'ALL'
)

ERR_FILE=/tmp/.profile_builds_err.txt

case "$(uname -s)" in
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE=LINUX
esac

if [[ "MACHINE" = "Mac" ]]; then
    MEM_UNIT=kB
else
    MEM_UNIT=MB
fi
TIMEFMT="Took %U, max memory usage was %M $MEM_UNIT"

for FEATURE_SET in "${FEATURE_SETS[@]}"; do
    if [[ "$FEATURE_SET" = "NONE" ]]; then
        FEATURE_MSG='no features'
        FEATURE_ARGS='--no-default-features'
    elif [[ "$FEATURE_SET" = "ALL" ]]; then
        FEATURE_MSG='all features'
        FEATURE_ARGS='--all-features'
    else
        FEATURE_MSG="features: $FEATURE_SET"
        FEATURE_ARGS="--no-default-features --features=$FEATURE_SET"
    fi
    echo -n "Building with $FEATURE_MSG... "
    cargo clean

    time bash $PROJECT_DIR/setup_backstaff --no-to-all build --release $FEATURE_ARGS > /dev/null 2> $ERR_FILE

    if [[ ! "$?" = 0 ]]; then
        echo Failed >&2
        cat $ERR_FILE
        rm -f $ERR_FILE
        exit 1
    fi

    rm -f $ERR_FILE

    if [[ -f "$PROJECT_DIR/target/release/backstaff" ]]; then
        echo "Binary size is $(du -h "$PROJECT_DIR/target/release/backstaff" | cut -f1)"
    fi
done
