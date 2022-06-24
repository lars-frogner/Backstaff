mod common;

use backstaff::{
    exit_on_error, field_value_computer,
    geometry::Dim3::{X, Y, Z},
    grid::{fgr, Grid3},
    io::{
        snapshot::{
            fdt,
            native::NATIVE_COORD_PRECISION,
            utils::{self as snapshot_utils},
        },
        Verbosity,
    },
};
use common::ENDIANNESS;
use std::path::PathBuf;

#[cfg(feature = "cli")]
use common::run;

#[cfg(feature = "for-testing")]
use approx::RelativeEq;

const MINIMAL_NATIVE_MESH: &str = "minimal.mesh";
const MINIMAL_NATIVE_SNAP: &str = "minimal_001.idl";
const MINIMAL_NETCDF_SNAP: &str = "minimal_001.nc";
const MINIMAL_REGULAR_NATIVE_SNAP: &str = "minimal_regular_001.idl";
const REALISTIC_NATIVE_SNAP: &str = "realistic_042.idl";
const REALISTIC_REGULAR_NATIVE_SNAP: &str = "realistic_regular_042.idl";

#[cfg(all(feature = "cli", feature = "for-testing"))]
def_test!(
IN[]
OUT[regular="regular.mesh", hor_regular="hor_regular.mesh"]
fn regular_hor_regular_mesh_is_regular() {
    const SHAPE: &str = "--shape=11,8,10";
    const X_BOUNDS: &str = "--x-bounds=-10.1,0";
    const Y_BOUNDS: &str = "--y-bounds=0.1,3";
    const Z_BOUNDS: &str = "--z-bounds=1,1.5";
    run(["create_mesh",
         regular,
         "regular",
         SHAPE,
         X_BOUNDS,
         Y_BOUNDS,
         Z_BOUNDS,
    ]);
    run(["create_mesh",
         hor_regular,
         "horizontally_regular",
         SHAPE,
         X_BOUNDS,
         Y_BOUNDS,
         Z_BOUNDS,
    ]);
    common::assert_mesh_files_equal(regular, hor_regular, fgr::default_max_relative());
});

#[cfg(all(feature = "cli", feature = "for-testing"))]
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_NATIVE_SNAP]
fn write_preserves_native_input_snapshot() {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, fdt::default_max_relative());
});

#[cfg(all(feature = "cli", feature = "for-testing", feature = "netcdf"))]
def_test!(
IN[input_snapshot=MINIMAL_NETCDF_SNAP]
OUT[output_snapshot=MINIMAL_NETCDF_SNAP]
fn write_preserves_netcdf_input_snapshot() {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, fdt::default_max_relative());
});

#[cfg(all(feature = "cli", feature = "for-testing", feature = "netcdf"))]
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_NETCDF_SNAP]
fn conversion_to_netcdf_preserves_native_input_snapshot() {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, fdt::default_max_relative());
});

#[cfg(all(feature = "cli", feature = "for-testing", feature = "netcdf"))]
def_test!(
IN[input_snapshot=MINIMAL_NETCDF_SNAP]
OUT[output_snapshot=MINIMAL_NATIVE_SNAP]
fn conversion_to_native_preserves_netcdf_input_snapshot() {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, fdt::default_max_relative());
});

#[cfg(all(feature = "cli", feature = "for-testing"))]
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[  tmp_1="tmp_101.idl",     tmp_2="tmp_102.idl",     tmp_3="tmp_103.idl",
    final_1="final_042.idl", final_2="final_043.idl", final_3="final_044.idl"]
fn snap_num_mapping_works() {
    for output_snapshot in [tmp_1, tmp_2, tmp_3] {
        run(["snapshot",
             input_snapshot,
             "write",
             output_snapshot,
             "--no-overwrite",
        ]);
    }
    run(["snapshot",
         tmp_1,
         "--snap-range=101,103",
         "write",
         final_1,
    ]);
    for output_snapshot in [final_1, final_2, final_3] {
        common::assert_file_exists(output_snapshot);
    }
});

#[cfg(all(feature = "cli", feature = "for-testing"))]
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_NATIVE_SNAP]
fn extract_with_original_bounds_preserves_input_snapshot() {
    run(["snapshot",
         input_snapshot,
         "extract",
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, fdt::default_max_relative());
});

#[cfg(all(feature = "cli", feature = "for-testing"))]
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_NATIVE_SNAP]
fn extract_with_too_large_subgrid_preserves_input_snapshot() {
    let grid = exit_on_error!(
        snapshot_utils::read_snapshot_grid(PathBuf::from(input_snapshot), ENDIANNESS, Verbosity::Quiet),
        "Error: {}"
    );
    let lower_bounds = grid.lower_bounds();
    let upper_bounds = grid.upper_bounds();
    let extents = grid.extents();
    run(["snapshot",
         input_snapshot,
         "extract",
         &format!("--x-bounds={:.precision$E},{:.precision$E}",
                  lower_bounds[X] - extents[X], upper_bounds[X] + extents[X],
                  precision = NATIVE_COORD_PRECISION
         ),
         &format!("--y-bounds={:.precision$E},{:.precision$E}",
                  lower_bounds[Y] - extents[Y], upper_bounds[Y] + extents[Y],
                  precision = NATIVE_COORD_PRECISION
         ),
         &format!("--z-bounds={:.precision$E},{:.precision$E}",
                  lower_bounds[Z] - extents[Z], upper_bounds[Z] + extents[Z],
                  precision = NATIVE_COORD_PRECISION
         ),
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, fdt::default_max_relative());
});

macro_rules! define_test_for_each_resampling_method {
    ($test_macro:ident) => {
        #[cfg(all(feature = "cli", feature = "for-testing"))]
        $test_macro!(sample_averaging);
        #[cfg(all(feature = "cli", feature = "for-testing"))]
        $test_macro!(direct_sampling);
        #[cfg(all(feature = "cli", feature = "for-testing"))]
        $test_macro!(cell_averaging);
    };
}

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_NATIVE_SNAP]
fn [<resampling_to_reshaped_of_same_shape_with_ $resampling_method _preserves_input_snapshot>] () {
    run(["snapshot",
        input_snapshot,
        "resample",
        "reshaped_grid",
        stringify!($resampling_method),
        "write",
        output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, 1e-4);
});
}};}
define_test_for_each_resampling_method!(resampling_test);

// Shadow the above macro with the same name
macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_mesh=MINIMAL_NATIVE_MESH, input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_NATIVE_SNAP]
fn [<resampling_to_original_mesh_file_with_ $resampling_method _preserves_input_snapshot>] () {
    run(["snapshot",
            input_snapshot,
            "resample",
            "mesh_file",
            input_mesh,
            stringify!($resampling_method),
            "write",
            output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, 1e-4);
});
}};}
define_test_for_each_resampling_method!(resampling_test);

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
fn [<resampling_regular_to_regular_of_same_bounds_and_shape_with_ $resampling_method _preserves_input_snapshot>] () {
    run(["snapshot",
        input_snapshot,
        "resample",
        "regular_grid",
        stringify!($resampling_method),
        "write",
        output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot, 1e-6);
});
}};}
define_test_for_each_resampling_method!(resampling_test);

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
fn [<resampling_regular_to_regular_of_same_shape_and_shifted_bounds_with_ $resampling_method _preserves_input_snapshot>] () {
    let grid = exit_on_error!(
        snapshot_utils::read_snapshot_grid(PathBuf::from(input_snapshot), ENDIANNESS, Verbosity::Quiet),
        "Error: {}"
    );
    let upper_bounds = grid.upper_bounds();
    let extents = grid.extents();
    run(["snapshot",
        input_snapshot,
        "resample",
        "regular_grid",
        "--shape=same,same,same",
        &format!(
            "--x-bounds={:.precision$E},{:.precision$E}",
            upper_bounds[X], upper_bounds[X] + extents[X],
            precision = NATIVE_COORD_PRECISION
        ),
        stringify!($resampling_method),
        "write",
        output_snapshot,
    ]);
    common::assert_snapshot_field_values_equal(input_snapshot, output_snapshot, 1e-5);
});
}};}
define_test_for_each_resampling_method!(resampling_test);

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_mesh=MINIMAL_NATIVE_MESH, input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot_1="a/out_001.idl", output_snapshot_2="b/out_001.idl"]
fn [<resampling_to_reshaped_and_reshaped_original_mesh_file_with_ $resampling_method _gives_same_result>] () {
    const SHAPE_ARG: &str ="--shape=15,9,12";
    run(["snapshot",
        input_snapshot,
        "resample",
        "reshaped_grid",
        SHAPE_ARG,
        stringify!($resampling_method),
        "write",
        output_snapshot_1,
    ]);
    run(["snapshot",
        input_snapshot,
        "resample",
        "mesh_file",
        input_mesh,
        SHAPE_ARG,
        stringify!($resampling_method),
        "write",
        output_snapshot_2,
    ]);
    common::assert_snapshot_files_equal(output_snapshot_1, output_snapshot_2, fdt::default_max_relative());
});
}};}
define_test_for_each_resampling_method!(resampling_test);

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[output_snapshot_1="a/out_001.idl", output_snapshot_2="b/out_001.idl"]
fn [<resampling_to_regular_and_rotated_regular_without_rotation_with_ $resampling_method _gives_same_result>] () {
    let grid = exit_on_error!(
        snapshot_utils::read_snapshot_grid(PathBuf::from(input_snapshot), ENDIANNESS, Verbosity::Quiet),
        "Error: {}"
    );
    let lower_bounds = grid.lower_bounds();
    let upper_bounds = grid.upper_bounds();
    let extents = grid.extents();
    run(["snapshot",
        input_snapshot,
        "resample",
        "regular_grid",
        stringify!($resampling_method),
        "write",
        output_snapshot_1,
    ]);
    run(["snapshot",
        input_snapshot,
        "resample",
        "rotated_regular_grid",
        &format!(
            "--x-start={:.precision$E},{:.precision$E}",
            lower_bounds[X], lower_bounds[Y],
            precision = NATIVE_COORD_PRECISION
        ),
        &format!(
            "--x-end={:.precision$E},{:.precision$E}",
            upper_bounds[X], lower_bounds[X],
            precision = NATIVE_COORD_PRECISION
        ),
        &format!(
            "--y-extent={:.precision$E}", extents[Y],
            precision = NATIVE_COORD_PRECISION
        ),
        stringify!($resampling_method),
        "write",
        output_snapshot_2,
    ]);
    common::assert_snapshot_field_values_equal(output_snapshot_1, output_snapshot_2, 1e-4);
});
}};}
define_test_for_each_resampling_method!(resampling_test);

def_test!(
IN[input_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
OUT[output_snapshot_1="a/out_001.idl", output_snapshot_2="b/out_001.idl"]
fn upsampling_regular_with_sample_averaging_and_direct_sampling_gives_same_result() {
    const SCALES_ARG: &str ="--scales=2,2,2";
    run(["snapshot",
        input_snapshot,
        "resample",
        "reshaped_grid",
        SCALES_ARG,
        "sample_averaging",
        "write",
        output_snapshot_1,
    ]);
    run(["snapshot",
        input_snapshot,
        "resample",
        "reshaped_grid",
        SCALES_ARG,
        "direct_sampling",
        "write",
        output_snapshot_2,
    ]);
    common::assert_snapshot_files_equal(output_snapshot_1, output_snapshot_2, fdt::default_max_relative());
});

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
OUT[output_snapshot=MINIMAL_REGULAR_NATIVE_SNAP]
fn [<resampling_to_single_cell_for_regular_with_ $resampling_method _gives_mean_value>] () {
    run(["snapshot",
        input_snapshot,
        "resample",
        "reshaped_grid",
        "--shape=1,1,1",
        stringify!($resampling_method),
        "write",
        output_snapshot,
    ]);
    common::assert_snapshot_field_values_custom_equal(output_snapshot, input_snapshot, &|resampled_values, input_values| {
        let mean_of_input_values = input_values.iter().sum::<fdt>() / (input_values.len() as fdt);
        resampled_values.len() == 1 && approx::relative_eq!(resampled_values[0], mean_of_input_values, max_relative=1e-4)
    });
});
}};}
#[cfg(all(feature = "cli", feature = "for-testing"))]
resampling_test!(sample_averaging);
#[cfg(all(feature = "cli", feature = "for-testing"))]
resampling_test!(cell_averaging);

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[generated_output_snapshot="gen/out_001.idl", resampled_output_snapshot="res/out_001.idl"]
fn [<resampling_uniform_valued_to_regular_with_ $resampling_method _preserves_input_values>] () {
    let grid = exit_on_error!(
        snapshot_utils::read_snapshot_grid(PathBuf::from(input_snapshot), ENDIANNESS, Verbosity::Quiet),
        "Error: {}"
    );
    common::write_snapshot_with_computed_variable(
        grid.clone(),
        "tg", field_value_computer!(constant = 1e3; (fdt)),
        generated_output_snapshot
    );

    let lower_bounds = grid.lower_bounds();
    let upper_bounds = grid.upper_bounds();
    let extents = grid.extents();
    let bounds_arg = |dim| format!(
        "--{}-bounds={:.precision$E},{:.precision$E}",
        dim,
        lower_bounds[dim] + 0.25 * extents[dim],
        upper_bounds[dim] - 0.25 * extents[dim],
        precision = NATIVE_COORD_PRECISION
    );
    run(["snapshot",
        generated_output_snapshot,
        "resample",
        "regular_grid",
        "--shape=same,same,same",
        &bounds_arg(X),
        &bounds_arg(Y),
        &bounds_arg(Z),
        stringify!($resampling_method),
        "write",
        resampled_output_snapshot,
    ]);

    common::assert_snapshot_field_values_equal(
        generated_output_snapshot, resampled_output_snapshot,
        fdt::default_max_relative()
    );
});
}};}
define_test_for_each_resampling_method!(resampling_test);

macro_rules! resampling_test { ($resampling_method:ident) => { paste::paste! {
def_test!(
IN[input_snapshot=MINIMAL_NATIVE_SNAP]
OUT[generated_output_snapshot="gen/out_001.idl", resampled_output_snapshot="res/out_001.idl"]
fn [<resampling_uniform_valued_to_rotated_regular_with_ $resampling_method _preserves_input_values>] () {
    let grid = exit_on_error!(
        snapshot_utils::read_snapshot_grid(PathBuf::from(input_snapshot), ENDIANNESS, Verbosity::Quiet),
        "Error: {}"
    );
    common::write_snapshot_with_computed_variable(
        grid.clone(),
        "tg", field_value_computer!(constant = 1e3; (fdt)),
        generated_output_snapshot
    );

    let lower_bounds = grid.lower_bounds();
    let upper_bounds = grid.upper_bounds();
    let extents = grid.extents();
    run(["snapshot",
        generated_output_snapshot,
        "resample",
        "rotated_regular_grid",
        "--shape=same,same,same",
        &format!(
            "--x-start={:.precision$E},{:.precision$E}",
            upper_bounds[X] - 0.25 * extents[X],
            lower_bounds[Y] + 0.25 * extents[Y],
            precision = NATIVE_COORD_PRECISION
        ),
        &format!(
            "--x-end={:.precision$E},{:.precision$E}",
            upper_bounds[X] - 0.25 * extents[X],
            upper_bounds[Y] - 0.25 * extents[Y],
            precision = NATIVE_COORD_PRECISION
        ),
        &format!(
            "--y-extent={:.precision$E}",
            0.5 * extents[X],
            precision = NATIVE_COORD_PRECISION
        ),
        &format!(
            "--z-bounds={:.precision$E},{:.precision$E}",
            lower_bounds[Z] + 0.25 * extents[Z],
            upper_bounds[Z] - 0.25 * extents[Z],
            precision = NATIVE_COORD_PRECISION
        ),
        stringify!($resampling_method),
        "write",
        resampled_output_snapshot,
    ]);

    common::assert_snapshot_field_values_equal(
        generated_output_snapshot, resampled_output_snapshot,
        fdt::default_max_relative()
    );
});
}};}
define_test_for_each_resampling_method!(resampling_test);

// Resampling a field where the value is an n-th order polynomial function of the grid coordinates using direct sampling with n-th order (or higher) polynomial fitting interpolation gives (a part of) the same field

macro_rules! create_combination_test {
    ($fn_name:ident, $commands:expr) => {
def_test!(
IN[input_snapshot=REALISTIC_NATIVE_SNAP]
OUT[output_snapshot="snapshot_001.idl",
    output_stats="stats.txt",
    output_slice="slice.pickle",
    output_corks_pickle="corks.pickle",
    output_corks_json="corks.json",
    output_field_lines="lines.fl"]
fn $fn_name() {
    macro_rules! snapshot {
        () => { vec!["-N", "snapshot", input_snapshot] };
    }
    macro_rules! inspect {
        () => { vec!["inspect",
                    "--included-quantities=uz",
                    "statistics",
                    &format!("--output-file={}", output_stats),
                    "--overwrite"
                ] };
    }
    macro_rules! write {
        () => { vec!["write",
                    output_snapshot,
                    "--included-quantities=uz",
                    "--overwrite"
                ] };
    }
    macro_rules! extract {
        () => { vec!["extract"] };
    }
    macro_rules! resample {
        () => { vec!["resample", "reshaped_grid" ] };
    }
    macro_rules! slice {
        () => { vec!["slice",
                    output_slice,
                    "--quantity=uz",
                    "--axis=z",
                    "--coord=0.0"
                ] };
    }
    macro_rules! corks {
        () => { vec!["corks",
                    if cfg!(feature = "pickle") {
                        output_corks_pickle
                    } else {
                        output_corks_json
                    },
                    "--sampled-scalar-quantities=uz",
                    "volume_seeder",
                    "regular",
                    "--shape=1,1,1"
                ] };
    }
    macro_rules! trace {
        () => { vec!["trace",
                    output_field_lines,
                    "--extracted-quantities=uz",
                    "volume_seeder",
                    "regular",
                    "--shape=1,1,1"
                ] };
    }
    macro_rules! ebeam {
        () => { vec!["ebeam",
                    "simulate",
                    output_field_lines,
                    "--extra-varying-scalars=uz",
                    "--overwrite"
                ] };
    }

    // Command runs before inspect
    #[cfg(feature = "statistics")]
    run([snapshot!(), $commands, inspect!()].concat());

    // Command runs before write
    run([snapshot!(), $commands, write!()].concat());

    // Command runs before extract
    run([snapshot!(), $commands, extract!(), write!()].concat());

    // Command runs before resample
    run([snapshot!(), $commands, resample!(), write!()].concat());

    // Command runs before slice
    #[cfg(feature = "pickle")]
    run([snapshot!(), $commands, slice!()].concat());

    // Command runs after extract
    run([snapshot!(), extract!(), $commands, write!()].concat());

    // Command runs after resample
    run([snapshot!(), resample!(), $commands, write!()].concat());

    // Command runs before corks
    #[cfg(all(feature = "corks", any(feature = "pickle", feature = "json")))]
    run([snapshot!(), $commands, corks!()].concat());

    // Command runs before trace
    #[cfg(feature = "tracing")]
    run([snapshot!(), $commands, trace!()].concat());

    // Command runs before ebeam
    #[cfg(feature = "ebeam")]
    run([snapshot!(), $commands, ebeam!()].concat());
});
};
}

#[cfg(all(feature = "cli", feature = "for-testing", feature = "derivation"))]
create_combination_test!(derive_combinations_work, vec!["derive"]);

// #[cfg(all(feature = "cli", feature = "for-testing", feature = "synthesis"))]
// create_combination_test!(synthesize_combinations_work, vec!["synthesize"]);
