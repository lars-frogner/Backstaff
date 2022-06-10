mod common;

use common::run;

const REALISTIC_NATIVE_SNAP: &str = "realistic_042.idl";
const REALISTIC_NETCDF_SNAP: &str = "realistic_042.nc";
const REALISTIC_REGULAR_NATIVE_SNAP: &str = "realistic_regular_042.idl";
const REALISTIC_REGULAR_NETCDF_SNAP: &str = "realistic_regular_042.nc";

def_test!(
IN[]
OUT[regular="regular.mesh", hor_regular="hor_regular.mesh"]
fn regular_hor_regular_mesh_is_regular {
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
    common::assert_mesh_files_equal(regular, hor_regular);
});

def_test!(
IN[input_snapshot=REALISTIC_NATIVE_SNAP]
OUT[output_snapshot=REALISTIC_NATIVE_SNAP]
fn write_preserves_native_input_snapshot {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot);
});

#[cfg(feature = "netcdf")]
def_test!(
IN[input_snapshot=REALISTIC_NETCDF_SNAP]
OUT[output_snapshot=REALISTIC_NETCDF_SNAP]
fn write_preserves_netcdf_input_snapshot {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot);
});

#[cfg(feature = "netcdf")]
def_test!(
IN[input_snapshot=REALISTIC_NATIVE_SNAP]
OUT[output_snapshot=REALISTIC_NETCDF_SNAP]
fn conversion_to_netcdf_preserves_native_input_snapshot {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot);
});

#[cfg(feature = "netcdf")]
def_test!(
IN[input_snapshot=REALISTIC_NETCDF_SNAP]
OUT[output_snapshot=REALISTIC_NATIVE_SNAP]
fn conversion_to_native_preserves_netcdf_input_snapshot {
    run(["snapshot",
         input_snapshot,
         "write",
         output_snapshot,
    ]);
    common::assert_snapshot_files_equal(input_snapshot, output_snapshot);
});

def_test!(
IN[input_snapshot=REALISTIC_NATIVE_SNAP]
OUT[  tmp_1="tmp_101.idl",     tmp_2="tmp_102.idl",     tmp_3="tmp_103.idl",
    final_1="final_042.idl", final_2="final_043.idl", final_3="final_044.idl"]
fn snap_num_mapping_works {
    for output_snapshot in [tmp_1, tmp_2, tmp_3] {
        run(["snapshot",
             input_snapshot,
             "write",
             output_snapshot,
             "--included-quantities=tg",
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

// Derive works at all possible places in the pipeline

// Synthesize works at all possible places in the pipeline

// Extract with same index bounds gives unchanged snapshot

// Extract with bounds shifted in periodic direction exactly the full extent gives unchanged snapshot

// For regular input snapshot: extract with grid interpreted as horizontally regular gives the same as when interpreted as regular

// Resampling to reshaped grid with same shape gives unchanged snapshot

// Resampling to mesh file of the snapshot with unchanged shape gives unchanged snapshot

// For regular input snapshot: resampling to regular grid with same bounds and shape gives unchanged snapshot

// For regular input snapshot: resampling to regular grid with same shape and bounds shifted in periodic direction exactly the full extent gives unchanged snapshot

// Resampling to mesh file of the snapshot with new shape gives same snapshot as resampling to reshaped grid with that shape

// Resampling to rotated regular grid with no rotation gives same snapshot as resampling to regular grid

// Resampling to regular grid with only one grid cell using sample averaging gives the grid cell a value equal to the mean of the resampled field

// Resampling to regular grid with only one grid cell using cell averaging gives the grid cell a value equal to the mean of the resampled field

// Resampling a field where the value is an n-th order polynomial function of the grid coordinates using direct sampling with n-th order (or higher) polynomial fitting interpolation gives (a part of) the same field

// Upsampling with sample averaging gives same snapshot as upsampling with direct sampling

// Resampling a uniform valued field with sample averaging gives the same field

// Resampling a uniform valued field with cell averaging gives the same field

// For regular input snapshot: resampling with grid interpreted as horizontally regular gives the same as when interpreted as regular
