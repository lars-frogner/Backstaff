mod common;

use common::run;

def_test!(
IN[]
OUT[regular = "regular.mesh", hor_regular = "hor_regular.mesh"]
fn regular_hor_regular_mesh_is_regular {
    const SHAPE: &str = "--shape=8,8,8";
    const X_BOUNDS: &str = "--x-bounds=0,1";
    const Y_BOUNDS: &str = "--y-bounds=-1,2";
    const Z_BOUNDS: &str = "--z-bounds=1,1.5";
    run([
        "create_mesh",
        regular,
        "--overwrite",
        "regular",
        SHAPE,
        X_BOUNDS,
        Y_BOUNDS,
        Z_BOUNDS,
    ]);
    run([
        "create_mesh",
        hor_regular,
        "--overwrite",
        "horizontally_regular",
        SHAPE,
        X_BOUNDS,
        Y_BOUNDS,
        Z_BOUNDS,
        "--boundary-dz-scales=1.0,1.0",
    ]);
    common::assert_mesh_files_similar(regular, hor_regular);
});
