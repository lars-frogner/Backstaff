mod regression;

use regression::{run, Actual, Expected, RegressionTest};

#[test]
fn regular_mesh_is_correct() {
    let test = RegressionTest::for_output_file("regular.mesh");
    run([
        "create_mesh",
        test.output_path(),
        "--overwrite",
        "regular",
        "--shape=3,4,5",
        "--x-bounds=0,1",
        "--y-bounds=-1,2",
        "--z-bounds=1,1.5",
    ]);
    test.assert_files_identical();
}
