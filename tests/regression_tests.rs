mod regression;

use regression::{assert_file_identical, in_output_dir, run};

#[test]
fn regular_mesh_is_correct() {
    let output_path = in_output_dir("regular.mesh");
    run([
        "create_mesh",
        &output_path.to_string_lossy(),
        "--overwrite",
        "regular",
        "--shape=3,4,5",
        "--x-bounds=0,1",
        "--y-bounds=-1,2",
        "--z-bounds=1,1.5",
    ]);
    assert_file_identical(output_path, None);
}
