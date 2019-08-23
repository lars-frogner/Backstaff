use crate::geometry::{In3D, Point3, Vec3};
use crate::grid::{BoundsCrossing};
use super::{ftr};

enum StepResult3 {
    Ok,
    Null,
    OutOfBounds(In3D<BoundsCrossing>)
}

trait Stepper3 {
    fn step() -> StepResult3;
    fn current_position(&self) -> &Point3<ftr>;
    fn previous_position(&self) -> &Point3<ftr>;
}

struct RKF23Stepper3 {
    step_size: ftr,
    previous_position: Point3<ftr>,
    current_position: Point3<ftr>,
    previous_direction: Vec3<ftr>,
    intermediate_directions: [Vec3<ftr>; 2]
}

impl  RKF23Stepper3 {
    const A21: ftr = 1.0/2.0;
    const A32: ftr = 3.0/4.0;
    const A41: ftr = 2.0/9.0;
    const A42: ftr = 1.0/3.0;
    const A43: ftr = 4.0/9.0;

    const E1: ftr = -5.0/72.0;
    const E2: ftr =  1.0/12.0;
    const E3: ftr =  1.0/9.0;
    const E4: ftr = -1.0/8.0;
}

impl Stepper3 for RKF23Stepper3 {
    fn step() -> StepResult3 { StepResult3::Ok }
    fn current_position(&self) -> &Point3<ftr> { &self.current_position }
    fn previous_position(&self) -> &Point3<ftr> { &self.previous_position }
}
