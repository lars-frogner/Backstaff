//! Tracing field lines of a Bifrost vector field.

pub mod seeding;
pub mod stepping;
pub mod field_line;

use num;
use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::field::VectorField3;
use crate::interpolation::Interpolator3;
use self::stepping::{Stepper3, StepperResult, StepperInstruction};

/// Floating-point precision to use for tracing.
#[allow(non_camel_case_types)]
pub type ftr = f64;

/// Traces a field line through the given vector field from the given start position,
/// using the given interpolator and stepper. The given callback closure is called with
/// the natural position of each step.
pub fn trace<F, G, I, S, C>(field: &VectorField3<F, G>, interpolator: &I, mut stepper: S, start_position: &Point3<ftr>, callback: &mut C)
where F: num::Float + std::fmt::Display,
        G: Grid3<F> + Clone,
        I: Interpolator3,
        S: Stepper3,
        C: FnMut(&Point3<ftr>) -> StepperInstruction
{
    match stepper.place(field, interpolator, start_position, callback) {
        StepperResult::Ok(_) => {},
        StepperResult::Stopped(_) => return
    };
    while let StepperResult::Ok(_) = stepper.step(field, interpolator, callback) {}
}

/// Traces a field line through the given vector field from the given start position,
/// using the given interpolator and stepper. The given callback closure is called with
/// regularly spaced positions along the field line.
pub fn trace_dense<F, G, I, S, C>(field: &VectorField3<F, G>, interpolator: &I, mut stepper: S, start_position: &Point3<ftr>, callback: &mut C)
where F: num::Float + std::fmt::Display,
        G: Grid3<F> + Clone,
        I: Interpolator3,
        S: Stepper3,
        C: FnMut(&Point3<ftr>) -> StepperInstruction
{
    match stepper.place(field, interpolator, start_position, callback) {
        StepperResult::Ok(_) => {},
        StepperResult::Stopped(_) => return
    };
    while let StepperResult::Ok(_) = stepper.step_dense_output(field, interpolator, callback) {}
}