//! Tracing field lines of a Bifrost vector field.

pub mod seeding;
pub mod stepping;

use num;
use crate::geometry::{Point3};
use crate::grid::Grid3;
use crate::field::VectorField3;
use crate::interpolation::Interpolator3;
use self::stepping::{Stepper3, StepperResult, StepperInstruction};

/// Floating-point precision to use for tracing.
#[allow(non_camel_case_types)]
pub type ftr = f64;

pub struct Tracer3;

impl Tracer3 {
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

    pub fn trace_dense_output<F, G, I, S, C>(field: &VectorField3<F, G>, interpolator: &I, mut stepper: S, start_position: &Point3<ftr>, callback: &mut C)
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
}