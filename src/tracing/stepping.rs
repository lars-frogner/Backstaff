//! Stepping along field lines of a Bifrost vector field.

pub mod rkf;

use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::field::VectorField3;
use crate::interpolation::Interpolator3;
use super::ftr;

/// A stepper result which is either OK (with an an abitrary value) or stopped (with a cause).
pub enum StepperResult<T> {
    Ok(T),
    Stopped(StoppingCause)
}

/// Reason for terminating stepping.
pub enum StoppingCause {
    Null,
    Sink,
    OutOfBounds,
    TooManyAttempts,
    StoppedByCallback
}

/// Lets the stepper callback communicate whether tracing should
/// continue or terminate.
pub enum StepperInstruction {
    Continue,
    Terminate
}

/// Defines the properties of a stepping scheme.
pub trait Stepper3 {
    /// Places the stepper at the given position in the field,
    /// and calls the callback with the position if successful.
    fn place<F, G, I, C>(&mut self, field: &VectorField3<F, G>, interpolator: &I, position: &Point3<ftr>, callback: &mut C) -> StepperResult<()>
    where F: num::Float + std::fmt::Display,
          G: Grid3<F> + Clone,
          I: Interpolator3,
          C: FnMut(&Point3<ftr>) -> StepperInstruction;

    /// Performs a step and calls the callback with the resulting position if successful.
    fn step<F, G, I, C>(&mut self, field: &VectorField3<F, G>, interpolator: &I, callback: &mut C) -> StepperResult<()>
    where F: num::Float + std::fmt::Display,
          G: Grid3<F> + Clone,
          I: Interpolator3,
          C: FnMut(&Point3<ftr>) -> StepperInstruction;

    /// Performs a step and calls the callback with the resulting dense positions if successful.
    fn step_dense_output<F, G, I, C>(&mut self, field: &VectorField3<F, G>, interpolator: &I, callback: &mut C) -> StepperResult<()>
    where F: num::Float + std::fmt::Display,
          G: Grid3<F> + Clone,
          I: Interpolator3,
          C: FnMut(&Point3<ftr>) -> StepperInstruction;

    /// Returns a reference to the current stepper position.
    fn position(&self) -> &Point3<ftr>;

    /// Retuns the current distance of the stepper along the field line.
    fn distance(&self) -> ftr;
}