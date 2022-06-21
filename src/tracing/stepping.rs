//! Stepping along field lines of a vector field.

pub mod rkf;

use super::ftr;
use crate::{
    field::VectorField3,
    geometry::{Point3, Vec3},
    interpolation::Interpolator3,
    num::BFloat,
};

/// Stepping along the field line in the same direction as the field or opposite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteppingSense {
    Same,
    Opposite,
}

/// A stepper result which is either OK (with an an abitrary value) or stopped (with a cause).
#[derive(Clone, Debug)]
pub enum StepperResult<T> {
    Ok(T),
    Stopped(StoppingCause),
}

/// Reason for terminating stepping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoppingCause {
    Null,
    Sink,
    OutOfBounds,
    TooManyAttempts,
    StoppedByCallback,
}

/// Lets the stepper callback communicate whether tracing should
/// continue or terminate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepperInstruction {
    Continue,
    Terminate,
}

pub type DynStepper3<F> = Box<dyn Stepper3<F>>;

pub type SteppingCallback<'a> =
    dyn 'a + FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction;

/// Defines the properties of a stepping scheme.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
pub trait Stepper3<F>: Sync {
    /// Places the stepper inside the field.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to step in.
    /// - `interpolator`: Interpolator to use.
    /// - `sense`: Whether the field line should be traced in the same or opposite direction as the field.
    /// - `position`: Position where the stepper should be placed.
    /// - `callback`: Closure that will be called with a zero-length displacement vector,
    /// the placed position and a zero-valued distance, if successful.
    ///
    /// # Returns
    ///
    /// A `StepperResult<()>` which is either:
    ///
    /// - `Ok`: Stepper placement succeeded.
    /// - `Stopped`: Contains a `StoppingCause` indicating why stepper placement failed.
    fn place(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        position: &Point3<ftr>,
        callback: &mut SteppingCallback,
    ) -> StepperResult<()>
    where
        F: BFloat;

    /// Performs a step.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to step in.
    /// - `interpolator`: Interpolator to use.
    /// - `sense`: Whether the field line should be traced in the same or opposite direction as the field.
    /// - `callback`: Closure that will be called with the displacement vector from the previous to
    /// the current position, the current position and the total traced distance, if successful.
    ///
    /// # Returns
    ///
    /// A `StepperResult<()>` which is either:
    ///
    /// - `Ok`: Stepper placement succeeded.
    /// - `Stopped`: Contains a `StoppingCause` indicating why the step failed.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn step(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut SteppingCallback,
    ) -> StepperResult<()>
    where
        F: BFloat;

    /// Performs a step, producing regularly spaced output positions.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to step in.
    /// - `interpolator`: Interpolator to use.
    /// - `sense`: Whether the field line should be traced in the same or opposite direction as the field.
    /// - `callback`: Closure that will be called with the displacement vector from the previous to
    /// the current output position, the current output position and the total traced distance, if successful.
    ///
    /// # Returns
    ///
    /// A `StepperResult<()>` which is either:
    ///
    /// - `Ok`: Stepper placement succeeded.
    /// - `Stopped`: Contains a `StoppingCause` indicating why the step failed.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    fn step_dense_output(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut SteppingCallback,
    ) -> StepperResult<()>
    where
        F: BFloat;

    /// Returns a reference to the current stepper position.
    fn position(&self) -> &Point3<ftr>;

    /// Retuns the current distance of the stepper along the field line.
    fn distance(&self) -> ftr;

    /// Returns a mutable reference to a clone of this stepper living on the heap.
    fn heap_clone(&self) -> DynStepper3<F>;
}
