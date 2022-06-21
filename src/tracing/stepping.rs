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

/// Defines the properties of a stepping scheme.
pub trait Stepper3: Clone {
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
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
    fn place<F, C>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        position: &Point3<ftr>,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction;

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
    /// - `D`: Function type taking a mutable reference to a field vector.
    /// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
    fn step<F, C>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction;

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
    /// - `D`: Function type taking a mutable reference to a field vector.
    /// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
    fn step_dense_output<F, C>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction;

    /// Returns a reference to the current stepper position.
    fn position(&self) -> &Point3<ftr>;

    /// Retuns the current distance of the stepper along the field line.
    fn distance(&self) -> ftr;
}

/// Defines the properties of a 3D stepper factory structure.
pub trait StepperFactory3 {
    type Output: Stepper3;

    /// Creates a new 3D stepper.
    fn produce(&self) -> Self::Output;
}
