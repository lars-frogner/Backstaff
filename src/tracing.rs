//! Tracing field lines of a vector field.

pub mod field_line;
pub mod stepping;

use self::stepping::{Stepper3, StepperInstruction, StepperResult, SteppingSense, StoppingCause};
use crate::{
    field::VectorField3,
    geometry::{Point3, Vec3},
    interpolation::Interpolator3,
    num::BFloat,
};

/// Floating-point precision to use for tracing.
#[allow(non_camel_case_types)]
pub type ftr = f64;

/// A tracer result which is either OK or void.
#[derive(Clone, Debug)]
pub enum TracerResult {
    Ok(Option<StoppingCause>),
    Void,
}

/// Traces a field line through a 3D vector field.
///
/// # Parameters
///
/// - `field`: Vector field to trace.
/// - `interpolator`: Interpolator to use.
/// - `stepper`: Stepper to use (will be consumed).
/// - `start_position`: Position where the tracing should start.
/// - `sense`: Whether the field line should be traced in the same or opposite direction as the field.
/// - `callback`: Closure that for each natural step will be called with the displacement vector from the
/// previous to the current position, the current position and the total traced distance.
///
/// # Returns
///
/// A `TracerResult` which is either:
///
/// - `Ok`: Contains an `Option<StoppingCause>`, possibly indicating why tracing was terminated.
/// - `Void`: No field line was traced.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `I`: Type of interpolator.
/// - `S`: Type of stepper.
/// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
pub fn trace_3d_field_line<F, I, St, C>(
    field: &VectorField3<F>,
    interpolator: &I,
    stepper: St,
    start_position: &Point3<ftr>,
    sense: SteppingSense,
    callback: &mut C,
) -> TracerResult
where
    F: BFloat,
    I: Interpolator3,
    St: Stepper3,
    C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
{
    match sense {
        SteppingSense::Same => custom_trace_3d_field_line(
            field,
            interpolator,
            &|dir| {
                dir.normalize();
            },
            stepper,
            start_position,
            callback,
        ),
        SteppingSense::Opposite => custom_trace_3d_field_line(
            field,
            interpolator,
            &|dir| {
                dir.normalize();
                dir.reverse();
            },
            stepper,
            start_position,
            callback,
        ),
    }
}

/// Traces a field line through a 3D vector field, producing regularly spaced output.
///
/// # Parameters
///
/// - `field`: Vector field to trace.
/// - `interpolator`: Interpolator to use.
/// - `stepper`: Stepper to use (will be consumed).
/// - `start_position`: Position where the tracing should start.
/// - `sense`: Whether the field line should be traced in the same or opposite direction as the field.
/// - `callback`: Closure that for each regularly spaced step will be called with the displacement vector from the
/// previous to the current position, the current position and the total traced distance.
///
/// # Returns
///
/// A `TracerResult` which is either:
///
/// - `Ok`: Contains an `Option<StoppingCause>`, possibly indicating why tracing was terminated.
/// - `Void`: No field line was traced.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `I`: Type of interpolator.
/// - `St`: Type of stepper.
/// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
pub fn trace_3d_field_line_dense<F, I, St, C>(
    field: &VectorField3<F>,
    interpolator: &I,
    stepper: St,
    start_position: &Point3<ftr>,
    sense: SteppingSense,
    callback: &mut C,
) -> TracerResult
where
    F: BFloat,
    I: Interpolator3,
    St: Stepper3,
    C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
{
    match sense {
        SteppingSense::Same => custom_trace_3d_field_line_dense(
            field,
            interpolator,
            &|dir| {
                dir.normalize();
            },
            stepper,
            start_position,
            callback,
        ),
        SteppingSense::Opposite => custom_trace_3d_field_line_dense(
            field,
            interpolator,
            &|dir| {
                dir.normalize();
                dir.reverse();
            },
            stepper,
            start_position,
            callback,
        ),
    }
}

/// Traces a field line through a 3D vector field, using a provided closure
/// to compute directions.
///
/// # Parameters
///
/// - `field`: Vector field to trace.
/// - `interpolator`: Interpolator to use.
/// - `direction_computer`: Closure used to compute a stepping direction from a field vector.
/// - `stepper`: Stepper to use (will be consumed).
/// - `start_position`: Position where the tracing should start.
/// - `callback`: Closure that for each natural step will be called with the displacement vector from the
/// previous to the current position, the current position and the total traced distance.
///
/// # Returns
///
/// A `TracerResult` which is either:
///
/// - `Ok`: Contains an `Option<StoppingCause>`, possibly indicating why tracing was terminated.
/// - `Void`: No field line was traced.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `I`: Type of interpolator.
/// - `D`: Function type taking a mutable reference to a field vector.
/// - `St`: Type of stepper.
/// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
pub fn custom_trace_3d_field_line<F, I, D, St, C>(
    field: &VectorField3<F>,
    interpolator: &I,
    direction_computer: &D,
    mut stepper: St,
    start_position: &Point3<ftr>,
    callback: &mut C,
) -> TracerResult
where
    F: BFloat,
    I: Interpolator3,
    D: Fn(&mut Vec3<ftr>),
    St: Stepper3,
    C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
{
    match stepper.place(
        field,
        interpolator,
        direction_computer,
        start_position,
        callback,
    ) {
        StepperResult::Ok(_) => {}
        StepperResult::Stopped(_) => return TracerResult::Void,
    };
    loop {
        if let StepperResult::Stopped(cause) =
            stepper.step(field, interpolator, direction_computer, callback)
        {
            return TracerResult::Ok(Some(cause));
        }
    }
}

/// Traces a field line through a 3D vector field, producing regularly spaced output
/// and using a provided closure to compute directions.
///
/// # Parameters
///
/// - `field`: Vector field to trace.
/// - `interpolator`: Interpolator to use.
/// - `direction_computer`: Closure used to compute a stepping direction from a field vector.
/// - `stepper`: Stepper to use (will be consumed).
/// - `start_position`: Position where the tracing should start.
/// - `callback`: Closure that for each regularly spaced step will be called with the displacement vector from the
/// previous to the current position, the current position and the total traced distance.
///
/// # Returns
///
/// A `TracerResult` which is either:
///
/// - `Ok`: Contains an `Option<StoppingCause>`, possibly indicating why tracing was terminated.
/// - `Void`: No field line was traced.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `I`: Type of interpolator.
/// - `D`: Function type taking a mutable reference to a field vector.
/// - `St`: Type of stepper.
/// - `C`: Mutable function type taking a displacement, a direction, a position and a distance and returning a `StepperInstruction`.
pub fn custom_trace_3d_field_line_dense<F, I, D, St, C>(
    field: &VectorField3<F>,
    interpolator: &I,
    direction_computer: &D,
    mut stepper: St,
    start_position: &Point3<ftr>,
    callback: &mut C,
) -> TracerResult
where
    F: BFloat,
    I: Interpolator3,
    D: Fn(&mut Vec3<ftr>),
    St: Stepper3,
    C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
{
    match stepper.place(
        field,
        interpolator,
        direction_computer,
        start_position,
        callback,
    ) {
        StepperResult::Ok(_) => {}
        StepperResult::Stopped(_) => return TracerResult::Void,
    };
    loop {
        if let StepperResult::Stopped(cause) =
            stepper.step_dense_output(field, interpolator, direction_computer, callback)
        {
            return TracerResult::Ok(Some(cause));
        }
    }
}
