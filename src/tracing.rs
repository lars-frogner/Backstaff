//! Tracing field lines of a vector field.

pub mod field_line;
pub mod stepping;

use self::stepping::{Stepper3, StepperResult, SteppingCallback, SteppingSense, StoppingCause};
use crate::{field::VectorField3, geometry::Point3, interpolation::Interpolator3, num::BFloat};

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
/// - `S`: Type of stepper.
pub fn trace_3d_field_line<F, St>(
    field: &VectorField3<F>,
    interpolator: &dyn Interpolator3<F>,
    mut stepper: St,
    start_position: &Point3<ftr>,
    sense: SteppingSense,
    callback: &mut SteppingCallback,
) -> TracerResult
where
    F: BFloat,
    St: Stepper3,
{
    match stepper.place(field, interpolator, sense, start_position, callback) {
        StepperResult::Ok(_) => {}
        StepperResult::Stopped(_) => return TracerResult::Void,
    };
    loop {
        if let StepperResult::Stopped(cause) = stepper.step(field, interpolator, sense, callback) {
            return TracerResult::Ok(Some(cause));
        }
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
/// - `St`: Type of stepper.
pub fn trace_3d_field_line_dense<F, St>(
    field: &VectorField3<F>,
    interpolator: &dyn Interpolator3<F>,
    mut stepper: St,
    start_position: &Point3<ftr>,
    sense: SteppingSense,
    callback: &mut SteppingCallback,
) -> TracerResult
where
    F: BFloat,
    St: Stepper3,
{
    match stepper.place(field, interpolator, sense, start_position, callback) {
        StepperResult::Ok(_) => {}
        StepperResult::Stopped(_) => return TracerResult::Void,
    };
    loop {
        if let StepperResult::Stopped(cause) =
            stepper.step_dense_output(field, interpolator, sense, callback)
        {
            return TracerResult::Ok(Some(cause));
        }
    }
}
