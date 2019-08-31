//! Tracing field lines of a Bifrost vector field.

pub mod seeding;
pub mod stepping;
pub mod field_line;

use num;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::VectorField3;
use crate::interpolation::Interpolator3;
use self::stepping::{Stepper3, SteppingSense, StepperResult, StepperInstruction};

/// Floating-point precision to use for tracing.
#[allow(non_camel_case_types)]
pub type ftr = f64;

/// Traces a field line through a 3D vector field.
///
/// # Parameters
///
/// - `field`: Vector field to trace.
/// - `interpolator`: Interpolator to use.
/// - `stepper`: Stepper to use (will be consumed).
/// - `start_position`: Position where the tracing should start.
/// - `sense`: Whether the field line should be traced in the same or opposite direction as the field.
/// - `callback`: Closure that will be called with the natural position of each step.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `G`: Type of grid.
/// - `I`: Type of interpolator.
/// - `S`: Type of stepper.
/// - `C`: Mutable function type taking a reference to a position and returning a `StepperInstruction`.
pub fn trace_3d_field_line<F, G, I, S, C>(field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>, sense: SteppingSense, callback: &mut C)
where F: num::Float + std::fmt::Display,
        G: Grid3<F> + Clone,
        I: Interpolator3,
        S: Stepper3,
        C: FnMut(&Point3<ftr>) -> StepperInstruction
{
    match sense {
        SteppingSense::Same => {
            custom_trace_3d_field_line(field, interpolator, &|dir| { dir.normalize(); }, stepper, start_position, callback);
        },
        SteppingSense::Opposite => {
            custom_trace_3d_field_line(field, interpolator, &|dir| { dir.normalize(); dir.reverse(); }, stepper, start_position, callback);
        },
    };
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
/// - `callback`: Closure that will be called with regularly spaced positions along the field line.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `G`: Type of grid.
/// - `I`: Type of interpolator.
/// - `S`: Type of stepper.
/// - `C`: Mutable function type taking a reference to a position and returning a `StepperInstruction`.
pub fn trace_3d_field_line_dense<F, G, I, S, C>(field: &VectorField3<F, G>, interpolator: &I, stepper: S, start_position: &Point3<ftr>, sense: SteppingSense, callback: &mut C)
where F: num::Float + std::fmt::Display,
        G: Grid3<F> + Clone,
        I: Interpolator3,
        S: Stepper3,
        C: FnMut(&Point3<ftr>) -> StepperInstruction
{
    match sense {
        SteppingSense::Same => {
            custom_trace_3d_field_line_dense(field, interpolator, &|dir| { dir.normalize(); }, stepper, start_position, callback);
        },
        SteppingSense::Opposite => {
            custom_trace_3d_field_line_dense(field, interpolator, &|dir| { dir.normalize(); dir.reverse(); }, stepper, start_position, callback);
        },
    };
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
/// - `callback`: Closure that will be called with the natural position of each step.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `G`: Type of grid.
/// - `I`: Type of interpolator.
/// - `D`: Function type taking a mutable reference to a field vector.
/// - `S`: Type of stepper.
/// - `C`: Mutable function type taking a reference to a position and returning a `StepperInstruction`.
pub fn custom_trace_3d_field_line<F, G, I, D, S, C>(field: &VectorField3<F, G>, interpolator: &I, direction_computer: &D, mut stepper: S, start_position: &Point3<ftr>, callback: &mut C)
where F: num::Float + std::fmt::Display,
        G: Grid3<F> + Clone,
        I: Interpolator3,
        D: Fn(&mut Vec3<ftr>),
        S: Stepper3,
        C: FnMut(&Point3<ftr>) -> StepperInstruction
{
    match stepper.place(field, interpolator, direction_computer, start_position, callback) {
        StepperResult::Ok(_) => {},
        StepperResult::Stopped(_) => return
    };
    while let StepperResult::Ok(_) = stepper.step(field, interpolator, direction_computer, callback) {}
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
/// - `callback`: Closure that will be called with regularly spaced positions along the field line.
///
/// # Type parameters
///
/// - `F`: Floating point type of the field data.
/// - `G`: Type of grid.
/// - `I`: Type of interpolator.
/// - `D`: Function type taking a mutable reference to a field vector.
/// - `S`: Type of stepper.
/// - `C`: Mutable function type taking a reference to a position and returning a `StepperInstruction`.
pub fn custom_trace_3d_field_line_dense<F, G, I, D, S, C>(field: &VectorField3<F, G>, interpolator: &I, direction_computer: &D, mut stepper: S, start_position: &Point3<ftr>, callback: &mut C)
where F: num::Float + std::fmt::Display,
        G: Grid3<F> + Clone,
        I: Interpolator3,
        D: Fn(&mut Vec3<ftr>),
        S: Stepper3,
        C: FnMut(&Point3<ftr>) -> StepperInstruction
{
    match stepper.place(field, interpolator, direction_computer, start_position, callback) {
        StepperResult::Ok(_) => {},
        StepperResult::Stopped(_) => return
    };
    while let StepperResult::Ok(_) = stepper.step_dense_output(field, interpolator, direction_computer, callback) {}
}