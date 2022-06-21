//! Stepping using Runge–Kutta–Fehlberg methods,
//! a set of RK methods with step size adaptation driven by
//! error estimation through an embedded lower-order step.

pub mod rkf23;
pub mod rkf45;

use super::{StepperInstruction, StepperResult, SteppingSense, StoppingCause};
use crate::{
    field::{FieldGrid3, VectorField3},
    geometry::{
        Dim3::{X, Y, Z},
        Point3, Vec3,
    },
    grid::{Grid3, GridPointQuery3},
    interpolation::Interpolator3,
    num::BFloat,
    tracing::ftr,
};

/// Type of RKF stepper.
#[derive(Clone, Copy, Debug)]
pub enum RKFStepperType {
    RKF23,
    RKF45,
}

#[derive(Clone, Debug)]
struct RKFStepperState3 {
    /// Configuration parameters for the stepper.
    config: RKFStepperConfig,
    /// PI control parameters for the stepper.
    pi_control: PIControlParams,
    /// Current position of the stepper.
    position: Point3<ftr>,
    /// Field direction at the current position of the stepper.
    direction: Vec3<ftr>,
    /// Current distance of the stepper along the field line.
    distance: ftr,
    /// Step length to use in the next step.
    step_length: ftr,
    /// The estimated error of the step from the previous to the current position.
    error: ftr,
    /// How many consecutive successful steps have been in opposite directions.
    n_sudden_reversals: u32,
    /// The step size used to get from the previous to the current position.
    previous_step_length: ftr,
    /// Position of the stepper directly before the previous step was taken.
    previous_position: Point3<ftr>,
    /// Field direction at the previous position of the stepper.
    previous_direction: Vec3<ftr>,
    /// Intermediate step directions used during the previous step.
    intermediate_directions: Vec<Vec3<ftr>>,
    /// Position of the stepper directly before the previous step was taken,
    /// not wrapped around periodic boundaries.
    previous_unwrapped_position: Point3<ftr>,
    /// Displacement vector from the previous to the current position
    /// (with periodic boundaries taken into account).
    previous_step_displacement: Vec3<ftr>,
    /// Whether the previous step had to wrap around a periodic boundary.
    previous_step_wrapped: bool,
    /// Distance along the field line where the next dense output position
    /// should be computed.
    next_output_distance: ftr,
    /// Previous dense output was position, not wrapped around periodic boundaries.
    previous_unwrapped_output_position: Point3<ftr>,
}

/// Configuration parameters for RKF steppers.
#[derive(Clone, Debug)]
pub struct RKFStepperConfig {
    /// Step length to use for dense (uniform) output positions.
    pub dense_step_length: ftr,
    /// Maximum number of step attempts before terminating.
    pub max_step_attempts: u32,
    /// Absolute error tolerance.
    pub absolute_tolerance: ftr,
    /// Relative error tolerance.
    pub relative_tolerance: ftr,
    /// Scaling factor for the error to reduce oscillations.
    pub safety_factor: ftr,
    /// Smallest allowed scaling of the step size in one step.
    pub min_step_scale: ftr,
    /// Largest allowed scaling of the step size in one step.
    pub max_step_scale: ftr,
    /// Start value for error.
    pub initial_error: ftr,
    /// Initial step size.
    pub initial_step_length: ftr,
    /// Number of sudden direction reversals before the area is considered a sink.
    pub sudden_reversals_for_sink: u32,
    /// Whether to use Proportional Integral (PI) control for stabilizing the stepping.
    pub use_pi_control: bool,
}

#[derive(Clone, Debug)]
struct PIControlParams {
    k_i: ftr,
    k_p: ftr,
}

#[derive(Clone, Debug)]
enum ComputedDirection3 {
    Standard(Vec3<ftr>),
    WithWrappedPosition((Point3<ftr>, Vec3<ftr>)),
}

#[derive(Clone, Debug)]
enum StepError {
    Acceptable(ftr),
    TooLarge(ftr),
}

#[derive(Clone, Debug)]
struct StepAttempt3 {
    next_position: Point3<ftr>,
    next_direction: Vec3<ftr>,
    intermediate_directions: Vec<Vec3<ftr>>,
    step_displacement: Vec3<ftr>,
    step_wrapped: bool,
}

trait RKFStepper3 {
    fn state(&self) -> &RKFStepperState3;
    fn state_mut(&mut self) -> &mut RKFStepperState3;

    fn attempt_step<F>(
        &self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
    ) -> StepperResult<StepAttempt3>
    where
        F: BFloat;

    fn compute_error_deltas(&self, attempt: &StepAttempt3) -> Vec3<ftr>;

    fn compute_dense_interpolation_coefs(&self) -> Vec<Vec3<ftr>>;

    /// Computes the interpolated position at the given fraction between
    /// the previous and current natural output position.
    ///
    /// The returned position is not wrapped around periodic boundaries.
    fn interpolate_dense_position(&self, coefs: &[Vec3<ftr>], fraction: ftr) -> Point3<ftr>;

    fn reset_state(&mut self, position: &Point3<ftr>, direction: &Vec3<ftr>) {
        let state = self.state_mut();
        state.position = position.clone();
        state.direction = direction.clone();
        state.distance = 0.0;
        state.step_length = state.config.initial_step_length;
        state.error = state.config.initial_error;
        state.n_sudden_reversals = 0;
        state.previous_step_length = 0.0;
        state.previous_position = position.clone();
        state.previous_direction = direction.clone();
        state.intermediate_directions = Vec::new();
        state.previous_unwrapped_position = position.clone();
        state.previous_step_displacement = Vec3::zero();
        state.previous_step_wrapped = false;
        state.next_output_distance = state.config.dense_step_length;
        state.previous_unwrapped_output_position = position.clone();
    }

    fn place_with_callback<F, C>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        position: &Point3<ftr>,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        let place_result = self.perform_place(field, interpolator, sense, position);
        if let StepperResult::Ok(_) = place_result {
            if let StepperInstruction::Terminate = callback(
                &self.state().previous_step_displacement,
                &self.state().direction,
                &self.state().position,
                self.state().distance,
            ) {
                return StepperResult::Stopped(StoppingCause::StoppedByCallback);
            }
        }
        place_result
    }

    fn step_with_callback<F, C>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        let step_result = self.perform_step(field, interpolator, sense);
        if let StepperResult::Ok(_) = step_result {
            let effective_direction =
                &self.state().previous_step_displacement / self.state().previous_step_length;
            if let StepperInstruction::Terminate = callback(
                &self.state().previous_step_displacement,
                &effective_direction,
                &self.state().position,
                self.state().distance,
            ) {
                return StepperResult::Stopped(StoppingCause::StoppedByCallback);
            }
        }
        step_result
    }

    fn step_with_callback_dense_output<F, C>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        let step_result = self.perform_step(field, interpolator, sense);
        if let StepperResult::Ok(_) = step_result {
            self.compute_dense_output(field.grid(), callback)
        } else {
            step_result
        }
    }

    fn perform_place<F>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        position: &Point3<ftr>,
    ) -> StepperResult<()>
    where
        F: BFloat,
    {
        match Self::compute_direction(field, interpolator, sense, position) {
            StepperResult::Ok(ComputedDirection3::Standard(direction)) => {
                self.reset_state(position, &direction)
            }
            StepperResult::Ok(ComputedDirection3::WithWrappedPosition((
                wrapped_position,
                direction,
            ))) => self.reset_state(&wrapped_position, &direction),
            StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
        };
        StepperResult::Ok(())
    }

    fn perform_step<F>(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
    ) -> StepperResult<()>
    where
        F: BFloat,
    {
        let grid = field.grid();
        let mut attempts = 0;

        while attempts < self.state().config.max_step_attempts {
            let step_attempt = match self.attempt_step(field, interpolator, sense) {
                StepperResult::Ok(step_attempt) => step_attempt,
                StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
            };

            attempts += 1;

            match self.compute_error(grid, &step_attempt) {
                StepError::Acceptable(new_error) => {
                    let mut new_step_length = self.compute_step_length_accepted(new_error);

                    // Don't increase step size if the previous attempt was rejected
                    if attempts > 1 && new_step_length > self.state().step_length {
                        new_step_length = self.state().step_length;
                    }

                    if self.check_for_sink(&step_attempt) {
                        return StepperResult::Stopped(StoppingCause::Sink);
                    }

                    self.apply_step_attempt(step_attempt);
                    self.update_step_length(new_step_length, new_error);
                    break;
                }
                StepError::TooLarge(new_error) => {
                    let new_step_length = self.compute_step_length_rejected(new_error);
                    self.update_step_length(new_step_length, new_error);
                }
            };
        }

        if attempts < self.state().config.max_step_attempts {
            StepperResult::Ok(())
        } else {
            StepperResult::Stopped(StoppingCause::TooManyAttempts)
        }
    }

    fn compute_direction<F>(
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        position: &Point3<ftr>,
    ) -> StepperResult<ComputedDirection3>
    where
        F: BFloat,
    {
        match interpolator.interp_vector_field(field, &Point3::from(position)) {
            GridPointQuery3::Inside(field_vector) => {
                if field_vector.is_zero() {
                    StepperResult::Stopped(StoppingCause::Null)
                } else {
                    StepperResult::Ok(ComputedDirection3::Standard(
                        Self::compute_direction_from_field_vector(field_vector, sense),
                    ))
                }
            }
            GridPointQuery3::MovedInside((field_vector, wrapped_position)) => {
                if field_vector.is_zero() {
                    StepperResult::Stopped(StoppingCause::Null)
                } else {
                    StepperResult::Ok(ComputedDirection3::WithWrappedPosition((
                        Point3::from(&wrapped_position),
                        Self::compute_direction_from_field_vector(field_vector, sense),
                    )))
                }
            }
            GridPointQuery3::Outside => StepperResult::Stopped(StoppingCause::OutOfBounds),
        }
    }

    fn compute_direction_from_field_vector<F>(
        field_vector: Vec3<F>,
        sense: SteppingSense,
    ) -> Vec3<ftr>
    where
        F: BFloat,
    {
        let mut direction = Vec3::from(&field_vector);
        direction.normalize();
        if sense == SteppingSense::Opposite {
            direction.reverse();
        }
        direction
    }

    fn compute_error(&self, grid: &FieldGrid3, attempt: &StepAttempt3) -> StepError {
        let state = self.state();
        let error_deltas = self.compute_error_deltas(attempt);

        let grid_extents: Vec3<ftr> = Vec3::from(grid.extents());
        let errors = Vec3::new(
            error_deltas[X]
                / (state.config.absolute_tolerance
                    + state.config.relative_tolerance * grid_extents[X]),
            error_deltas[Y]
                / (state.config.absolute_tolerance
                    + state.config.relative_tolerance * grid_extents[Y]),
            error_deltas[Z]
                / (state.config.absolute_tolerance
                    + state.config.relative_tolerance * grid_extents[Z]),
        );

        let error = ftr::sqrt(0.5 * errors.squared_length());

        if error <= 1.0 {
            StepError::Acceptable(error)
        } else {
            StepError::TooLarge(error)
        }
    }

    fn compute_step_length_accepted(&self, new_error: ftr) -> ftr {
        let state = self.state();
        let step_scale = if new_error < 1e-9 {
            // Use max step scale directly for very small error to avoid division by zero
            state.config.max_step_scale
        } else {
            let step_scale = state.config.safety_factor * (state.error.powf(state.pi_control.k_i))
                / (new_error.powf(state.pi_control.k_p));
            if step_scale < state.config.min_step_scale {
                state.config.min_step_scale
            } else if step_scale > state.config.max_step_scale {
                state.config.max_step_scale
            } else {
                step_scale
            }
        };
        state.step_length * step_scale
    }

    fn compute_step_length_rejected(&self, new_error: ftr) -> ftr {
        let state = self.state();
        ftr::max(
            state.config.safety_factor / (new_error.powf(state.pi_control.k_p)),
            state.config.min_step_scale,
        ) * state.step_length
    }

    fn check_for_sink(&mut self, attempt: &StepAttempt3) -> bool {
        let state = self.state_mut();
        if attempt.next_direction.dot(&state.direction) < 0.0 {
            state.n_sudden_reversals += 1;
            state.n_sudden_reversals >= state.config.sudden_reversals_for_sink
        } else {
            state.n_sudden_reversals = 0;
            false
        }
    }

    fn apply_step_attempt(&mut self, attempt: StepAttempt3) {
        let state = self.state_mut();
        state.previous_position = state.position.clone();
        state.previous_direction = state.direction.clone();
        state.position = attempt.next_position;
        state.direction = attempt.next_direction;
        // Advance distance with step size *prior to* calling `update_step_length`
        state.distance += state.step_length;
        state.intermediate_directions = attempt.intermediate_directions;
        // Advance previous unwrapped position before updating previous step displacement.
        state.previous_unwrapped_position =
            &state.previous_unwrapped_position + &state.previous_step_displacement;
        state.previous_step_displacement = attempt.step_displacement;
        state.previous_step_wrapped = attempt.step_wrapped;
    }

    fn update_step_length(&mut self, new_step_length: ftr, new_error: ftr) {
        let state = self.state_mut();
        state.previous_step_length = state.step_length;
        state.step_length = new_step_length;
        state.error = new_error;
    }

    fn compute_dense_output<C>(&mut self, grid: &FieldGrid3, callback: &mut C) -> StepperResult<()>
    where
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        #![allow(clippy::float_cmp)] // Allows the float comparison with zero
        let state = self.state();
        let previous_distance = state.distance - state.previous_step_length;
        debug_assert_ne!(state.previous_step_length, 0.0);
        debug_assert!(state.next_output_distance > previous_distance);

        let mut next_output_distance = state.next_output_distance;
        let mut previous_unwrapped_output_position =
            state.previous_unwrapped_output_position.clone();

        if next_output_distance <= state.distance {
            let coefs = self.compute_dense_interpolation_coefs();
            loop {
                let fraction =
                    (next_output_distance - previous_distance) / state.previous_step_length;
                let unwrapped_output_position = self.interpolate_dense_position(&coefs, fraction);
                let output_position =
                    match grid.wrap_point(&Point3::from(&unwrapped_output_position)) {
                        Some(ref position) => Point3::from(position),
                        None => return StepperResult::Stopped(StoppingCause::OutOfBounds),
                    };
                let dense_step_displacement =
                    &unwrapped_output_position - &previous_unwrapped_output_position;
                previous_unwrapped_output_position = unwrapped_output_position;
                let effective_direction = &dense_step_displacement / state.config.dense_step_length;
                if let StepperInstruction::Terminate = callback(
                    &dense_step_displacement,
                    &effective_direction,
                    &output_position,
                    next_output_distance,
                ) {
                    return StepperResult::Stopped(StoppingCause::StoppedByCallback);
                }
                next_output_distance += state.config.dense_step_length;
                if next_output_distance > state.distance {
                    break;
                }
            }
        }

        // Reborrow state as mutable to update distance and position
        let state = self.state_mut();
        state.next_output_distance = next_output_distance;
        state.previous_unwrapped_output_position = previous_unwrapped_output_position;

        StepperResult::Ok(())
    }
}

impl RKFStepperConfig {
    pub const DEFAULT_DENSE_STEP_LENGTH: ftr = 1e-2;
    pub const DEFAULT_MAX_STEP_ATTEMPTS: u32 = 16;
    pub const DEFAULT_ABSOLUTE_TOLERANCE: ftr = 1e-6;
    pub const DEFAULT_RELATIVE_TOLERANCE: ftr = 1e-6;
    pub const DEFAULT_SAFETY_FACTOR: ftr = 0.9;
    pub const DEFAULT_MIN_STEP_SCALE: ftr = 0.2;
    pub const DEFAULT_MAX_STEP_SCALE: ftr = 10.0;
    pub const DEFAULT_INITIAL_ERROR: ftr = 1e-4;
    pub const DEFAULT_INITIAL_STEP_LENGTH: ftr = 1e-4;
    pub const DEFAULT_SUDDEN_REVERSALS_FOR_SINK: u32 = 3;
    pub const DEFAULT_USE_PI_CONTROL: bool = true;

    fn validate(&self) {
        assert!(
            self.dense_step_length > 0.0,
            "Dense step size must be larger than zero."
        );
        assert!(
            self.max_step_attempts > 0,
            "Maximum number of step attempts must be larger than zero."
        );
        assert!(
            self.absolute_tolerance > 0.0,
            "Absolute error tolerance must be larger than zero."
        );
        assert!(
            self.relative_tolerance >= 0.0,
            "Relative error tolerance must be larger than or equal to zero."
        );
        assert!(
            self.safety_factor > 0.0 && self.safety_factor <= 1.0,
            "Safety factor must be in the range (0, 1]."
        );
        assert!(
            self.min_step_scale > 0.0,
            "Minimum step scale must be larger than zero."
        );
        assert!(
            self.max_step_scale >= self.min_step_scale,
            "Maximum step scale must be larger than or equal to the minimum step scale."
        );
        assert!(
            self.initial_step_length > 0.0,
            "Initial step size must be larger than zero."
        );
        assert!(
            self.initial_error > 0.0 && self.initial_error <= 1.0,
            "Initial error must be in the range (0, 1]."
        );
        assert!(
            self.sudden_reversals_for_sink > 0,
            "Number of sudden reversals for sink must be larger than zero."
        );
    }
}

impl Default for RKFStepperConfig {
    fn default() -> Self {
        RKFStepperConfig {
            dense_step_length: Self::DEFAULT_DENSE_STEP_LENGTH,
            max_step_attempts: Self::DEFAULT_MAX_STEP_ATTEMPTS,
            absolute_tolerance: Self::DEFAULT_ABSOLUTE_TOLERANCE,
            relative_tolerance: Self::DEFAULT_RELATIVE_TOLERANCE,
            safety_factor: Self::DEFAULT_SAFETY_FACTOR,
            min_step_scale: Self::DEFAULT_MIN_STEP_SCALE,
            max_step_scale: Self::DEFAULT_MAX_STEP_SCALE,
            initial_step_length: Self::DEFAULT_INITIAL_STEP_LENGTH,
            initial_error: Self::DEFAULT_INITIAL_ERROR,
            sudden_reversals_for_sink: Self::DEFAULT_SUDDEN_REVERSALS_FOR_SINK,
            use_pi_control: Self::DEFAULT_USE_PI_CONTROL,
        }
    }
}

impl PIControlParams {
    fn activated(scheme_order: u8) -> Self {
        #[allow(clippy::cast_lossless)]
        let order = scheme_order as ftr;
        let k_i = 0.4 / order;
        let k_p = 1.0 / order - 0.75 * k_i;
        PIControlParams { k_i, k_p }
    }

    fn deactivated(scheme_order: u8) -> Self {
        #[allow(clippy::cast_lossless)]
        let order = scheme_order as ftr;
        let k_i = 0.0;
        let k_p = 1.0 / order;
        PIControlParams { k_i, k_p }
    }
}
