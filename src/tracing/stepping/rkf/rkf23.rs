//! Stepping using the the Bogacki–Shampine scheme,
//! a third-order Runge-Kutta method with error
//! estimation through an embedded second-order step.

use super::{
    super::{Stepper3, StepperFactory3, StepperResult, SteppingCallback, SteppingSense},
    ComputedDirection3, PIControlParams, RKFStepper3, RKFStepperConfig, RKFStepperState3,
    StepAttempt3,
};
use crate::{
    field::VectorField3,
    geometry::{Point3, Vec3},
    interpolation::Interpolator3,
    num::BFloat,
    tracing::ftr,
};

/// A stepper using the third order Runge–Kutta–Fehlberg method.
#[derive(Clone, Debug)]
pub struct RKF23Stepper3(RKFStepperState3);

/// Factory for `RKF23Stepper3` objects.
#[derive(Clone, Debug)]
pub struct RKF23StepperFactory3 {
    config: RKFStepperConfig,
}

impl RKF23Stepper3 {
    const ORDER: u8 = 3;
    const N_INTERMEDIATE_STEPS: usize = 2;

    const A21: ftr = 1.0 / 2.0;
    const A32: ftr = 3.0 / 4.0;
    const A41: ftr = 2.0 / 9.0;
    const A42: ftr = 1.0 / 3.0;
    const A43: ftr = 4.0 / 9.0;

    const E1: ftr = -5.0 / 72.0;
    const E2: ftr = 1.0 / 12.0;
    const E3: ftr = 1.0 / 9.0;
    const E4: ftr = -1.0 / 8.0;

    /// Creates a new RKF23 stepper with the given configuration.
    pub fn new(config: RKFStepperConfig) -> Self {
        config.validate();

        let pi_control = if config.use_pi_control {
            PIControlParams::activated(Self::ORDER)
        } else {
            PIControlParams::deactivated(Self::ORDER)
        };
        let position = Point3::origin();
        let direction = Vec3::zero();
        let distance = 0.0;
        let step_length = config.initial_step_length;
        let error = config.initial_error;
        let n_sudden_reversals = 0;
        let previous_step_length = 0.0;
        let previous_position = Point3::origin();
        let previous_direction = Vec3::zero();
        let intermediate_directions = Vec::with_capacity(Self::N_INTERMEDIATE_STEPS);
        let previous_unwrapped_position = Point3::origin();
        let previous_step_displacement = Vec3::zero();
        let previous_step_wrapped = false;
        let next_output_distance = config.dense_step_length;
        let previous_unwrapped_output_position = Point3::origin();

        RKF23Stepper3(RKFStepperState3 {
            config,
            pi_control,
            position,
            direction,
            distance,
            step_length,
            error,
            n_sudden_reversals,
            previous_step_length,
            previous_position,
            previous_direction,
            intermediate_directions,
            previous_unwrapped_position,
            previous_step_displacement,
            previous_step_wrapped,
            next_output_distance,
            previous_unwrapped_output_position,
        })
    }
}

impl RKFStepper3 for RKF23Stepper3 {
    fn state(&self) -> &RKFStepperState3 {
        &self.0
    }
    fn state_mut(&mut self) -> &mut RKFStepperState3 {
        &mut self.0
    }

    fn attempt_step<F>(
        &self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
    ) -> StepperResult<StepAttempt3>
    where
        F: BFloat,
    {
        let state = self.state();

        let mut next_position =
            &state.position + &state.direction * (Self::A21 * state.step_length);

        let intermediate_direction_1 =
            match Self::compute_direction(field, interpolator, sense, &next_position) {
                StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
                StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => {
                    direction
                }
                StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
            };

        next_position =
            &state.position + &intermediate_direction_1 * (Self::A32 * state.step_length);

        let intermediate_direction_2 =
            match Self::compute_direction(field, interpolator, sense, &next_position) {
                StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
                StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => {
                    direction
                }
                StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
            };

        let step_displacement = (&state.direction * Self::A41
            + &intermediate_direction_1 * Self::A42
            + &intermediate_direction_2 * Self::A43)
            * state.step_length;

        next_position = &state.position + &step_displacement;

        let mut step_wrapped = false;

        let next_direction =
            match Self::compute_direction(field, interpolator, sense, &next_position) {
                StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
                StepperResult::Ok(ComputedDirection3::WithWrappedPosition((
                    wrapped_position,
                    direction,
                ))) => {
                    step_wrapped = true;
                    next_position = wrapped_position;
                    direction
                }
                StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
            };

        StepperResult::Ok(StepAttempt3 {
            step_displacement,
            next_position,
            next_direction,
            intermediate_directions: vec![intermediate_direction_1, intermediate_direction_2],
            step_wrapped,
        })
    }

    fn compute_error_deltas(&self, attempt: &StepAttempt3) -> Vec3<ftr> {
        let state = self.state();
        (&state.direction * Self::E1
            + &attempt.intermediate_directions[0] * Self::E2
            + &attempt.intermediate_directions[1] * Self::E3
            + &attempt.next_direction * Self::E4)
            * state.step_length
    }

    fn compute_dense_interpolation_coefs(&self) -> Vec<Vec3<ftr>> {
        let state = self.state();
        let coef_vec_1 = state.previous_unwrapped_position.to_vec3();
        let coef_vec_2 = state.previous_step_displacement.clone();
        let coef_vec_3 = &state.previous_direction * state.previous_step_length;
        let coef_vec_4 = &state.direction * state.previous_step_length;
        vec![coef_vec_1, coef_vec_2, coef_vec_3, coef_vec_4]
    }

    fn interpolate_dense_position(&self, coefs: &[Vec3<ftr>], fraction: ftr) -> Point3<ftr> {
        debug_assert!(fraction > 0.0 && fraction <= 1.0);
        let fraction_minus_one = fraction - 1.0;
        coefs[0].to_point3()
            + &coefs[1] * fraction
            + (&coefs[1] * (-(fraction + fraction_minus_one))
                + &coefs[2] * fraction_minus_one
                + &coefs[3] * fraction)
                * (fraction * fraction_minus_one)
    }
}

impl<F> Stepper3<F> for RKF23Stepper3 {
    fn place(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        position: &Point3<ftr>,
        callback: &mut SteppingCallback,
    ) -> StepperResult<()>
    where
        F: BFloat,
    {
        self.place_with_callback(field, interpolator, sense, position, callback)
    }

    fn step(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut SteppingCallback,
    ) -> StepperResult<()>
    where
        F: BFloat,
    {
        self.step_with_callback(field, interpolator, sense, callback)
    }

    fn step_dense_output(
        &mut self,
        field: &VectorField3<F>,
        interpolator: &dyn Interpolator3<F>,
        sense: SteppingSense,
        callback: &mut SteppingCallback,
    ) -> StepperResult<()>
    where
        F: BFloat,
    {
        self.step_with_callback_dense_output(field, interpolator, sense, callback)
    }

    fn position(&self) -> &Point3<ftr> {
        &self.state().position
    }

    fn distance(&self) -> ftr {
        self.state().distance
    }

    fn heap_clone(&self) -> Box<dyn Stepper3<F>> {
        Box::new(self.clone())
    }
}

impl RKF23StepperFactory3 {
    /// Creates a new factory for producing steppers with the given configuration parameters.
    pub fn new(config: RKFStepperConfig) -> Self {
        RKF23StepperFactory3 { config }
    }
}

impl<F> StepperFactory3<F> for RKF23StepperFactory3 {
    type Output = RKF23Stepper3;
    fn produce(&self) -> Self::Output {
        RKF23Stepper3::new(self.config.clone())
    }
}
