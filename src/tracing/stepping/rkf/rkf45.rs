//! Stepping using the the Dormand-Prince scheme,
//! a fifth-order Runge-Kutta method with error
//! estimation through an embedded fourth-order step.

use super::super::{Stepper3, StepperFactory3, StepperInstruction, StepperResult};
use super::{
    ComputedDirection3, PIControlParams, RKFStepper3, RKFStepperConfig, RKFStepperState3,
    StepAttempt3,
};
use crate::field::VectorField3;
use crate::geometry::{Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::num::BFloat;
use crate::tracing::ftr;

/// A stepper using the fifth order Runge–Kutta–Fehlberg method.
#[derive(Clone, Debug)]
pub struct RKF45Stepper3(RKFStepperState3);

/// Factory for `RKF45Stepper3` objects.
#[derive(Clone, Debug)]
pub struct RKF45StepperFactory3 {
    config: RKFStepperConfig,
}

impl RKF45Stepper3 {
    const ORDER: u8 = 5;
    const N_INTERMEDIATE_STEPS: usize = 5;

    const A21: ftr = 1.0 / 5.0;
    const A31: ftr = 3.0 / 40.0;
    const A32: ftr = 9.0 / 40.0;
    const A41: ftr = 44.0 / 45.0;
    const A42: ftr = -56.0 / 15.0;
    const A43: ftr = 32.0 / 9.0;
    const A51: ftr = 19_372.0 / 6561.0;
    const A52: ftr = -25_360.0 / 2187.0;
    const A53: ftr = 64_448.0 / 6561.0;
    const A54: ftr = -212.0 / 729.0;
    const A61: ftr = 9017.0 / 3168.0;
    const A62: ftr = -355.0 / 33.0;
    const A63: ftr = 46_732.0 / 5247.0;
    const A64: ftr = 49.0 / 176.0;
    const A65: ftr = -5103.0 / 18_656.0;
    const A71: ftr = 35.0 / 384.0;
    //  const A72: ftr =       0.0         ;
    const A73: ftr = 500.0 / 1113.0;
    const A74: ftr = 125.0 / 192.0;
    const A75: ftr = -2187.0 / 6784.0;
    const A76: ftr = 11.0 / 84.0;

    const E1: ftr = 71.0 / 57_600.0;
    //  const E2: ftr =       0.0          ;
    const E3: ftr = -71.0 / 16_695.0;
    const E4: ftr = 71.0 / 1920.0;
    const E5: ftr = -17_253.0 / 339_200.0;
    const E6: ftr = 22.0 / 525.0;
    const E7: ftr = -1.0 / 40.0;

    const D1: ftr = -12_715_105_075.0 / 11_282_082_432.0;
    //  const D2: ftr =               0.0                  ;
    const D3: ftr = 87_487_479_700.0 / 32_700_410_799.0;
    const D4: ftr = -10_690_763_975.0 / 1_880_347_072.0;
    const D5: ftr = 701_980_252_875.0 / 199_316_789_632.0;
    const D6: ftr = -1_453_857_185.0 / 822_651_844.0;
    const D7: ftr = 69_997_945.0 / 29_380_423.0;

    /// Creates a new RKF45 stepper with the given configuration.
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

        RKF45Stepper3(RKFStepperState3 {
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

impl RKFStepper3 for RKF45Stepper3 {
    fn state(&self) -> &RKFStepperState3 {
        &self.0
    }
    fn state_mut(&mut self) -> &mut RKFStepperState3 {
        &mut self.0
    }

    fn attempt_step<F, G, I, D>(
        &self,
        field: &VectorField3<F, G>,
        interpolator: &I,
        direction_computer: &D,
    ) -> StepperResult<StepAttempt3>
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        D: Fn(&mut Vec3<ftr>),
    {
        let state = self.state();

        let mut next_position =
            &state.position + &state.direction * (Self::A21 * state.step_length);

        let intermediate_direction_1 = match Self::compute_direction(
            field,
            interpolator,
            direction_computer,
            &next_position,
        ) {
            StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
            StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => direction,
            StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
        };

        next_position = &state.position
            + (&state.direction * Self::A31 + &intermediate_direction_1 * Self::A32)
                * state.step_length;

        let intermediate_direction_2 = match Self::compute_direction(
            field,
            interpolator,
            direction_computer,
            &next_position,
        ) {
            StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
            StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => direction,
            StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
        };

        next_position = &state.position
            + (&state.direction * Self::A41
                + &intermediate_direction_1 * Self::A42
                + &intermediate_direction_2 * Self::A43)
                * state.step_length;

        let intermediate_direction_3 = match Self::compute_direction(
            field,
            interpolator,
            direction_computer,
            &next_position,
        ) {
            StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
            StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => direction,
            StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
        };

        next_position = &state.position
            + (&state.direction * Self::A51
                + &intermediate_direction_1 * Self::A52
                + &intermediate_direction_2 * Self::A53
                + &intermediate_direction_3 * Self::A54)
                * state.step_length;

        let intermediate_direction_4 = match Self::compute_direction(
            field,
            interpolator,
            direction_computer,
            &next_position,
        ) {
            StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
            StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => direction,
            StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
        };

        next_position = &state.position
            + (&state.direction * Self::A61
                + &intermediate_direction_1 * Self::A62
                + &intermediate_direction_2 * Self::A63
                + &intermediate_direction_3 * Self::A64
                + &intermediate_direction_4 * Self::A65)
                * state.step_length;

        let intermediate_direction_5 = match Self::compute_direction(
            field,
            interpolator,
            direction_computer,
            &next_position,
        ) {
            StepperResult::Ok(ComputedDirection3::Standard(direction)) => direction,
            StepperResult::Ok(ComputedDirection3::WithWrappedPosition((_, direction))) => direction,
            StepperResult::Stopped(cause) => return StepperResult::Stopped(cause),
        };

        let step_displacement = (&state.direction * Self::A71
            + &intermediate_direction_2 * Self::A73
            + &intermediate_direction_3 * Self::A74
            + &intermediate_direction_4 * Self::A75
            + &intermediate_direction_5 * Self::A76)
            * state.step_length;

        next_position = &state.position + &step_displacement;

        let mut step_wrapped = false;

        let next_direction = match Self::compute_direction(
            field,
            interpolator,
            direction_computer,
            &next_position,
        ) {
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
            intermediate_directions: vec![
                intermediate_direction_1,
                intermediate_direction_2,
                intermediate_direction_3,
                intermediate_direction_4,
                intermediate_direction_5,
            ],
            step_wrapped,
        })
    }

    fn compute_error_deltas(&self, attempt: &StepAttempt3) -> Vec3<ftr> {
        let state = self.state();
        (&state.direction * Self::E1
            + &attempt.intermediate_directions[1] * Self::E3
            + &attempt.intermediate_directions[2] * Self::E4
            + &attempt.intermediate_directions[3] * Self::E5
            + &attempt.intermediate_directions[4] * Self::E6
            + &attempt.next_direction * Self::E7)
            * state.step_length
    }

    fn compute_dense_interpolation_coefs(&self) -> Vec<Vec3<ftr>> {
        let state = self.state();
        let coef_vec_1 = state.previous_unwrapped_position.to_vec3();
        let coef_vec_2 = state.previous_step_displacement.clone();
        let coef_vec_3 = &state.previous_direction * state.previous_step_length - &coef_vec_2;
        let coef_vec_4 = &coef_vec_2 - &state.direction * state.previous_step_length - &coef_vec_3;
        let coef_vec_5 = (&state.previous_direction * Self::D1
            + &state.intermediate_directions[1] * Self::D3
            + &state.intermediate_directions[2] * Self::D4
            + &state.intermediate_directions[3] * Self::D5
            + &state.intermediate_directions[4] * Self::D6
            + &state.direction * Self::D7)
            * state.previous_step_length;
        vec![coef_vec_1, coef_vec_2, coef_vec_3, coef_vec_4, coef_vec_5]
    }

    fn interpolate_dense_position(&self, coefs: &[Vec3<ftr>], fraction: ftr) -> Point3<ftr> {
        debug_assert!(fraction > 0.0 && fraction <= 1.0);
        let one_minus_fraction = 1.0 - fraction;
        let position = &coefs[3] + &coefs[4] * one_minus_fraction;
        let position = &coefs[2] + position * fraction;
        let position = &coefs[1] + position * one_minus_fraction;
        (&coefs[0] + position * fraction).to_point3()
    }
}

impl Stepper3 for RKF45Stepper3 {
    fn place<F, G, I, D, C>(
        &mut self,
        field: &VectorField3<F, G>,
        interpolator: &I,
        direction_computer: &D,
        position: &Point3<ftr>,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        D: Fn(&mut Vec3<ftr>),
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        self.place_with_callback(field, interpolator, direction_computer, position, callback)
    }

    fn step<F, G, I, D, C>(
        &mut self,
        field: &VectorField3<F, G>,
        interpolator: &I,
        direction_computer: &D,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        D: Fn(&mut Vec3<ftr>),
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        self.step_with_callback(field, interpolator, direction_computer, callback)
    }

    fn step_dense_output<F, G, I, D, C>(
        &mut self,
        field: &VectorField3<F, G>,
        interpolator: &I,
        direction_computer: &D,
        callback: &mut C,
    ) -> StepperResult<()>
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        D: Fn(&mut Vec3<ftr>),
        C: FnMut(&Vec3<ftr>, &Vec3<ftr>, &Point3<ftr>, ftr) -> StepperInstruction,
    {
        self.step_with_callback_dense_output(field, interpolator, direction_computer, callback)
    }

    fn position(&self) -> &Point3<ftr> {
        &self.state().position
    }
    fn distance(&self) -> ftr {
        self.state().distance
    }
}

impl RKF45StepperFactory3 {
    /// Creates a new factory for producing steppers with the given configuration parameters.
    pub fn new(config: RKFStepperConfig) -> Self {
        RKF45StepperFactory3 { config }
    }
}

impl StepperFactory3 for RKF45StepperFactory3 {
    type Output = RKF45Stepper3;
    fn produce(&self) -> Self::Output {
        RKF45Stepper3::new(self.config.clone())
    }
}
