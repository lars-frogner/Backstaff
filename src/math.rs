//! Math utilities.

use num::Float;
use special::Beta;

/// Floating-point precision to use for integration.
#[allow(non_camel_case_types)]
pub type fin = f64;

/// Evaluates the beta function B(a, b) = int t^(a-1)*(1-t)^(b-1) dt from t=0 to t=1.
pub fn beta<F: Float + Beta>(a: F, b: F) -> F {
    F::exp(a.ln_beta(b))
}

/// Evaluates the unregularized incomplete beta function
/// B(x; a, b) = int t^(a-1)*(1-t)^(b-1) dt from t=0 to t=x.
pub fn incomplete_beta<F: Float + Beta>(x: F, a: F, b: F) -> F {
    let ln_beta = a.ln_beta(b);
    x.inc_beta(a, b, ln_beta) * F::exp(ln_beta)
}

/// Estimates the integral of the given function over the given interval using a
/// two-point Gauss-Legendre quadrature.
pub fn integrate_two_point_gauss_legendre<E>(evaluate_integrand: E, start: fin, end: fin) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORD_1: fin = -0.577_350_269_189_625_8; // -1/sqrt(3)
    const COORD_2: fin = -COORD_1; //  1/sqrt(3)

    const WEIGHT_1: fin = 1.0;
    const WEIGHT_2: fin = WEIGHT_1;

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    interval_scale
        * (WEIGHT_1 * evaluate_integrand(interval_offset + interval_scale * COORD_1)
            + WEIGHT_2 * evaluate_integrand(interval_offset + interval_scale * COORD_2))
}

/// Estimates the integral of the given function over the given interval using a
/// three-point Gauss-Legendre quadrature.
pub fn integrate_three_point_gauss_legendre<E>(evaluate_integrand: E, start: fin, end: fin) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORD_1: fin = -0.774_596_669_241_483_4; // -sqrt(3/5)
    const COORD_2: fin = 0.0;
    const COORD_3: fin = -COORD_1; //  sqrt(3/5)

    const WEIGHT_1: fin = 5.0 / 9.0;
    const WEIGHT_2: fin = 8.0 / 9.0;
    const WEIGHT_3: fin = WEIGHT_1;

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    interval_scale
        * (WEIGHT_1 * evaluate_integrand(interval_offset + interval_scale * COORD_1)
            + WEIGHT_2 * evaluate_integrand(interval_offset + interval_scale * COORD_2)
            + WEIGHT_3 * evaluate_integrand(interval_offset + interval_scale * COORD_3))
}

/// Estimates the integral of the given function over the given interval using a
/// four-point Gauss-Legendre quadrature.
pub fn integrate_four_point_gauss_legendre<E>(evaluate_integrand: E, start: fin, end: fin) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORD_1: fin = -0.861_136_311_594_052_6; // -sqrt(3/7 + (2/7)*sqrt(6/5))
    const COORD_2: fin = -0.339_981_043_584_856_3; // -sqrt(3/7 - (2/7)*sqrt(6/5))
    const COORD_3: fin = -COORD_2; //  sqrt(3/7 - (2/7)*sqrt(6/5))
    const COORD_4: fin = -COORD_1; //  sqrt(3/7 + (2/7)*sqrt(6/5))

    const WEIGHT_1: fin = 0.347_854_845_137_453_85; // (18 - sqrt(30))/36
    const WEIGHT_2: fin = 0.652_145_154_862_546_2; // (18 + sqrt(30))/36
    const WEIGHT_3: fin = WEIGHT_2; // (18 + sqrt(30))/36
    const WEIGHT_4: fin = WEIGHT_1; // (18 - sqrt(30))/36

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    interval_scale
        * (WEIGHT_1 * evaluate_integrand(interval_offset + interval_scale * COORD_1)
            + WEIGHT_2 * evaluate_integrand(interval_offset + interval_scale * COORD_2)
            + WEIGHT_3 * evaluate_integrand(interval_offset + interval_scale * COORD_3)
            + WEIGHT_4 * evaluate_integrand(interval_offset + interval_scale * COORD_4))
}

/// Estimates the integral of the given function over the given interval using a
/// five-point Gauss-Legendre quadrature.
pub fn integrate_five_point_gauss_legendre<E>(evaluate_integrand: E, start: fin, end: fin) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORD_1: fin = -0.906_179_845_938_664; // -(1/3)*sqrt(5 + 2*sqrt(10/7))
    const COORD_2: fin = -0.538_469_310_105_683; // -(1/3)*sqrt(5 - 2*sqrt(10/7))
    const COORD_3: fin = 0.0;
    const COORD_4: fin = -COORD_2; // (1/3)*sqrt(5 - 2*sqrt(10/7))
    const COORD_5: fin = -COORD_1; // (1/3)*sqrt(5 + 2*sqrt(10/7))

    const WEIGHT_1: fin = 0.236_926_885_056_189_08; // (322 - 13*sqrt(70))/900
    const WEIGHT_2: fin = 0.478_628_670_499_366_47; // (322 + 13*sqrt(70))/900
    const WEIGHT_3: fin = 128.0 / 225.0;
    const WEIGHT_4: fin = WEIGHT_2; // (322 + 13*sqrt(70))/900
    const WEIGHT_5: fin = WEIGHT_1; // (322 - 13*sqrt(70))/900

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    interval_scale
        * (WEIGHT_1 * evaluate_integrand(interval_offset + interval_scale * COORD_1)
            + WEIGHT_2 * evaluate_integrand(interval_offset + interval_scale * COORD_2)
            + WEIGHT_3 * evaluate_integrand(interval_offset + interval_scale * COORD_3)
            + WEIGHT_4 * evaluate_integrand(interval_offset + interval_scale * COORD_4)
            + WEIGHT_5 * evaluate_integrand(interval_offset + interval_scale * COORD_5))
}
