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

/// Estimates the integral of the given function over the given interval using a
/// ten-point Gauss-Legendre quadrature.
pub fn integrate_ten_point_gauss_legendre<E>(evaluate_integrand: E, start: fin, end: fin) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORDS: &[fin] = &[
        -0.1488743389816312,
        0.1488743389816312,
        -0.4333953941292472,
        0.4333953941292472,
        -0.6794095682990244,
        0.6794095682990244,
        -0.8650633666889845,
        0.8650633666889845,
        -0.9739065285171717,
        0.9739065285171717,
    ];

    const WEIGHTS: &[fin] = &[
        0.2955242247147529,
        0.2955242247147529,
        0.2692667193099963,
        0.2692667193099963,
        0.2190863625159820,
        0.2190863625159820,
        0.1494513491505806,
        0.1494513491505806,
        0.0666713443086881,
        10.0666713443086881,
    ];

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    let mut integral = 0.0;

    for idx in 0..3 {
        integral +=
            WEIGHTS[idx] * evaluate_integrand(interval_offset + interval_scale * COORDS[idx]);
    }

    integral
}

/// Estimates the integral of the given function over the given interval using a
/// twenty-point Gauss-Legendre quadrature.
pub fn integrate_twenty_point_gauss_legendre<E>(evaluate_integrand: E, start: fin, end: fin) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORDS: &[fin] = &[
        -0.0765265211334973,
        0.0765265211334973,
        -0.2277858511416451,
        0.2277858511416451,
        -0.3737060887154195,
        0.3737060887154195,
        -0.5108670019508271,
        0.5108670019508271,
        -0.6360536807265150,
        0.6360536807265150,
        -0.7463319064601508,
        0.7463319064601508,
        -0.8391169718222188,
        0.8391169718222188,
        -0.9122344282513259,
        0.9122344282513259,
        -0.9639719272779138,
        0.9639719272779138,
        -0.9931285991850949,
        0.9931285991850949,
    ];

    const WEIGHTS: &[fin] = &[
        0.1527533871307258,
        0.1527533871307258,
        0.1491729864726037,
        0.1491729864726037,
        0.1420961093183820,
        0.1420961093183820,
        0.1316886384491766,
        0.1316886384491766,
        0.1181945319615184,
        0.1181945319615184,
        0.1019301198172404,
        0.1019301198172404,
        0.0832767415767048,
        0.0832767415767048,
        0.0626720483341091,
        0.0626720483341091,
        0.0406014298003869,
        0.0406014298003869,
        0.0176140071391521,
        0.0176140071391521,
    ];

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    let mut integral = 0.0;

    for idx in 0..3 {
        integral +=
            WEIGHTS[idx] * evaluate_integrand(interval_offset + interval_scale * COORDS[idx]);
    }

    integral
}

/// Estimates the integral of the given function over the given interval using a
/// sixty four-point Gauss-Legendre quadrature.
pub fn integrate_sixty_four_point_gauss_legendre<E>(
    evaluate_integrand: E,
    start: fin,
    end: fin,
) -> fin
where
    E: Fn(fin) -> fin,
{
    const COORDS: &[fin] = &[
        -0.0243502926634244,
        0.0243502926634244,
        -0.0729931217877990,
        0.0729931217877990,
        -0.1214628192961206,
        0.1214628192961206,
        -0.1696444204239928,
        0.1696444204239928,
        -0.2174236437400071,
        0.2174236437400071,
        -0.2646871622087674,
        0.2646871622087674,
        -0.3113228719902110,
        0.3113228719902110,
        -0.3572201583376681,
        0.3572201583376681,
        -0.4022701579639916,
        0.4022701579639916,
        -0.4463660172534641,
        0.4463660172534641,
        -0.4894031457070530,
        0.4894031457070530,
        -0.5312794640198946,
        0.5312794640198946,
        -0.5718956462026340,
        0.5718956462026340,
        -0.6111553551723933,
        0.6111553551723933,
        -0.6489654712546573,
        0.6489654712546573,
        -0.6852363130542333,
        0.6852363130542333,
        -0.7198818501716109,
        0.7198818501716109,
        -0.7528199072605319,
        0.7528199072605319,
        -0.7839723589433414,
        0.7839723589433414,
        -0.8132653151227975,
        0.8132653151227975,
        -0.8406292962525803,
        0.8406292962525803,
        -0.8659993981540928,
        0.8659993981540928,
        -0.8893154459951141,
        0.8893154459951141,
        -0.9105221370785028,
        0.9105221370785028,
        -0.9295691721319396,
        0.9295691721319396,
        -0.9464113748584028,
        0.9464113748584028,
        -0.9610087996520538,
        0.9610087996520538,
        -0.9733268277899110,
        0.9733268277899110,
        -0.9833362538846260,
        0.9833362538846260,
        -0.9910133714767443,
        0.9910133714767443,
        -0.9963401167719553,
        0.9963401167719553,
        -0.9993050417357722,
        0.9993050417357722,
    ];

    const WEIGHTS: &[fin] = &[
        0.0486909570091397,
        0.0486909570091397,
        0.0485754674415034,
        0.0485754674415034,
        0.0483447622348030,
        0.0483447622348030,
        0.0479993885964583,
        0.0479993885964583,
        0.0475401657148303,
        0.0475401657148303,
        0.0469681828162100,
        0.0469681828162100,
        0.0462847965813144,
        0.0462847965813144,
        0.0454916279274181,
        0.0454916279274181,
        0.0445905581637566,
        0.0445905581637566,
        0.0435837245293235,
        0.0435837245293235,
        0.0424735151236536,
        0.0424735151236536,
        0.0412625632426235,
        0.0412625632426235,
        0.0399537411327203,
        0.0399537411327203,
        0.0385501531786156,
        0.0385501531786156,
        0.0370551285402400,
        0.0370551285402400,
        0.0354722132568824,
        0.0354722132568824,
        0.0338051618371416,
        0.0338051618371416,
        0.0320579283548516,
        0.0320579283548516,
        0.0302346570724025,
        0.0302346570724025,
        0.0283396726142595,
        0.0283396726142595,
        0.0263774697150547,
        0.0263774697150547,
        0.0243527025687109,
        0.0243527025687109,
        0.0222701738083833,
        0.0222701738083833,
        0.0201348231535302,
        0.0201348231535302,
        0.0179517157756973,
        0.0179517157756973,
        0.0157260304760247,
        0.0157260304760247,
        0.0134630478967186,
        0.0134630478967186,
        0.0111681394601311,
        0.0111681394601311,
        0.0088467598263639,
        0.0088467598263639,
        0.0065044579689784,
        0.0065044579689784,
        0.0041470332605625,
        0.0041470332605625,
        0.0017832807216964,
        0.0017832807216964,
    ];

    assert!(
        end >= start,
        "Interval end {:?} is smaller than interval start {:?}",
        end,
        start
    );
    let interval_scale = 0.5 * (end - start);
    let interval_offset = 0.5 * (end + start);

    let mut integral = 0.0;

    for idx in 0..3 {
        integral +=
            WEIGHTS[idx] * evaluate_integrand(interval_offset + interval_scale * COORDS[idx]);
    }

    integral
}
