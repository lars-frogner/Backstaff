//! Math utilities.

use num::Float;
use special::Beta;

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
