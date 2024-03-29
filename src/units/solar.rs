//! Solar units.

use super::fun;
use crate::constants::{CLIGHT, PI};
use lazy_static::lazy_static;

/// Unit for length [cm].
pub const U_L: fun = 1e8;
/// Unit for time [s].
pub const U_T: fun = 1e2;
/// Unit for mass density [g/cm^3].
pub const U_R: fun = 1e-7;
/// Unit for speed [cm/s].
pub const U_U: fun = U_L / U_T;
/// Unit for pressure [dyn/cm^2].
pub const U_P: fun = U_R * (U_L / U_T) * (U_L / U_T);
/// Unit for Rosseland opacity [cm^2/g].
pub const U_KR: fun = 1.0 / (U_R * U_L);
/// Unit for energy per mass [erg/g].
pub const U_EE: fun = U_U * U_U;
/// Unit for energy per volume [erg/cm^3].
pub const U_E: fun = U_R * U_EE;
/// Unit for thermal emission [erg/(s ster cm^2)].
pub const U_TE: fun = U_E / U_T * U_L;
/// Unit for volume [cm^3].
pub const U_L3: fun = U_L * U_L * U_L;

lazy_static! {
    /// Unit for magnetic flux density [gauss].
    pub static ref U_B: fun = U_U * fun::sqrt(4.0 * PI * U_R);
    /// Unit for electric field strength [statV/cm]. (The first factor is to
    /// account for that E is premultiplied with c in Bifrost)
    pub static ref U_EL: fun = (1.0 / (CLIGHT / U_U)) * U_U * (*U_B) / CLIGHT;
}
