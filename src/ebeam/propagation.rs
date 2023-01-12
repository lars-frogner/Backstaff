//! Propagation of non-thermal electron distributions.

pub mod analytical;
pub mod fp_characteristics;

use super::{distribution::Distribution, feb};
use crate::{
    field::CachingScalarFieldProvider3,
    geometry::{Point3, Vec3},
    interpolation::Interpolator3,
    io::snapshot::fdt,
    tracing::ftr,
};
use ndarray::prelude::*;

/// Whether or not a distribution is depleted.
#[derive(Clone, Copy, Debug)]
pub enum DepletionStatus {
    Undepleted,
    Depleted,
}

/// Holds the result of propagating the electron distribution.
#[derive(Clone, Debug)]
pub struct PropagationResult {
    /// Total power deposited during propagation.
    pub deposited_power: feb,
    /// Total power density deposited during propagation.
    pub deposited_power_density: feb,
    /// Average position where the power density was deposited.
    pub deposition_position: Point3<ftr>,
    /// Whether or not the distribution is now depleted.
    pub depletion_status: DepletionStatus,
}

/// Defines the properties of a propagator for a non-thermal electron
/// distribution.
pub trait Propagator<D: Distribution>: Sized + Sync + Send {
    type Config: Clone + Sync;

    fn new(config: Self::Config, distribution: D) -> Option<Self>;

    fn distribution(&self) -> &D;

    fn into_distribution(self) -> D;

    /// Returns the maximum distance the distribution can propagate before propagation should be terminated.
    fn max_propagation_distance(&self) -> ftr;

    /// Propagates the electron distribution for the given displacement
    /// and returns the power density deposited during the propagation.
    fn propagate(
        &mut self,
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        acceleration_map: &Array3<bool>,
        interpolator: &dyn Interpolator3<fdt>,
        displacement: &Vec3<ftr>,
        new_position: &Point3<ftr>,
    ) -> PropagationResult;
}
