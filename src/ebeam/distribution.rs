//! Non-thermal electron distributions.

pub mod power_law;

use super::{feb, BeamPropertiesCollection};
use crate::{
    geometry::{Idx3, Point3, Vec3},
    grid::Grid3,
    interpolation::Interpolator3,
    io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
    tracing::{ftr, stepping::SteppingSense},
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
    /// Factor governing the reduction of heating due to energy depletion.
    pub residual_factor: feb,
    /// Total power deposited during propagation.
    pub deposited_power: feb,
    /// Total power density deposited during propagation.
    pub deposited_power_density: feb,
    /// Average position where the power density was deposited.
    pub deposition_position: Point3<ftr>,
    /// Whether or not the distribution is now depleted.
    pub depletion_status: DepletionStatus,
}

/// Defines the properties of a non-thermal electron distribution.
pub trait Distribution {
    type PropertiesCollectionType: BeamPropertiesCollection;

    /// Returns the position where the distribution originates.
    fn acceleration_position(&self) -> &Point3<fdt>;

    /// Returns the indices of the position where the distribution originates.
    fn acceleration_indices(&self) -> &Idx3<usize>;

    /// Returns the direction of propagation of the electrons relative to the magnetic field direction.
    fn propagation_sense(&self) -> SteppingSense;

    /// Returns the maximum distance the distribution can propagate before propagation should be terminated.
    fn max_propagation_distance(&self) -> ftr;

    /// Returns an object holding properties associated with the distribution.
    fn properties(&self) -> <Self::PropertiesCollectionType as BeamPropertiesCollection>::Item;

    /// Propagates the electron distribution for the given displacement
    /// and returns the power density deposited during the propagation.
    fn propagate<G, R, I>(
        &mut self,
        snapshot: &SnapshotCacher3<G, R>,
        acceleration_map: &Array3<bool>,
        interpolator: &I,
        displacement: &Vec3<ftr>,
        new_position: &Point3<ftr>,
    ) -> PropagationResult
    where
        G: Grid3<fdt>,
        R: SnapshotReader3<G>,
        I: Interpolator3;
}
