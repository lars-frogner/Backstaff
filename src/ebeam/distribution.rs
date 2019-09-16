//! Non-thermal electron distributions.

pub mod power_law;

use std::collections::HashMap;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::ftr;
use crate::tracing::stepping::SteppingSense;
use super::feb;

/// Direction of propagation of the non-thermal electrons with respect to the magnetic field.
pub type PropagationSense = Option<SteppingSense>;

/// Whether or not a distribution is depleted.
#[derive(Clone, Copy, Debug)]
pub enum DepletionStatus {
    Undepleted,
    Depleted
}

/// Holds the result of propagating the electron distribution.
#[derive(Clone, Debug)]
pub struct PropagationResult {
    /// Total power density deposited during propagation.
    pub deposited_power_density: feb,
    /// Average position where the power density was deposited.
    pub deposition_position: Point3<ftr>,
    /// Whether or not the distribution is now depleted.
    pub depletion_status: DepletionStatus
}

/// Defines the properties of a non-thermal electron distribution.
pub trait Distribution {
    /// Returns the position where the distribution originates.
    fn acceleration_position(&self) -> &Point3<fdt>;

    /// Finds the direction that the distribution will move along the magnetic field.
    ///
    /// Returns `None` if the direction is not sufficiently well-defined.
    fn determine_propagation_sense(&self, magnetic_field_direction: &Vec3<fdt>) -> PropagationSense;

    /// Creates a hash map containing scalar-valued properties of the distribution.
    fn scalar_properties(&self) -> HashMap<String, feb>;

    /// Creates a hash map containing vector-valued properties of the distribution.
    fn vector_properties(&self) -> HashMap<String, Vec3<feb>>;

    /// Propagates the electron distribution for the given displacement
    /// and returns the power density deposited during the propagation.
    fn propagate<G, I>(&mut self, snapshot: &SnapshotCacher3<G>, interpolator: &I, displacement: &Vec3<ftr>, new_position: &Point3<ftr>) -> PropagationResult
    where G: Grid3<fdt>,
          I: Interpolator3;
}
