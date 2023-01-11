//! Non-thermal electron distributions.

pub mod power_law;

use super::BeamPropertiesCollection;
use crate::{
    geometry::{Idx3, Point3},
    grid::fgr,
    tracing::stepping::SteppingSense,
};

/// Defines the properties of a non-thermal electron distribution.
pub trait Distribution {
    type PropertiesCollectionType: BeamPropertiesCollection;

    /// Returns the position where the distribution originates.
    fn acceleration_position(&self) -> &Point3<fgr>;

    /// Returns the indices of the position where the distribution originates.
    fn acceleration_indices(&self) -> &Idx3<usize>;

    /// Returns the direction of propagation of the electrons relative to the magnetic field direction.
    fn propagation_sense(&self) -> SteppingSense;

    /// Returns an object holding properties associated with the distribution.
    fn properties(&self) -> <Self::PropertiesCollectionType as BeamPropertiesCollection>::Item;
}
