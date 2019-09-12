//! Simple model for acceleration of non-thermal electron beams.

use std::sync::Arc;
use crate::units::solar::{U_T, U_E, U_L3};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Dim3, In3D, Vec3, Point3};
use crate::grid::Grid3;
use crate::grid::regular::RegularGrid3;
use crate::interpolation::Interpolator3;
use super::{AccelerationEvent, AccelerationEventGenerator};
use super::super::feb;
use Dim3::{X, Y, Z};

/// Direction of acceleration of non-thermal electrons.
pub type AccelerationDirection = Option<Vec3<fdt>>;

/// Simple representation of an event where particles are accelerated.
#[derive(Clone, Debug)]
pub struct SimpleAccelerationEvent {
    /// Central position of the acceleration event [Mm].
    position: Point3<fdt>,
    /// Grid specifying the extent and resolution of the acceleration event.
    grid: Arc<RegularGrid3<fdt>>,
    /// Duration of the acceleration event [s].
    duration: feb,
    /// Temperature at the acceleration site [K].
    temperature: feb,
    /// Total electron number density at the acceleration site [electrons/cm^3].
    electron_density: feb,
    /// Fraction of the total energy release going into acceleration of non-thermal particles.
    particle_energy_fraction: feb,
    /// Total energy per volume going into acceleration of non-thermal particles during the event [erg/cm^3].
    particle_energy_density: feb,
    /// Average energy per volume and time going into acceleration of non-thermal particles [erg/(cm^3 s)].
    particle_power_density: feb,
    /// Direction of acceleration of the electrons.
    acceleration_direction: AccelerationDirection
}

/// Generator for acceleration events with the same duration and particle energy fraction.
#[derive(Clone, Debug)]
pub struct SimpleAccelerationEventGenerator {
    /// Spatial extent of the acceleration event [cm].
    extent: fdt,
    /// Duration of the acceleration events [s].
    duration: feb,
    /// Fraction of the total energy release going into acceleration of non-thermal particles.
    particle_energy_fraction: feb
}

impl SimpleAccelerationEvent {
    /// Spatial resolution of the acceleration event (number of points in each direction).
    const RESOLUTION: usize = 5;

    /// Returns a reference to the position of the acceleration event [Mm].
    pub fn position(&self) -> &Point3<fdt> { &self.position }

    /// Returns a reference to the grid specifying the extent and resolution of the acceleration event.
    pub fn grid(&self) -> &RegularGrid3<fdt> { self.grid.as_ref() }

    /// Returns the duration of the acceleration event [s].
    pub fn duration(&self) -> feb { self.duration }

    /// Returns the temperature at the acceleration site [K].
    pub fn temperature(&self) -> feb { self.temperature }

    /// Returns the total electron number density at the acceleration site [electrons/cm^3].
    pub fn electron_density(&self) -> feb { self.electron_density }

    /// Returns the fraction of the total energy release
    /// going into acceleration of non-thermal particles.
    pub fn particle_energy_fraction(&self) -> feb { self.particle_energy_fraction }

    /// Returns the total energy per volume going into acceleration
    /// of non-thermal particles during the event [erg/cm^3].
    pub fn particle_energy_density(&self) -> feb { self.particle_energy_density }

    /// Returns the average energy per volume and time going into
    /// acceleration of non-thermal particles [erg/(cm^3 s)].
    pub fn particle_power_density(&self) -> feb { self.particle_power_density }

    /// Returns the direction of acceleration of the electrons.
    pub fn acceleration_direction(&self) -> &AccelerationDirection { &self.acceleration_direction }

    fn new<G, I>(snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>, extent: fdt, duration: feb, particle_energy_fraction: feb) -> Self
    where G: Grid3<fdt> ,
          I: Interpolator3
    {
        let grid = Arc::new(Self::construct_grid(position, extent));

        assert!(duration >= 0.0, "Duration must be larger than or equal to zero.");
        assert!(particle_energy_fraction >= 0.0, "Particle energy fraction must be larger than or equal to zero.");

        let temperature = feb::from(interpolator.interp_scalar_field(snapshot.expect_scalar_field("tg"), position).expect_inside());
        assert!(temperature > 0.0, "Temperature must be larger than zero.");

        let electron_density = interpolator.interp_scalar_field(snapshot.expect_scalar_field("ne"), position).expect_inside();
        let electron_density = feb::from(electron_density)/U_L3; // [electrons/cm^3]
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");

        let particle_power_density = Self::compute_particle_power_density(snapshot, interpolator, &position, particle_energy_fraction);
        let particle_energy_density = particle_power_density*duration;

        let acceleration_direction = Self::determine_acceleration_direction(snapshot, interpolator, Arc::clone(&grid));

        SimpleAccelerationEvent{
            position: position.clone(),
            duration,
            grid,
            temperature,
            electron_density,
            particle_energy_fraction,
            particle_energy_density,
            particle_power_density,
            acceleration_direction
        }
    }

    fn construct_grid(position: &Point3<fdt>, extent: fdt) -> RegularGrid3<fdt> {
        let shape = In3D::same(Self::RESOLUTION);
        let extent_vec = Vec3::equal_components(0.5*extent);
        let lower_bounds = position - &extent_vec;
        let upper_bounds = position + &extent_vec;
        let is_periodic = In3D::same(false);
        RegularGrid3::from_bounds(shape, lower_bounds.to_vec3(), upper_bounds.to_vec3(), is_periodic)
    }

    fn compute_particle_power_density<G, I>(snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>, particle_energy_fraction: feb) -> feb
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let joule_heating_field = snapshot.expect_scalar_field("qjoule");
        let joule_heating = feb::from(interpolator.interp_scalar_field(joule_heating_field, position).expect_inside());
        let joule_heating = joule_heating*U_E/U_T; // [erg/(cm^3 s)]

        assert!(joule_heating >= 0.0, "Joule heating must be larger than or equal to zero.");

        particle_energy_fraction*joule_heating
    }

    fn determine_acceleration_direction<G, I>(snapshot: &mut SnapshotCacher3<G>, interpolator: &I, grid: Arc<RegularGrid3<fdt>>) -> AccelerationDirection
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let local_electric_field = snapshot.expect_vector_field("e").resampled_to_grid(grid, interpolator);

        let total_electric_vector = Vec3::new(
            local_electric_field.values(X).sum(),
            local_electric_field.values(Y).sum(),
            local_electric_field.values(Z).sum()
        );
        let squared_total_electric_vector = total_electric_vector.squared_length();

        if squared_total_electric_vector > std::f32::EPSILON {
            // Electrons are accelerated in the opposite direction as the electric field.
            Some(total_electric_vector/(-fdt::sqrt(squared_total_electric_vector)))
        } else {
            None
        }
    }
}

impl AccelerationEvent for SimpleAccelerationEvent {}

impl SimpleAccelerationEventGenerator {
    /// Creates a new generator for acceleration events with the given
    /// extent, duration and particle energy fraction.
    pub fn new(extent: fdt, duration: feb, particle_energy_fraction: feb) -> Self {
        SimpleAccelerationEventGenerator{ extent, duration, particle_energy_fraction }
    }
}

impl AccelerationEventGenerator for SimpleAccelerationEventGenerator {
    type AccelerationEventType = SimpleAccelerationEvent;

    fn generate<G, I>(&self, snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> Self::AccelerationEventType
    where G: Grid3<fdt> ,
          I: Interpolator3
    {
        SimpleAccelerationEvent::new(snapshot, interpolator, position, self.extent, self.duration, self.particle_energy_fraction)
    }
}
