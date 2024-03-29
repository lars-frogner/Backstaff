//! Tracing of corks in an evolving velocity field.

use crate::{
    field::{CachingScalarFieldProvider3, ScalarField3, VectorField3},
    geometry::{Idx3, Point3, Vec3},
    grid::{Grid3, GridPointQuery3},
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, MASS_DENSITY_VARIABLE_NAME, MOMENTUM_VARIABLE_NAME},
        utils, Verbosity,
    },
    seeding::Seeder3,
};
use rayon::prelude::*;
use std::{io, iter, path::Path};

#[cfg(feature = "serialization")]
use serde::{
    ser::{SerializeStruct, Serializer},
    Serialize,
};

/// Floating-point precision to use for cork tracing.
#[allow(non_camel_case_types)]
pub type fco = f64;

type ScalarFieldValues = Vec<Vec<fdt>>;
type VectorFieldValues = Vec<Vec<Vec3<fdt>>>;

/// Represents the evolution of a single cork.
#[derive(Clone, Debug)]
pub struct Cork {
    /// Position at each point in time.
    positions: Vec<Point3<fco>>,
    /// Velocity at each point in time.
    velocities: Vec<Vec3<fco>>,
    /// Indices of the position into the snapshot grid at the latest point in time.
    last_position_indices: Idx3<usize>,
    /// Index of the point in time when the cork was created.
    first_time_idx: usize,
    /// Whether the cork should not be traced further.
    terminated: bool,
    /// Values of any scalar fields sampled along the cork trajectory.
    scalar_field_values: ScalarFieldValues,
    /// Values of any vector fields sampled along the cork trajectory.
    vector_field_values: VectorFieldValues,
}

/// Represents a collection of corks.
#[derive(Clone, Debug)]
pub struct CorkSet {
    /// All the corks in the collection.
    corks: Vec<Cork>,
    /// The points in time through which the corks have been evolved.
    times: Vec<fco>,
    /// Lower bounds of the simulation domain.
    lower_bounds: Vec3<fco>,
    /// Upper bounds of the simulation domain.
    upper_bounds: Vec3<fco>,
    /// Names of the scalar fields that should be sampled along the cork trajectories.
    scalar_quantity_names: Vec<String>,
    /// Names of the vector fields that should be sampled along the cork trajectories.
    vector_quantity_names: Vec<String>,
    /// Whether the mass density should not be sampled and thus can be dropped after each step.
    /// (The momentum field is always dropped since it can be recomputed from velocity and density.)
    can_drop_mass_density_field: bool,
    /// Whether and how to pass non-essential information to user.
    verbosity: Verbosity,
}

/// Uses the Heun method for advecting corks.
#[derive(Clone, Copy, Debug)]
pub struct HeunCorkStepper;

/// Traces a constant number of initial corks.
#[derive(Clone, Copy, Debug)]
pub struct ConstantCorkAdvector {
    step_duration: fco,
}

impl ConstantCorkAdvector {
    pub fn new(step_duration: fco) -> Self {
        assert!(step_duration > 0.0);
        Self { step_duration }
    }
}

impl Cork {
    fn new(
        position: Point3<fco>,
        position_indices: Idx3<usize>,
        velocity: Vec3<fco>,
        current_time_idx: usize,
        number_of_scalar_quantities: usize,
        number_of_vector_quantities: usize,
    ) -> Self {
        Self {
            positions: vec![position],
            last_position_indices: position_indices,
            velocities: vec![velocity],
            first_time_idx: current_time_idx,
            terminated: false,
            scalar_field_values: iter::repeat_with(Vec::new)
                .take(number_of_scalar_quantities)
                .collect(),
            vector_field_values: iter::repeat_with(Vec::new)
                .take(number_of_vector_quantities)
                .collect(),
        }
    }

    fn new_from_fields(
        mut position: Point3<fco>,
        current_time_idx: usize,
        number_of_scalar_quantities: usize,
        number_of_vector_quantities: usize,
        mass_density_field: &ScalarField3<fdt>,
        momentum_field: &VectorField3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    ) -> Self {
        let (position_indices, velocity) = evaluate_velocity(
            &mut position,
            mass_density_field,
            momentum_field,
            interpolator,
        )
        .expect("Initial position must be inside grid boundaries");
        Self::new(
            position,
            position_indices,
            velocity,
            current_time_idx,
            number_of_scalar_quantities,
            number_of_vector_quantities,
        )
    }

    #[allow(dead_code)]
    fn number_of_times(&self) -> usize {
        debug_assert!(
            !self.positions.is_empty(),
            "Cork trajectory should never be empty"
        );
        self.positions.len()
    }

    #[allow(dead_code)]
    fn last_time_idx(&self) -> usize {
        self.first_time_idx + self.number_of_times() - 1
    }

    fn last_position(&self) -> &Point3<fco> {
        self.positions
            .last()
            .expect("Cork trajectory should never be empty")
    }

    fn last_position_indices(&self) -> &Idx3<usize> {
        &self.last_position_indices
    }

    fn last_velocity(&self) -> &Vec3<fco> {
        self.velocities
            .last()
            .expect("Cork trajectory should never be empty")
    }

    fn is_terminated(&self) -> bool {
        self.terminated
    }

    fn add_next_position_and_velocity(
        &mut self,
        position: Point3<fco>,
        position_indices: Idx3<usize>,
        velocity: Vec3<fco>,
    ) {
        debug_assert_eq!(
            self.positions.len(),
            self.velocities.len(),
            "Number of positions and velocities should always be equal"
        );
        assert!(
            !self.is_terminated(),
            "Cannot add data after cork has been terminated"
        );
        self.positions.push(position);
        self.velocities.push(velocity);
        self.last_position_indices = position_indices;
    }

    fn add_scalar_quantity_value(
        &mut self,
        quantity_idx: usize,
        field: &ScalarField3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    ) {
        if self.is_terminated() {
            return;
        }
        let last_position = self.last_position().converted();
        let value = interpolator.interp_scalar_field_known_cell(
            field,
            &last_position,
            self.last_position_indices(),
        );
        self.scalar_field_values[quantity_idx].push(value as fdt);
        debug_assert_eq!(
            self.scalar_field_values[quantity_idx].len(),
            self.positions.len(),
            "Number of scalar field values should match number of positions"
        );
    }

    fn add_vector_quantity_value(
        &mut self,
        quantity_idx: usize,
        field: &VectorField3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    ) {
        if self.is_terminated() {
            return;
        }
        let last_position = self.last_position().converted();
        let value = interpolator.interp_vector_field_known_cell(
            field,
            &last_position,
            self.last_position_indices(),
        );
        self.vector_field_values[quantity_idx].push(value.cast());
        debug_assert_eq!(
            self.vector_field_values[quantity_idx].len(),
            self.positions.len(),
            "Number of vector field values should match number of positions"
        );
    }

    fn terminate(&mut self) {
        self.terminated = true;
    }
}

#[cfg(feature = "serialization")]
impl Serialize for Cork {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Cork", 5)?;
        s.serialize_field("positions", &self.positions)?;
        s.serialize_field("velocities", &self.velocities)?;
        s.serialize_field("scalar_field_values", &self.scalar_field_values)?;
        s.serialize_field("vector_field_values", &self.vector_field_values)?;
        s.serialize_field("first_time_idx", &self.first_time_idx)?;
        s.end()
    }
}

impl CorkSet {
    /// Creates a new collection of corks at the specified positions.
    ///
    /// # Parameters
    ///
    /// - `number_of_corks`: The total number of corks.
    /// - `seeder`: Seeder providing the positions of the corks at the initial time.
    /// - `initial_snapshot`: Snapshot representing the atmosphere at the initial time.
    /// - `interpolator`: Interpolator to use.
    /// - `scalar_quantity_names`: List of scalar quantities to sample along cork trajectories.
    /// - `vector_quantity_names`: List of vector quantities to sample along cork trajectories.
    /// - `verbosity`: Whether and how to pass non-essential information to user.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a new `CorkSet` with initialized corks.
    /// - `Err`: Contains an error encountered while trying to read or interpret the initial snapshot.
    pub fn new(
        number_of_corks: usize,
        seeder: &dyn Seeder3,
        initial_snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
        scalar_quantity_names: Vec<String>,
        vector_quantity_names: Vec<String>,
        verbosity: Verbosity,
    ) -> io::Result<Self> {
        initial_snapshot.cache_scalar_field(MASS_DENSITY_VARIABLE_NAME)?;
        initial_snapshot.cache_vector_field(MOMENTUM_VARIABLE_NAME)?;

        let mass_density_field = initial_snapshot.cached_scalar_field(MASS_DENSITY_VARIABLE_NAME);
        let momentum_field = initial_snapshot.cached_vector_field(MOMENTUM_VARIABLE_NAME);

        let lower_bounds = initial_snapshot.grid().lower_bounds().cast();
        let upper_bounds = initial_snapshot.grid().upper_bounds().cast();

        let number_of_scalar_quantities = scalar_quantity_names.len();
        let number_of_vector_quantities = vector_quantity_names.len();

        let can_drop_mass_density_field = !scalar_quantity_names
            .iter()
            .any(|name| name == MASS_DENSITY_VARIABLE_NAME);

        let progress_bar = verbosity.create_progress_bar(number_of_corks);

        let mut corks = Self {
            corks: seeder
                .points()
                .par_iter()
                .map(|position| {
                    let corks = Cork::new_from_fields(
                        position.clone(),
                        0,
                        number_of_scalar_quantities,
                        number_of_vector_quantities,
                        mass_density_field,
                        momentum_field,
                        interpolator,
                    );
                    progress_bar.inc();
                    corks
                })
                .collect(),
            times: vec![0.0],
            lower_bounds,
            upper_bounds,
            scalar_quantity_names,
            vector_quantity_names,
            can_drop_mass_density_field,
            verbosity,
        };
        if corks.verbosity().print_messages() {
            println!("Initialized {} corks", corks.number_of_corks());
        }

        initial_snapshot.drop_vector_field(MOMENTUM_VARIABLE_NAME);
        if corks.can_drop_mass_density_field() {
            initial_snapshot.drop_scalar_field(MASS_DENSITY_VARIABLE_NAME);
        }

        corks.sample_field_values(initial_snapshot, interpolator)?;

        Ok(corks)
    }

    pub fn verbosity(&self) -> &Verbosity {
        &self.verbosity
    }

    fn number_of_corks(&self) -> usize {
        self.corks.len()
    }

    #[allow(dead_code)]
    fn number_of_times(&self) -> usize {
        debug_assert!(!self.times.is_empty());
        self.times.len()
    }

    fn current_time(&self) -> fco {
        *self.times.last().unwrap()
    }

    #[allow(dead_code)]
    fn current_time_idx(&self) -> usize {
        self.number_of_times() - 1
    }

    fn scalar_quantity_names(&self) -> &[String] {
        &self.scalar_quantity_names
    }

    fn vector_quantity_names(&self) -> &[String] {
        &self.vector_quantity_names
    }

    fn can_drop_mass_density_field(&self) -> bool {
        self.can_drop_mass_density_field
    }

    #[allow(dead_code)]
    fn create_cork(
        &mut self,
        position: Point3<fco>,
        mass_density_field: &ScalarField3<fdt>,
        momentum_field: &VectorField3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    ) {
        self.corks.push(Cork::new_from_fields(
            position,
            self.current_time_idx(),
            self.scalar_quantity_names().len(),
            self.vector_quantity_names().len(),
            mass_density_field,
            momentum_field,
            interpolator,
        ));
    }

    fn advance<A>(&mut self, advancer: &A, step_duration: fco)
    where
        A: Fn(&mut Cork) + Sync,
    {
        if self.verbosity().print_messages() {
            println!("Advancing corks");
        }
        self.times.push(self.current_time() + step_duration);
        self.update(advancer);
    }

    fn update<U>(&mut self, updater: &U)
    where
        U: Fn(&mut Cork) + Sync,
    {
        self.corks.par_iter_mut().for_each(updater);
    }

    fn sample_field_values(
        &mut self,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    ) -> io::Result<()> {
        if self.verbosity().print_messages() {
            println!("Sampling field values");
        }
        #[allow(clippy::unnecessary_to_owned)]
        for (quantity_idx, name) in self
            .scalar_quantity_names()
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let field = snapshot.provide_scalar_field(&name)?;
            self.update(&|cork: &mut Cork| {
                cork.add_scalar_quantity_value(quantity_idx, field.as_ref(), interpolator)
            });
        }
        #[allow(clippy::unnecessary_to_owned)]
        for (quantity_idx, name) in self
            .vector_quantity_names()
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let field = snapshot.provide_vector_field(&name)?;
            self.update(&|cork: &mut Cork| {
                cork.add_vector_quantity_value(quantity_idx, field.as_ref(), interpolator)
            });
        }
        Ok(())
    }

    /// Serializes the cork data into JSON format and writes to the given writer.
    #[cfg(feature = "json")]
    pub fn write_as_json<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        utils::write_data_as_json(writer, &self)
    }

    /// Serializes the cork data into JSON format and saves at the given path.
    #[cfg(feature = "json")]
    pub fn save_as_json(&self, output_file_path: &Path) -> io::Result<()> {
        utils::save_data_as_json(output_file_path, &self)
    }

    /// Serializes the cork data into pickle format and writes to the given writer.
    ///
    /// All the cork data is saved as a single pickled structure.
    #[cfg(feature = "pickle")]
    pub fn write_as_pickle<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        utils::write_data_as_pickle(writer, &self)
    }

    /// Serializes the cork data into pickle format and saves at the given path.
    ///
    /// All the cork data is saved as a single pickled structure.
    #[cfg(feature = "pickle")]
    pub fn save_as_pickle(&self, output_file_path: &Path) -> io::Result<()> {
        utils::save_data_as_pickle(output_file_path, &self)
    }
}

#[cfg(feature = "serialization")]
impl Serialize for CorkSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("CorkSet", 7)?;
        s.serialize_field("corks", &self.corks)?;
        s.serialize_field("times", &self.times)?;
        s.serialize_field("lower_bounds", &self.lower_bounds)?;
        s.serialize_field("upper_bounds", &self.upper_bounds)?;
        s.serialize_field("scalar_quantity_names", &self.scalar_quantity_names)?;
        s.serialize_field("vector_quantity_names", &self.vector_quantity_names)?;
        s.end()
    }
}

/// Defines a method for stepping corks forward in time.
pub trait CorkStepper: Sync {
    fn step_one_cork(
        &self,
        cork: &mut Cork,
        step_duration: fco,
        mass_density_field: &ScalarField3<fdt>,
        momentum_field: &VectorField3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    );

    fn step_all_corks(
        &self,
        corks: &mut CorkSet,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
        step_duration: fco,
    ) -> io::Result<()> {
        snapshot.cache_scalar_field(MASS_DENSITY_VARIABLE_NAME)?;
        snapshot
            .cache_vector_field(MOMENTUM_VARIABLE_NAME)
            .or_else(|_| snapshot.cache_vector_field(MOMENTUM_VARIABLE_NAME))?;

        let mass_density_field = snapshot.cached_scalar_field(MASS_DENSITY_VARIABLE_NAME);
        let momentum_field = snapshot.cached_vector_field(MOMENTUM_VARIABLE_NAME);

        corks.advance(
            &|cork: &mut Cork| {
                self.step_one_cork(
                    cork,
                    step_duration,
                    mass_density_field,
                    momentum_field,
                    interpolator,
                )
            },
            step_duration,
        );

        snapshot.drop_vector_field(MOMENTUM_VARIABLE_NAME);
        if corks.can_drop_mass_density_field() {
            snapshot.drop_scalar_field(MASS_DENSITY_VARIABLE_NAME);
        }

        corks.sample_field_values(snapshot, interpolator)
    }
}

impl CorkStepper for HeunCorkStepper {
    fn step_one_cork(
        &self,
        cork: &mut Cork,
        step_duration: fco,
        mass_density_field: &ScalarField3<fdt>,
        momentum_field: &VectorField3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
    ) {
        if cork.is_terminated() {
            return;
        }
        let mut next_position = cork.last_position() + cork.last_velocity() * step_duration;

        match evaluate_velocity(
            &mut next_position,
            mass_density_field,
            momentum_field,
            interpolator,
        ) {
            Some((_, next_velocity)) => {
                next_position = cork.last_position()
                    + (cork.last_velocity() + next_velocity) * (0.5 * step_duration);
                match evaluate_velocity(
                    &mut next_position,
                    mass_density_field,
                    momentum_field,
                    interpolator,
                ) {
                    Some((next_position_indices, next_velocity)) => {
                        cork.add_next_position_and_velocity(
                            next_position,
                            next_position_indices,
                            next_velocity,
                        );
                    }
                    None => cork.terminate(),
                }
            }
            None => cork.terminate(),
        }
    }
}

pub trait CorkAdvector {
    /// Advects the given corks in the velocity field of the given snapshot,
    /// for one step corresponding to the duration between subsequent snapshots.
    ///
    /// # Parameters
    ///
    /// - `corks`: The set of corks to advect.
    /// - `snapshot`: Snapshot representing the atmosphere at the time to advect to.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper`: Time stepping scheme.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Empty.
    /// - `Err`: Contains an error encountered while trying to read or interpret the snapshot.
    ///
    /// # Type parameters
    ///
    /// - `I`: Type of interpolator.
    /// - `S`: Type of stepper.
    fn advect_corks<S>(
        &self,
        corks: &mut CorkSet,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
        stepper: &S,
    ) -> io::Result<()>
    where
        S: CorkStepper;
}

impl CorkAdvector for ConstantCorkAdvector {
    fn advect_corks<S>(
        &self,
        corks: &mut CorkSet,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
        stepper: &S,
    ) -> io::Result<()>
    where
        S: CorkStepper,
    {
        stepper.step_all_corks(corks, snapshot, interpolator, self.step_duration)
    }
}

fn evaluate_velocity(
    position: &mut Point3<fco>,
    mass_density_field: &ScalarField3<fdt>,
    momentum_field: &VectorField3<fdt>,
    interpolator: &dyn Interpolator3<fdt>,
) -> Option<(Idx3<usize>, Vec3<fco>)> {
    let interp_point = position.converted();
    let grid_point_query = mass_density_field.grid().find_grid_cell(&interp_point);

    if grid_point_query == GridPointQuery3::Outside {
        None
    } else {
        let interp_indices = grid_point_query.unwrap_and_update_position(position);

        let mass_density = interpolator.interp_scalar_field_known_cell(
            mass_density_field,
            &interp_point,
            &interp_indices,
        );
        let momentum = interpolator.interp_vector_field_known_cell(
            momentum_field,
            &interp_point,
            &interp_indices,
        );
        let velocity = momentum / mass_density;

        Some((interp_indices, velocity.cast()))
    }
}
