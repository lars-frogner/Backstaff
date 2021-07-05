//! Tracing of corks in an evolving velocity field.

use crate::{
    field::{ScalarField3, VectorField3},
    geometry::{Idx3, Point3, Vec3},
    grid::{Grid3, GridPointQuery3},
    interpolation::Interpolator3,
    io::{
        snapshot::{
            self, fdt, SnapshotCacher3, SnapshotParameters, SnapshotReader3,
            MASS_DENSITY_VARIABLE_NAME, MOMENTUM_VARIABLE_NAME, OUTPUT_TIME_STEP_NAME,
        },
        utils, Verbose,
    },
};
use rayon::prelude::*;
use serde::{
    ser::{SerializeStruct, Serializer},
    Serialize,
};
use std::{io, iter, path::Path};

/// Floating-point precision to use for cork tracing.
#[allow(non_camel_case_types)]
pub type fco = f32;

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
    /// Names of the vector fields whos magnitude should be sampled along the cork trajectories.
    vector_magnitude_names: Vec<String>,
    /// Names of the vector fields that should be sampled along the cork trajectories.
    vector_quantity_names: Vec<String>,
    /// Whether the mass density should not be sampled and thus can be dropped after each step.
    /// (The momentum field is always dropped since it can be recomputed from velocity and density.)
    can_drop_mass_density_field: bool,
    /// Whether to print status messages.
    verbose: Verbose,
}

/// Uses the Heun method for advecting corks.
#[derive(Clone, Copy, Debug)]
pub struct HeunCorkStepper;

/// Traces a constant number of initial corks.
#[derive(Clone, Copy, Debug)]
pub struct ConstantCorkAdvector;

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
            scalar_field_values: iter::repeat_with(|| Vec::new())
                .take(number_of_scalar_quantities)
                .collect(),
            vector_field_values: iter::repeat_with(|| Vec::new())
                .take(number_of_vector_quantities)
                .collect(),
        }
    }

    fn new_from_fields<G, I>(
        mut position: Point3<fco>,
        current_time_idx: usize,
        number_of_scalar_quantities: usize,
        number_of_vector_quantities: usize,
        mass_density_field: &ScalarField3<fdt, G>,
        momentum_field: &VectorField3<fdt, G>,
        interpolator: &I,
    ) -> Self
    where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
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

    fn number_of_times(&self) -> usize {
        debug_assert!(
            !self.positions.is_empty(),
            "Cork trajectory should never be empty"
        );
        self.positions.len()
    }

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

    fn add_scalar_quantity_value<G, I>(
        &mut self,
        quantity_idx: usize,
        field: &ScalarField3<fdt, G>,
        interpolator: &I,
    ) where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
        if self.is_terminated() {
            return;
        }
        let value = interpolator.interp_scalar_field_known_cell(
            field,
            self.last_position(),
            self.last_position_indices(),
        );
        self.scalar_field_values[quantity_idx].push(value);
        debug_assert_eq!(
            self.scalar_field_values[quantity_idx].len(),
            self.positions.len(),
            "Number of scalar field values should match number of positions"
        );
    }

    fn add_vector_quantity_value<G, I>(
        &mut self,
        quantity_idx: usize,
        field: &VectorField3<fdt, G>,
        interpolator: &I,
    ) where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
        if self.is_terminated() {
            return;
        }
        let value = interpolator.interp_vector_field_known_cell(
            field,
            self.last_position(),
            self.last_position_indices(),
        );
        self.vector_field_values[quantity_idx].push(value);
        debug_assert_eq!(
            self.vector_field_values[quantity_idx].len(),
            self.positions.len(),
            "Number of vector field values should match number of positions"
        );
    }

    fn add_vector_magnitude_value<G, I>(
        &mut self,
        quantity_idx: usize,
        field: &VectorField3<fdt, G>,
        interpolator: &I,
    ) where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
        if self.is_terminated() {
            return;
        }
        let value = interpolator
            .interp_vector_field_known_cell(
                field,
                self.last_position(),
                self.last_position_indices(),
            )
            .length();
        self.scalar_field_values[quantity_idx].push(value);
        debug_assert_eq!(
            self.scalar_field_values[quantity_idx].len(),
            self.positions.len(),
            "Number of scalar field values should match number of positions"
        );
    }

    fn terminate(&mut self) {
        self.terminated = true;
    }
}

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
    /// - `initial_positions`: Iterator over the positions of the corks at the initial time.
    /// - `initial_snapshot`: Snapshot representing the atmosphere at the initial time.
    /// - `interpolator`: Interpolator to use.
    /// - `scalar_quantity_names`: List of scalar quantities to sample along cork trajectories.
    /// - `vector_quantity_names`: List of vector quantities to sample along cork trajectories.
    /// - `vector_magnitude_names`: List of vector quantities to sample magnitude of along cork trajectories.
    /// - `verbose`: Whether to print status messages.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a new `CorkSet` with initialized corks.
    /// - `Err`: Contains an error encountered while trying to read or interpret the initial snapshot.
    ///
    /// # Type parameters
    ///
    /// - `G`: Type of grid.
    /// - `R`: Type of snapshot reader.
    /// - `I`: Type of interpolator.
    pub fn new<P, G, R, I>(
        initial_positions: P,
        initial_snapshot: &mut SnapshotCacher3<G, R>,
        interpolator: &I,
        scalar_quantity_names: Vec<String>,
        vector_magnitude_names: Vec<String>,
        vector_quantity_names: Vec<String>,
        verbose: Verbose,
    ) -> io::Result<Self>
    where
        P: IntoParallelIterator<Item = Point3<fco>>,
        G: Grid3<fdt>,
        R: SnapshotReader3<G> + Sync,
        I: Interpolator3,
    {
        initial_snapshot.cache_scalar_field(MASS_DENSITY_VARIABLE_NAME)?;
        initial_snapshot.cache_vector_field(MOMENTUM_VARIABLE_NAME)?;

        let mass_density_field = initial_snapshot.cached_scalar_field(MASS_DENSITY_VARIABLE_NAME);
        let momentum_field = initial_snapshot.cached_vector_field(MOMENTUM_VARIABLE_NAME);

        let lower_bounds = initial_snapshot.reader().grid().lower_bounds().clone();
        let upper_bounds = initial_snapshot.reader().grid().upper_bounds().clone();

        let number_of_scalar_quantities =
            scalar_quantity_names.len() + vector_magnitude_names.len();
        let number_of_vector_quantities = vector_quantity_names.len();

        let can_drop_mass_density_field = !scalar_quantity_names
            .iter()
            .any(|name| name == MASS_DENSITY_VARIABLE_NAME);

        let mut corks = Self {
            corks: initial_positions
                .into_par_iter()
                .map(|position| {
                    Cork::new_from_fields(
                        position,
                        0,
                        number_of_scalar_quantities,
                        number_of_vector_quantities,
                        mass_density_field,
                        momentum_field,
                        interpolator,
                    )
                })
                .collect(),
            times: vec![0.0],
            lower_bounds,
            upper_bounds,
            scalar_quantity_names,
            vector_quantity_names,
            vector_magnitude_names,
            can_drop_mass_density_field,
            verbose,
        };
        if corks.verbose().is_yes() {
            println!("Initialized {} corks", corks.number_of_corks());
        }

        initial_snapshot.drop_vector_field(MOMENTUM_VARIABLE_NAME);
        if corks.can_drop_mass_density_field() {
            initial_snapshot.drop_scalar_field(MASS_DENSITY_VARIABLE_NAME);
        }

        corks.sample_field_values(initial_snapshot, interpolator)?;

        Ok(corks)
    }

    /// Whether the cork set is verbose.
    pub fn verbose(&self) -> Verbose {
        self.verbose
    }

    fn number_of_corks(&self) -> usize {
        self.corks.len()
    }

    fn number_of_times(&self) -> usize {
        debug_assert!(!self.times.is_empty());
        self.times.len()
    }

    fn current_time(&self) -> fco {
        *self.times.last().unwrap()
    }

    fn current_time_idx(&self) -> usize {
        self.number_of_times() - 1
    }

    fn scalar_quantity_names(&self) -> &[String] {
        &self.scalar_quantity_names
    }

    fn vector_magnitude_names(&self) -> &[String] {
        &self.vector_magnitude_names
    }

    fn piped_vector_magnitude_names(&self) -> Vec<String> {
        self.vector_magnitude_names
            .iter()
            .map(snapshot::add_magnitude_pipes)
            .collect()
    }

    fn vector_quantity_names(&self) -> &[String] {
        &self.vector_quantity_names
    }

    fn can_drop_mass_density_field(&self) -> bool {
        self.can_drop_mass_density_field
    }

    fn create_cork<G, I>(
        &mut self,
        position: Point3<fco>,
        mass_density_field: &ScalarField3<fdt, G>,
        momentum_field: &VectorField3<fdt, G>,
        interpolator: &I,
    ) where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
        self.corks.push(Cork::new_from_fields(
            position,
            self.current_time_idx(),
            self.scalar_quantity_names().len() + self.vector_magnitude_names().len(),
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
        if self.verbose().is_yes() {
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

    fn sample_field_values<G, R, I>(
        &mut self,
        snapshot: &mut SnapshotCacher3<G, R>,
        interpolator: &I,
    ) -> io::Result<()>
    where
        G: Grid3<fdt>,
        R: SnapshotReader3<G> + Sync,
        I: Interpolator3,
    {
        if self.verbose().is_yes() {
            println!("Sampling field values");
        }
        for (quantity_idx, name) in self
            .scalar_quantity_names()
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let field = snapshot.obtain_scalar_field(&name)?;
            self.update(&|cork: &mut Cork| {
                cork.add_scalar_quantity_value(quantity_idx, field, interpolator)
            });
            snapshot.drop_scalar_field(&name);
        }
        for (idx, name) in self
            .vector_magnitude_names()
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let quantity_idx = self.scalar_quantity_names().len() + idx;
            let field = snapshot.obtain_vector_field(&name)?;
            self.update(&|cork: &mut Cork| {
                cork.add_vector_magnitude_value(quantity_idx, field, interpolator)
            });
            snapshot.drop_vector_field(&name);
        }
        for (quantity_idx, name) in self
            .vector_quantity_names()
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let field = snapshot.obtain_vector_field(&name)?;
            self.update(&|cork: &mut Cork| {
                cork.add_vector_quantity_value(quantity_idx, field, interpolator)
            });
            snapshot.drop_vector_field(&name);
        }
        Ok(())
    }

    /// Serializes the cork data into JSON format and writes to the given writer.
    pub fn write_as_json<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        utils::write_data_as_json(writer, &self)
    }

    /// Serializes the cork data into JSON format and saves at the given path.
    pub fn save_as_json<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()> {
        utils::save_data_as_json(output_file_path, &self)
    }

    /// Serializes the cork data into pickle format and writes to the given writer.
    ///
    /// All the cork data is saved as a single pickled structure.
    pub fn write_as_pickle<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        utils::write_data_as_pickle(writer, &self)
    }

    /// Serializes the cork data into pickle format and saves at the given path.
    ///
    /// All the cork data is saved as a single pickled structure.
    pub fn save_as_pickle<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()> {
        utils::save_data_as_pickle(output_file_path, &self)
    }
}

impl Serialize for CorkSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("CorkSet", 7)?;
        s.serialize_field("corks", &self.corks)?;
        s.serialize_field("times", &self.times)?;
        s.serialize_field("lower_bounds", &self.lower_bounds)?;
        s.serialize_field("upper_bounds", &self.upper_bounds)?;
        s.serialize_field("scalar_quantity_names", &self.scalar_quantity_names)?;
        s.serialize_field(
            "vector_magnitude_names",
            &self.piped_vector_magnitude_names(),
        )?;
        s.serialize_field("vector_quantity_names", &self.vector_quantity_names)?;
        s.end()
    }
}

/// Defines a method for stepping corks forward in time.
pub trait CorkStepper: Sync {
    fn step_one_cork<G, I>(
        &self,
        cork: &mut Cork,
        step_duration: fco,
        mass_density_field: &ScalarField3<fdt, G>,
        momentum_field: &VectorField3<fdt, G>,
        interpolator: &I,
    ) where
        G: Grid3<fdt>,
        I: Interpolator3;

    fn step_all_corks<G, R, I>(
        &self,
        corks: &mut CorkSet,
        snapshot: &mut SnapshotCacher3<G, R>,
        interpolator: &I,
    ) -> io::Result<()>
    where
        G: Grid3<fdt>,
        R: SnapshotReader3<G> + Sync,
        I: Interpolator3,
    {
        let step_duration = snapshot
            .reader()
            .parameters()
            .get_value(OUTPUT_TIME_STEP_NAME)?
            .try_as_float()?;

        snapshot.cache_scalar_field(MASS_DENSITY_VARIABLE_NAME)?;
        snapshot.cache_vector_field(MOMENTUM_VARIABLE_NAME)?;

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
    fn step_one_cork<G, I>(
        &self,
        cork: &mut Cork,
        step_duration: fco,
        mass_density_field: &ScalarField3<fdt, G>,
        momentum_field: &VectorField3<fdt, G>,
        interpolator: &I,
    ) where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
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
    /// - `G`: Type of grid.
    /// - `R`: Type of snapshot reader.
    /// - `I`: Type of interpolator.
    /// - `S`: Type of stepper.
    fn advect_corks<G, R, I, S>(
        &self,
        corks: &mut CorkSet,
        snapshot: &mut SnapshotCacher3<G, R>,
        interpolator: &I,
        stepper: &S,
    ) -> io::Result<()>
    where
        G: Grid3<fdt>,
        R: SnapshotReader3<G> + Sync,
        I: Interpolator3,
        S: CorkStepper;
}

impl CorkAdvector for ConstantCorkAdvector {
    fn advect_corks<G, R, I, S>(
        &self,
        corks: &mut CorkSet,
        snapshot: &mut SnapshotCacher3<G, R>,
        interpolator: &I,
        stepper: &S,
    ) -> io::Result<()>
    where
        G: Grid3<fdt>,
        R: SnapshotReader3<G> + Sync,
        I: Interpolator3,
        S: CorkStepper,
    {
        stepper.step_all_corks(corks, snapshot, interpolator)
    }
}

fn evaluate_velocity<G, I>(
    position: &mut Point3<fco>,
    mass_density_field: &ScalarField3<fdt, G>,
    momentum_field: &VectorField3<fdt, G>,
    interpolator: &I,
) -> Option<(Idx3<usize>, Vec3<fco>)>
where
    G: Grid3<fdt>,
    I: Interpolator3,
{
    let grid_point_query = mass_density_field.grid().find_grid_cell(position);

    if grid_point_query == GridPointQuery3::Outside {
        None
    } else {
        let interp_indices = grid_point_query.unwrap_and_update_position(position);

        let mass_density = interpolator.interp_scalar_field_known_cell(
            mass_density_field,
            position,
            &interp_indices,
        );
        let momentum =
            interpolator.interp_vector_field_known_cell(momentum_field, position, &interp_indices);
        let velocity = momentum / mass_density;

        Some((interp_indices, velocity))
    }
}
