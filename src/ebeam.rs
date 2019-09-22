//! Non-thermal electron beam physics in Bifrost simulations.

pub mod distribution;
pub mod accelerator;
pub mod execution;

use std::{io, path, fs};
use std::io::Write;
use std::collections::HashMap;
use serde::Serialize;
use serde::ser::{Serializer, SerializeStruct};
use rayon::prelude::*;
use crate::num::BFloat;
use crate::io::Verbose;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::io::utils;
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::field::{ScalarField3, VectorField3};
use crate::interpolation::Interpolator3;
use crate::tracing::{self, ftr, TracerResult};
use crate::tracing::seeding::IndexSeeder3;
use crate::tracing::stepping::{StepperInstruction, Stepper3, StepperFactory3};
use self::distribution::{DepletionStatus, PropagationResult, Distribution};
use self::accelerator::Accelerator;

/// Floating-point precision to use for electron beam physics.
#[allow(non_camel_case_types)]
pub type feb = f64;

pub type FixedBeamScalarValues = HashMap<String, Vec<feb>>;
pub type FixedBeamVectorValues = HashMap<String, Vec<Vec3<feb>>>;
pub type VaryingBeamScalarValues = HashMap<String, Vec<Vec<feb>>>;
pub type VaryingBeamVectorValues = HashMap<String, Vec<Vec<Vec3<feb>>>>;

/// Defines the required behaviour of a type representing electron beam distribution properties.
pub trait ElectronBeamProperties {
    type NestedTuplesOfValues: Send;
    type NestedTuplesOfVecs: Default + ParallelExtend<Self::NestedTuplesOfValues> + Send;

    /// Aranges the property values into a set of nested tuples.
    ///
    /// This is required in order to unzip a vector of properties structs into
    /// a set of vectors each holding values for a single property.
    fn into_nested_tuples_of_values(self) -> Self::NestedTuplesOfValues;

    /// Takes the property values from a nested tuple of vectors
    /// (acquired by unzipping a vector of `NestedTuplesOfValues`)
    /// and inserts them with the approptiate keys into the
    /// appropriate hash maps.
    fn distribute_nested_tuples_of_vecs_into_maps(vecs: Self::NestedTuplesOfVecs, scalar_values: &mut FixedBeamScalarValues, vector_values: &mut FixedBeamVectorValues);
}

/// Marker trait for electron beam metadata types.
pub trait ElectronBeamMetadata: Clone + std::fmt::Debug + Serialize + Send + Sync {}

/// A set of non-thermal electron beams.
#[derive(Clone, Debug)]
pub struct ElectronBeamSwarm<A: Accelerator> {
    trajectories: Vec<Vec<Point3<ftr>>>,
    fixed_scalar_values: FixedBeamScalarValues,
    fixed_vector_values: FixedBeamVectorValues,
    varying_scalar_values: VaryingBeamScalarValues,
    varying_vector_values: VaryingBeamVectorValues,
    metadata: Vec<<A::DistributionType as Distribution>::MetadataType>
}

struct ElectronBeam<D: Distribution>  {
    trajectory: Vec<Point3<ftr>>,
    distribution_properties: D::PropertiesType,
    total_propagation_distance: feb,
    deposited_power_densities: Vec<feb>,
    metadata: D::MetadataType
}

impl<A> FromParallelIterator<ElectronBeam<A::DistributionType>> for ElectronBeamSwarm<A>
where A: Accelerator,
      <A::DistributionType as Distribution>::PropertiesType: Send
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where I: IntoParallelIterator<Item=ElectronBeam<A::DistributionType>>
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(
            |data| (data.trajectory,
                    (data.total_propagation_distance,
                     (data.deposited_power_densities,
                      (data.metadata,
                        data.distribution_properties.into_nested_tuples_of_values()))))
        );

        #[allow(clippy::type_complexity)]
        let (trajectories,
              (total_propagation_distances,
               (deposited_power_densities,
                (metadata,
                 distribution_properties)))):
            (Vec<_>,
             (Vec<_>,
              (Vec<_>,
               (Vec<_>,
                <<A::DistributionType as Distribution>::PropertiesType as ElectronBeamProperties>::NestedTuplesOfVecs))))
            = nested_tuples_iter.unzip();

        let mut fixed_scalar_values = HashMap::new();
        let mut fixed_vector_values = HashMap::new();
        let mut varying_scalar_values = HashMap::new();
        let varying_vector_values = HashMap::new();

        <A::DistributionType as Distribution>::PropertiesType::distribute_nested_tuples_of_vecs_into_maps(
            distribution_properties,
            &mut fixed_scalar_values,
            &mut fixed_vector_values
        );

        fixed_scalar_values.insert("total_propagation_distance".to_string(), total_propagation_distances);
        varying_scalar_values.insert("deposited_power_density".to_string(), deposited_power_densities);

        ElectronBeamSwarm{
            trajectories,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
            metadata
        }
    }
}

impl<A: Accelerator> ElectronBeamSwarm<A> {
    /// Generates a set of electron beams using the given seeder and accelerator
    /// but does not propagate them.
    ///
    /// # Parameters
    ///
    /// - `seeder`: Seeder to use for generating acceleration positions.
    /// - `snapshot`: Snapshot representing the atmosphere.
    /// - `accelerator`: Accelerator to use for generating electron distributions.
    /// - `interpolator`: Interpolator to use.
    /// - `verbose`: Whether to print status messages.
    ///
    /// # Returns
    ///
    /// A new `ElectronBeamSwarm` with unpropagated electron beams.
    ///
    /// # Type parameters
    ///
    /// - `Sd`: Type of index seeder.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    pub fn generate_unpropagated<Sd, G, I>(seeder: Sd, snapshot: &mut SnapshotCacher3<G>, accelerator: A, interpolator: &I, verbose: Verbose) -> Self
    where Sd: IndexSeeder3,
          G: Grid3<fdt>,
          A: Accelerator + Sync + Send,
          A::DistributionType: Send,
          <A::DistributionType as Distribution>::PropertiesType: Send,
          I: Interpolator3
    {
        A::prepare_snapshot_for_generation(snapshot).unwrap_or_else(|err| panic!("Snapshot preparation failed: {}", err));

        if verbose.is_yes() { println!("Generating electron distributions at {} acceleration sites", seeder.number_of_indices()); }
        let seed_iter = seeder.into_par_iter();
        seed_iter.filter_map(
            |indices| {
                accelerator.generate_distribution(snapshot, interpolator, &indices)
                           .map(Self::generate_unpropagated_beam)
            }
        ).collect()
    }

    /// Generates a set of electron beams using the given seeder and accelerator,
    /// and propagates them through the atmosphere in the given snapshot.
    ///
    /// # Parameters
    ///
    /// - `seeder`: Seeder to use for generating start positions.
    /// - `snapshot`: Snapshot representing the atmosphere.
    /// - `accelerator`: Accelerator to use for generating initial electron distributions.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper_factory`: Factory structure to use for producing steppers.
    /// - `verbose`: Whether to print status messages.
    ///
    /// # Returns
    ///
    /// A new `ElectronBeamSwarm` with propagated electron beams.
    ///
    /// # Type parameters
    ///
    /// - `Sd`: Type of index seeder.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `StF`: Type of stepper factory.
    pub fn generate_propagated<Sd, G, I, StF>(seeder: Sd, snapshot: &mut SnapshotCacher3<G>, accelerator: A, interpolator: &I, stepper_factory: StF, verbose: Verbose) -> Self
    where Sd: IndexSeeder3,
          G: Grid3<fdt>,
          A: Accelerator + Sync + Send,
          A::DistributionType: Send,
          <A::DistributionType as Distribution>::PropertiesType: Send,
          I: Interpolator3,
          StF: StepperFactory3 + Sync
    {
        A::prepare_snapshot_for_generation(snapshot).unwrap_or_else(|err| panic!("Snapshot preparation failed: {}", err));

        if verbose.is_yes() { println!("Generating electron distributions at {} acceleration sites", seeder.number_of_indices()); }
        let seed_iter = seeder.into_par_iter();
        let distributions: Vec<_> = seed_iter.filter_map(
            |indices| {
                accelerator.generate_distribution(snapshot, interpolator, &indices)
            }
        ).collect();

        A::prepare_snapshot_for_propagation(snapshot).unwrap_or_else(|err| panic!("Snapshot preparation failed: {}", err));

        if verbose.is_yes() { println!("Attempting to propagate {} electron distributions", distributions.len()); }
        let electron_beam_swarm: Self = distributions.into_par_iter().filter_map(
            |distribution| Self::generate_propagated_beam(distribution, snapshot, interpolator, stepper_factory.produce())
        ).collect();

        if verbose.is_yes() { println!("Successfully propagated {} electron distributions", electron_beam_swarm.number_of_beams()); }
        electron_beam_swarm
    }

    /// Returns the number of beams making up the electron beam set.
    pub fn number_of_beams(&self) -> usize { self.trajectories.len() }

    /// Extracts and stores the value of the given scalar field at the initial position for each beam.
    pub fn extract_fixed_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        let values = self.trajectories.par_iter().map(
            |trajectory| {
                let value = interpolator.interp_scalar_field(field, &Point3::from(&trajectory[0])).expect_inside();
                num::NumCast::from(value).expect("Conversion failed.")
            }
        ).collect();
        self.fixed_scalar_values.insert(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at the initial position for each beam.
    pub fn extract_fixed_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        let vectors = self.trajectories.par_iter().map(
            |trajectory| {
                let vector = interpolator.interp_vector_field(field, &Point3::from(&trajectory[0])).expect_inside();
                Vec3::from(&vector)
            }
        ).collect();
        self.fixed_vector_values.insert(field.name().to_string(), vectors);
    }

    /// Extracts and stores the value of the given scalar field at each position for each beam.
    pub fn extract_varying_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        let values = self.trajectories.par_iter().map(
            |trajectory| {
                let mut values_single_beam = Vec::with_capacity(trajectory.len());
                for pos in trajectory {
                    let value = interpolator.interp_scalar_field(field, &Point3::from(pos)).expect_inside();
                    values_single_beam.push(num::NumCast::from(value).expect("Conversion failed."));
                }
                values_single_beam
            }
        ).collect();
        self.varying_scalar_values.insert(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at each position for each beam.
    pub fn extract_varying_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        let vectors = self.trajectories.par_iter().map(
            |trajectory| {
                let mut vectors_single_beam = Vec::with_capacity(trajectory.len());
                for pos in trajectory {
                    let vector = interpolator.interp_vector_field(field, &Point3::from(pos)).expect_inside();
                    vectors_single_beam.push(Vec3::from(&vector));
                }
                vectors_single_beam
            }
        ).collect();
        self.varying_vector_values.insert(field.name().to_string(), vectors);
    }

    /// Serializes the electron beam data into pickle format and saves at the given path.
    ///
    /// All the electron beam data is saved as a single pickled structure.
    pub fn save_as_pickle<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        utils::save_data_as_pickle(file_path, &self)
    }

    /// Serializes the electron beam data fields in parallel into pickle format and saves at the given path.
    ///
    /// The data fields are saved as separate pickle objects in the same file.
    pub fn save_as_combined_pickles<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        let (mut result_1, mut result_2, mut result_3, mut result_4, mut result_5, mut result_6) = (Ok(()), Ok(()), Ok(()), Ok(()), Ok(()), Ok(()));
        let (mut buffer_1, mut buffer_2, mut buffer_3, mut buffer_4, mut buffer_5, mut buffer_6) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        rayon::scope(|s| {
            s.spawn(|_| result_1 = utils::write_data_as_pickle(&mut buffer_1, &self.trajectories));
            s.spawn(|_| result_2 = utils::write_data_as_pickle(&mut buffer_2, &self.fixed_scalar_values));
            s.spawn(|_| result_3 = utils::write_data_as_pickle(&mut buffer_3, &self.fixed_vector_values));
            s.spawn(|_| result_4 = utils::write_data_as_pickle(&mut buffer_4, &self.varying_scalar_values));
            s.spawn(|_| result_5 = utils::write_data_as_pickle(&mut buffer_5, &self.varying_vector_values));
            s.spawn(|_| result_6 = utils::write_data_as_pickle(&mut buffer_6, &self.metadata));
        });
        result_1?; result_2?; result_3?; result_4?; result_5?; result_6?;
        let mut file = fs::File::create(file_path)?;
        file.write_all(&[buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, buffer_6].concat())?;
        Ok(())
    }

    fn generate_unpropagated_beam(distribution: A::DistributionType) -> ElectronBeam<A::DistributionType> {
        ElectronBeam{
            trajectory: vec![Point3::from(distribution.acceleration_position())],
            distribution_properties: distribution.properties(),
            total_propagation_distance: 0.0,
            deposited_power_densities: vec![0.0],
            metadata: distribution.metadata()
        }
    }

    fn generate_propagated_beam<G, I, S>(mut distribution: A::DistributionType, snapshot: &SnapshotCacher3<G>, interpolator: &I, stepper: S) -> Option<ElectronBeam<A::DistributionType>>
    where G: Grid3<fdt>,
          I: Interpolator3,
          S: Stepper3
    {
        let mut trajectory = Vec::new();
        let mut deposited_power_densities = Vec::new();
        let mut total_propagation_distance = 0.0;

        let magnetic_field = snapshot.cached_vector_field("b");
        let start_position = Point3::from(distribution.acceleration_position());

        let tracer_result = tracing::trace_3d_field_line_dense(magnetic_field, interpolator, stepper, &start_position, distribution.propagation_sense(),
            &mut |displacement, position, distance| {
                let PropagationResult{
                    deposited_power_density,
                    deposition_position,
                    depletion_status
                } = distribution.propagate(snapshot, interpolator, displacement, position);

                trajectory.push(deposition_position);
                deposited_power_densities.push(deposited_power_density);
                total_propagation_distance = distance;

                match depletion_status {
                    DepletionStatus::Undepleted => StepperInstruction::Continue,
                    DepletionStatus::Depleted => StepperInstruction::Terminate
                }
            }
        );

        let distribution_properties = distribution.properties();
        let metadata = distribution.metadata();

        match tracer_result {
            TracerResult::Ok(_) => Some(ElectronBeam{
                trajectory,
                distribution_properties,
                total_propagation_distance,
                deposited_power_densities,
                metadata
            }),
            TracerResult::Void => None
        }
    }
}

impl<A: Accelerator> Serialize for ElectronBeamSwarm<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("ElectronBeamSwarm", 6)?;
        s.serialize_field("trajectories", &self.trajectories)?;
        s.serialize_field("fixed_scalar_values", &self.fixed_scalar_values)?;
        s.serialize_field("fixed_vector_values", &self.fixed_vector_values)?;
        s.serialize_field("varying_scalar_values", &self.varying_scalar_values)?;
        s.serialize_field("varying_vector_values", &self.varying_vector_values)?;
        s.serialize_field("metadata", &self.metadata)?;
        s.end()
    }
}
