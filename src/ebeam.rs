//! Non-thermal electron beam physics in Bifrost simulations.

pub mod accelerator;
pub mod detection;
pub mod distribution;

use self::{
    accelerator::Accelerator,
    detection::ReconnectionSiteDetector,
    distribution::{DepletionStatus, Distribution, PropagationResult},
};
use crate::{
    field::{ScalarField3, VectorField3},
    geometry::{
        Dim3::{X, Y, Z},
        Point3, Vec3,
    },
    grid::Grid3,
    interpolation::Interpolator3,
    io::{
        snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
        utils, Verbose,
    },
    num::BFloat,
    tracing::{
        self,
        field_line::{self, FieldLineSetProperties3},
        ftr,
        stepping::{Stepper3, StepperFactory3, StepperInstruction},
        TracerResult,
    },
};
use rayon::prelude::*;
use serde::{
    ser::{SerializeStruct, Serializer},
    Serialize,
};
use std::{
    collections::HashMap,
    io::{self, Write},
    path::Path,
};

/// Floating-point precision to use for electron beam physics.
#[allow(non_camel_case_types)]
pub type feb = f64;

type BeamTrajectory = (Vec<ftr>, Vec<ftr>, Vec<ftr>);
type FixedBeamScalarValues = HashMap<String, Vec<feb>>;
type FixedBeamVectorValues = HashMap<String, Vec<Vec3<feb>>>;
type VaryingBeamScalarValues = HashMap<String, Vec<Vec<feb>>>;
type VaryingBeamVectorValues = HashMap<String, Vec<Vec<Vec3<feb>>>>;

/// Defines the required behaviour of a type representing
/// a collection of objects holding electron beam properties.
pub trait BeamPropertiesCollection: Default + Sync + Send {
    type Item: Send;

    /// Moves the property values into the appropriate entries of the
    /// gives hash maps.
    fn distribute_into_maps(
        self,
        scalar_values: &mut FixedBeamScalarValues,
        vector_values: &mut FixedBeamVectorValues,
    );
}

/// Defines the required behaviour of a type representing
/// a collection of objects holding electron beam acceleration data.
pub trait AccelerationDataCollection: Serialize + Sync {
    /// Writes the acceleration data into the given writer.
    fn write<W: io::Write>(&self, format_hint: &str, writer: &mut W) -> io::Result<()>;

    /// Writes the acceleration data into the given writer,
    /// consuming the data object.
    fn write_into<W: io::Write>(self, format_hint: &str, writer: &mut W) -> io::Result<()>;
}

impl AccelerationDataCollection for () {
    fn write<W: io::Write>(&self, _format_hint: &str, _writer: &mut W) -> io::Result<()> {
        Ok(())
    }
    fn write_into<W: io::Write>(self, _format_hint: &str, _writer: &mut W) -> io::Result<()> {
        Ok(())
    }
}

/// A set of non-thermal electron beams.
#[derive(Clone, Debug)]
pub struct ElectronBeamSwarm<A: Accelerator> {
    lower_bounds: Vec3<ftr>,
    upper_bounds: Vec3<ftr>,
    properties: ElectronBeamSwarmProperties,
    acceleration_data: A::AccelerationDataCollectionType,
    verbose: Verbose,
}

#[derive(Clone, Debug)]
struct ElectronBeamSwarmProperties {
    number_of_beams: usize,
    fixed_scalar_values: FixedBeamScalarValues,
    fixed_vector_values: FixedBeamVectorValues,
    varying_scalar_values: VaryingBeamScalarValues,
    varying_vector_values: VaryingBeamVectorValues,
}

struct UnpropagatedElectronBeam<D: Distribution> {
    acceleration_position: Point3<ftr>,
    distribution_properties: <D::PropertiesCollectionType as BeamPropertiesCollection>::Item,
}

struct PropagatedElectronBeam<D: Distribution> {
    trajectory: BeamTrajectory,
    distribution_properties: <D::PropertiesCollectionType as BeamPropertiesCollection>::Item,
    total_propagation_distance: feb,
    residual_factors: Vec<feb>,
    deposited_powers: Vec<feb>,
    deposited_power_densities: Vec<feb>,
}

impl ElectronBeamSwarmProperties {
    fn into_field_line_set_properties(self) -> FieldLineSetProperties3 {
        let ElectronBeamSwarmProperties {
            number_of_beams,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
        } = self;
        FieldLineSetProperties3 {
            number_of_field_lines: number_of_beams,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
        }
    }
}

impl<D> FromParallelIterator<UnpropagatedElectronBeam<D>> for ElectronBeamSwarmProperties
where
    D: Distribution,
    D::PropertiesCollectionType:
        ParallelExtend<<D::PropertiesCollectionType as BeamPropertiesCollection>::Item>,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = UnpropagatedElectronBeam<D>>,
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(|beam| {
            (
                beam.acceleration_position[X],
                (
                    beam.acceleration_position[Y],
                    (beam.acceleration_position[Z], beam.distribution_properties),
                ),
            )
        });

        // Unzip the iterator of nested tuples into individual collections.
        // The unzipping has to be performed in multiple stages to avoid excessive
        // compilation times.

        let (
            acceleration_positions_x,
            (acceleration_positions_y, (acceleration_positions_z, nested_tuples)),
        ): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = nested_tuples_iter.unzip();

        let mut distribution_properties = D::PropertiesCollectionType::default();
        distribution_properties.par_extend(nested_tuples.into_par_iter());

        let number_of_beams = acceleration_positions_x.len();
        let mut fixed_scalar_values = HashMap::new();
        let mut fixed_vector_values = HashMap::new();
        let varying_scalar_values = HashMap::new();
        let varying_vector_values = HashMap::new();

        distribution_properties
            .distribute_into_maps(&mut fixed_scalar_values, &mut fixed_vector_values);

        fixed_scalar_values.insert("x0".to_string(), acceleration_positions_x);
        fixed_scalar_values.insert("y0".to_string(), acceleration_positions_y);
        fixed_scalar_values.insert("z0".to_string(), acceleration_positions_z);

        ElectronBeamSwarmProperties {
            number_of_beams,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
        }
    }
}

impl<D> FromParallelIterator<PropagatedElectronBeam<D>> for ElectronBeamSwarmProperties
where
    D: Distribution,
    D::PropertiesCollectionType:
        ParallelExtend<<D::PropertiesCollectionType as BeamPropertiesCollection>::Item>,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = PropagatedElectronBeam<D>>,
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(|beam| {
            (
                beam.trajectory.0,
                (
                    beam.trajectory.1,
                    (
                        beam.trajectory.2,
                        (
                            beam.distribution_properties,
                            (
                                beam.total_propagation_distance,
                                (
                                    beam.residual_factors,
                                    (beam.deposited_powers, beam.deposited_power_densities),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        });

        // Unzip the iterator of nested tuples into individual collections.
        // The unzipping has to be performed in multiple stages to avoid excessive
        // compilation times.

        let (trajectories_x, (trajectories_y, (trajectories_z, nested_tuples))): (
            Vec<_>,
            (Vec<_>, (Vec<_>, Vec<_>)),
        ) = nested_tuples_iter.unzip();

        let (distribution_properties, nested_tuples): (D::PropertiesCollectionType, Vec<_>) =
            nested_tuples.into_par_iter().unzip();

        let (total_propagation_distances, (residual_factors, nested_tuples)): (
            Vec<_>,
            (Vec<_>, Vec<_>),
        ) = nested_tuples.into_par_iter().unzip();

        let (deposited_powers, deposited_power_densities): (Vec<_>, Vec<_>) =
            nested_tuples.into_par_iter().unzip();

        let number_of_beams = trajectories_x.len();
        let mut fixed_scalar_values = HashMap::new();
        let mut fixed_vector_values = HashMap::new();
        let mut varying_scalar_values = HashMap::new();
        let varying_vector_values = HashMap::new();

        distribution_properties
            .distribute_into_maps(&mut fixed_scalar_values, &mut fixed_vector_values);

        fixed_scalar_values.insert(
            "x0".to_string(),
            trajectories_x
                .par_iter()
                .map(|trajectory_x| trajectory_x[0])
                .collect(),
        );
        fixed_scalar_values.insert(
            "y0".to_string(),
            trajectories_y
                .par_iter()
                .map(|trajectory_y| trajectory_y[0])
                .collect(),
        );
        fixed_scalar_values.insert(
            "z0".to_string(),
            trajectories_z
                .par_iter()
                .map(|trajectory_z| trajectory_z[0])
                .collect(),
        );
        fixed_scalar_values.insert(
            "total_propagation_distance".to_string(),
            total_propagation_distances,
        );

        varying_scalar_values.insert("x".to_string(), trajectories_x);
        varying_scalar_values.insert("y".to_string(), trajectories_y);
        varying_scalar_values.insert("z".to_string(), trajectories_z);
        varying_scalar_values.insert("residual_factor".to_string(), residual_factors);
        varying_scalar_values.insert("deposited_power".to_string(), deposited_powers);
        varying_scalar_values.insert(
            "deposited_power_density".to_string(),
            deposited_power_densities,
        );

        ElectronBeamSwarmProperties {
            number_of_beams,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
        }
    }
}

impl<A: Accelerator> ElectronBeamSwarm<A> {
    /// Generates a set of electron beams using the given seeder and accelerator
    /// but does not propagate them.
    ///
    /// # Parameters
    ///
    /// - `snapshot`: Snapshot representing the atmosphere.
    /// - `detector`: Reconnection site detector to use for obtaining acceleration positions.
    /// - `accelerator`: Accelerator to use for generating electron distributions.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper_factory`: Factory structure to use for producing steppers.
    /// - `verbose`: Whether to print status messages.
    ///
    /// # Returns
    ///
    /// A new `ElectronBeamSwarm` with unpropagated electron beams.
    ///
    /// # Type parameters
    ///
    /// - `G`: Type of grid.
    /// - `D`: Type of reconnection site detector.
    /// - `I`: Type of interpolator.
    /// - `StF`: Type of stepper factory.
    pub fn generate_unpropagated<G, R, D, I, StF>(snapshot: &mut SnapshotCacher3<G, R>, detector: D, accelerator: A, interpolator: &I, stepper_factory: &StF, verbose: Verbose) -> Self
    where G: Grid3<fdt>,
          R: SnapshotReader3<G> + Sync,
          D: ReconnectionSiteDetector,
          A: Accelerator + Sync,
          A::DistributionType: Send,
          <A::DistributionType as Distribution>::PropertiesCollectionType: ParallelExtend<<<A::DistributionType as Distribution>::PropertiesCollectionType as BeamPropertiesCollection>::Item>,
          I: Interpolator3,
          StF: StepperFactory3 + Sync
    {
        let (distributions, acceleration_data) = accelerator
            .generate_distributions(snapshot, detector, interpolator, stepper_factory, verbose)
            .unwrap_or_else(|err| panic!("Could not read field from snapshot: {}", err));

        let properties: ElectronBeamSwarmProperties = distributions
            .into_par_iter()
            .map(UnpropagatedElectronBeam::<A::DistributionType>::generate)
            .collect();

        let lower_bounds = Vec3::from(snapshot.reader().grid().lower_bounds());
        let upper_bounds = Vec3::from(snapshot.reader().grid().upper_bounds());

        ElectronBeamSwarm {
            lower_bounds,
            upper_bounds,
            properties,
            acceleration_data,
            verbose,
        }
    }

    /// Generates a set of electron beams using the given seeder and accelerator,
    /// and propagates them through the atmosphere in the given snapshot.
    ///
    /// # Parameters
    ///
    /// - `snapshot`: Snapshot representing the atmosphere.
    /// - `detector`: Reconnection site detector to use for obtaining acceleration positions.
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
    /// - `G`: Type of grid.
    /// - `D`: Type of reconnection site detector.
    /// - `I`: Type of interpolator.
    /// - `StF`: Type of stepper factory.
    pub fn generate_propagated<G, R, D, I, StF>(snapshot: &mut SnapshotCacher3<G, R>, detector: D, accelerator: A, interpolator: &I, stepper_factory: &StF, verbose: Verbose) -> Self
    where G: Grid3<fdt>,
          R: SnapshotReader3<G> + Sync,
          D: ReconnectionSiteDetector,
          A: Accelerator + Sync + Send,
          A::DistributionType: Send,
          <A::DistributionType as Distribution>::PropertiesCollectionType: ParallelExtend<<<A::DistributionType as Distribution>::PropertiesCollectionType as BeamPropertiesCollection>::Item>,
          I: Interpolator3,
          StF: StepperFactory3 + Sync
    {
        let (distributions, acceleration_data) = accelerator
            .generate_distributions(snapshot, detector, interpolator, stepper_factory, verbose)
            .unwrap_or_else(|err| panic!("Could not read field from snapshot: {}", err));

        if verbose.is_yes() {
            println!(
                "Attempting to propagate {} electron beams",
                distributions.len()
            );
        }

        let properties: ElectronBeamSwarmProperties = distributions
            .into_par_iter()
            .filter_map(|distribution| {
                PropagatedElectronBeam::<A::DistributionType>::generate(
                    distribution,
                    snapshot,
                    interpolator,
                    stepper_factory.produce(),
                )
            })
            .collect();

        if verbose.is_yes() {
            println!(
                "Successfully propagated {} electron beams",
                properties.number_of_beams
            );
        }

        let lower_bounds = Vec3::from(snapshot.reader().grid().lower_bounds());
        let upper_bounds = Vec3::from(snapshot.reader().grid().upper_bounds());

        ElectronBeamSwarm {
            lower_bounds,
            upper_bounds,
            properties,
            acceleration_data,
            verbose,
        }
    }

    /// Whether the electron beam swarm is verbose.
    pub fn verbose(&self) -> Verbose {
        self.verbose
    }

    /// Returns the number of beams making up the electron beam set.
    pub fn number_of_beams(&self) -> usize {
        self.properties.number_of_beams
    }

    /// Returns a reference to the acceleration data collection associated with the electron beams.
    pub fn acceleration_data(&self) -> &A::AccelerationDataCollectionType {
        &self.acceleration_data
    }

    /// Returns a mutable reference to the acceleration data collection associated with the electron beams.
    pub fn acceleration_data_mut(&mut self) -> &mut A::AccelerationDataCollectionType {
        &mut self.acceleration_data
    }

    /// Extracts and stores the value of the given scalar field at the initial position for each beam.
    pub fn extract_fixed_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting {} at acceleration sites", field.name());
        }
        let initial_coords_x = &self.properties.fixed_scalar_values["x0"];
        let initial_coords_y = &self.properties.fixed_scalar_values["y0"];
        let initial_coords_z = &self.properties.fixed_scalar_values["z0"];
        let values = initial_coords_x
            .into_par_iter()
            .zip(initial_coords_y)
            .zip(initial_coords_z)
            .map(|((&beam_x0, &beam_y0), &beam_z0)| {
                let acceleration_position = Point3::from_components(beam_x0, beam_y0, beam_z0);
                let value = interpolator
                    .interp_scalar_field(field, &acceleration_position)
                    .expect_inside();
                num::NumCast::from(value).expect("Conversion failed")
            })
            .collect();
        self.properties
            .fixed_scalar_values
            .insert(format!("{}0", field.name()), values);
    }

    /// Extracts and stores the value of the given vector field at the initial position for each beam.
    pub fn extract_fixed_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting {} at acceleration sites", field.name());
        }
        let initial_coords_x = &self.properties.fixed_scalar_values["x0"];
        let initial_coords_y = &self.properties.fixed_scalar_values["y0"];
        let initial_coords_z = &self.properties.fixed_scalar_values["z0"];
        let vectors = initial_coords_x
            .into_par_iter()
            .zip(initial_coords_y)
            .zip(initial_coords_z)
            .map(|((&beam_x0, &beam_y0), &beam_z0)| {
                let acceleration_position = Point3::from_components(beam_x0, beam_y0, beam_z0);
                let vector = interpolator
                    .interp_vector_field(field, &acceleration_position)
                    .expect_inside();
                Vec3::from(&vector)
            })
            .collect();
        self.properties
            .fixed_vector_values
            .insert(format!("{}0", field.name()), vectors);
    }

    /// Extracts and stores the value of the given scalar field at each position for each beam.
    pub fn extract_varying_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting {} along beam trajectories", field.name());
        }
        let coords_x = &self.properties.varying_scalar_values["x"];
        let coords_y = &self.properties.varying_scalar_values["y"];
        let coords_z = &self.properties.varying_scalar_values["z"];
        let values = coords_x
            .into_par_iter()
            .zip(coords_y)
            .zip(coords_z)
            .map(|((beam_coords_x, beam_coords_y), beam_coords_z)| {
                beam_coords_x
                    .iter()
                    .zip(beam_coords_y)
                    .zip(beam_coords_z)
                    .map(|((&beam_x, &beam_y), &beam_z)| {
                        let position = Point3::from_components(beam_x, beam_y, beam_z);
                        let value = interpolator
                            .interp_scalar_field(field, &position)
                            .expect_inside();
                        num::NumCast::from(value).expect("Conversion failed")
                    })
                    .collect()
            })
            .collect();
        self.properties
            .varying_scalar_values
            .insert(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at each position for each beam.
    pub fn extract_varying_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting {} along beam trajectories", field.name());
        }
        let coords_x = &self.properties.varying_scalar_values["x"];
        let coords_y = &self.properties.varying_scalar_values["y"];
        let coords_z = &self.properties.varying_scalar_values["z"];
        let vectors = coords_x
            .into_par_iter()
            .zip(coords_y)
            .zip(coords_z)
            .map(|((beam_coords_x, beam_coords_y), beam_coords_z)| {
                beam_coords_x
                    .iter()
                    .zip(beam_coords_y)
                    .zip(beam_coords_z)
                    .map(|((&beam_x, &beam_y), &beam_z)| {
                        let position = Point3::from_components(beam_x, beam_y, beam_z);
                        let vector = interpolator
                            .interp_vector_field(field, &position)
                            .expect_inside();
                        Vec3::from(&vector)
                    })
                    .collect()
            })
            .collect();
        self.properties
            .varying_vector_values
            .insert(field.name().to_string(), vectors);
    }

    /// Extracts and stores the magnitude of the given vector field at each position for each beam.
    pub fn extract_varying_vector_magnitudes<F, G, I>(
        &mut self,
        field: &VectorField3<F, G>,
        interpolator: &I,
    ) where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting |{}| along beam trajectories", field.name());
        }
        let coords_x = &self.properties.varying_scalar_values["x"];
        let coords_y = &self.properties.varying_scalar_values["y"];
        let coords_z = &self.properties.varying_scalar_values["z"];
        let values = coords_x
            .into_par_iter()
            .zip(coords_y)
            .zip(coords_z)
            .map(|((beam_coords_x, beam_coords_y), beam_coords_z)| {
                beam_coords_x
                    .iter()
                    .zip(beam_coords_y)
                    .zip(beam_coords_z)
                    .map(|((&beam_x, &beam_y), &beam_z)| {
                        let position = Point3::from_components(beam_x, beam_y, beam_z);
                        let value = interpolator
                            .interp_vector_field(field, &position)
                            .expect_inside()
                            .length();
                        num::NumCast::from(value).expect("Conversion failed")
                    })
                    .collect()
            })
            .collect();
        self.properties
            .varying_scalar_values
            .insert(field.name().to_string(), values);
    }

    /// Serializes the electron beam data into JSON format and saves at the given path.
    pub fn save_as_json<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()> {
        utils::save_data_as_json(output_file_path, &self)
    }

    /// Serializes the electron beam data into pickle format and saves at the given path.
    ///
    /// All the electron beam data is saved as a single pickled structure.
    pub fn save_as_pickle<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()> {
        utils::save_data_as_pickle(output_file_path, &self)
    }

    /// Serializes the electron beam data fields in parallel into pickle format and saves at the given path.
    ///
    /// The data fields are saved as separate pickle objects in the same file.
    pub fn save_as_combined_pickles<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()> {
        let mut buffer_1 = Vec::new();
        utils::write_data_as_pickle(&mut buffer_1, &self.lower_bounds)?;
        let mut buffer_2 = Vec::new();
        utils::write_data_as_pickle(&mut buffer_2, &self.upper_bounds)?;
        let mut buffer_3 = Vec::new();
        utils::write_data_as_pickle(&mut buffer_3, &self.number_of_beams())?;

        let (mut result_4, mut result_5, mut result_6, mut result_7, mut result_8) =
            (Ok(()), Ok(()), Ok(()), Ok(()), Ok(()));
        let (mut buffer_4, mut buffer_5, mut buffer_6, mut buffer_7, mut buffer_8) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        rayon::scope(|s| {
            s.spawn(|_| {
                result_4 =
                    utils::write_data_as_pickle(&mut buffer_4, &self.properties.fixed_scalar_values)
            });
            s.spawn(|_| {
                result_5 =
                    utils::write_data_as_pickle(&mut buffer_5, &self.properties.fixed_vector_values)
            });
            s.spawn(|_| {
                result_6 = utils::write_data_as_pickle(
                    &mut buffer_6,
                    &self.properties.varying_scalar_values,
                )
            });
            s.spawn(|_| {
                result_7 = utils::write_data_as_pickle(
                    &mut buffer_7,
                    &self.properties.varying_vector_values,
                )
            });
            s.spawn(|_| result_8 = self.acceleration_data.write("pickle", &mut buffer_8));
        });
        result_4?;
        result_5?;
        result_6?;
        result_7?;
        result_8?;

        let mut file = utils::create_file_and_required_directories(output_file_path)?;
        file.write_all(
            &[
                buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, buffer_6, buffer_7, buffer_8,
            ]
            .concat(),
        )?;
        Ok(())
    }

    /// Serializes the electron beam data into a custom binary format and saves at the given path.
    pub fn save_as_custom_binary<P: AsRef<Path>>(&self, output_file_path: P) -> io::Result<()> {
        let mut file = field_line::save_field_line_data_as_custom_binary(
            output_file_path,
            &self.lower_bounds,
            &self.upper_bounds,
            self.properties.clone().into_field_line_set_properties(),
        )?;
        self.acceleration_data.write("fl", &mut file)
    }

    /// Serializes the electron beam data into a H5Part format and saves to the given path.
    #[cfg(feature = "hdf5")]
    pub fn save_as_h5part<P: AsRef<Path>>(
        &self,
        output_file_path: P,
        output_acceleration_site_file_path: P,
        drop_id: bool,
    ) -> io::Result<()> {
        field_line::save_field_line_data_as_h5part(
            output_file_path,
            output_acceleration_site_file_path,
            self.properties.clone().into_field_line_set_properties(),
            drop_id,
        )
    }

    /// Serializes the electron beam data into a custom binary format and saves at the given path,
    /// consuming the electron beam swarm in the process.
    pub fn save_into_custom_binary<P: AsRef<Path>>(self, output_file_path: P) -> io::Result<()> {
        let mut file = field_line::save_field_line_data_as_custom_binary(
            output_file_path,
            &self.lower_bounds,
            &self.upper_bounds,
            self.properties.into_field_line_set_properties(),
        )?;
        self.acceleration_data.write_into("fl", &mut file)
    }

    /// Serializes the electron beam data into a H5Part format and saves to the given path,
    /// consuming the electron beam swarm in the process.
    #[cfg(feature = "hdf5")]
    pub fn save_into_h5part<P: AsRef<Path>>(
        self,
        output_file_path: P,
        output_acceleration_site_file_path: P,
        drop_id: bool,
    ) -> io::Result<()> {
        field_line::save_field_line_data_as_h5part(
            output_file_path,
            output_acceleration_site_file_path,
            self.properties.into_field_line_set_properties(),
            drop_id,
        )
    }
}

impl<D: Distribution> UnpropagatedElectronBeam<D> {
    fn generate(distribution: D) -> Self {
        let acceleration_position = Point3::from(distribution.acceleration_position());
        UnpropagatedElectronBeam {
            acceleration_position,
            distribution_properties: distribution.properties(),
        }
    }
}

impl<D: Distribution> PropagatedElectronBeam<D> {
    fn generate<G, R, I, S>(
        mut distribution: D,
        snapshot: &SnapshotCacher3<G, R>,
        interpolator: &I,
        stepper: S,
    ) -> Option<Self>
    where
        G: Grid3<fdt>,
        R: SnapshotReader3<G>,
        I: Interpolator3,
        S: Stepper3,
    {
        let magnetic_field = snapshot.cached_vector_field("b");
        let start_position = Point3::from(distribution.acceleration_position());

        let mut trajectory = (
            vec![start_position[X]],
            vec![start_position[Y]],
            vec![start_position[Z]],
        );
        let mut residual_factors = vec![0.0];
        let mut deposited_powers = vec![0.0];
        let mut deposited_power_densities = vec![0.0];
        let mut total_propagation_distance = 0.0;

        let tracer_result = tracing::trace_3d_field_line_dense(
            magnetic_field,
            interpolator,
            stepper,
            &start_position,
            distribution.propagation_sense(),
            &mut |displacement, _, position, distance| {
                if distance > distribution.max_propagation_distance() {
                    StepperInstruction::Terminate
                } else if distance > 0.0 {
                    let PropagationResult {
                        residual_factor,
                        deposited_power,
                        deposited_power_density,
                        deposition_position,
                        depletion_status,
                    } = distribution.propagate(snapshot, interpolator, displacement, position);

                    trajectory.0.push(deposition_position[X]);
                    trajectory.1.push(deposition_position[Y]);
                    trajectory.2.push(deposition_position[Z]);
                    residual_factors.push(residual_factor);
                    deposited_powers.push(deposited_power);
                    deposited_power_densities.push(deposited_power_density);
                    total_propagation_distance = distance;

                    match depletion_status {
                        DepletionStatus::Undepleted => StepperInstruction::Continue,
                        DepletionStatus::Depleted => StepperInstruction::Terminate,
                    }
                } else {
                    StepperInstruction::Continue
                }
            },
        );

        let distribution_properties = distribution.properties();

        match tracer_result {
            TracerResult::Ok(_) => Some(PropagatedElectronBeam {
                trajectory,
                distribution_properties,
                total_propagation_distance,
                residual_factors,
                deposited_powers,
                deposited_power_densities,
            }),
            TracerResult::Void => None,
        }
    }
}

impl<A: Accelerator> Serialize for ElectronBeamSwarm<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("ElectronBeamSwarm", 7)?;
        s.serialize_field("lower_bounds", &self.lower_bounds)?;
        s.serialize_field("upper_bounds", &self.upper_bounds)?;
        s.serialize_field("number_of_beams", &self.number_of_beams())?;
        s.serialize_field("fixed_scalar_values", &self.properties.fixed_scalar_values)?;
        s.serialize_field("fixed_vector_values", &self.properties.fixed_vector_values)?;
        s.serialize_field(
            "varying_scalar_values",
            &self.properties.varying_scalar_values,
        )?;
        s.serialize_field(
            "varying_vector_values",
            &self.properties.varying_vector_values,
        )?;
        s.serialize_field("acceleration_data", &self.acceleration_data)?;
        s.end()
    }
}
