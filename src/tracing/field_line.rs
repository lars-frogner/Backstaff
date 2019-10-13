//! Field lines in vector fields.

pub mod basic;

use super::ftr;
use super::seeding::Seeder3;
use super::stepping::{Stepper3, StepperFactory3};
use crate::field::{ScalarField3, VectorField3};
use crate::geometry::{Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::{utils, Endianness, Verbose};
use crate::num::BFloat;
use rayon::prelude::*;
use serde::ser::{SerializeStruct, Serializer};
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::{fs, io, mem, path};

type FieldLinePath3 = (Vec<ftr>, Vec<ftr>, Vec<ftr>);
type FixedScalarValues = HashMap<String, Vec<ftr>>;
type FixedVector3Values = HashMap<String, Vec<Vec3<ftr>>>;
type VaryingScalarValues = HashMap<String, Vec<Vec<ftr>>>;
type VaryingVector3Values = HashMap<String, Vec<Vec<Vec3<ftr>>>>;

/// Defines the properties of a field line tracer for a 3D vector field.
pub trait FieldLineTracer3 {
    /// Traces a field line through a 3D vector field.
    ///
    /// # Parameters
    ///
    /// - `field`: Vector field to trace.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper`: Stepper to use (will be consumed).
    /// - `start_position`: Position where the tracing should start.
    ///
    /// # Returns
    ///
    /// An `Option` which is either:
    ///
    /// - `Some`: Contains a `FieldLineData3` object representing the traced field line.
    /// - `None`: No field line was traced.
    ///
    /// # Type parameters
    ///
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `St`: Type of stepper.
    fn trace<F, G, I, St>(
        &self,
        field: &VectorField3<F, G>,
        interpolator: &I,
        stepper: St,
        start_position: &Point3<ftr>,
    ) -> Option<FieldLineData3>
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        St: Stepper3;
}

/// Data required to represent a 3D field line.
pub struct FieldLineData3 {
    path: FieldLinePath3,
    total_length: ftr,
}

/// Collection of 3D field lines.
#[derive(Clone, Debug)]
pub struct FieldLineSet3 {
    properties: FieldLineSetProperties3,
    verbose: Verbose,
}

/// Holds the data associated with a set of 3D field lines.
#[derive(Clone, Debug)]
pub struct FieldLineSetProperties3 {
    /// Number of field lines in the set.
    pub number_of_field_lines: usize,
    /// Scalar values defined at the start positions of the field lines.
    pub fixed_scalar_values: FixedScalarValues,
    /// Vector values defined at the start positions of the field lines.
    pub fixed_vector_values: FixedVector3Values,
    /// Scalar values defined along the paths of the field lines.
    pub varying_scalar_values: VaryingScalarValues,
    /// Vector values defined along the paths of the field lines.
    pub varying_vector_values: VaryingVector3Values,
}

impl FromParallelIterator<FieldLineData3> for FieldLineSetProperties3 {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = FieldLineData3>,
    {
        let nested_tuples_iter = par_iter.into_par_iter().map(|field_line| {
            (
                field_line.path.0,
                (
                    field_line.path.1,
                    (field_line.path.2, field_line.total_length),
                ),
            )
        });

        let (paths_x, (paths_y, (paths_z, total_lengths))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) =
            nested_tuples_iter.unzip();

        let number_of_field_lines = paths_x.len();
        let mut fixed_scalar_values = HashMap::new();
        let fixed_vector_values = HashMap::new();
        let mut varying_scalar_values = HashMap::new();
        let varying_vector_values = HashMap::new();

        fixed_scalar_values.insert(
            "x0".to_string(),
            paths_x.par_iter().map(|path_x| path_x[0]).collect(),
        );
        fixed_scalar_values.insert(
            "y0".to_string(),
            paths_y.par_iter().map(|path_y| path_y[0]).collect(),
        );
        fixed_scalar_values.insert(
            "z0".to_string(),
            paths_z.par_iter().map(|path_z| path_z[0]).collect(),
        );
        fixed_scalar_values.insert("total_length".to_string(), total_lengths);

        varying_scalar_values.insert("x".to_string(), paths_x);
        varying_scalar_values.insert("y".to_string(), paths_y);
        varying_scalar_values.insert("z".to_string(), paths_z);

        FieldLineSetProperties3 {
            number_of_field_lines,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
        }
    }
}

impl FieldLineSet3 {
    /// Traces all the field lines in the set from positions generated by the given seeder.
    ///
    /// # Parameters
    ///
    /// - `seeder`: Seeder to use for generating start positions.
    /// - `tracer`: Field line tracer to use.
    /// - `field`: Vector field to trace.
    /// - `interpolator`: Interpolator to use.
    /// - `stepper_factory`: Factory structure to use for producing steppers.
    /// - `verbose`: Whether to print status messages.
    ///
    /// # Returns
    ///
    /// A new `FieldLineSet3` with traced field lines.
    ///
    /// # Type parameters
    ///
    /// - `Sd`: Type of seeder.
    /// - `Tr`: Type of field line tracer.
    /// - `F`: Floating point type of the field data.
    /// - `G`: Type of grid.
    /// - `I`: Type of interpolator.
    /// - `StF`: Type of stepper factory.
    pub fn trace<Sd, Tr, F, G, I, StF>(
        seeder: Sd,
        tracer: &Tr,
        field: &VectorField3<F, G>,
        interpolator: &I,
        stepper_factory: StF,
        verbose: Verbose,
    ) -> Self
    where
        Sd: Seeder3,
        Tr: FieldLineTracer3 + Sync,
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
        StF: StepperFactory3 + Sync,
    {
        if verbose.is_yes() {
            println!("Found {} start positions", seeder.number_of_points());
        }

        let properties: FieldLineSetProperties3 = seeder
            .into_par_iter()
            .filter_map(|start_position| {
                tracer.trace(
                    field,
                    interpolator,
                    stepper_factory.produce(),
                    &start_position,
                )
            })
            .collect();

        if verbose.is_yes() {
            println!(
                "Successfully traced {} field lines",
                properties.number_of_field_lines
            );
        }

        FieldLineSet3 {
            properties,
            verbose,
        }
    }

    /// Returns the number of field lines making up the field line set.
    pub fn number_of_field_lines(&self) -> usize {
        self.properties.number_of_field_lines
    }

    /// Extracts and stores the value of the given scalar field at the initial position for each field line.
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
            .map(|((&field_line_x0, &field_line_y0), &field_line_z0)| {
                let acceleration_position =
                    Point3::from_components(field_line_x0, field_line_y0, field_line_z0);
                let value = interpolator
                    .interp_scalar_field(field, &acceleration_position)
                    .expect_inside();
                num::NumCast::from(value).expect("Conversion failed.")
            })
            .collect();
        self.properties
            .fixed_scalar_values
            .insert(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at the initial position for each field line.
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
            .map(|((&field_line_x0, &field_line_y0), &field_line_z0)| {
                let acceleration_position =
                    Point3::from_components(field_line_x0, field_line_y0, field_line_z0);
                let vector = interpolator
                    .interp_vector_field(field, &acceleration_position)
                    .expect_inside();
                Vec3::from(&vector)
            })
            .collect();
        self.properties
            .fixed_vector_values
            .insert(field.name().to_string(), vectors);
    }

    /// Extracts and stores the value of the given scalar field at each position for each field line.
    pub fn extract_varying_scalars<F, G, I>(&mut self, field: &ScalarField3<F, G>, interpolator: &I)
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting {} along field line paths", field.name());
        }
        let coords_x = &self.properties.varying_scalar_values["x"];
        let coords_y = &self.properties.varying_scalar_values["y"];
        let coords_z = &self.properties.varying_scalar_values["z"];
        let values = coords_x
            .into_par_iter()
            .zip(coords_y)
            .zip(coords_z)
            .map(
                |((field_line_coords_x, field_line_coords_y), field_line_coords_z)| {
                    field_line_coords_x
                        .iter()
                        .zip(field_line_coords_y)
                        .zip(field_line_coords_z)
                        .map(|((&field_line_x, &field_line_y), &field_line_z)| {
                            let position =
                                Point3::from_components(field_line_x, field_line_y, field_line_z);
                            let value = interpolator
                                .interp_scalar_field(field, &position)
                                .expect_inside();
                            num::NumCast::from(value).expect("Conversion failed.")
                        })
                        .collect()
                },
            )
            .collect();
        self.properties
            .varying_scalar_values
            .insert(field.name().to_string(), values);
    }

    /// Extracts and stores the value of the given vector field at each position for each field line.
    pub fn extract_varying_vectors<F, G, I>(&mut self, field: &VectorField3<F, G>, interpolator: &I)
    where
        F: BFloat,
        G: Grid3<F>,
        I: Interpolator3,
    {
        if self.verbose.is_yes() {
            println!("Extracting {} along field line paths", field.name());
        }
        let coords_x = &self.properties.varying_scalar_values["x"];
        let coords_y = &self.properties.varying_scalar_values["y"];
        let coords_z = &self.properties.varying_scalar_values["z"];
        let vectors = coords_x
            .into_par_iter()
            .zip(coords_y)
            .zip(coords_z)
            .map(
                |((field_line_coords_x, field_line_coords_y), field_line_coords_z)| {
                    field_line_coords_x
                        .iter()
                        .zip(field_line_coords_y)
                        .zip(field_line_coords_z)
                        .map(|((&field_line_x, &field_line_y), &field_line_z)| {
                            let position =
                                Point3::from_components(field_line_x, field_line_y, field_line_z);
                            let vector = interpolator
                                .interp_vector_field(field, &position)
                                .expect_inside();
                            Vec3::from(&vector)
                        })
                        .collect()
                },
            )
            .collect();
        self.properties
            .varying_vector_values
            .insert(field.name().to_string(), vectors);
    }

    /// Serializes the field line data into JSON format and saves at the given path.
    pub fn save_as_json<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        if self.verbose.is_yes() {
            println!(
                "Saving field line data in JSON format in {}",
                file_path.as_ref().display()
            );
        }
        utils::save_data_as_json(file_path, &self)
    }

    /// Serializes the field line data into pickle format and saves at the given path.
    ///
    /// All the field line data is saved as a single pickled structure.
    pub fn save_as_pickle<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        if self.verbose.is_yes() {
            println!(
                "Saving field lines as single pickle object in {}",
                file_path.as_ref().display()
            );
        }
        utils::save_data_as_pickle(file_path, &self)
    }

    /// Serializes the field line data fields in parallel into pickle format and saves at the given path.
    ///
    /// The data fields are saved as separate pickle objects in the same file.
    pub fn save_as_combined_pickles<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        if self.verbose.is_yes() {
            println!("Saving field lines in {}", file_path.as_ref().display());
        }
        let mut buffer_1 = Vec::new();
        utils::write_data_as_pickle(&mut buffer_1, &self.number_of_field_lines())?;

        let (mut result_2, mut result_3, mut result_4, mut result_5) =
            (Ok(()), Ok(()), Ok(()), Ok(()));
        let (mut buffer_2, mut buffer_3, mut buffer_4, mut buffer_5) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        rayon::scope(|s| {
            s.spawn(|_| {
                result_2 =
                    utils::write_data_as_pickle(&mut buffer_2, &self.properties.fixed_scalar_values)
            });
            s.spawn(|_| {
                result_3 =
                    utils::write_data_as_pickle(&mut buffer_3, &self.properties.fixed_vector_values)
            });
            s.spawn(|_| {
                result_4 = utils::write_data_as_pickle(
                    &mut buffer_4,
                    &self.properties.varying_scalar_values,
                )
            });
            s.spawn(|_| {
                result_5 = utils::write_data_as_pickle(
                    &mut buffer_5,
                    &self.properties.varying_vector_values,
                )
            });
        });
        result_2?;
        result_3?;
        result_4?;
        result_5?;

        let mut file = fs::File::create(file_path)?;
        file.write_all(&[buffer_1, buffer_2, buffer_3, buffer_4, buffer_5].concat())?;
        Ok(())
    }

    /// Serializes the field line data into a custom binary format and saves at the given path.
    pub fn save_as_custom_binary_file<P: AsRef<path::Path>>(&self, file_path: P) -> io::Result<()> {
        write_field_line_data_in_custom_binary_format(file_path, self.properties.clone())
    }

    /// Serializes the field line data into a custom binary format and saves at the given path,
    /// consuming the field line set in the process.
    pub fn into_custom_binary_file<P: AsRef<path::Path>>(self, file_path: P) -> io::Result<()> {
        write_field_line_data_in_custom_binary_format(file_path, self.properties)
    }
}

impl Serialize for FieldLineSet3 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("FieldLineSet3", 5)?;
        s.serialize_field("number_of_field_lines", &self.number_of_field_lines())?;
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
        s.end()
    }
}

/// Writes the given field line data in a custom binary format at the
/// given path.
pub fn write_field_line_data_in_custom_binary_format<P: AsRef<path::Path>>(
    output_path: P,
    properties: FieldLineSetProperties3,
) -> io::Result<()> {
    // Field line file format:
    // [HEADER]
    // float_size: u64
    // number_of_field_lines: u64
    // number_of_field_line_elements: u64
    // number_of_fixed_scalar_quantities: u64
    // number_of_fixed_vector_quantities: u64
    // number_of_varying_scalar_quantities: u64
    // number_of_varying_vector_quantities: u64
    // number_of_fixed_scalar_name_bytes:   u64
    // number_of_fixed_vector_name_bytes:   u64
    // number_of_varying_scalar_name_bytes: u64
    // number_of_varying_vector_name_bytes: u64
    // start_indices_of_field_line_elements: [u64; number_of_field_lines]
    // fixed_scalar_name_bytes:   [u8; number_of_fixed_scalar_name_bytes  ]
    // fixed_vector_name_bytes:   [u8; number_of_fixed_vector_name_bytes  ]
    // varying_scalar_name_bytes: [u8; number_of_varying_scalar_name_bytes]
    // varying_vector_name_bytes: [u8; number_of_varying_vector_name_bytes]
    // [BODY]
    // flat_fixed_scalar_values:   [ftr: number_of_fixed_scalar_quantities*number_of_field_lines  ]
    // flat_fixed_vector_values:   [ftr: number_of_fixed_vector_quantities*number_of_field_lines*3]
    // flat_varying_scalar_values: [ftr: number_of_varying_scalar_quantities*number_of_field_line_elements  ]
    // flat_varying_vector_values: [ftr: number_of_varying_vector_quantities*number_of_field_line_elements*3]

    const ENDIANNESS: Endianness = Endianness::Little;

    let FieldLineSetProperties3 {
        number_of_field_lines,
        fixed_scalar_values,
        fixed_vector_values,
        varying_scalar_values,
        varying_vector_values,
    } = properties;

    let number_of_field_lines = number_of_field_lines;
    let number_of_fixed_scalar_quantities = fixed_scalar_values.len();
    let number_of_fixed_vector_quantities = fixed_vector_values.len();
    let number_of_varying_scalar_quantities = varying_scalar_values.len();
    let number_of_varying_vector_quantities = varying_vector_values.len();

    let (_, varying_scalars) = varying_scalar_values
        .iter()
        .next()
        .expect("No varying scalar values for field line set.");

    let number_of_field_line_elements: usize = varying_scalars.iter().map(|vec| vec.len()).sum();

    let start_indices_of_field_line_elements: Vec<_> = varying_scalars
        .iter()
        .scan(0, |count, vec| {
            let idx = *count;
            *count += vec.len();
            Some(idx as u64)
        })
        .collect();

    let mut number_of_fixed_scalar_name_bytes = 0;
    let mut fixed_scalar_name_bytes = Vec::new();
    let mut flat_fixed_scalar_values = Vec::new();

    let set_fixed_scalar_variables =
        |number_of_fixed_scalar_name_bytes: &mut usize,
         fixed_scalar_name_bytes: &mut Vec<_>,
         flat_fixed_scalar_values: &mut Vec<_>| {
            flat_fixed_scalar_values
                .reserve_exact(number_of_fixed_scalar_quantities * number_of_field_lines);
            let mut fixed_scalar_names = Vec::new();
            for (name, values) in fixed_scalar_values {
                fixed_scalar_names.push(name);
                flat_fixed_scalar_values.extend(values.into_iter());
            }
            let fixed_scalar_names = fixed_scalar_names.join("\n");
            fixed_scalar_name_bytes.extend_from_slice(fixed_scalar_names.as_bytes());
            *number_of_fixed_scalar_name_bytes = fixed_scalar_name_bytes.len();
        };

    let mut number_of_fixed_vector_name_bytes = 0;
    let mut fixed_vector_name_bytes = Vec::new();
    let mut flat_fixed_vector_values = Vec::new();

    let set_fixed_vector_variables =
        |number_of_fixed_vector_name_bytes: &mut usize,
         fixed_vector_name_bytes: &mut Vec<_>,
         flat_fixed_vector_values: &mut Vec<ftr>| {
            flat_fixed_vector_values
                .reserve_exact(number_of_fixed_vector_quantities * number_of_field_lines * 3);
            let mut fixed_vector_names = Vec::new();
            for (name, values) in fixed_vector_values {
                fixed_vector_names.push(name);
                for vec3 in values {
                    flat_fixed_vector_values.extend(vec3.into_iter());
                }
            }
            let fixed_vector_names = fixed_vector_names.join("\n");
            fixed_vector_name_bytes.extend_from_slice(fixed_vector_names.as_bytes());
            *number_of_fixed_vector_name_bytes = fixed_vector_name_bytes.len();
        };

    let mut number_of_varying_scalar_name_bytes = 0;
    let mut varying_scalar_name_bytes = Vec::new();
    let mut flat_varying_scalar_values = Vec::new();

    let set_varying_scalar_variables =
        |number_of_varying_scalar_name_bytes: &mut usize,
         varying_scalar_name_bytes: &mut Vec<_>,
         flat_varying_scalar_values: &mut Vec<_>| {
            flat_varying_scalar_values
                .reserve_exact(number_of_varying_scalar_quantities * number_of_field_line_elements);
            let mut varying_scalar_names = Vec::new();
            for (name, values) in varying_scalar_values {
                varying_scalar_names.push(name);
                for vec in values {
                    flat_varying_scalar_values.extend(vec.into_iter());
                }
            }
            let varying_scalar_names = varying_scalar_names.join("\n");
            varying_scalar_name_bytes.extend_from_slice(varying_scalar_names.as_bytes());
            *number_of_varying_scalar_name_bytes = varying_scalar_name_bytes.len();
        };

    let mut number_of_varying_vector_name_bytes = 0;
    let mut varying_vector_name_bytes = Vec::new();
    let mut flat_varying_vector_values = Vec::new();

    let set_varying_vector_variables =
        |number_of_varying_vector_name_bytes: &mut usize,
         varying_vector_name_bytes: &mut Vec<_>,
         flat_varying_vector_values: &mut Vec<ftr>| {
            flat_varying_vector_values.reserve_exact(
                number_of_varying_vector_quantities * number_of_field_line_elements * 3,
            );
            let mut varying_vector_names = Vec::new();
            for (name, values) in varying_vector_values {
                varying_vector_names.push(name);
                for vec in values {
                    for vec3 in vec {
                        flat_varying_vector_values.extend(vec3.into_iter());
                    }
                }
            }
            let varying_vector_names = varying_vector_names.join("\n");
            varying_vector_name_bytes.extend_from_slice(varying_vector_names.as_bytes());
            *number_of_varying_vector_name_bytes = varying_vector_name_bytes.len();
        };

    rayon::scope(|s| {
        s.spawn(|_| {
            set_fixed_scalar_variables(
                &mut number_of_fixed_scalar_name_bytes,
                &mut fixed_scalar_name_bytes,
                &mut flat_fixed_scalar_values,
            );
        });
        s.spawn(|_| {
            set_fixed_vector_variables(
                &mut number_of_fixed_vector_name_bytes,
                &mut fixed_vector_name_bytes,
                &mut flat_fixed_vector_values,
            );
        });
        s.spawn(|_| {
            set_varying_scalar_variables(
                &mut number_of_varying_scalar_name_bytes,
                &mut varying_scalar_name_bytes,
                &mut flat_varying_scalar_values,
            );
        });
        s.spawn(|_| {
            set_varying_vector_variables(
                &mut number_of_varying_vector_name_bytes,
                &mut varying_vector_name_bytes,
                &mut flat_varying_vector_values,
            );
        });
    });

    let u64_size = mem::size_of::<u64>();
    let u8_size = mem::size_of::<u8>();
    let float_size = mem::size_of::<ftr>();

    let section_sizes = [
        11 * u64_size,
        number_of_field_lines * u64_size,
        number_of_fixed_scalar_name_bytes * u8_size,
        number_of_fixed_vector_name_bytes * u8_size,
        number_of_varying_scalar_name_bytes * u8_size,
        number_of_varying_vector_name_bytes * u8_size,
        number_of_fixed_scalar_quantities * number_of_field_lines * float_size,
        number_of_fixed_vector_quantities * number_of_field_lines * 3 * float_size,
        number_of_varying_scalar_quantities * number_of_field_line_elements * float_size,
        number_of_varying_vector_quantities * number_of_field_line_elements * 3 * float_size,
    ];

    let byte_buffer_size = *section_sizes.iter().max().unwrap();
    let mut byte_buffer = vec![0_u8; byte_buffer_size];

    let file_size: usize = section_sizes.iter().sum();
    let mut file = fs::File::create(output_path)?;
    file.set_len(file_size as u64)?;

    let byte_offset = utils::write_into_byte_buffer(
        &[
            float_size as u64,
            number_of_field_lines as u64,
            number_of_field_line_elements as u64,
            number_of_fixed_scalar_quantities as u64,
            number_of_fixed_vector_quantities as u64,
            number_of_varying_scalar_quantities as u64,
            number_of_varying_vector_quantities as u64,
            number_of_fixed_scalar_name_bytes as u64,
            number_of_fixed_vector_name_bytes as u64,
            number_of_varying_scalar_name_bytes as u64,
            number_of_varying_vector_name_bytes as u64,
        ],
        &mut byte_buffer,
        0,
        ENDIANNESS,
    );
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset = utils::write_into_byte_buffer(
        &start_indices_of_field_line_elements,
        &mut byte_buffer,
        0,
        ENDIANNESS,
    );
    mem::drop(start_indices_of_field_line_elements);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&fixed_scalar_name_bytes, &mut byte_buffer, 0, ENDIANNESS);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&fixed_vector_name_bytes, &mut byte_buffer, 0, ENDIANNESS);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&varying_scalar_name_bytes, &mut byte_buffer, 0, ENDIANNESS);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&varying_vector_name_bytes, &mut byte_buffer, 0, ENDIANNESS);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&flat_fixed_scalar_values, &mut byte_buffer, 0, ENDIANNESS);
    mem::drop(flat_fixed_scalar_values);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&flat_fixed_vector_values, &mut byte_buffer, 0, ENDIANNESS);
    mem::drop(flat_fixed_vector_values);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&flat_varying_scalar_values, &mut byte_buffer, 0, ENDIANNESS);
    mem::drop(flat_varying_scalar_values);
    file.write_all(&byte_buffer[..byte_offset])?;

    let byte_offset =
        utils::write_into_byte_buffer(&flat_varying_vector_values, &mut byte_buffer, 0, ENDIANNESS);
    mem::drop(flat_varying_vector_values);
    file.write_all(&byte_buffer[..byte_offset])
}
