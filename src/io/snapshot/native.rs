//! Reading and writing of Bifrost simulation data in native format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self},
        Endianness, Verbosity,
    },
    fdt, SnapshotMetadata, SnapshotParameters, FALLBACK_SNAP_NUM, PRIMARY_VARIABLE_NAMES_HD,
    PRIMARY_VARIABLE_NAMES_MHD,
};
use crate::{
    field::{FieldGrid3, ScalarField3, ScalarFieldProvider3},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{
        CoordLocation::{self, Center, LowerEdge},
        Grid3,
    },
    io::utils::IOContext,
};
use ndarray::prelude::*;
use std::{
    collections::HashMap,
    io::{self, Write},
    mem,
    path::{Path, PathBuf},
    str,
    sync::Arc,
};

pub use mesh::{
    create_grid_from_mesh_file, write_mesh_file_from_grid, NATIVE_COORD_PRECISION,
    NATIVE_COORD_WIDTH,
};
pub use param::NativeSnapshotParameters;

#[cfg(feature = "for-testing")]
pub use mesh::parsed_mesh_files_eq;

/// Configuration parameters for native snapshot reader.
#[derive(Clone, Debug)]
pub struct NativeSnapshotReaderConfig {
    /// Path to the parameter (.idl) file.
    param_file_path: PathBuf,
    /// Order of bytes in the binary data files.
    endianness: Endianness,
    /// Whether and how to pass non-essential information to user while reading fields.
    verbosity: Verbosity,
}

impl NativeSnapshotReaderConfig {
    /// Creates a new set of snapshot reader configuration parameters.
    pub fn new(param_file_path: PathBuf, endianness: Endianness, verbosity: Verbosity) -> Self {
        NativeSnapshotReaderConfig {
            param_file_path,
            endianness,
            verbosity,
        }
    }
}

/// Information associated with a Bifrost 3D simulation snapshot
/// in native format.
#[derive(Clone, Debug)]
pub struct NativeSnapshotMetadata {
    snap_name: String,
    snap_num: Option<u64>,
    endianness: Endianness,
    parameters: Box<NativeSnapshotParameters>,
}

impl NativeSnapshotMetadata {
    fn new(parameters: NativeSnapshotParameters, endianness: Endianness) -> Self {
        let (snap_name, snap_num) =
            super::extract_name_and_num_from_snapshot_path(parameters.original_path());
        Self {
            snap_name,
            snap_num,
            endianness,
            parameters: Box::new(parameters),
        }
    }

    /// Returns the path of the parameter (.idl) file.
    pub fn parameter_file_path(&self) -> &Path {
        self.parameters.original_path()
    }
}

impl SnapshotMetadata for NativeSnapshotMetadata {
    fn snap_name(&self) -> &str {
        &self.snap_name
    }

    fn snap_num(&self) -> Option<u64> {
        self.snap_num
    }

    fn parameters(&self) -> &dyn SnapshotParameters {
        self.parameters.as_ref()
    }

    fn endianness(&self) -> Endianness {
        self.endianness
    }
}

/// Reader for the native output files associated with Bifrost 3D simulation snapshots.
#[derive(Clone, Debug)]
pub struct NativeSnapshotReader3 {
    snap_path: PathBuf,
    aux_path: PathBuf,
    endianness: Endianness,
    grid: Arc<FieldGrid3>,
    variable_descriptors: HashMap<String, VariableDescriptor>,
    all_variable_names: Vec<String>,
    verbosity: Verbosity,
}

impl NativeSnapshotReader3 {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: NativeSnapshotReaderConfig) -> io::Result<(Self, NativeSnapshotMetadata)> {
        let NativeSnapshotReaderConfig {
            param_file_path,
            endianness,
            verbosity,
        } = config;

        let parameters = NativeSnapshotParameters::new(param_file_path, &verbosity)?;

        let (snap_path, aux_path) = parameters.determine_snap_path()?;

        // Determine the set of primary variables to use
        let is_mhd = parameters.determine_if_mhd()?;

        let primary_variable_names: Vec<_> = if snap_path.exists() {
            if is_mhd {
                PRIMARY_VARIABLE_NAMES_MHD.iter()
            } else {
                PRIMARY_VARIABLE_NAMES_HD.iter()
            }
            .map(|&name| name.to_string())
            .collect()
        } else {
            Vec::new()
        };

        let auxiliary_variable_names = parameters.determine_aux_names()?;
        let mut all_variable_names = primary_variable_names;
        all_variable_names.append(&mut auxiliary_variable_names.clone());

        let mut variable_descriptors = HashMap::new();
        Self::insert_primary_variable_descriptors(is_mhd, &mut variable_descriptors);
        Self::insert_auxiliary_variable_descriptors(
            &auxiliary_variable_names,
            &mut variable_descriptors,
        )?;

        let mesh_path = parameters.determine_mesh_path()?;
        let is_periodic = parameters.determine_grid_periodicity()?;

        let grid = create_grid_from_mesh_file(&mesh_path, is_periodic, &verbosity)?;

        let metadata = NativeSnapshotMetadata::new(parameters, endianness);

        Ok((
            Self {
                snap_path,
                aux_path,
                endianness,
                grid: Arc::new(grid),
                variable_descriptors,
                all_variable_names,
                verbosity,
            },
            metadata,
        ))
    }

    /// Returns the path of the primary variable (.snap) file.
    pub fn primary_variable_file_path(&self) -> &Path {
        self.snap_path.as_path()
    }

    /// Returns the path of the auxiliary variable (.aux) file.
    pub fn auxiliary_variable_file_path(&self) -> &Path {
        self.aux_path.as_path()
    }

    pub fn verbosity(&self) -> &Verbosity {
        &self.verbosity
    }

    fn insert_primary_variable_descriptors(
        is_mhd: bool,
        variable_descriptors: &mut HashMap<String, VariableDescriptor>,
    ) {
        let is_primary = true;
        variable_descriptors.insert(
            "r".to_string(),
            VariableDescriptor {
                is_primary,
                locations: In3D::new(Center, Center, Center),
                index: 0,
            },
        );
        variable_descriptors.insert(
            "px".to_string(),
            VariableDescriptor {
                is_primary,
                locations: In3D::new(LowerEdge, Center, Center),
                index: 1,
            },
        );
        variable_descriptors.insert(
            "py".to_string(),
            VariableDescriptor {
                is_primary,
                locations: In3D::new(Center, LowerEdge, Center),
                index: 2,
            },
        );
        variable_descriptors.insert(
            "pz".to_string(),
            VariableDescriptor {
                is_primary,
                locations: In3D::new(Center, Center, LowerEdge),
                index: 3,
            },
        );
        variable_descriptors.insert(
            "e".to_string(),
            VariableDescriptor {
                is_primary,
                locations: In3D::new(Center, Center, Center),
                index: 4,
            },
        );
        if is_mhd {
            variable_descriptors.insert(
                "bx".to_string(),
                VariableDescriptor {
                    is_primary,
                    locations: In3D::new(LowerEdge, Center, Center),
                    index: 5,
                },
            );
            variable_descriptors.insert(
                "by".to_string(),
                VariableDescriptor {
                    is_primary,
                    locations: In3D::new(Center, LowerEdge, Center),
                    index: 6,
                },
            );
            variable_descriptors.insert(
                "bz".to_string(),
                VariableDescriptor {
                    is_primary,
                    locations: In3D::new(Center, Center, LowerEdge),
                    index: 7,
                },
            );
        }
    }

    fn insert_auxiliary_variable_descriptors(
        aux_variable_names: &[String],
        variable_descriptors: &mut HashMap<String, VariableDescriptor>,
    ) -> io::Result<()> {
        let is_primary = false;

        for (index, name) in aux_variable_names.iter().enumerate() {
            let name_length = name.len();
            let ends_with_x = name.ends_with('x');
            let ends_with_y = name.ends_with('y');
            let ends_with_z = name.ends_with('z');

            let locations = if name_length == 2 && (ends_with_x || ends_with_y || ends_with_z) {
                if name.starts_with('p') || name.starts_with('b') {
                    In3D::new(
                        if ends_with_x { LowerEdge } else { Center },
                        if ends_with_y { LowerEdge } else { Center },
                        if ends_with_z { LowerEdge } else { Center },
                    )
                } else if name.starts_with('e') || name.starts_with('i') {
                    In3D::new(
                        if ends_with_x { Center } else { LowerEdge },
                        if ends_with_y { Center } else { LowerEdge },
                        if ends_with_z { Center } else { LowerEdge },
                    )
                } else {
                    In3D::same(Center)
                }
            } else {
                In3D::same(Center)
            };

            variable_descriptors.insert(
                name.to_string(),
                VariableDescriptor {
                    is_primary,
                    locations,
                    index,
                },
            );
        }

        Ok(())
    }

    fn get_variable_descriptor(&self, name: &str) -> io::Result<&VariableDescriptor> {
        match self.variable_descriptors.get(name) {
            Some(variable) => Ok(variable),
            None => Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )),
        }
    }
}

impl ScalarFieldProvider3<fdt> for NativeSnapshotReader3 {
    fn grid(&self) -> &FieldGrid3 {
        self.grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<FieldGrid3> {
        Arc::clone(&self.grid)
    }

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn has_variable(&self, variable_name: &str) -> bool {
        self.all_variable_names()
            .contains(&variable_name.to_string())
    }

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<fdt>> {
        let variable_descriptor = self.get_variable_descriptor(variable_name)?;
        let file_path = if variable_descriptor.is_primary {
            self.primary_variable_file_path()
        } else {
            self.auxiliary_variable_file_path()
        };
        if self.verbosity().print_messages() {
            println!(
                "Reading {} from {}",
                variable_name,
                file_path.file_name().unwrap().to_string_lossy()
            );
        }
        let shape = self.grid().shape();
        let number_of_values = shape[X] * shape[Y] * shape[Z];
        let byte_offset = number_of_values * variable_descriptor.index * mem::size_of::<fdt>();
        let buffer = utils::read_from_binary_file(
            file_path,
            number_of_values,
            byte_offset,
            self.endianness,
        )?;
        let values = Array::from_shape_vec((shape[X], shape[Y], shape[Z]).f(), buffer).unwrap();
        Ok(ScalarField3::new(
            variable_name.to_string(),
            self.arc_with_grid(),
            variable_descriptor.locations.clone(),
            values,
        ))
    }
}

/// Writes the data associated with the given snapshot to native snapshot files at the given path.
pub fn write_new_snapshot(
    input_metadata: &dyn SnapshotMetadata,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    output_param_path: &Path,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> io::Result<()> {
    let quantity_names = provider.all_variable_names().to_vec();
    write_modified_snapshot(
        input_metadata,
        provider,
        &quantity_names,
        output_param_path,
        false,
        true,
        io_context,
        verbosity,
    )
}

/// Writes modified data associated with the given snapshot to native snapshot files at the given path.
pub fn write_modified_snapshot(
    input_metadata: &dyn SnapshotMetadata,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    quantity_names: &[String],
    output_param_path: &Path,
    is_scratch: bool,
    write_mesh_file: bool,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> io::Result<()> {
    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(output_param_path);
    let mut signed_snap_num =
        snap_num.unwrap_or(if is_scratch { 1 } else { FALLBACK_SNAP_NUM }) as i64;
    if is_scratch {
        signed_snap_num = -signed_snap_num;
    }

    let (included_primary_variable_names, included_auxiliary_variable_names, is_mhd) =
        input_metadata.classify_variable_names(quantity_names);

    let has_primary = !included_primary_variable_names.is_empty();
    let has_auxiliary = !included_auxiliary_variable_names.is_empty();

    let atomic_param_file =
        io_context.create_atomic_output_file(output_param_path.to_path_buf())?;
    let mut atomic_mesh_file = if write_mesh_file {
        Some(
            io_context.create_atomic_output_file(
                atomic_param_file
                    .target_path()
                    .with_file_name(format!("{}.mesh", snap_name)),
            )?,
        )
    } else {
        None
    };
    let atomic_snap_file = io_context.create_atomic_output_file(if is_scratch {
        atomic_param_file
            .target_path()
            .with_file_name(format!("{}.snap.scr", snap_name))
    } else {
        atomic_param_file.target_path().with_extension("snap")
    })?;
    let atomic_aux_file = io_context.create_atomic_output_file(if is_scratch {
        atomic_param_file
            .target_path()
            .with_file_name(format!("{}.aux.scr", snap_name))
    } else {
        atomic_param_file.target_path().with_extension("aux")
    })?;

    let write_param_file = atomic_param_file.check_if_write_allowed(io_context, verbosity);
    let write_mesh_file = if let Some(atomic_mesh_path) = &atomic_mesh_file {
        atomic_mesh_path.check_if_write_allowed(io_context, verbosity)
    } else {
        false
    };
    let write_snap_file =
        has_primary && atomic_snap_file.check_if_write_allowed(io_context, verbosity);
    let write_aux_file =
        has_auxiliary && atomic_aux_file.check_if_write_allowed(io_context, verbosity);

    if write_param_file {
        let output_param_file_name = atomic_param_file
            .target_path()
            .file_name()
            .unwrap()
            .to_string_lossy();

        let new_parameters = input_metadata.create_updated_parameters(
            provider.grid(),
            snap_name.as_str(),
            signed_snap_num,
            &included_auxiliary_variable_names,
            is_mhd,
        );

        if verbosity.print_messages() {
            println!("Writing parameters to {}", output_param_file_name);
        }
        utils::write_text_file(
            &new_parameters.borrow().native_text_representation(),
            atomic_param_file.temporary_path(),
        )?;
    }
    if write_mesh_file {
        let atomic_mesh_path = atomic_mesh_file.as_ref().unwrap();
        if verbosity.print_messages() {
            println!(
                "Writing grid to {}",
                atomic_mesh_path
                    .target_path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
            );
        }
        mesh::write_mesh_file_from_grid(provider.grid(), atomic_mesh_path.temporary_path())?;
    }

    let endianness = input_metadata.endianness();

    if write_snap_file {
        let output_snap_file_name = atomic_snap_file
            .target_path()
            .file_name()
            .unwrap()
            .to_string_lossy();

        write_3d_snapfile(
            atomic_snap_file.temporary_path(),
            &included_primary_variable_names,
            &mut |name| {
                provider.produce_scalar_field(name).map(|field| {
                    if verbosity.print_messages() {
                        println!("Writing {} to {}", name, output_snap_file_name);
                    }
                    field.into_values()
                })
            },
            endianness,
        )?;
    }

    if write_aux_file {
        let output_aux_file_name = atomic_aux_file
            .target_path()
            .file_name()
            .unwrap()
            .to_string_lossy();

        write_3d_snapfile(
            atomic_aux_file.temporary_path(),
            &included_auxiliary_variable_names,
            &mut |name| {
                provider.produce_scalar_field(name).map(|field| {
                    if verbosity.print_messages() {
                        println!("Writing {} to {}", name, output_aux_file_name);
                    }
                    field.into_values()
                })
            },
            endianness,
        )?;
    }

    if write_param_file {
        io_context.close_atomic_output_file(atomic_param_file)?;
    }
    if write_mesh_file {
        io_context.close_atomic_output_file(atomic_mesh_file.take().unwrap())?;
    }
    if write_snap_file {
        io_context.close_atomic_output_file(atomic_snap_file)?;
    }
    if write_aux_file {
        io_context.close_atomic_output_file(atomic_aux_file)?;
    }

    Ok(())
}

/// Writes arrays of variable values sequentially into a binary file.
///
/// # Parameters
///
/// - `output_file_path`: Path where the output file should be written.
/// - `variable_names`: Names of the variables to write.
/// - `variable_value_producer`: Closure producing an array of values given the variable name.
/// - `endianness`: Endianness of the output data.
///
/// # Returns
///
/// A `Result` which is either:
///
/// - `Ok`: Writing was completed successfully.
/// - `Err`: Contains an error encountered while trying to create or write to the file.
fn write_3d_snapfile(
    output_file_path: &Path,
    variable_names: &[String],
    variable_value_producer: &mut dyn FnMut(&str) -> io::Result<Array3<fdt>>,
    endianness: Endianness,
) -> io::Result<()> {
    let number_of_variables = variable_names.len();
    assert!(
        number_of_variables > 0,
        "Number of variables must larger than zero."
    );

    let name = variable_names[0].as_ref();
    let variable_values = variable_value_producer(name)?;
    let array_length = variable_values.len();
    let float_size = mem::size_of::<fdt>();
    let byte_buffer_size = array_length * float_size;
    let mut byte_buffer = vec![0_u8; byte_buffer_size];

    let mut file = utils::create_file_and_required_directories(output_file_path)?;
    file.set_len(byte_buffer_size as u64)?;

    utils::write_into_byte_buffer(
        variable_values
            .as_slice_memory_order()
            .expect("Values array not contiguous"),
        &mut byte_buffer,
        0,
        endianness,
    );
    file.write_all(&byte_buffer)?;

    for name in variable_names.iter().skip(1) {
        let name = name.as_ref();
        let variable_values = variable_value_producer(name)?;
        assert_eq!(
            variable_values.len(),
            array_length,
            "All variable arrays must have the same length."
        );
        utils::write_into_byte_buffer(
            variable_values
                .as_slice_memory_order()
                .expect("Values array not contiguous"),
            &mut byte_buffer,
            0,
            endianness,
        );
        file.write_all(&byte_buffer)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct VariableDescriptor {
    is_primary: bool,
    locations: In3D<CoordLocation>,
    index: usize,
}
