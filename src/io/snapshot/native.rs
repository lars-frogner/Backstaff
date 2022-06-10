//! Reading and writing of Bifrost simulation data in native format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self, AtomicOutputPath},
        Endianness, OverwriteMode, Verbose,
    },
    fdt, SnapshotParameters, SnapshotProvider3, SnapshotReader3, FALLBACK_SNAP_NUM,
    PRIMARY_VARIABLE_NAMES_HD, PRIMARY_VARIABLE_NAMES_MHD,
};
use crate::{
    field::{ScalarField3, ScalarFieldProvider3},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{
        fgr,
        CoordLocation::{self, Center, LowerEdge},
        Grid3, GridType,
    },
    with_io_err_msg,
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
    create_grid_from_mesh_file, parse_mesh_file, write_mesh_file_from_grid, NativeGridData,
};
pub use param::NativeSnapshotParameters;

#[cfg(feature = "comparison")]
pub use mesh::parsed_mesh_files_eq;

/// Configuration parameters for native snapshot reader.
#[derive(Clone, Debug)]
pub struct NativeSnapshotReaderConfig {
    /// Path to the parameter (.idl) file.
    param_file_path: PathBuf,
    /// Order of bytes in the binary data files.
    endianness: Endianness,
    /// Whether to print status messages while reading fields.
    verbose: Verbose,
}

/// Reader for the native output files associated with Bifrost 3D simulation snapshots.
#[derive(Clone, Debug)]
pub struct NativeSnapshotReader3<G> {
    config: NativeSnapshotReaderConfig,
    parameters: NativeSnapshotParameters,
    snap_path: PathBuf,
    aux_path: PathBuf,
    grid: Arc<G>,
    all_variable_names: Vec<String>,
    variable_descriptors: HashMap<String, VariableDescriptor>,
}

impl<G: Grid3<fgr>> NativeSnapshotReader3<G> {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: NativeSnapshotReaderConfig) -> io::Result<Self> {
        let parameters =
            NativeSnapshotParameters::new(config.param_file_path.clone(), config.verbose())?;

        let mesh_path = parameters.determine_mesh_path()?;
        let is_periodic = parameters.determine_grid_periodicity()?;

        let grid = create_grid_from_mesh_file(&mesh_path, is_periodic, config.verbose())?;

        Self::new_from_parameters_and_grid(config, parameters, grid)
    }

    /// Returns the path to the file representing the snapshot.
    pub fn path(&self) -> &Path {
        self.config.param_file_path()
    }

    /// Whether the reader is verbose.
    pub fn verbose(&self) -> Verbose {
        self.config.verbose()
    }

    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new_from_parameters_and_grid(
        config: NativeSnapshotReaderConfig,
        parameters: NativeSnapshotParameters,
        grid: G,
    ) -> io::Result<Self> {
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

        Ok(Self {
            config,
            parameters,
            snap_path,
            aux_path,
            grid: Arc::new(grid),
            all_variable_names,
            variable_descriptors,
        })
    }

    /// Returns the path of the parameter (.idl) file.
    pub fn parameter_file_path(&self) -> &Path {
        self.config.param_file_path.as_path()
    }

    /// Returns the path of the primary variable (.snap) file.
    pub fn primary_variable_file_path(&self) -> &Path {
        self.snap_path.as_path()
    }

    /// Returns the path of the auxiliary variable (.aux) file.
    pub fn auxiliary_variable_file_path(&self) -> &Path {
        self.aux_path.as_path()
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

    #[allow(dead_code)]
    fn reread(&mut self) -> io::Result<()> {
        let Self {
            parameters,
            snap_path,
            aux_path,
            grid,
            variable_descriptors,
            ..
        } = Self::new(self.config.clone())?;

        self.parameters = parameters;
        self.snap_path = snap_path;
        self.aux_path = aux_path;
        self.grid = grid;
        self.variable_descriptors = variable_descriptors;

        Ok(())
    }
}

impl<G: Grid3<fgr>> ScalarFieldProvider3<fdt, G> for NativeSnapshotReader3<G> {
    fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
        self.read_scalar_field(variable_name)
    }
}

impl<G: Grid3<fgr>> SnapshotProvider3<G> for NativeSnapshotReader3<G> {
    type Parameters = NativeSnapshotParameters;

    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }

    fn endianness(&self) -> Endianness {
        self.config.endianness
    }

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn has_variable(&self, variable_name: &str) -> bool {
        self.all_variable_names()
            .contains(&variable_name.to_string())
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u64>) {
        super::extract_name_and_num_from_snapshot_path(self.config.param_file_path())
    }
}

impl<G: Grid3<fgr>> SnapshotReader3<G> for NativeSnapshotReader3<G> {
    fn read_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
        let variable_descriptor = self.get_variable_descriptor(variable_name)?;
        let file_path = if variable_descriptor.is_primary {
            &self.snap_path
        } else {
            &self.aux_path
        };
        if self.config.verbose.is_yes() {
            println!(
                "Reading {} from {}",
                variable_name,
                file_path.file_name().unwrap().to_string_lossy()
            );
        }
        let shape = self.grid.shape();
        let number_of_values = shape[X] * shape[Y] * shape[Z];
        let byte_offset = number_of_values * variable_descriptor.index * mem::size_of::<fdt>();
        let buffer = utils::read_from_binary_file(
            file_path,
            number_of_values,
            byte_offset,
            self.config.endianness,
        )?;
        let values = Array::from_shape_vec((shape[X], shape[Y], shape[Z]).f(), buffer).unwrap();
        Ok(ScalarField3::new(
            variable_name.to_string(),
            Arc::clone(&self.grid),
            variable_descriptor.locations.clone(),
            values,
        ))
    }
}

impl NativeSnapshotReaderConfig {
    /// Creates a new set of snapshot reader configuration parameters.
    pub fn new(param_file_path: PathBuf, endianness: Endianness, verbose: Verbose) -> Self {
        NativeSnapshotReaderConfig {
            param_file_path,
            endianness,
            verbose,
        }
    }

    pub fn verbose(&self) -> Verbose {
        self.verbose
    }

    pub fn param_file_path(&self) -> &Path {
        self.param_file_path.as_path()
    }
}

/// Helper object for interpreting native snapshot metadata and creating
/// the corresponding reader object.
#[derive(Debug, Clone)]
pub struct NativeSnapshotMetadata {
    reader_config: NativeSnapshotReaderConfig,
    parameters: NativeSnapshotParameters,
    is_periodic: In3D<bool>,
    grid_data: NativeGridData,
}

impl NativeSnapshotMetadata {
    /// Gathers the metadata for the snapshot specified in the given
    /// reader configuration and creates a new metadata object.
    pub fn new(reader_config: NativeSnapshotReaderConfig) -> io::Result<Self> {
        let parameters = with_io_err_msg!(
            NativeSnapshotParameters::new(
                reader_config.param_file_path().to_path_buf(),
                reader_config.verbose()
            ),
            "Could not read parameter file: {}"
        )?;
        let mesh_path = with_io_err_msg!(
            parameters.determine_mesh_path(),
            "Could not obtain path to mesh file: {}"
        )?;
        let is_periodic = with_io_err_msg!(
            parameters.determine_grid_periodicity(),
            "Could not determine grid periodicity: {}"
        )?;
        let grid_data = with_io_err_msg!(
            parse_mesh_file(&mesh_path, reader_config.verbose()),
            "Could not parse mesh file: {}"
        )?;
        Ok(Self {
            reader_config,
            parameters,
            is_periodic,
            grid_data,
        })
    }

    /// Returns the type of the grid used by this snapshot.
    pub fn grid_type(&self) -> GridType {
        self.grid_data.detected_grid_type
    }

    /// Creates a new snapshot reader from this metadata.
    pub fn into_reader<G: Grid3<fgr>>(self) -> io::Result<NativeSnapshotReader3<G>> {
        let (reader_config, parameters, grid) = self.into_parameters_and_grid();

        with_io_err_msg!(
            NativeSnapshotReader3::new_from_parameters_and_grid(reader_config, parameters, grid),
            "Could not create snapshot reader: {}"
        )
    }

    /// Creates a new grid from this metadata.
    pub fn into_grid<G: Grid3<fgr>>(self) -> G {
        let (_, _, grid) = self.into_parameters_and_grid();
        grid
    }

    fn into_parameters_and_grid<G: Grid3<fgr>>(
        self,
    ) -> (NativeSnapshotReaderConfig, NativeSnapshotParameters, G) {
        let Self {
            reader_config,
            parameters,
            is_periodic,
            grid_data,
        } = self;

        let NativeGridData {
            detected_grid_type: _,
            center_coords,
            lower_edge_coords,
            up_derivatives,
            down_derivatives,
        } = grid_data;

        let grid = G::from_coords(
            center_coords,
            lower_edge_coords,
            is_periodic,
            Some(up_derivatives),
            Some(down_derivatives),
        );
        (reader_config, parameters, grid)
    }
}

/// Writes the data associated with the given snapshot to native snapshot files at the given path.
pub fn write_new_snapshot<G, P>(
    provider: &mut P,
    output_param_path: &Path,
    verbose: Verbose,
) -> io::Result<()>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let quantity_names = provider.all_variable_names().to_vec();
    write_modified_snapshot(
        provider,
        &quantity_names,
        output_param_path,
        false,
        true,
        OverwriteMode::Always,
        &[],
        verbose,
    )
}

/// Writes modified data associated with the given snapshot to native snapshot files at the given path.
pub fn write_modified_snapshot<G, P>(
    provider: &mut P,
    quantity_names: &[String],
    output_param_path: &Path,
    is_scratch: bool,
    write_mesh_file: bool,
    overwrite_mode: OverwriteMode,
    protected_file_types: &[&str],
    verbose: Verbose,
) -> io::Result<()>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(output_param_path);
    let mut signed_snap_num =
        snap_num.unwrap_or(if is_scratch { 1 } else { FALLBACK_SNAP_NUM }) as i64;
    if is_scratch {
        signed_snap_num = -signed_snap_num;
    }

    let (included_primary_variable_names, included_auxiliary_variable_names, is_mhd) =
        provider.classify_variable_names(quantity_names);

    let has_primary = !included_primary_variable_names.is_empty();
    let has_auxiliary = !included_auxiliary_variable_names.is_empty();

    let atomic_param_path = AtomicOutputPath::new(output_param_path.to_path_buf())?;
    let mut atomic_mesh_path = if write_mesh_file {
        Some(AtomicOutputPath::new(
            atomic_param_path
                .target_path()
                .with_file_name(format!("{}.mesh", snap_name)),
        )?)
    } else {
        None
    };
    let atomic_snap_path = AtomicOutputPath::new(if is_scratch {
        atomic_param_path
            .target_path()
            .with_file_name(format!("{}.snap.scr", snap_name))
    } else {
        atomic_param_path.target_path().with_extension("snap")
    })?;
    let atomic_aux_path = AtomicOutputPath::new(if is_scratch {
        atomic_param_path
            .target_path()
            .with_file_name(format!("{}.aux.scr", snap_name))
    } else {
        atomic_param_path.target_path().with_extension("aux")
    })?;

    let write_param_file =
        atomic_param_path.check_if_write_allowed(overwrite_mode, protected_file_types);
    let write_mesh_file = if let Some(atomic_mesh_path) = &atomic_mesh_path {
        atomic_mesh_path.check_if_write_allowed(overwrite_mode, protected_file_types)
    } else {
        false
    };
    let write_snap_file = has_primary
        && atomic_snap_path.check_if_write_allowed(overwrite_mode, protected_file_types);
    let write_aux_file = has_auxiliary
        && atomic_aux_path.check_if_write_allowed(overwrite_mode, protected_file_types);

    if write_param_file {
        let output_param_file_name = atomic_param_path
            .target_path()
            .file_name()
            .unwrap()
            .to_string_lossy();

        let new_parameters = provider.create_updated_parameters(
            snap_name.as_str(),
            signed_snap_num,
            &included_auxiliary_variable_names,
            is_mhd,
        );

        if verbose.is_yes() {
            println!("Writing parameters to {}", output_param_file_name);
        }
        utils::write_text_file(
            &new_parameters.native_text_representation(),
            atomic_param_path.temporary_path(),
        )?;
    }
    if write_mesh_file {
        let atomic_mesh_path = atomic_mesh_path.as_ref().unwrap();
        if verbose.is_yes() {
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

    let endianness = provider.endianness();

    if write_snap_file {
        let output_snap_file_name = atomic_snap_path
            .target_path()
            .file_name()
            .unwrap()
            .to_string_lossy();

        write_3d_snapfile(
            atomic_snap_path.temporary_path(),
            &included_primary_variable_names,
            &mut |name| {
                provider.produce_scalar_field(name).map(|field| {
                    if verbose.is_yes() {
                        println!("Writing {} to {}", name, output_snap_file_name);
                    }
                    field.into_values()
                })
            },
            endianness,
        )?;
    }

    if write_aux_file {
        let output_aux_file_name = atomic_aux_path
            .target_path()
            .file_name()
            .unwrap()
            .to_string_lossy();

        write_3d_snapfile(
            atomic_aux_path.temporary_path(),
            &included_auxiliary_variable_names,
            &mut |name| {
                provider.produce_scalar_field(name).map(|field| {
                    if verbose.is_yes() {
                        println!("Writing {} to {}", name, output_aux_file_name);
                    }
                    field.into_values()
                })
            },
            endianness,
        )?;
    }

    if write_param_file {
        atomic_param_path.perform_replace()?;
    }
    if write_mesh_file {
        atomic_mesh_path.take().unwrap().perform_replace()?;
    }
    if write_snap_file {
        atomic_snap_path.perform_replace()?;
    }
    if write_aux_file {
        atomic_aux_path.perform_replace()?;
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
