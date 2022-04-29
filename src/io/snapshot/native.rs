//! Reading and writing of Bifrost simulation data in native format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self, AtomicOutputPath},
        Endianness, OverwriteMode, Verbose,
    },
    fdt, ParameterValue, SnapshotParameters, SnapshotProvider3, FALLBACK_SNAP_NUM,
    PRIMARY_VARIABLE_NAMES_HD, PRIMARY_VARIABLE_NAMES_MHD,
};
use crate::{
    field::{ScalarField3, ScalarFieldProvider3},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{
        CoordLocation::{self, Center, LowerEdge},
        Grid3,
    },
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

pub use mesh::{create_grid_from_mesh_file, parse_mesh_file, write_mesh_file_from_grid};
pub use param::NativeSnapshotParameters;

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
    primary_variable_names: Vec<String>,
    auxiliary_variable_names: Vec<String>,
    all_variable_names: Vec<String>,
    variable_descriptors: HashMap<String, VariableDescriptor>,
}

impl<G: Grid3<fdt>> NativeSnapshotReader3<G> {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: NativeSnapshotReaderConfig) -> io::Result<Self> {
        let parameters = NativeSnapshotParameters::new(&config.param_file_path, config.verbose())?;

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
        let is_mhd = parameters.determine_if_mhd()?;

        let (snap_path, aux_path) = parameters.determine_snap_path()?;

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
        let mut all_variable_names = primary_variable_names.clone();
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
            primary_variable_names,
            auxiliary_variable_names,
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
            let ends_with_x = name.ends_with('x');
            let ends_with_y = name.ends_with('y');
            let ends_with_z = name.ends_with('z');

            let locations = if (ends_with_x || ends_with_y || ends_with_z)
                && (name.starts_with('e') || name.starts_with('i'))
            {
                In3D::new(
                    if ends_with_x { Center } else { LowerEdge },
                    if ends_with_y { Center } else { LowerEdge },
                    if ends_with_z { Center } else { LowerEdge },
                )
            } else {
                In3D::new(
                    if ends_with_x { LowerEdge } else { Center },
                    if ends_with_y { LowerEdge } else { Center },
                    if ends_with_z { LowerEdge } else { Center },
                )
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
            primary_variable_names,
            auxiliary_variable_names,
            variable_descriptors,
            ..
        } = Self::new(self.config.clone())?;

        self.parameters = parameters;
        self.snap_path = snap_path;
        self.aux_path = aux_path;
        self.grid = grid;
        self.primary_variable_names = primary_variable_names;
        self.auxiliary_variable_names = auxiliary_variable_names;
        self.variable_descriptors = variable_descriptors;

        Ok(())
    }

    fn read_scalar_field<S: AsRef<str>>(
        &self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
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

impl<G: Grid3<fdt>> ScalarFieldProvider3<fdt, G> for NativeSnapshotReader3<G> {
    fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    fn produce_scalar_field<S: AsRef<str>>(
        &mut self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        self.read_scalar_field(variable_name)
    }
}

impl<G: Grid3<fdt>> SnapshotProvider3<G> for NativeSnapshotReader3<G> {
    type Parameters = NativeSnapshotParameters;

    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }

    fn endianness(&self) -> Endianness {
        self.config.endianness
    }

    fn primary_variable_names(&self) -> &[String] {
        &self.primary_variable_names
    }

    fn auxiliary_variable_names(&self) -> &[String] {
        &self.auxiliary_variable_names
    }

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn has_variable<S: AsRef<str>>(&self, variable_name: S) -> bool {
        self.all_variable_names()
            .contains(&variable_name.as_ref().to_string())
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        super::extract_name_and_num_from_snapshot_path(self.config.param_file_path())
    }
}

impl NativeSnapshotReaderConfig {
    /// Creates a new set of snapshot reader configuration parameters.
    pub fn new<P: AsRef<Path>>(
        param_file_path: P,
        endianness: Endianness,
        verbose: Verbose,
    ) -> Self {
        NativeSnapshotReaderConfig {
            param_file_path: param_file_path.as_ref().to_path_buf(),
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

/// Writes modified data associated with the given snapshot to native snapshot files at the given path.
pub fn write_modified_snapshot<Pa, G, P>(
    provider: &mut P,
    quantity_names: &[String],
    mut modified_parameters: HashMap<&str, ParameterValue>,
    output_param_path: Pa,
    is_scratch: bool,
    write_mesh_file: bool,
    overwrite_mode: OverwriteMode,
    protected_file_types: &[&str],
    verbose: Verbose,
) -> io::Result<()>
where
    Pa: AsRef<Path>,
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    let output_param_path = output_param_path.as_ref();

    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(output_param_path);
    let snap_num = snap_num.unwrap_or(if is_scratch { 1 } else { FALLBACK_SNAP_NUM });

    modified_parameters.insert(
        "snapname",
        ParameterValue::Str(format!("\"{}\"", snap_name)),
    );
    modified_parameters.insert(
        "isnap",
        ParameterValue::Str(format!("{}{}", if is_scratch { "-" } else { "" }, snap_num)),
    );
    modified_parameters.insert(
        "meshfile",
        ParameterValue::Str(format!("\"{}.mesh\"", snap_name)),
    );

    let (included_primary_variable_names, included_auxiliary_variable_names) =
        provider.classify_variable_names(quantity_names);

    modified_parameters.insert(
        "aux",
        ParameterValue::Str(format!(
            "\"{}\"",
            included_auxiliary_variable_names.join(" ")
        )),
    );

    let grid = provider.grid();

    let shape = grid.shape();
    let average_grid_cell_extents = grid.average_grid_cell_extents();
    modified_parameters.insert("mx", ParameterValue::Int(shape[X] as i64));
    modified_parameters.insert("my", ParameterValue::Int(shape[Y] as i64));
    modified_parameters.insert("mz", ParameterValue::Int(shape[Z] as i64));
    modified_parameters.insert("dx", ParameterValue::Float(average_grid_cell_extents[X]));
    modified_parameters.insert("dy", ParameterValue::Float(average_grid_cell_extents[Y]));
    modified_parameters.insert("dz", ParameterValue::Float(average_grid_cell_extents[Z]));

    let has_primary = !included_primary_variable_names.is_empty();
    let has_auxiliary = !included_auxiliary_variable_names.is_empty();

    let atomic_param_path = AtomicOutputPath::new(output_param_path)?;
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

        let mut new_parameters = provider.parameters().clone();
        new_parameters.modify_values(modified_parameters);

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
        mesh::write_mesh_file_from_grid(grid, atomic_mesh_path.temporary_path())?;
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
///
/// # Type parameters
///
/// - `P`: A type that can be treated as a reference to a `Path`.
/// - `N`: A type that can be treated as a reference to a `str`.
/// - `V`: A function type taking a reference to a string slice and returning a reference to a 3D array.
fn write_3d_snapfile<P, N, V>(
    output_file_path: P,
    variable_names: &[N],
    variable_value_producer: &mut V,
    endianness: Endianness,
) -> io::Result<()>
where
    P: AsRef<Path>,
    N: AsRef<str>,
    V: FnMut(&str) -> io::Result<Array3<fdt>>,
{
    let output_file_path = output_file_path.as_ref();

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
