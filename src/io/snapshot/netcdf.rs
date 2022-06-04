//! Reading and writing of Bifrost simulation data in NetCDF format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self, AtomicOutputPath},
        Endianness, OverwriteMode, Verbose,
    },
    fdt, fpa, ParameterValue, SnapshotParameters, SnapshotProvider3, SnapshotReader3,
    COORDINATE_NAMES, FALLBACK_SNAP_NUM, PRIMARY_VARIABLE_NAMES_MHD,
};
use crate::{
    field::{ScalarField3, ScalarFieldProvider3},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{fgr, CoordLocation, Grid3, GridType},
    io_result,
    num::BFloat,
    with_io_err_msg,
};
use ndarray::prelude::*;
use netcdf_rs::{self as nc, File, Group, GroupMut, MutableFile, Numeric};
use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub use mesh::{read_grid_data, NetCDFGridData};
pub use param::NetCDFSnapshotParameters;

/// Configuration parameters for NetCDF snapshot reader.
#[derive(Clone, Debug)]
pub struct NetCDFSnapshotReaderConfig {
    /// Path to the file.
    file_path: PathBuf,
    /// Whether to print status messages while reading fields.
    verbose: Verbose,
}

/// Reader for NetCDF files associated with Bifrost 3D simulation snapshots.
#[derive(Debug)]
pub struct NetCDFSnapshotReader3<G> {
    config: NetCDFSnapshotReaderConfig,
    file: File,
    grid: Arc<G>,
    parameters: NetCDFSnapshotParameters,
    endianness: Endianness,
    primary_variable_names: Vec<String>,
    auxiliary_variable_names: Vec<String>,
    all_variable_names: Vec<String>,
}

impl<G: Grid3<fgr>> NetCDFSnapshotReader3<G> {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: NetCDFSnapshotReaderConfig) -> io::Result<Self> {
        let file = open_file(&config.file_path)?;

        let parameters = NetCDFSnapshotParameters::new(&file, config.verbose())?;

        let is_periodic = parameters.determine_grid_periodicity()?;
        let (grid, endianness) = mesh::read_grid::<G>(&file, is_periodic, config.verbose())?;

        Ok(Self::new_from_parameters_and_grid(
            config, file, parameters, grid, endianness,
        ))
    }

    /// Returns the path to the file representing the snapshot.
    pub fn path(&self) -> &Path {
        self.config.file_path()
    }

    /// Whether the reader is verbose.
    pub fn verbose(&self) -> Verbose {
        self.config.verbose
    }

    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new_from_parameters_and_grid(
        config: NetCDFSnapshotReaderConfig,
        file: File,
        parameters: NetCDFSnapshotParameters,
        grid: G,
        endianness: Endianness,
    ) -> Self {
        let root_group = file.root().unwrap();
        let (primary_variable_names, auxiliary_variable_names) =
            Self::read_variable_names(&root_group);
        let mut all_variable_names = primary_variable_names.clone();
        all_variable_names.append(&mut auxiliary_variable_names.clone());

        Self {
            config,
            file,
            grid: Arc::new(grid),
            parameters,
            endianness,
            primary_variable_names,
            auxiliary_variable_names,
            all_variable_names,
        }
    }

    fn read_variable_names(group: &Group) -> (Vec<String>, Vec<String>) {
        let variable_names = read_all_non_coord_variable_names(group);
        let primary_variable_names: Vec<_> = PRIMARY_VARIABLE_NAMES_MHD
            .iter()
            .map(|&name| name.to_string())
            .collect();
        if primary_variable_names
            .iter()
            .all(|primary_variable_name| variable_names.contains(primary_variable_name))
        {
            let secondary_variable_names = variable_names
                .into_iter()
                .filter(|name| !primary_variable_names.contains(name))
                .collect();
            (primary_variable_names, secondary_variable_names)
        } else {
            (Vec::new(), variable_names)
        }
    }

    #[allow(dead_code)]
    fn reread(&mut self) -> io::Result<()> {
        let Self {
            file,
            grid,
            parameters,
            endianness,
            primary_variable_names,
            auxiliary_variable_names,
            ..
        } = Self::new(self.config.clone())?;

        self.file = file;
        self.grid = grid;
        self.parameters = parameters;
        self.endianness = endianness;
        self.primary_variable_names = primary_variable_names;
        self.auxiliary_variable_names = auxiliary_variable_names;

        Ok(())
    }
}

impl<G: Grid3<fgr>> ScalarFieldProvider3<fdt, G> for NetCDFSnapshotReader3<G> {
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

impl<G: Grid3<fgr>> SnapshotProvider3<G> for NetCDFSnapshotReader3<G> {
    type Parameters = NetCDFSnapshotParameters;

    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }

    fn endianness(&self) -> Endianness {
        self.endianness
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
        super::extract_name_and_num_from_snapshot_path(self.config.file_path())
    }
}

impl<G: Grid3<fgr>> SnapshotReader3<G> for NetCDFSnapshotReader3<G> {
    fn read_scalar_field<S: AsRef<str>>(
        &self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
        if self.config.verbose.is_yes() {
            println!(
                "Reading {} from {}",
                variable_name,
                self.config.file_path.file_name().unwrap().to_string_lossy()
            );
        }
        let (values, locations, endianness) = read_snapshot_3d_variable::<fdt, G>(
            &self.file.root().unwrap(),
            self.grid(),
            variable_name,
        )?;
        if endianness != self.endianness {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Inconsistent endianness for variable {} in NetCDF file",
                    variable_name
                ),
            ));
        }
        Ok(ScalarField3::new(
            variable_name.to_string(),
            Arc::clone(&self.grid),
            locations,
            values,
        ))
    }
}

/// Helper object for interpreting NetCDF snapshot metadata and creating
/// the corresponding reader object.
#[derive(Debug)]
pub struct NetCDFSnapshotMetadata {
    reader_config: NetCDFSnapshotReaderConfig,
    file: File,
    parameters: NetCDFSnapshotParameters,
    is_periodic: In3D<bool>,
    grid_data: NetCDFGridData,
}

impl NetCDFSnapshotMetadata {
    /// Gathers the metadata for the snapshot specified in the given
    /// reader configuration and creates a new metadata object.
    pub fn new(reader_config: NetCDFSnapshotReaderConfig) -> io::Result<Self> {
        let file = with_io_err_msg!(
            open_file(reader_config.file_path()),
            "Could not open NetCDF file: {}"
        )?;
        let parameters = with_io_err_msg!(
            NetCDFSnapshotParameters::new(&file, reader_config.verbose()),
            "Could not read snapshot parameters from NetCDF file: {}"
        )?;
        let grid_data = with_io_err_msg!(
            read_grid_data(&file, reader_config.verbose()),
            "Could not read grid data from NetCDF file: {}"
        )?;
        let is_periodic = with_io_err_msg!(
            parameters.determine_grid_periodicity(),
            "Could not determine grid periodicity: {}"
        )?;
        Ok(Self {
            reader_config,
            file,
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
    pub fn into_reader<G: Grid3<fgr>>(self) -> NetCDFSnapshotReader3<G> {
        let (reader_config, file, parameters, grid, endianness) = self.into_parameters_and_grid();

        NetCDFSnapshotReader3::new_from_parameters_and_grid(
            reader_config,
            file,
            parameters,
            grid,
            endianness,
        )
    }

    /// Creates a new grid from this metadata.
    pub fn into_grid<G: Grid3<fgr>>(self) -> G {
        let (_, _, _, grid, _) = self.into_parameters_and_grid();
        grid
    }

    fn into_parameters_and_grid<G: Grid3<fgr>>(
        self,
    ) -> (
        NetCDFSnapshotReaderConfig,
        File,
        NetCDFSnapshotParameters,
        G,
        Endianness,
    ) {
        let Self {
            reader_config,
            file,
            parameters,
            is_periodic,
            grid_data,
        } = self;

        let NetCDFGridData {
            detected_grid_type: _,
            center_coords,
            lower_edge_coords,
            up_derivatives,
            down_derivatives,
            endianness,
        } = grid_data;

        let grid = G::from_coords(
            center_coords,
            lower_edge_coords,
            is_periodic,
            up_derivatives,
            down_derivatives,
        );
        (reader_config, file, parameters, grid, endianness)
    }
}

impl NetCDFSnapshotReaderConfig {
    /// Creates a new set of snapshot reader configuration parameters.
    pub fn new<P: AsRef<Path>>(file_path: P, verbose: Verbose) -> Self {
        NetCDFSnapshotReaderConfig {
            file_path: file_path.as_ref().to_path_buf(),
            verbose,
        }
    }

    pub fn verbose(&self) -> Verbose {
        self.verbose
    }

    pub fn file_path(&self) -> &Path {
        self.file_path.as_path()
    }
}

/// Writes all data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_identical_snapshot<Pa, G, P>(
    provider: &mut P,
    output_file_path: Pa,
    strip_metadata: bool,
    overwrite_mode: OverwriteMode,
    protected_file_types: &[&str],
    verbose: Verbose,
) -> io::Result<()>
where
    Pa: AsRef<Path>,
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let mut quantity_names = provider.primary_variable_names().to_vec();
    quantity_names.append(&mut provider.auxiliary_variable_names().to_vec());
    write_modified_snapshot(
        provider,
        &quantity_names,
        HashMap::new(),
        output_file_path,
        strip_metadata,
        overwrite_mode,
        protected_file_types,
        verbose,
    )
}

/// Writes modified data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_modified_snapshot<Pa, G, P>(
    provider: &mut P,
    quantity_names: &[String],
    mut modified_parameters: HashMap<&str, ParameterValue>,
    output_file_path: Pa,
    strip_metadata: bool,
    overwrite_mode: OverwriteMode,
    protected_file_types: &[&str],
    verbose: Verbose,
) -> io::Result<()>
where
    Pa: AsRef<Path>,
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let output_file_path = output_file_path.as_ref();

    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(output_file_path);
    let snap_num = snap_num.unwrap_or(FALLBACK_SNAP_NUM);

    modified_parameters.insert(
        "snapname",
        ParameterValue::Str(format!("\"{}\"", snap_name)),
    );
    modified_parameters.insert("isnap", ParameterValue::Str(format!("{}", snap_num)));
    modified_parameters.insert(
        "meshfile",
        ParameterValue::Str(format!("\"{}.mesh\"", snap_name)),
    );

    let (_, included_auxiliary_variable_names) = provider.classify_variable_names(quantity_names);

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
    modified_parameters.insert(
        "dx",
        ParameterValue::Float(average_grid_cell_extents[X] as fpa),
    );
    modified_parameters.insert(
        "dy",
        ParameterValue::Float(average_grid_cell_extents[Y] as fpa),
    );
    modified_parameters.insert(
        "dz",
        ParameterValue::Float(average_grid_cell_extents[Z] as fpa),
    );

    let atomic_output_path = AtomicOutputPath::new(output_file_path)?;
    if !atomic_output_path.check_if_write_allowed(overwrite_mode, protected_file_types) {
        return Ok(());
    }

    let output_file_name = atomic_output_path
        .target_path()
        .file_name()
        .unwrap()
        .to_string_lossy();

    let mut file = create_file(atomic_output_path.temporary_path())?;
    let mut root_group = file.root_mut().unwrap();

    let mut new_parameters = provider.parameters().clone();
    new_parameters.modify_values(modified_parameters);

    if !strip_metadata {
        if verbose.is_yes() {
            println!("Writing parameters to {}", output_file_name);
        }
        param::write_snapshot_parameters(&mut root_group, &new_parameters)?;
    }

    if verbose.is_yes() {
        println!("Writing grid to {}", output_file_name);
    }
    mesh::write_grid(&mut root_group, grid, strip_metadata)?;

    for name in quantity_names {
        let field = provider.provide_scalar_field(name)?;
        if verbose.is_yes() {
            println!("Writing {} to {}", name, output_file_name);
        }
        write_3d_scalar_field(&mut root_group, &field)?;
    }

    drop(file);
    atomic_output_path.perform_replace()?;

    Ok(())
}

/// Opens an existing NetCDF file at the given path.
pub fn open_file<P: AsRef<Path>>(path: P) -> io::Result<File> {
    io_result!(nc::open(path))
}

/// Returns a list of all non-coordinate variables in the given NetCDF group.
pub fn read_all_non_coord_variable_names(group: &Group) -> Vec<String> {
    group
        .variables()
        .filter_map(|var| {
            let name = var.name();
            if COORDINATE_NAMES.contains(&name.as_str()) {
                None
            } else {
                Some(name)
            }
        })
        .collect()
}

/// Reads the given 1D variable from the given NetCDF group.
fn read_snapshot_1d_variable<F: Numeric + BFloat + Default>(
    group: &Group,
    name: &str,
) -> io::Result<(Vec<F>, Endianness)> {
    let var = group.variable(name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Variable {} not found in NetCDF file", name),
        )
    })?;
    let endianness = match io_result!(var.endian_value())? {
        nc::Endianness::Native => Endianness::Native,
        nc::Endianness::Little => Endianness::Little,
        nc::Endianness::Big => Endianness::Big,
    };
    let dimensions = var.dimensions();
    if dimensions.len() != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Variable {} in NetCDF file is not 1D", name),
        ));
    }
    let number_of_values = dimensions[0].len();
    let mut values = vec![F::default(); number_of_values];
    io_result!(var.values_to(&mut values, None, None))?;
    Ok((values, endianness))
}

/// Reads the given 3D variable from the given NetCDF group.
fn read_snapshot_3d_variable<F: Numeric + BFloat + Default, G: Grid3<fgr>>(
    group: &Group,
    grid: &G,
    name: &str,
) -> io::Result<(Array3<F>, In3D<CoordLocation>, Endianness)> {
    let var = group.variable(name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Variable {} not found in NetCDF file", name),
        )
    })?;
    let endianness = match io_result!(var.endian_value())? {
        nc::Endianness::Native => Endianness::Native,
        nc::Endianness::Little => Endianness::Little,
        nc::Endianness::Big => Endianness::Big,
    };

    let dimensions = var.dimensions();
    if dimensions.len() != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Variable {} in NetCDF file is not 3D", name),
        ));
    }
    let locations = In3D::new(
        match dimensions[2].name().as_str() {
            "xm" => CoordLocation::Center,
            "xmdn" => CoordLocation::LowerEdge,
            invalid => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid x-coordinate {} for variable {} in NetCDF file",
                        invalid, name
                    ),
                ))
            }
        },
        match dimensions[1].name().as_str() {
            "ym" => CoordLocation::Center,
            "ymdn" => CoordLocation::LowerEdge,
            invalid => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid y-coordinate {} for variable {} in NetCDF file",
                        invalid, name
                    ),
                ))
            }
        },
        match dimensions[0].name().as_str() {
            "zm" => CoordLocation::Center,
            "zmdn" => CoordLocation::LowerEdge,
            invalid => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid z-coordinate {} for variable {} in NetCDF file",
                        invalid, name
                    ),
                ))
            }
        },
    );

    let shape = (
        dimensions[2].len(),
        dimensions[1].len(),
        dimensions[0].len(),
    );
    let grid_shape = grid.shape();
    if shape.0 != grid_shape[X] || shape.1 != grid_shape[Y] || shape.2 != grid_shape[Z] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Variable {} in NetCDF file does not have the same shape as the grid",
                name
            ),
        ));
    }

    let number_of_values = shape.0 * shape.1 * shape.2;
    let mut buffer = vec![F::default(); number_of_values];
    io_result!(var.values_to(&mut buffer, None, None))?;
    let values = Array::from_shape_vec(shape.f(), buffer).unwrap();
    Ok((values, locations, endianness))
}

/// Creates a new NetCDF file at the given path.
pub fn create_file<P: AsRef<Path>>(path: P) -> io::Result<MutableFile> {
    utils::create_directory_if_missing(&path)?;
    let file = io_result!(nc::create(path))?;
    Ok(file)
}

/// Writes all the primary variables of the given snapshot to the given NetCDF group.
pub fn write_snapshot_primary_variables<G, P>(
    group: &mut GroupMut,
    provider: &mut P,
) -> io::Result<()>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    for name in &provider.primary_variable_names().to_vec() {
        let field = provider.provide_scalar_field(name)?;
        write_3d_scalar_field(group, &field)?;
    }
    Ok(())
}

/// Writes all the auxiliary variables of the given snapshot to the given NetCDF group.
pub fn write_snapshot_auxiliary_variables<G, P>(
    group: &mut GroupMut,
    provider: &mut P,
) -> io::Result<()>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    for name in &provider.auxiliary_variable_names().to_vec() {
        let field = provider.provide_scalar_field(name)?;
        write_3d_scalar_field(group, &field)?;
    }
    Ok(())
}

/// Writes a representation of the given 3D scalar field to the given NetCDF group.
pub fn write_3d_scalar_field<G: Grid3<fgr>>(
    group: &mut GroupMut,
    field: &ScalarField3<fdt, G>,
) -> io::Result<()> {
    let field_name = field.name();
    let locations = field.locations();
    let dimension_names = [
        match locations[Z] {
            CoordLocation::Center => "zm",
            CoordLocation::LowerEdge => "zmdn",
        },
        match locations[Y] {
            CoordLocation::Center => "ym",
            CoordLocation::LowerEdge => "ymdn",
        },
        match locations[X] {
            CoordLocation::Center => "xm",
            CoordLocation::LowerEdge => "xmdn",
        },
    ];
    let values = field
        .values()
        .as_slice_memory_order()
        .expect("Values array not contiguous");
    io_result!(
        io_result!(group.add_variable::<fdt>(field_name, &dimension_names))?
            .put_values(values, None, None)
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::hor_regular::HorRegularGrid3;
    use crate::io::{
        snapshot::native::{NativeSnapshotReader3, NativeSnapshotReaderConfig},
        Endianness, Verbose,
    };
    #[test]
    fn snapshot_writing_works() {
        let mut reader =
            NativeSnapshotReader3::<HorRegularGrid3<_>>::new(NativeSnapshotReaderConfig::new(
                "data/test_snapshot.idl",
                Endianness::Little,
                Verbose::No,
            ))
            .unwrap();
        write_identical_snapshot(
            &mut reader,
            "data/test_snapshot.nc",
            false,
            OverwriteMode::Always,
            &[],
            Verbose::No,
        )
        .unwrap();
    }
}
