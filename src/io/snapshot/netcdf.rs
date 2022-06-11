//! Reading and writing of Bifrost simulation data in NetCDF format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self},
        Endianness, Verbosity,
    },
    fdt, SnapshotProvider3, SnapshotReader3, COORDINATE_NAMES, FALLBACK_SNAP_NUM,
};
use crate::{
    field::{ScalarField3, ScalarFieldProvider3},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{fgr, CoordLocation, Grid3, GridType},
    io::utils::IOContext,
    io_result,
    num::BFloat,
    with_io_err_msg,
};
use ndarray::prelude::*;
use netcdf_rs::{self as nc, File, Group, GroupMut, MutableFile, Numeric};
use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub use mesh::{read_grid_data, NetCDFGridData};
pub use param::{read_netcdf_snapshot_parameters, NetCDFSnapshotParameters};

/// Configuration parameters for NetCDF snapshot reader.
#[derive(Clone, Debug)]
pub struct NetCDFSnapshotReaderConfig {
    /// Path to the file.
    file_path: PathBuf,
    /// Whether and how to pass non-essential information to user while reading fields.
    verbosity: Verbosity,
}

/// Reader for NetCDF files associated with Bifrost 3D simulation snapshots.
#[derive(Debug)]
pub struct NetCDFSnapshotReader3<G> {
    config: NetCDFSnapshotReaderConfig,
    file: File,
    grid: Arc<G>,
    parameters: NetCDFSnapshotParameters,
    endianness: Endianness,
    all_variable_names: Vec<String>,
}

impl<G: Grid3<fgr>> NetCDFSnapshotReader3<G> {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: NetCDFSnapshotReaderConfig) -> io::Result<Self> {
        let file = open_file(&config.file_path)?;

        let parameters = read_netcdf_snapshot_parameters(&file, config.verbosity())?;

        let is_periodic = parameters.determine_grid_periodicity()?;
        let (grid, endianness) = mesh::read_grid::<G>(&file, is_periodic, config.verbosity())?;

        Ok(Self::new_from_parameters_and_grid(
            config, file, parameters, grid, endianness,
        ))
    }

    /// Returns the path to the file representing the snapshot.
    pub fn path(&self) -> &Path {
        self.config.file_path()
    }

    pub fn verbosity(&self) -> &Verbosity {
        self.config.verbosity()
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

        let all_variable_names = read_all_non_coord_variable_names(&root_group);

        Self {
            config,
            file,
            grid: Arc::new(grid),
            parameters,
            endianness,
            all_variable_names,
        }
    }

    #[allow(dead_code)]
    fn reread(&mut self) -> io::Result<()> {
        let Self {
            file,
            grid,
            parameters,
            endianness,
            ..
        } = Self::new(self.config.clone())?;

        self.file = file;
        self.grid = grid;
        self.parameters = parameters;
        self.endianness = endianness;

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

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
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

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn has_variable(&self, variable_name: &str) -> bool {
        self.all_variable_names()
            .contains(&variable_name.to_string())
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u64>) {
        super::extract_name_and_num_from_snapshot_path(self.config.file_path())
    }
}

impl<G: Grid3<fgr>> SnapshotReader3<G> for NetCDFSnapshotReader3<G> {
    fn read_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
        if self.config.verbosity.print_messages() {
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
            read_netcdf_snapshot_parameters(&file, reader_config.verbosity()),
            "Could not read snapshot parameters from NetCDF file: {}"
        )?;
        let grid_data = with_io_err_msg!(
            read_grid_data(&file, reader_config.verbosity()),
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
    pub fn new(file_path: PathBuf, verbosity: Verbosity) -> Self {
        NetCDFSnapshotReaderConfig {
            file_path,
            verbosity,
        }
    }

    pub fn verbosity(&self) -> &Verbosity {
        &self.verbosity
    }

    pub fn file_path(&self) -> &Path {
        self.file_path.as_path()
    }
}

/// Writes data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_new_snapshot<G, P>(
    provider: &mut P,
    output_file_path: &Path,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> io::Result<()>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let quantity_names = provider.all_variable_names().to_vec();
    write_modified_snapshot(
        provider,
        &quantity_names,
        output_file_path,
        false,
        io_context,
        verbosity,
    )
}

/// Writes modified data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_modified_snapshot<G, P>(
    provider: &mut P,
    quantity_names: &[String],
    output_file_path: &Path,
    strip_metadata: bool,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> io::Result<()>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(output_file_path);
    let snap_num = snap_num.unwrap_or(FALLBACK_SNAP_NUM) as i64;

    let (_, included_auxiliary_variable_names, is_mhd) =
        provider.classify_variable_names(quantity_names);

    let atomic_output_file =
        io_context.create_atomic_output_file(output_file_path.to_path_buf())?;
    if !atomic_output_file.check_if_write_allowed(io_context, verbosity) {
        return Ok(());
    }

    let output_file_name = atomic_output_file
        .target_path()
        .file_name()
        .unwrap()
        .to_string_lossy();

    let mut file = create_file(atomic_output_file.temporary_path())?;
    let mut root_group = file.root_mut().unwrap();

    let new_parameters = provider.create_updated_parameters(
        snap_name.as_str(),
        snap_num,
        &included_auxiliary_variable_names,
        is_mhd,
    );

    if !strip_metadata {
        if verbosity.print_messages() {
            println!("Writing parameters to {}", output_file_name);
        }
        param::write_snapshot_parameters(&mut root_group, &new_parameters)?;
    }

    if verbosity.print_messages() {
        println!("Writing grid to {}", output_file_name);
    }
    mesh::write_grid(&mut root_group, provider.grid(), strip_metadata)?;

    for name in quantity_names {
        let field = provider.provide_scalar_field(name)?;
        if verbosity.print_messages() {
            println!("Writing {} to {}", name, output_file_name);
        }
        write_3d_scalar_field(&mut root_group, &field)?;
    }

    drop(file);
    io_context.close_atomic_output_file(atomic_output_file)?;

    Ok(())
}

/// Opens an existing NetCDF file at the given path.
pub fn open_file(path: &Path) -> io::Result<File> {
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
pub fn create_file(path: &Path) -> io::Result<MutableFile> {
    utils::create_directory_if_missing(&path)?;
    let file = io_result!(nc::create(path))?;
    Ok(file)
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
