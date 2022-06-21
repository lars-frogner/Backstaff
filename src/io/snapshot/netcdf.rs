//! Reading and writing of Bifrost simulation data in NetCDF format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self},
        Endianness, Verbosity,
    },
    fdt, MapOfSnapshotParameters, SnapshotMetadata, SnapshotParameters, COORDINATE_NAMES,
    FALLBACK_SNAP_NUM,
};
use crate::{
    field::{FieldGrid3, ScalarField3, ScalarFieldProvider3},
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{CoordLocation, Grid3},
    io::utils::IOContext,
    io_result,
    num::BFloat,
};
use ndarray::prelude::*;
use netcdf_rs::{self as nc, File, Group, GroupMut, MutableFile, Numeric};
use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub use mesh::create_grid_from_netcdf_file;
pub use param::{read_netcdf_snapshot_parameters, NetCDFSnapshotParameters};

/// Configuration parameters for NetCDF snapshot reader.
#[derive(Clone, Debug)]
pub struct NetCDFSnapshotReaderConfig {
    /// Path to the file.
    file_path: PathBuf,
    /// Whether and how to pass non-essential information to user while reading fields.
    verbosity: Verbosity,
}

impl NetCDFSnapshotReaderConfig {
    /// Creates a new set of snapshot reader configuration parameters.
    pub fn new(file_path: PathBuf, verbosity: Verbosity) -> Self {
        NetCDFSnapshotReaderConfig {
            file_path,
            verbosity,
        }
    }
}

/// Information associated with a Bifrost 3D simulation snapshot
/// in NetCDF format.
#[derive(Clone, Debug)]
pub struct NetCDFSnapshotMetadata {
    snap_name: String,
    snap_num: Option<u64>,
    endianness: Endianness,
    parameters: Box<MapOfSnapshotParameters>,
}

impl NetCDFSnapshotMetadata {
    fn new(file_path: &Path, parameters: MapOfSnapshotParameters, endianness: Endianness) -> Self {
        let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(file_path);
        Self {
            snap_name,
            snap_num,
            endianness,
            parameters: Box::new(parameters),
        }
    }
}

impl SnapshotMetadata for NetCDFSnapshotMetadata {
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

/// Reader for NetCDF files associated with Bifrost 3D simulation snapshots.
#[derive(Debug)]
pub struct NetCDFSnapshotReader3 {
    file: File,
    file_path: PathBuf,
    endianness: Endianness,
    grid: Arc<FieldGrid3>,
    all_variable_names: Vec<String>,
    verbosity: Verbosity,
}

impl NetCDFSnapshotReader3 {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: NetCDFSnapshotReaderConfig) -> io::Result<(Self, NetCDFSnapshotMetadata)> {
        let NetCDFSnapshotReaderConfig {
            file_path,
            verbosity,
        } = config;

        let file = open_netcdf_file(&file_path)?;

        let parameters = read_netcdf_snapshot_parameters(&file, &verbosity)?;

        let root_group = file.root().unwrap();
        let all_variable_names = read_all_non_coord_variable_names(&root_group);

        let is_periodic = parameters.determine_grid_periodicity()?;
        let (grid, endianness) =
            mesh::create_grid_from_open_netcdf_file(&file, is_periodic, &verbosity)?;

        let metadata = NetCDFSnapshotMetadata::new(file_path.as_path(), parameters, endianness);

        Ok((
            Self {
                file,
                file_path,
                endianness,
                grid: Arc::new(grid),
                all_variable_names,
                verbosity,
            },
            metadata,
        ))
    }

    pub fn verbosity(&self) -> &Verbosity {
        &self.verbosity
    }
}

impl ScalarFieldProvider3<fdt> for NetCDFSnapshotReader3 {
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
        if self.verbosity().print_messages() {
            println!(
                "Reading {} from {}",
                variable_name,
                self.file_path.file_name().unwrap().to_string_lossy()
            );
        }
        let (values, locations, endianness) = read_snapshot_3d_variable::<fdt>(
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
            self.arc_with_grid(),
            locations,
            values,
        ))
    }
}

/// Writes data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_new_snapshot(
    input_metadata: &dyn SnapshotMetadata,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    output_file_path: &Path,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> io::Result<()> {
    let quantity_names = provider.all_variable_names().to_vec();
    write_modified_snapshot(
        input_metadata,
        provider,
        &quantity_names,
        output_file_path,
        false,
        io_context,
        verbosity,
    )
}

/// Writes modified data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_modified_snapshot(
    input_metadata: &dyn SnapshotMetadata,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    quantity_names: &[String],
    output_file_path: &Path,
    strip_metadata: bool,
    io_context: &IOContext,
    verbosity: &Verbosity,
) -> io::Result<()> {
    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(output_file_path);
    let snap_num = snap_num.unwrap_or(FALLBACK_SNAP_NUM) as i64;

    let (_, included_auxiliary_variable_names, is_mhd) =
        input_metadata.classify_variable_names(quantity_names);

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

    let new_parameters = input_metadata.create_updated_parameters(
        provider.grid(),
        snap_name.as_str(),
        snap_num,
        &included_auxiliary_variable_names,
        is_mhd,
    );

    if !strip_metadata {
        if verbosity.print_messages() {
            println!("Writing parameters to {}", output_file_name);
        }
        param::write_snapshot_parameters(&mut root_group, new_parameters.borrow())?;
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
pub fn open_netcdf_file(path: &Path) -> io::Result<File> {
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
fn read_snapshot_3d_variable<F: Numeric + BFloat + Default>(
    group: &Group,
    grid: &FieldGrid3,
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
pub fn write_3d_scalar_field(group: &mut GroupMut, field: &ScalarField3<fdt>) -> io::Result<()> {
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
