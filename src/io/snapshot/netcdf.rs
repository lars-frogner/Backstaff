//! Reading and writing of Bifrost simulation data in NetCDF format.

mod mesh;
mod param;

use super::{
    super::{
        utils::{self, AtomicOutputPath},
        Endianness, Verbose,
    },
    fdt, ParameterValue, SnapshotFormat, SnapshotParameters, SnapshotReader3, COORDINATE_NAMES,
    FALLBACK_SNAP_NUM, PRIMARY_VARIABLE_NAMES_MHD,
};
use crate::{
    field::ScalarField3,
    geometry::{
        Dim3::{X, Y, Z},
        In3D,
    },
    grid::{CoordLocation, Grid3},
    io_result,
    num::BFloat,
};
use ndarray::prelude::*;
use netcdf_rs::{self as nc, File, Group, GroupMut, MutableFile, Numeric};
use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub use mesh::read_grid_data;
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
}

impl<G: Grid3<fdt>> NetCDFSnapshotReader3<G> {
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

        Self {
            config,
            file,
            grid: Arc::new(grid),
            parameters,
            endianness,
            primary_variable_names,
            auxiliary_variable_names,
        }
    }

    fn read_variable_names(group: &Group) -> (Vec<String>, Vec<String>) {
        read_all_non_coord_variable_names(group)
            .into_iter()
            .partition(|name| PRIMARY_VARIABLE_NAMES_MHD.contains(&name.as_str()))
    }
}

impl<G: Grid3<fdt>> SnapshotReader3<G> for NetCDFSnapshotReader3<G> {
    type Parameters = NetCDFSnapshotParameters;

    const FORMAT: SnapshotFormat = SnapshotFormat::NetCDF;

    fn path(&self) -> &Path {
        self.config.file_path()
    }

    fn verbose(&self) -> Verbose {
        self.config.verbose
    }

    fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }

    fn endianness(&self) -> Endianness {
        self.endianness
    }

    fn primary_variable_names(&self) -> Vec<&str> {
        self.primary_variable_names
            .iter()
            .map(|s| s.as_str())
            .collect()
    }

    fn auxiliary_variable_names(&self) -> Vec<&str> {
        self.auxiliary_variable_names
            .iter()
            .map(|s| s.as_str())
            .collect()
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        super::extract_name_and_num_from_snapshot_path(self.config.file_path())
    }

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

    fn read_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
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
pub fn write_identical_snapshot<P, G, R>(
    reader: &R,
    output_file_path: P,
    strip_metadata: bool,
    automatic_overwrite: bool,
    protected_file_types: &[&str],
    verbose: Verbose,
) -> io::Result<()>
where
    P: AsRef<Path>,
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let mut quantity_names = reader.primary_variable_names();
    quantity_names.append(&mut reader.auxiliary_variable_names());
    write_modified_snapshot(
        reader,
        None,
        &quantity_names,
        HashMap::new(),
        |name| reader.read_scalar_field(name),
        output_file_path,
        strip_metadata,
        automatic_overwrite,
        protected_file_types,
        verbose,
    )
}

/// Writes modified data associated with the given snapshot to a NetCDF file at the given path.
pub fn write_modified_snapshot<P, GIN, RIN, GOUT, FP>(
    reader: &RIN,
    new_grid: Option<Arc<GOUT>>,
    quantity_names: &[&str],
    mut modified_parameters: HashMap<&str, ParameterValue>,
    field_producer: FP,
    output_file_path: P,
    strip_metadata: bool,
    automatic_overwrite: bool,
    protected_file_types: &[&str],
    verbose: Verbose,
) -> io::Result<()>
where
    P: AsRef<Path>,
    GIN: Grid3<fdt>,
    GOUT: Grid3<fdt>,
    RIN: SnapshotReader3<GIN>,
    FP: Fn(&str) -> io::Result<ScalarField3<fdt, GOUT>>,
{
    let output_file_path = output_file_path.as_ref().with_extension("nc");

    let (snap_name, snap_num) = super::extract_name_and_num_from_snapshot_path(&output_file_path);
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

    let (_, included_auxiliary_variable_names) = reader.classify_variable_names(quantity_names);

    modified_parameters.insert(
        "aux",
        ParameterValue::Str(format!(
            "\"{}\"",
            included_auxiliary_variable_names.join(" ")
        )),
    );

    let atomic_output_path = AtomicOutputPath::new(output_file_path)?;
    atomic_output_path.ensure_write_allowed(automatic_overwrite, protected_file_types);

    let output_file_name = atomic_output_path
        .target_path()
        .file_name()
        .unwrap()
        .to_string_lossy();

    let mut file = create_file(atomic_output_path.temporary_path())?;
    let mut root_group = file.root_mut().unwrap();

    macro_rules! perform_writing {
        ($grid:expr) => {{
            let mut new_parameters = reader.parameters().clone();
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
            mesh::write_grid(&mut root_group, $grid, strip_metadata)?;

            for &name in quantity_names {
                let field = field_producer(name)?;
                if verbose.is_yes() {
                    println!("Writing {} to {}", name, output_file_name);
                }
                write_3d_scalar_field(&mut root_group, &field)?;
            }
        }};
    }

    if let Some(new_grid) = new_grid {
        let shape = new_grid.shape();
        let average_grid_cell_extents = new_grid.average_grid_cell_extents();
        modified_parameters.insert("mx", ParameterValue::Int(shape[X] as i64));
        modified_parameters.insert("my", ParameterValue::Int(shape[Y] as i64));
        modified_parameters.insert("mz", ParameterValue::Int(shape[Z] as i64));
        modified_parameters.insert("dx", ParameterValue::Float(average_grid_cell_extents[X]));
        modified_parameters.insert("dy", ParameterValue::Float(average_grid_cell_extents[Y]));
        modified_parameters.insert("dz", ParameterValue::Float(average_grid_cell_extents[Z]));
        perform_writing!(new_grid.as_ref());
    } else {
        perform_writing!(reader.grid());
    };

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
fn read_snapshot_3d_variable<F: Numeric + BFloat + Default, G: Grid3<F>>(
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
        match dimensions[0].name().as_str() {
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
        match dimensions[2].name().as_str() {
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
        dimensions[0].len(),
        dimensions[1].len(),
        dimensions[2].len(),
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
pub fn write_snapshot_primary_variables<G, R>(group: &mut GroupMut, reader: &R) -> io::Result<()>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    for name in reader.primary_variable_names() {
        let field = reader.read_scalar_field(name)?;
        write_3d_scalar_field(group, &field)?;
    }
    Ok(())
}

/// Writes all the auxiliary variables of the given snapshot to the given NetCDF group.
pub fn write_snapshot_auxiliary_variables<G, R>(group: &mut GroupMut, reader: &R) -> io::Result<()>
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    for name in reader.auxiliary_variable_names() {
        let field = reader.read_scalar_field(name)?;
        write_3d_scalar_field(group, &field)?;
    }
    Ok(())
}

/// Writes a representation of the given 3D scalar field to the given NetCDF group.
pub fn write_3d_scalar_field<G: Grid3<fdt>>(
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
        let reader =
            NativeSnapshotReader3::<HorRegularGrid3<_>>::new(NativeSnapshotReaderConfig::new(
                "data/cb24ni_ebeam_offline/cb24ni_ebeam_offline_462.idl",
                Endianness::Little,
                Verbose::No,
            ))
            .unwrap();
        write_identical_snapshot(&reader, "test.nc", false, true, &[], Verbose::No).unwrap();
    }
}
