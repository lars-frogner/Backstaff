//! Reading and writing of Bifrost simulation data.

use super::{mesh, utils};
use super::{Endianness, Verbose};
use crate::field::{ScalarField3, VectorField3};
use crate::geometry::{Dim3, In3D};
use crate::grid::{CoordLocation, Grid3};
use ndarray::prelude::*;
use num;
use regex;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::{fs, io, mem, path, str, string};
use CoordLocation::{Center, LowerEdge};
use Dim3::{X, Y, Z};

/// Floating-point precision assumed for Bifrost data.
#[allow(non_camel_case_types)]
pub type fdt = f32;

/// Configuration parameters for snapshot reader.
#[derive(Clone, Debug)]
pub struct SnapshotReaderConfig {
    /// Path to the parameter (.idl) file.
    params_path: path::PathBuf,
    /// Order of bytes in the binary data files.
    endianness: Endianness,
    /// Whether to print status messages while reading fields.
    verbose: Verbose,
}

/// Reader for the output files assoicated with a single Bifrost 3D simulation snapshot.
#[derive(Clone, Debug)]
pub struct SnapshotReader3<G: Grid3<fdt>> {
    config: SnapshotReaderConfig,
    snap_path: path::PathBuf,
    aux_path: path::PathBuf,
    params: Params,
    grid: Arc<G>,
    primary_variable_names: Vec<&'static str>,
    auxiliary_variable_names: Vec<String>,
    variable_descriptors: HashMap<String, VariableDescriptor>,
}

/// Wrapper for `SnapshotReader3` that reads snapshot variables only on first request and
/// then caches the results.
#[derive(Clone, Debug)]
pub struct SnapshotCacher3<G: Grid3<fdt>> {
    reader: SnapshotReader3<G>,
    scalar_fields: HashMap<String, ScalarField3<fdt, G>>,
    vector_fields: HashMap<String, VectorField3<fdt, G>>,
}

impl<G: Grid3<fdt>> SnapshotReader3<G> {
    /// Creates a reader for a 3D Bifrost snapshot.
    pub fn new(config: SnapshotReaderConfig) -> io::Result<Self> {
        let params = Params::new(&config.params_path)?;
        let snap_num: u32 = params.get_numerical_param("isnap")?;
        let mesh_path = config
            .params_path
            .with_file_name(params.get_str_param("meshfile")?);
        let snap_path = config.params_path.with_file_name(format!(
            "{}_{}.snap",
            params.get_str_param("snapname")?,
            snap_num
        ));
        let aux_path = snap_path.with_extension("aux");

        let is_periodic = In3D::new(
            params.get_numerical_param::<u8>("periodic_x")? == 1,
            params.get_numerical_param::<u8>("periodic_y")? == 1,
            params.get_numerical_param::<u8>("periodic_z")? == 1,
        );
        let grid = Arc::new(mesh::create_grid_from_mesh_file(&mesh_path, is_periodic)?);

        let is_mhd = params.get_numerical_param::<u8>("do_mhd")? > 0;

        let primary_variable_names = if is_mhd {
            vec!["r", "px", "py", "pz", "e", "bx", "by", "bz"]
        } else {
            vec!["r", "px", "py", "pz", "e"]
        };
        let auxiliary_variable_names = Self::get_auxiliary_variable_names(&params)?;

        let mut variable_descriptors = HashMap::new();
        Self::insert_primary_variable_descriptors(is_mhd, &mut variable_descriptors);
        Self::insert_auxiliary_variable_descriptors(
            &auxiliary_variable_names,
            &mut variable_descriptors,
        )?;

        Ok(SnapshotReader3 {
            config,
            snap_path,
            aux_path,
            params,
            grid,
            primary_variable_names,
            auxiliary_variable_names,
            variable_descriptors,
        })
    }

    /// Wraps the reader in a snapshot cacher structure.
    pub fn into_cacher(self) -> SnapshotCacher3<G> {
        SnapshotCacher3::new(self)
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns the assumed endianness of the snapshot.
    pub fn endianness(&self) -> Endianness {
        self.config.endianness
    }

    /// Returns the names of the primary variables of the snapshot.
    pub fn primary_variable_names(&self) -> &[&str] {
        &self.primary_variable_names
    }

    /// Returns the names of the auxiliary variables of the snapshot.
    pub fn auxiliary_variable_names(&self) -> &[String] {
        &self.auxiliary_variable_names
    }

    /// Reads the specified primary or auxiliary 3D variable from the output files.
    pub fn read_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
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
        let length = shape[X] * shape[Y] * shape[Z];
        let offset = length * variable_descriptor.index;
        let buffer = super::utils::read_f32_from_binary_file(
            file_path,
            length,
            offset,
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

    /// Reads the component variables of the specified 3D vector quantity from the output files.
    pub fn read_vector_field(&self, variable_name: &str) -> io::Result<VectorField3<fdt, G>> {
        Ok(VectorField3::new(
            variable_name.to_string(),
            Arc::clone(&self.grid),
            In3D::new(
                self.read_scalar_field(&format!("{}x", variable_name))?,
                self.read_scalar_field(&format!("{}y", variable_name))?,
                self.read_scalar_field(&format!("{}z", variable_name))?,
            ),
        ))
    }

    /// Provides the string value of a parameter from the parameter file.
    pub fn get_str_param<'a, 'b>(&'a self, name: &'b str) -> io::Result<&'a str> {
        self.params.get_str_param(name)
    }

    /// Provides the numerical value of a parameter from the parameter file.
    pub fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
    where
        T: num::Num + str::FromStr,
        T::Err: string::ToString,
    {
        self.params.get_numerical_param(name)
    }

    /// Tries to read the given parameter from the parameter file.
    /// If successful, the value is converted with the given closure and
    /// returned, otherwise a warning is printed and the given default is returned.
    pub fn get_converted_numerical_param_or_fallback_to_default_with_warning<T, U, C>(
        &self,
        display_name: &str,
        name_in_param_file: &str,
        conversion_mapping: &C,
        default_value: U,
    ) -> U
    where
        T: num::Num + str::FromStr,
        T::Err: string::ToString,
        U: std::fmt::Display,
        C: Fn(T) -> U,
    {
        self.get_numerical_param(name_in_param_file)
            .map(conversion_mapping)
            .unwrap_or_else(|_| {
                println!(
                    "Could not find {} in param file, falling back to default for {}: {}",
                    name_in_param_file, display_name, default_value
                );
                default_value
            })
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

    fn get_auxiliary_variable_names(params: &Params) -> io::Result<Vec<String>> {
        Ok(params
            .get_str_param("aux")?
            .split_whitespace()
            .map(|name| name.to_string())
            .collect())
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
                io::ErrorKind::InvalidData,
                format!("Variable `{}` not found", name),
            )),
        }
    }
}

impl SnapshotReaderConfig {
    /// Creates a new set of snapshot reader configuration parameters.
    pub fn new<P: AsRef<path::Path>>(
        params_path: P,
        endianness: Endianness,
        verbose: Verbose,
    ) -> Self {
        SnapshotReaderConfig {
            params_path: params_path.as_ref().to_path_buf(),
            endianness,
            verbose,
        }
    }
}

impl<G: Grid3<fdt>> SnapshotCacher3<G> {
    /// Creates a new snapshot cacher from the given reader.
    pub fn new(reader: SnapshotReader3<G>) -> Self {
        SnapshotCacher3 {
            reader,
            scalar_fields: HashMap::new(),
            vector_fields: HashMap::new(),
        }
    }

    /// Returns a reference to the reader.
    pub fn reader(&self) -> &SnapshotReader3<G> {
        &self.reader
    }

    /// Returns a mutable reference to the reader.
    pub fn reader_mut(&mut self) -> &mut SnapshotReader3<G> {
        &mut self.reader
    }

    /// Returns a `Result` with a reference to the scalar field representing the given variable,
    /// reading it from file and caching it if has not already been cached.
    pub fn obtain_scalar_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<&ScalarField3<fdt, G>> {
        Ok(self
            .scalar_fields
            .entry(variable_name.to_string())
            .or_insert(self.reader.read_scalar_field(variable_name)?))
    }

    /// Returns a `Result` with a reference to the vector field representing the given variable,
    /// reading it from file and caching it if has not already been cached.
    pub fn obtain_vector_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<&VectorField3<fdt, G>> {
        Ok(self
            .vector_fields
            .entry(variable_name.to_string())
            .or_insert(self.reader.read_vector_field(variable_name)?))
    }

    /// Makes sure the scalar field representing the giveb variable is cached.
    pub fn cache_scalar_field(&mut self, variable_name: &str) -> io::Result<()> {
        self.obtain_scalar_field(variable_name).map(|_| ())
    }

    /// Makes sure the scalar field representing the giveb variable is cached.
    pub fn cache_vector_field(&mut self, variable_name: &str) -> io::Result<()> {
        self.obtain_vector_field(variable_name).map(|_| ())
    }

    /// Returns a reference to the scalar field representing the given variable.
    ///
    /// Panics if the field is not cached.
    pub fn cached_scalar_field(&self, variable_name: &str) -> &ScalarField3<fdt, G> {
        self.scalar_fields
            .get(variable_name)
            .expect("Scalar field is not cached.")
    }

    /// Returns a reference to the vector field representing the given variable.
    ///
    /// Panics if the field is not cached.
    pub fn cached_vector_field(&self, variable_name: &str) -> &VectorField3<fdt, G> {
        self.vector_fields
            .get(variable_name)
            .expect("Vector field is not cached.")
    }

    /// Whether the scalar field representing the given variable is cached.
    pub fn scalar_field_is_cached(&self, variable_name: &str) -> bool {
        self.scalar_fields.contains_key(variable_name)
    }

    /// Whether the vector field representing the given variable is cached.
    pub fn vector_field_is_cached(&self, variable_name: &str) -> bool {
        self.vector_fields.contains_key(variable_name)
    }

    /// Removes the scalar field representing the given variable from the cache.
    pub fn drop_scalar_field(&mut self, variable_name: &str) {
        self.scalar_fields.remove(variable_name);
    }

    /// Removes the vector field representing the given variable from the cache.
    pub fn drop_vector_field(&mut self, variable_name: &str) {
        self.vector_fields.remove(variable_name);
    }

    /// Removes all cached scalar and vector fields.
    pub fn drop_all_fields(&mut self) {
        self.scalar_fields.clear();
        self.vector_fields.clear();
    }
}

/// Writes arrays of variable values sequentially into a binary file.
///
/// # Parameters
///
/// - `output_path`: Path where the output file should be written.
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
pub fn write_3d_snapfile<P, N, V>(
    output_path: P,
    variable_names: &[N],
    variable_value_producer: &V,
    endianness: Endianness,
) -> io::Result<()>
where
    P: AsRef<path::Path>,
    N: AsRef<str>,
    V: Fn(&str) -> Array3<fdt>,
{
    let number_of_variables = variable_names.len();
    assert!(
        number_of_variables > 0,
        "Number of variables must larger than zero."
    );

    let variable_values = variable_value_producer(variable_names[0].as_ref());
    let array_length = variable_values.len();
    let float_size = mem::size_of::<fdt>();
    let byte_buffer_size = array_length * float_size;
    let mut byte_buffer = vec![0_u8; byte_buffer_size];

    let mut file = fs::File::create(output_path)?;
    file.set_len(byte_buffer_size as u64)?;

    super::utils::write_f32_into_byte_buffer(
        variable_values
            .as_slice_memory_order()
            .expect("Values array not contiguous."),
        &mut byte_buffer,
        0,
        endianness,
    );
    file.write_all(&byte_buffer)?;

    for name in variable_names.iter().skip(1) {
        let variable_values = variable_value_producer(name.as_ref());
        assert_eq!(
            variable_values.len(),
            array_length,
            "All variable arrays must have the same length."
        );
        super::utils::write_f32_into_byte_buffer(
            variable_values
                .as_slice_memory_order()
                .expect("Values array not contiguous."),
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

#[derive(Clone, Debug)]
struct Params {
    params_map: HashMap<String, String>,
}

impl Params {
    fn new<P: AsRef<path::Path>>(params_path: P) -> io::Result<Self> {
        let params_text = utils::read_text_file(params_path)?;
        let params_map = Self::parse_params_text(&params_text);
        Ok(Params { params_map })
    }

    fn parse_params_text(text: &str) -> HashMap<String, String> {
        let re = regex::Regex::new(r"(?m)^\s*([_\w]+)\s*=\s*(.+?)\s*$").unwrap();
        re.captures_iter(&text)
            .map(|captures| (captures[1].to_string(), captures[2].to_string()))
            .collect()
    }

    fn get_str_param<'a, 'b>(&'a self, name: &'b str) -> io::Result<&'a str> {
        match self.params_map.get(name) {
            Some(value) => Ok(value.trim_matches('"')),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Parameter `{}` not found in parameter file", name),
            )),
        }
    }

    fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
    where
        T: num::Num + str::FromStr,
        T::Err: string::ToString,
    {
        let str_value = self.get_str_param(name)?;
        match str_value.parse::<T>() {
            Ok(value) => Ok(value),
            Err(err) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Failed parsing string `{}` in parameter file: {}",
                    str_value,
                    err.to_string()
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::grid::hor_regular::HorRegularGrid3;

    #[test]
    fn param_parsing_works() {
        #![allow(clippy::float_cmp)]
        let text =
            "int = 12 \n file_str=\"file.ext\"\nfloat =  -1.02E-07\ninvalid = number\n;comment";
        let params = Params {
            params_map: Params::parse_params_text(text),
        };

        let correct_params: HashMap<_, _> = vec![
            ("int".to_string(), "12".to_string()),
            ("file_str".to_string(), "\"file.ext\"".to_string()),
            ("float".to_string(), "-1.02E-07".to_string()),
            ("invalid".to_string(), "number".to_string()),
        ]
        .into_iter()
        .collect();
        assert_eq!(params.params_map, correct_params);

        assert_eq!(params.get_str_param("file_str").unwrap(), "file.ext");
        assert_eq!(params.get_numerical_param::<u32>("int").unwrap(), 12);
        assert_eq!(
            params.get_numerical_param::<f32>("float").unwrap(),
            -1.02e-7
        );
        assert!(params.get_numerical_param::<f32>("invalid").is_err());
    }

    #[test]
    fn reading_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new(SnapshotReaderConfig::new(
            "data/en024031_emer3.0sml_ebeam_631.idl",
            Endianness::Little,
            Verbose::No,
        ))
        .unwrap();
        let _field = reader.read_scalar_field("r").unwrap();
    }
}
