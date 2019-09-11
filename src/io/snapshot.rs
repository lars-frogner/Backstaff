//! Reading of Bifrost simulation data.

use std::{io, path, fs, mem, str, string};
use std::io::{BufRead, Write};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use num;
use regex;
use ndarray::prelude::*;
use crate::geometry::{Dim3, In3D, Coords3};
use crate::grid::{CoordLocation, GridType, Grid3};
use crate::field::{ScalarField3, VectorField3};
use super::Endianness;
use super::utils::read_text_file;
use Dim3::{X, Y, Z};
use CoordLocation::{Center, LowerEdge};

/// Floating-point precision assumed for Bifrost data.
#[allow(non_camel_case_types)]
pub type fdt = f32;

/// Reader for the output files assoicated with a single Bifrost 3D simulation snapshot.
#[derive(Clone, Debug)]
pub struct SnapshotReader3<G: Grid3<fdt>> {
    snap_path: path::PathBuf,
    aux_path: path::PathBuf,
    params: Params,
    endianness: Endianness,
    grid: Arc<G>,
    variable_descriptors: HashMap<String, VariableDescriptor>,
}

/// Wrapper for `SnapshotReader3` that reads snapshot variables only on first request and
/// then caches the results.
#[derive(Clone, Debug)]
pub struct SnapshotCacher3<G: Grid3<fdt>> {
    reader: SnapshotReader3<G>,
    scalar_fields: HashMap<String, ScalarField3<fdt, G>>,
    vector_fields: HashMap<String, VectorField3<fdt, G>>
}

impl<G: Grid3<fdt>> SnapshotReader3<G> {
    /// Creates a reader for a 3D Bifrost snapshot.
    ///
    /// # Parameters
    ///
    /// - `params_path`: Path to the parameter (.idl) file.
    /// - `endianness`: Order of bytes in the binary data files.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a new `SnapshotReader3`.
    /// - `Err`: Contains an error encountered during opening, reading or parsing the relevant files.
    ///
    /// # Type parameters
    ///
    /// - `P`: A type that can be treated as a reference to a `Path`.
    pub fn new<P: AsRef<path::Path>>(params_path: P, endianness: Endianness) -> io::Result<Self> {
        let params = Params::new(&params_path)?;

        let params_path = params_path.as_ref().to_path_buf();
        let snap_num: u32 = params.get_numerical_param("isnap")?;
        let mesh_path = params_path.with_file_name(params.get_str_param("meshfile")?);
        let snap_path = params_path.with_file_name(format!("{}_{}.snap", params.get_str_param("snapname")?, snap_num));
        let aux_path = snap_path.with_extension("aux");

        let grid = Arc::new(Self::read_grid_from_mesh_file(&params, &mesh_path)?);

        let mut variable_descriptors = HashMap::new();
        Self::insert_primary_variable_descriptors(&mut variable_descriptors);
        Self::insert_aux_variable_descriptors(&params, &mut variable_descriptors)?;

        Ok(SnapshotReader3{
            snap_path,
            aux_path,
            params,
            endianness,
            grid,
            variable_descriptors
        })
    }

    /// Wraps the reader in a snapshot cacher structure.
    pub fn into_cacher(self) -> SnapshotCacher3<G> { SnapshotCacher3::new(self) }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G { self.grid.as_ref() }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> { Arc::clone(&self.grid) }

    /// Reads the specified primary or auxiliary 3D variable from the output files.
    ///
    /// # Parameters
    ///
    /// - `variable_name`: Name of the variable to read.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a `ScalarField3<fdt>` holding the field coordinates and values.
    /// - `Err`: Contains an error encountered while locating the variable or reading the data.
    pub fn read_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
        let variable_descriptor = self.get_variable_descriptor(variable_name)?;
        let values = self.read_variable_from_binary_file(variable_descriptor)?;
        Ok(ScalarField3::new(variable_name.to_string(), Arc::clone(&self.grid), variable_descriptor.locations.clone(), values))
    }

    /// Reads the component variables of the specified 3D vector quantity from the output files.
    ///
    /// # Parameters
    ///
    /// - `variable_name`: Root name of the variable to read (without trailing x, y or z).
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a `VectorField3<fdt>` holding the coordinates and values of the vector field components.
    /// - `Err`: Contains an error encountered while locating the variable or reading the data.
    pub fn read_vector_field(&self, variable_name: &str) -> io::Result<VectorField3<fdt, G>> {
        Ok(VectorField3::new(
            variable_name.to_string(),
            Arc::clone(&self.grid),
            In3D::new(
                self.read_scalar_field(&format!("{}x", variable_name))?,
                self.read_scalar_field(&format!("{}y", variable_name))?,
                self.read_scalar_field(&format!("{}z", variable_name))?
            )
        ))
    }

    /// Provides the string value of a parameter from the parameter file.
    ///
    /// # Parameters
    ///
    /// - `name`: Name of the parameter.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a reference to the string value of the parameter.
    /// - `Err`: Contains an error encountered while locating the parameter.
    ///
    /// # Lifetimes
    ///
    /// - `a`: The life time of `self` and the returned reference.
    /// - `b`: The life time of the `name` argument.
    pub fn get_str_param<'a, 'b>(&'a self, name: &'b str) -> io::Result<&'a str> {
        self.params.get_str_param(name)
    }

    /// Provides the numerical value of a parameter from the parameter file.
    ///
    /// # Parameters
    ///
    /// - `name`: Name of the parameter.
    ///
    /// # Returns
    ///
    /// A `Result` which is either:
    ///
    /// - `Ok`: Contains a reference to the numerical value of the parameter.
    /// - `Err`: Contains an error encountered while locating or parsing the parameter.
    ///
    /// # Type parameters
    ///
    /// - `T`: A numerical type that can be parsed from a string.
    pub fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
        where T: num::Num + str::FromStr,
              T::Err: string::ToString
    {
        self.params.get_numerical_param(name)
    }

    fn read_grid_from_mesh_file<P: AsRef<path::Path>>(params: &Params, mesh_path: P) -> io::Result<G> {
        let file = fs::File::open(mesh_path)?;
        let mut lines = io::BufReader::new(file).lines();
        let coord_names = ["x", "y", "z"];
        let mut center_coord_vecs = VecDeque::new();
        let mut lower_coord_vecs = VecDeque::new();
        let mut is_uniform = [true; 3];

        for dim in 0..3 {

            let mut center_coords = Vec::new();
            let mut lower_coords = Vec::new();
            let mut up_derivatives = Vec::new();
            let mut down_derivatives = Vec::new();

            let length = match lines.next() {
                Some(string) => match string {
                    Ok(s) => match s.trim().parse::<usize>() {
                        Ok(length) => length,
                        Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                              format!("Failed parsing string `{}` in mesh file: {}", s, err.to_string())))
                    },
                    Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
                },
                None => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                  format!("Number of {}-coordinates not found in mesh file", coord_names[dim])))
            };

            for coords in [&mut center_coords, &mut lower_coords, &mut up_derivatives, &mut down_derivatives].iter_mut() {
                match lines.next() {
                    Some(string) =>
                        for s in string?.split_whitespace() {
                            match s.parse::<fdt>() {
                                Ok(val) => coords.push(val),
                                Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                                      format!("Failed parsing string `{}` in mesh file: {}", s, err.to_string())))
                            };
                        },
                    None => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                      format!("{}-coordinates not found in mesh file", coord_names[dim])))
                };
            }

            if center_coords.len()    != length ||
               lower_coords.len()     != length ||
               up_derivatives.len()   != length ||
               down_derivatives.len() != length {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                                          format!("Inconsistent number of {}-coordinates in mesh file", coord_names[dim])))
            }

            if length < 4 {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                                          format!("Insufficient number of {}-coordinates in mesh file (must be at least 4)", coord_names[dim])))
            }

            let uniform_up = up_derivatives.iter().all(|&element| fdt::abs(element - up_derivatives[0]) < 1e-3);
            let uniform_down = down_derivatives.iter().all(|&element| fdt::abs(element - down_derivatives[0]) < 1e-3);

            if uniform_up != uniform_down {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                                          format!("Inconsistent uniformity of {}-coordinates in mesh file", coord_names[dim])))
            }

            is_uniform[dim] = uniform_up;

            center_coord_vecs.push_back(center_coords);
            lower_coord_vecs.push_back(lower_coords);
        }

        let detected_grid_type = match is_uniform {
            [true, true, true] => GridType::Regular,
            [true, true, false] => GridType::HorRegular,
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                           "Non-uniform x- or y-coordinates not supported"))
        };

        if detected_grid_type != G::TYPE {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                       "Wrong reader type for the specified mesh file"))
        }

        let center_coords = Coords3::new(
            center_coord_vecs.pop_front().unwrap(),
            center_coord_vecs.pop_front().unwrap(),
            center_coord_vecs.pop_front().unwrap()
        );

        let lower_edge_coords = Coords3::new(
            lower_coord_vecs.pop_front().unwrap(),
            lower_coord_vecs.pop_front().unwrap(),
            lower_coord_vecs.pop_front().unwrap()
        );

        let is_periodic = In3D::new(params.get_numerical_param::<u8>("periodic_x")? == 1,
                                    params.get_numerical_param::<u8>("periodic_y")? == 1,
                                    params.get_numerical_param::<u8>("periodic_z")? == 1);

        Ok(G::from_coords(center_coords, lower_edge_coords, is_periodic))
    }

    fn insert_primary_variable_descriptors(variable_descriptors: &mut HashMap<String, VariableDescriptor>) {
        let is_primary = true;
        variable_descriptors.insert("r" .to_string(), VariableDescriptor{ is_primary, locations: In3D::new(Center,    Center,    Center),    index: 0 });
        variable_descriptors.insert("px".to_string(), VariableDescriptor{ is_primary, locations: In3D::new(LowerEdge, Center,    Center),    index: 1 });
        variable_descriptors.insert("py".to_string(), VariableDescriptor{ is_primary, locations: In3D::new(Center,    LowerEdge, Center),    index: 2 });
        variable_descriptors.insert("pz".to_string(), VariableDescriptor{ is_primary, locations: In3D::new(Center,    Center,    LowerEdge), index: 3 });
        variable_descriptors.insert("e" .to_string(), VariableDescriptor{ is_primary, locations: In3D::new(Center,    Center,    Center),    index: 4 });
        variable_descriptors.insert("bx".to_string(), VariableDescriptor{ is_primary, locations: In3D::new(LowerEdge, Center,    Center),    index: 5 });
        variable_descriptors.insert("by".to_string(), VariableDescriptor{ is_primary, locations: In3D::new(Center,    LowerEdge, Center),    index: 6 });
        variable_descriptors.insert("bz".to_string(), VariableDescriptor{ is_primary, locations: In3D::new(Center,    Center,    LowerEdge), index: 7 });
    }

    fn insert_aux_variable_descriptors(params: &Params, variable_descriptors: &mut HashMap<String, VariableDescriptor>) -> io::Result<()> {
        let is_primary = false;

        for (index, name) in params.get_str_param("aux")?.split_whitespace().enumerate() {
            let ends_with_x = name.ends_with('x');
            let ends_with_y = name.ends_with('y');
            let ends_with_z = name.ends_with('z');

            let locations = if (ends_with_x || ends_with_y || ends_with_z) &&
                               (name.starts_with('e') || name.starts_with('i')) {
                In3D::new(if ends_with_x { Center } else { LowerEdge },
                          if ends_with_y { Center } else { LowerEdge },
                          if ends_with_z { Center } else { LowerEdge })
            } else {
                In3D::new(if ends_with_x { LowerEdge } else { Center },
                          if ends_with_y { LowerEdge } else { Center },
                          if ends_with_z { LowerEdge } else { Center })
            };

            variable_descriptors.insert(name.to_string(), VariableDescriptor{ is_primary, locations, index });
        }

        Ok(())
    }

    fn get_variable_descriptor(&self, name: &str) -> io::Result<&VariableDescriptor> {
        match self.variable_descriptors.get(name) {
            Some(variable) => Ok(variable),
            None => Err(io::Error::new(io::ErrorKind::InvalidData,
                                       format!("Variable `{}` not found", name)))
        }
    }

    fn read_variable_from_binary_file(&self, variable_descriptor: &VariableDescriptor) -> io::Result<Array3<fdt>> {
        let file_path = if variable_descriptor.is_primary { &self.snap_path } else { &self.aux_path };
        let shape = self.grid.shape();
        let length = shape[X]*shape[Y]*shape[Z];
        let offset = length*variable_descriptor.index;
        let buffer = super::utils::read_f32_from_binary_file(file_path, length, offset, self.endianness)?;
        Ok(Array::from_shape_vec((shape[X], shape[Y], shape[Z]).f(), buffer).unwrap())
    }
}

impl<G: Grid3<fdt>> SnapshotCacher3<G> {
    /// Creates a new snapshot cacher from the given reader.
    pub fn new(reader: SnapshotReader3<G>) -> Self {
        SnapshotCacher3{
            reader,
            scalar_fields: HashMap::new(),
            vector_fields: HashMap::new()
        }
    }

    /// Returns a reference to the reader.
    pub fn reader(&self) -> &SnapshotReader3<G> { &self.reader }

    /// Returns a `Result` with a reference to the scalar field representing the given variable,
    /// reading it from file and caching it if has not already been cached.
    pub fn get_scalar_field(&mut self, variable_name: &str) -> io::Result<&ScalarField3<fdt, G>> {
        Ok(self.scalar_fields.entry(variable_name.to_string()).or_insert(self.reader.read_scalar_field(variable_name)?))
    }

    /// Returns a `Result` with a reference to the vector field representing the given variable,
    /// reading it from file and caching it if has not already been cached.
    pub fn get_vector_field(&mut self, variable_name: &str) -> io::Result<&VectorField3<fdt, G>> {
        Ok(self.vector_fields.entry(variable_name.to_string()).or_insert(self.reader.read_vector_field(variable_name)?))
    }

    /// Calls `get_scalar_field` and unwraps the `Result`,
    /// panicking with the associated error message if the result is `Err`.
    pub fn expect_scalar_field(&mut self, variable_name: &str) -> &ScalarField3<fdt, G> {
        self.get_scalar_field(variable_name).unwrap_or_else(|err| panic!("{}", err))
    }

    /// Calls `get_vector_field` and unwraps the `Result`,
    /// panicking with the associated error message if the result is `Err`.
    pub fn expect_vector_field(&mut self, variable_name: &str) -> &VectorField3<fdt, G> {
        self.get_vector_field(variable_name).unwrap_or_else(|err| panic!("{}", err))
    }

    /// Removes the scalar field representing the given variable from the cache.
    pub fn drop_scalar_field(&mut self, variable_name: &str) {
        self.scalar_fields.remove(variable_name);
    }

    /// Removes the vector field representing the given variable from the cache.
    pub fn drop_vector_field(&mut self, variable_name: &str) {
        self.vector_fields.remove(variable_name);
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
/// - `Err`: Contains an error encountered trying to create or write to the file.
///
/// # Type parameters
///
/// - `P`: A type that can be treated as a reference to a `Path`.
/// - `V`: A function type taking a reference to a string slice and returning a reference to a 3D array.
pub fn write_3d_snapfile<P, V>(output_path: P, variable_names: &[&str], variable_value_producer: &V, endianness: Endianness) -> io::Result<()>
where P: AsRef<path::Path>,
      V: Fn(&str) -> Array3<fdt>
{
    let number_of_variables = variable_names.len();
    assert!(number_of_variables == 5 || number_of_variables == 8, "Number of variables must be 5 or 8.");

    let variable_values = variable_value_producer(variable_names[0]);
    let array_length = variable_values.len();
    let float_size = mem::size_of::<fdt>();
    let byte_buffer_size = array_length*float_size;
    let mut byte_buffer = vec![0_u8; byte_buffer_size];

    let mut file = fs::File::create(output_path)?;
    file.set_len(byte_buffer_size as u64)?;

    super::utils::write_f32_into_byte_buffer(variable_values.as_slice_memory_order().expect("Values array not contiguous."), &mut byte_buffer, 0, endianness);
    file.write_all(&byte_buffer)?;

    for name in variable_names.iter().skip(1) {
        let variable_values = variable_value_producer(name);
        assert_eq!(variable_values.len(), array_length, "All variable arrays must have the same length.");
        super::utils::write_f32_into_byte_buffer(variable_values.as_slice_memory_order().expect("Values array not contiguous."), &mut byte_buffer, 0, endianness);
        file.write_all(&byte_buffer)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct VariableDescriptor {
    is_primary: bool,
    locations: In3D<CoordLocation>,
    index: usize
}

#[derive(Clone, Debug)]
struct Params {
    params_map: HashMap<String, String>
}

impl Params {
    fn new<P: AsRef<path::Path>>(params_path: P) -> io::Result<Self> {
        let params_text = read_text_file(params_path)?;
        let params_map = Self::parse_params_text(&params_text);
        Ok(Params{ params_map })
    }

    fn parse_params_text(text: &str) -> HashMap<String, String> {
        let re = regex::Regex::new(r"(?m)^\s*([_\w]+)\s*=\s*(.+?)\s*$").unwrap();
        re.captures_iter(&text).map(|captures| (captures[1].to_string(), captures[2].to_string())).collect()
    }

    fn get_str_param<'a, 'b>(&'a self, name: &'b str) -> io::Result<&'a str> {
        match self.params_map.get(name) {
            Some(value) => Ok(value.trim_matches('"')),
            None => Err(io::Error::new(io::ErrorKind::InvalidData,
                                       format!("Parameter `{}` not found in parameter file", name)))
        }
    }

    fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
        where T: num::Num + str::FromStr,
              T::Err: string::ToString
    {
        let str_value = self.get_str_param(name)?;
        match str_value.parse::<T>() {
            Ok(value) => Ok(value),
            Err(err) => Err(io::Error::new(io::ErrorKind::InvalidData,
                                           format!("Failed parsing string `{}` in parameter file: {}", str_value, err.to_string())))
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
        let text = "int = 12 \n file_str=\"file.ext\"\nfloat =  -1.02E-07\ninvalid = number\n;comment";
        let params = Params{ params_map: Params::parse_params_text(text) };

        let correct_params: HashMap<_, _> = vec![(     "int".to_string(),           "12".to_string()),
                                                 ("file_str".to_string(), "\"file.ext\"".to_string()),
                                                 (   "float".to_string(),    "-1.02E-07".to_string()),
                                                 ( "invalid".to_string(),       "number".to_string())].into_iter().collect();
        assert_eq!(params.params_map, correct_params);

        assert_eq!(params.get_str_param("file_str").unwrap(), "file.ext");
        assert_eq!(params.get_numerical_param::<u32>("int").unwrap(), 12);
        assert_eq!(params.get_numerical_param::<f32>("float").unwrap(), -1.02e-7);
        assert!(params.get_numerical_param::<f32>("invalid").is_err());
    }

    #[test]
    fn reading_works() {
        let reader = SnapshotReader3::<HorRegularGrid3<_>>::new("data/en024031_emer3.0sml_ebeam_631.idl", Endianness::Little).unwrap();
        let _field = reader.read_scalar_field("r").unwrap();
    }
}
