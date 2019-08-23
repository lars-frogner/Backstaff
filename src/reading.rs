//! Reading of Bifrost simulation data.

use std::{io, path, fs, mem, str, string};
use std::io::{Read, BufRead, Seek, SeekFrom};
use std::convert::TryInto;
use std::collections::{HashMap, VecDeque};
use num;
use regex;
use byteorder;
use byteorder::ReadBytesExt;
use ndarray::prelude::*;
use crate::geometry::{Dim3, In3D, Coords3};
use crate::grid::{CoordsType, Grid3Type, Grid3};
use crate::field::{ScalarField3, VectorField3};
use Dim3::{X, Y, Z};

/// Little- or big-endian byte order.
#[derive(Debug, Copy, Clone)]
pub enum Endianness {
    Little,
    Big
}

/// Reader for the output files assoicated with a single Bifrost simulation snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotReader<G: Grid3<f32> + Clone> {
    snap_path: path::PathBuf,
    aux_path: path::PathBuf,
    params: Params,
    endianness: Endianness,
    grid: G,
    variables: HashMap<String, Variable>
}

impl<G: Grid3<f32> + Clone> SnapshotReader<G> {
    /// Creates a reader for a Bifrost snapshot.
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
    /// - `Ok`: Contains a new `SnapshotReader`.
    /// - `Err`: Contains an error encountered during opening, reading or parsing the relevant files.
    pub fn new(params_path: &path::Path, endianness: Endianness) -> io::Result<Self> {
        let params = Params::new(&params_path)?;

        let params_path = params_path.to_path_buf();
        let snap_num: u32 = params.get_numerical_param("isnap")?;
        let mesh_path = params_path.with_file_name(params.get_str_param("meshfile")?);
        let snap_path = params_path.with_file_name(format!("{}_{}.snap", params.get_str_param("snapname")?, snap_num));
        let aux_path = snap_path.with_extension("aux");

        let grid = Self::read_3d_grid_from_mesh_file(&params, &mesh_path)?;

        let mut variables = HashMap::new();
        Self::insert_primary_variables(&mut variables);
        Self::insert_aux_variables(&params, &mut variables)?;

        Ok(SnapshotReader{
            snap_path,
            aux_path,
            params,
            endianness,
            grid,
            variables
        })
    }

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
    /// - `Ok`: Contains a `ScalarField3<f32>` holding the field coordinates and values.
    /// - `Err`: Contains an error encountered while locating the variable or reading the data.
    pub fn read_3d_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<f32, G>> {
        let variable = self.get_variable(variable_name)?;
        let values = self.read_3d_variable_from_binary_file(variable)?;
        Ok(ScalarField3::new(self.grid.clone(), variable.coord_types.clone(), values))
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
    /// - `Ok`: Contains a `VectorField3<f32>` holding the coordinates and values of the vector field components.
    /// - `Err`: Contains an error encountered while locating the variable or reading the data.
    pub fn read_3d_vector_field(&self, variable_name: &str) -> io::Result<VectorField3<f32, G>> {
        let component_variables = [self.get_variable(&format!("{}x", variable_name))?,
                                   self.get_variable(&format!("{}y", variable_name))?,
                                   self.get_variable(&format!("{}z", variable_name))?];

        let values = In3D::new(self.read_3d_variable_from_binary_file(component_variables[0])?,
                               self.read_3d_variable_from_binary_file(component_variables[1])?,
                               self.read_3d_variable_from_binary_file(component_variables[2])?);

        let coord_types = In3D::new(component_variables[0].coord_types.clone(),
                                    component_variables[1].coord_types.clone(),
                                    component_variables[2].coord_types.clone());

        Ok(VectorField3::new(self.grid.clone(), coord_types, values))
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

    fn read_3d_grid_from_mesh_file(params: &Params, mesh_path: &path::Path) -> io::Result<G> {
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
                            match s.parse::<f32>() {
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

            let uniform_up = up_derivatives.iter().all(|element| element == &up_derivatives[0]);
            let uniform_down = down_derivatives.iter().all(|element| element == &down_derivatives[0]);

            if uniform_up != uniform_down {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                                          format!("Inconsistent uniformity of {}-coordinates in mesh file", coord_names[dim])))
            }

            is_uniform[dim] = uniform_up;

            center_coord_vecs.push_back(center_coords);
            lower_coord_vecs.push_back(lower_coords);
        }

        let detected_grid_type = match is_uniform {
            [true, true, true] => Grid3Type::Regular,
            [true, true, false] => Grid3Type::HorRegular,
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                           "Non-uniform x- or y-coordinates not supported"))
        };

        if detected_grid_type != G::TYPE {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                       "Wrong reader type for the specified mesh file"))
        }

        let center_coords = Coords3::new(
            Array::from_vec(center_coord_vecs.pop_front().unwrap()),
            Array::from_vec(center_coord_vecs.pop_front().unwrap()),
            Array::from_vec(center_coord_vecs.pop_front().unwrap())
        );

        let lower_edge_coords = Coords3::new(
            Array::from_vec( lower_coord_vecs.pop_front().unwrap()),
            Array::from_vec( lower_coord_vecs.pop_front().unwrap()),
            Array::from_vec( lower_coord_vecs.pop_front().unwrap())
        );

        let is_periodic = In3D::new(params.get_numerical_param::<u8>("periodic_x")? == 1,
                                    params.get_numerical_param::<u8>("periodic_y")? == 1,
                                    params.get_numerical_param::<u8>("periodic_z")? == 1);

        Ok(G::new(center_coords, lower_edge_coords, is_periodic))
    }

    fn insert_primary_variables(variables: &mut HashMap<String, Variable>) {
        let is_primary = true;
        variables.insert("r" .to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Center, CoordsType::Center, CoordsType::Center), index: 0 });
        variables.insert("px".to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Lower,  CoordsType::Center, CoordsType::Center), index: 1 });
        variables.insert("py".to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Center, CoordsType::Lower,  CoordsType::Center), index: 2 });
        variables.insert("pz".to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Center, CoordsType::Center, CoordsType::Lower ), index: 3 });
        variables.insert("e" .to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Center, CoordsType::Center, CoordsType::Center), index: 4 });
        variables.insert("bx".to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Lower,  CoordsType::Center, CoordsType::Center), index: 5 });
        variables.insert("by".to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Center, CoordsType::Lower,  CoordsType::Center), index: 6 });
        variables.insert("bz".to_string(), Variable{ is_primary, coord_types: In3D::new(CoordsType::Center, CoordsType::Center, CoordsType::Lower ), index: 7 });
    }

    fn insert_aux_variables(params: &Params, variables: &mut HashMap<String, Variable>) -> io::Result<()> {
        let is_primary = false;

        for (index, name) in params.get_str_param("aux")?.split_whitespace().enumerate() {
            let ends_with_x = name.ends_with('x');
            let ends_with_y = name.ends_with('y');
            let ends_with_z = name.ends_with('z');

            let coord_types = if (ends_with_x || ends_with_y || ends_with_z) &&
                               (name.starts_with('e') || name.starts_with('i')) {
                In3D::new(if ends_with_x { CoordsType::Center } else {CoordsType::Lower},
                          if ends_with_y { CoordsType::Center } else {CoordsType::Lower},
                          if ends_with_z { CoordsType::Center } else {CoordsType::Lower})
            } else {
                In3D::new(if ends_with_x { CoordsType::Lower } else {CoordsType::Center},
                          if ends_with_y { CoordsType::Lower } else {CoordsType::Center},
                          if ends_with_z { CoordsType::Lower } else {CoordsType::Center})
            };

            variables.insert(name.to_string(), Variable{ is_primary, coord_types, index });
        }

        Ok(())
    }

    fn get_variable(&self, name: &str) -> io::Result<&Variable> {
        match self.variables.get(name) {
            Some(variable) => Ok(variable),
            None => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                              format!("Variable `{}` not found", name)))
        }
    }

    fn read_3d_variable_from_binary_file(&self, variable: &Variable) -> io::Result<Array3<f32>> {
        let file_path = if variable.is_primary { &self.snap_path } else { &self.aux_path };
        let shape = self.grid.shape();
        let length = shape[X]*shape[Y]*shape[Z];
        let offset = length*variable.index;
        let buffer = self.read_floats_from_binary_file(file_path, length, offset)?;
        Ok(Array::from_shape_vec((shape[X], shape[Y], shape[Z]).f(), buffer).unwrap())
    }

    fn read_floats_from_binary_file(&self, file_path: &path::Path, length: usize, offset: usize) -> io::Result<Vec<f32>> {
        let mut file = fs::File::open(file_path)?;
        file.seek(SeekFrom::Start((offset*mem::size_of::<f32>()).try_into().unwrap()))?;
        let mut buffer = vec![0.0; length];
        match self.endianness {
            Endianness::Little => file.read_f32_into::<byteorder::LittleEndian>(&mut buffer)?,
            Endianness::Big => file.read_f32_into::<byteorder::BigEndian>(&mut buffer)?
        };
        Ok(buffer)
    }
}

fn read_text_file(file_path: &path::Path) -> io::Result<String> {
    let file = fs::File::open(file_path)?;
    let mut text = String::new();
    let _ = io::BufReader::new(file).read_to_string(&mut text)?;
    Ok(text)
}

#[derive(Debug, Clone)]
struct Variable {
    is_primary: bool,
    coord_types: In3D<CoordsType>,
    index: usize
}

#[derive(Debug, Clone)]
struct Params {
    params_map: HashMap<String, String>
}

impl Params {

    fn new(params_path: &path::Path) -> io::Result<Self> {
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
        let text = "int = 12 \n file_str=\"file.ext\"\nfloat =  -1.02E-07\ninvalid = number\n;comment";
        let params = Params{ params_map: Params::parse_params_text(text) };

        let correct_params: HashMap<_, _> = vec![(     "int".to_string(),           "12".to_string()),
                                                 ("file_str".to_string(), "\"file.ext\"".to_string()),
                                                 (   "float".to_string(),    "-1.02E-07".to_string()),
                                                 ( "invalid".to_string(),       "number".to_string())].into_iter().collect();
        assert_eq!(params.params_map, correct_params);

        assert_eq!(params.get_str_param("file_str").unwrap(), "file.ext");
        assert_eq!(params.get_numerical_param::<u32>("int").unwrap(), 12_u32);
        assert_eq!(params.get_numerical_param::<f32>("float").unwrap(), -1.02e-7_f32);
        assert!(params.get_numerical_param::<f32>("invalid").is_err());
    }

    #[test]
    fn reader_works() {
        let params_path = path::PathBuf::from("data/en024031_emer3.0sml_ebeam_631.idl");
        let reader: SnapshotReader<HorRegularGrid3<f32>> = SnapshotReader::new(&params_path, Endianness::Little).unwrap();
        let field = reader.read_3d_scalar_field("r").unwrap();
        println!("{:?}", field.values().sum()/(field.values().len() as f32));
    }
}
