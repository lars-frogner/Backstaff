pub mod data {

    use std::{io, path, fs, mem, str, string};
    use std::io::{Read, BufRead, Seek, SeekFrom};
    use std::convert::TryInto;
    use std::collections::{HashMap, VecDeque};
    use num;
    use regex;
    use byteorder;
    use byteorder::ReadBytesExt;
    use ndarray;
    use ndarray::ShapeBuilder;

    /// 3D spatial coordinates specifying where field values are defined.
    ///
    /// The coordinates can be non-uniform and do not necessarily correspond
    /// to grid cell centers.
    #[derive(Debug, Clone)]
    pub struct FieldCoordinates3<T: num::Float> {
        pub x: ndarray::Array1<T>,
        pub y: ndarray::Array1<T>,
        pub z: ndarray::Array1<T>
    }

    impl<T: num::Float> FieldCoordinates3<T> {
        pub fn shape(&self) -> (usize, usize, usize) {
            (self.x.len(), self.y.len(), self.z.len())
        }
    }

    /// A 3D scalar field.
    ///
    /// Holds the coordinates and values of a 3D scalar field.
    /// The array of values is laid out in column-major order in memory.
    #[derive(Debug, Clone)]
    pub struct ScalarField3<T: num::Float> {
        pub coords: FieldCoordinates3<T>,
        pub values: ndarray::Array3<T>
    }

    impl<T: num::Float> ScalarField3<T> {
        pub fn shape(&self) -> (usize, usize, usize) {
            self.coords.shape()
        }
    }

    /// A 3D vector field.
    ///
    /// Holds the coordinates and values of the three components of a 3D vector field.
    /// The arrays of component values are laid out in column-major order in memory.
    #[derive(Debug, Clone)]
    pub struct VectorField3<T: num::Float> {
        pub coords: [FieldCoordinates3<T>; 3],
        pub values: [ndarray::Array3<T>; 3]
    }

    impl<T: num::Float> VectorField3<T> {
        pub fn shape(&self) -> (usize, usize, usize) {
            self.coords[0].shape()
        }
    }

    /// Little- or big-endian byte order.
    #[derive(Debug, Clone)]
    pub enum Endianness {
        Little,
        Big
    }

    /// Reader for the output files assoicated with a single Bifrost simulation snapshot.
    pub struct SnapshotReader {
        snap_path: path::PathBuf,
        aux_path: path::PathBuf,
        params: Params,
        endianness: Endianness,
        shape: (usize, usize, usize),
        center_coords: FieldCoordinates3<f32>,
        lower_coords: FieldCoordinates3<f32>,
        variables: HashMap<String, Variable>
    }

    impl SnapshotReader {

        /// Creates a reader for the snapshot with the parameter (.idl) file
        /// specified by `params_path` and with the given `endianness`.
        pub fn new(params_path: &path::Path, endianness: Endianness) -> io::Result<Self> {
            let params = Params::new(&params_path)?;

            let params_path = params_path.to_path_buf();
            let snap_num: u32 = params.get_numerical_param("isnap")?;
            let mesh_path = params_path.with_file_name(params.get_str_param("meshfile")?);
            let snap_path = params_path.with_file_name(format!("{}_{}.snap", params.get_str_param("snapname")?, snap_num));
            let aux_path = snap_path.with_extension("aux");

            let (center_coords, lower_coords) = Self::read_3d_coordinates_from_mesh_file(&mesh_path)?;
            let shape = center_coords.shape();

            let mut variables = HashMap::new();
            Self::insert_primary_variables(&mut variables);
            Self::insert_aux_variables(&params, &mut variables)?;

            Ok(SnapshotReader{
                snap_path,
                aux_path,
                params,
                endianness,
                shape,
                center_coords,
                lower_coords,
                variables
            })
        }

        /// Reads the values of the primary or auxiliary variable named `variable_name`
        /// and returns them as a 3D scalar field.
        pub fn read_3d_scalar_field(&self, variable_name: &str) -> io::Result<ScalarField3<f32>> {
            let variable = self.get_variable(variable_name)?;
            let coords = self.get_variable_coordinates(variable);
            let values = self.read_3d_variable_from_binary_file(variable)?;
            Ok(ScalarField3{ coords, values })
        }

        /// Reads the three component fields of the primary or auxiliary variable with root `variable_name`
        /// and returns them as a 3D vector field.
        pub fn read_3d_vector_field(&self, variable_name: &str) -> io::Result<VectorField3<f32>> {
            let component_variables = [self.get_variable(&format!("{}x", variable_name))?,
                                       self.get_variable(&format!("{}y", variable_name))?,
                                       self.get_variable(&format!("{}z", variable_name))?];

            let coords = [self.get_variable_coordinates(component_variables[0]),
                          self.get_variable_coordinates(component_variables[1]),
                          self.get_variable_coordinates(component_variables[2])];

            let values = [self.read_3d_variable_from_binary_file(component_variables[0])?,
                          self.read_3d_variable_from_binary_file(component_variables[1])?,
                          self.read_3d_variable_from_binary_file(component_variables[2])?];

            Ok(VectorField3{ coords, values })
        }

        /// Returns a reference to the string value of the parameter with the given `name`.
        pub fn get_str_param<'a, 'b>(&'a self, name: &'b str) -> io::Result<&'a str> {
            self.params.get_str_param(name)
        }

        /// Returns a the numerical value of the parameter with the given `name`.
        pub fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
            where T: num::Num + str::FromStr,
                  T::Err: string::ToString
        {
            self.params.get_numerical_param(name)
        }

        fn read_3d_coordinates_from_mesh_file(mesh_path: &path::Path) -> io::Result<(FieldCoordinates3<f32>, FieldCoordinates3<f32>)> {
            let file = fs::File::open(mesh_path)?;
            let mut lines = io::BufReader::new(file).lines();
            let coord_names = ["x", "y", "z"];
            let mut center_coord_vecs = VecDeque::new();
            let mut lower_coord_vecs = VecDeque::new();

            for dim in 0..3 {

                let mut center_coords = Vec::new();
                let mut lower_coords = Vec::new();

                let length = match lines.next() {
                    Some(string) => match string?.parse::<usize>() {
                        Ok(length) => length,
                        Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
                    },
                    None => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                      format!("number of {}-coordinates not found in mesh file", coord_names[dim])))
                };

                for coords in [&mut center_coords, &mut lower_coords].iter_mut() {
                    match lines.next() {
                        Some(string) =>
                            for s in string?.split_whitespace() {
                                match s.parse::<f32>() {
                                    Ok(val) => coords.push(val),
                                    Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
                                };
                            },
                        None => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                          format!("{}-coordinates not found in mesh file", coord_names[dim])))
                    };
                }

                if center_coords.len() != length || lower_coords.len() != length {
                    return Err(io::Error::new(io::ErrorKind::InvalidData,
                                              format!("inconsistent number of {}-coordinates in mesh file", coord_names[dim])))
                }

                let _ = lines.next(); // Skip up-derivative of coordinates
                let _ = lines.next(); // Skip down-derivative of coordinates

                center_coord_vecs.push_back(center_coords);
                lower_coord_vecs.push_back(lower_coords);
            }

            Ok((FieldCoordinates3{
                    x: ndarray::Array::from_vec(center_coord_vecs.pop_front().unwrap()),
                    y: ndarray::Array::from_vec(center_coord_vecs.pop_front().unwrap()),
                    z: ndarray::Array::from_vec(center_coord_vecs.pop_front().unwrap())
                },
                FieldCoordinates3{
                    x: ndarray::Array::from_vec( lower_coord_vecs.pop_front().unwrap()),
                    y: ndarray::Array::from_vec( lower_coord_vecs.pop_front().unwrap()),
                    z: ndarray::Array::from_vec( lower_coord_vecs.pop_front().unwrap())
                }))
        }

        fn insert_primary_variables(variables: &mut HashMap<String, Variable>) {
            let is_primary = true;
            variables.insert("r" .to_string(), Variable{ is_primary, at_lower: (false, false, false), index: 0 });
            variables.insert("px".to_string(), Variable{ is_primary, at_lower: (true,  false, false), index: 1 });
            variables.insert("py".to_string(), Variable{ is_primary, at_lower: (false, true,  false), index: 2 });
            variables.insert("pz".to_string(), Variable{ is_primary, at_lower: (false, false, true ), index: 3 });
            variables.insert("e" .to_string(), Variable{ is_primary, at_lower: (false, false, false), index: 4 });
            variables.insert("bx".to_string(), Variable{ is_primary, at_lower: (true,  false, false), index: 5 });
            variables.insert("by".to_string(), Variable{ is_primary, at_lower: (false, true,  false), index: 6 });
            variables.insert("bz".to_string(), Variable{ is_primary, at_lower: (false, false, true ), index: 7 });
        }

        fn insert_aux_variables(params: &Params, variables: &mut HashMap<String, Variable>) -> io::Result<()> {
            let is_primary = false;

            for (index, name) in params.get_str_param("aux")?.split_whitespace().enumerate() {
                let ends_with_x = name.ends_with('x');
                let ends_with_y = name.ends_with('y');
                let ends_with_z = name.ends_with('z');

                let at_lower = if (ends_with_x || ends_with_y || ends_with_z) &&
                                  (name.starts_with('e') || name.starts_with('i')) {
                    (!ends_with_x, !ends_with_y, !ends_with_z)
                } else {
                    (ends_with_x, ends_with_y, ends_with_z)
                };

                variables.insert(name.to_string(), Variable{ is_primary, at_lower, index });
            }
            Ok(())
        }

        fn get_variable(&self, name: &str) -> io::Result<&Variable> {
            match self.variables.get(name) {
                Some(variable) => Ok(variable),
                None => return Err(io::Error::new(io::ErrorKind::InvalidData,
                                                  format!("variable {} not found", name)))
            }
        }

        fn get_variable_coordinates(&self, variable: &Variable) -> FieldCoordinates3<f32> {
            FieldCoordinates3{
                x: if variable.at_lower.0 { self.lower_coords.x.clone() } else { self.center_coords.x.clone() },
                y: if variable.at_lower.1 { self.lower_coords.y.clone() } else { self.center_coords.y.clone() },
                z: if variable.at_lower.2 { self.lower_coords.z.clone() } else { self.center_coords.z.clone() }
            }
        }

        fn read_3d_variable_from_binary_file(&self, variable: &Variable) -> io::Result<ndarray::Array3<f32>> {
            let file_path = if variable.is_primary { &self.snap_path } else { &self.aux_path };
            let (mx, my, mz) = self.shape;
            let length = mx*my*mz;
            let offset = length*variable.index;
            let buffer = self.read_floats_from_binary_file(file_path, length, offset)?;
            Ok(ndarray::Array::from_shape_vec((mx, my, mz).f(), buffer).unwrap())
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

    struct Variable {
        is_primary: bool,
        at_lower: (bool, bool, bool),
        index: usize
    }

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
                                           format!("parameter {} not found", name)))
            }
        }

        fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
            where T: num::Num + str::FromStr,
                  T::Err: string::ToString
        {
            let str_value = self.get_str_param(name)?;
            match str_value.parse::<T>() {
                Ok(value) => Ok(value),
                Err(err) => Err(io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
            }
        }
    }

    #[cfg(test)]
    mod tests {

        use super::*;

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
    }
}
