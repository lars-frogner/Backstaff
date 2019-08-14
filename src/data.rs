pub mod data {

    use std::{io, path, fs, mem, str, string};
    use std::io::{Read, Seek, SeekFrom};
    use std::convert::TryInto;
    use std::collections::HashMap;
    use num;
    use byteorder;
    use byteorder::ReadBytesExt;
    use ndarray;
    use ndarray::ShapeBuilder;
    use regex;

    pub struct Bifrost3DScalarField<T: num::Float> {
        data: ndarray::Array3<T>
    }

    pub struct Bifrost3DVectorField<T: num::Float>(
        Bifrost3DScalarField<T>,
        Bifrost3DScalarField<T>,
        Bifrost3DScalarField<T>
    );

    pub enum Endianness {
        Little,
        Big
    }

    pub struct BifrostOutput {
        params_path: path::PathBuf,
        mesh_path: path::PathBuf,
        snap_path: path::PathBuf,
        aux_path: path::PathBuf,
        endianness: Endianness,
        snap_num: u32,
        params: Params,
        primary_variable_indices: HashMap<String, usize>,
        aux_variable_indices: HashMap<String, usize>
    }

    impl BifrostOutput {

        pub fn new(params_path: &path::Path, endianness: Endianness) -> io::Result<Self> {
            let params = Params::new(params_path)?;
            let params_path = params_path.to_path_buf();
            let snap_num = params.get_numerical_param("isnap")?;
            let mesh_path = params_path.with_file_name(params.get_str_param("meshfile")?);
            let snap_path = params_path.with_file_name(format!("{}_{}.snap", params.get_str_param("snapname")?, snap_num));
            let aux_path = snap_path.with_extension("aux");
            let primary_variable_indices = Self::get_primary_variable_indices();
            let aux_variable_indices = Self::get_aux_variable_indices(&params)?;
            Ok(BifrostOutput{ params_path,
                              mesh_path,
                              snap_path,
                              aux_path,
                              endianness,
                              snap_num,
                              params,
                              primary_variable_indices,
                              aux_variable_indices })
        }

        pub fn read_3d_scalar_field(&self, variable_name: &str) -> io::Result<Bifrost3DScalarField<f32>> {
            let (mx, my, mz) = self.get_field_shape()?;
            let (file_path, index) = self.get_variable_location(variable_name)?;
            let length = mx*my*mz;
            let offset = length*index;
            let buffer = self.read_f32_values(file_path, length, offset)?;
            let data = ndarray::Array::from_shape_vec((mx, my, mz).f(), buffer).unwrap();
            Ok(Bifrost3DScalarField{ data })
        }

        pub fn read_3d_vector_field(&self, variable_name: &str) -> io::Result<Bifrost3DVectorField<f32>> {
            let component_names = (format!("{}x", variable_name),
                                   format!("{}y", variable_name),
                                   format!("{}z", variable_name));
            Ok(Bifrost3DVectorField(self.read_3d_scalar_field(&component_names.0)?,
                                    self.read_3d_scalar_field(&component_names.1)?,
                                    self.read_3d_scalar_field(&component_names.2)?))
        }

        fn get_primary_variable_indices() -> HashMap<String, usize> {
            let names = ["r", "px", "py", "pz", "e", "bx", "by", "bz"];
            names.iter().map(|&s| s.into()).zip(0..).collect()
        }

        fn get_aux_variable_indices(params: &Params) -> io::Result<HashMap<String, usize>> {
            let names = params.get_str_param("aux")?.split_whitespace();
            Ok(names.map(|s| s.into()).zip(0..).collect())
        }

        fn get_variable_location<'a, 'b>(&'a self, name: &'b str) -> io::Result<(&'a path::Path, usize)> {
            if self.primary_variable_indices.contains_key(name) {
                Ok((&self.snap_path, self.primary_variable_indices[name]))
            }
            else if self.aux_variable_indices.contains_key(name) {
                Ok((&self.aux_path, self.aux_variable_indices[name]))
            }
            else {
                Err(io::Error::new(io::ErrorKind::NotFound, format!("variable {} not found", name)))
            }
        }

        fn read_f32_values(&self, file_path: &path::Path, length: usize, offset: usize) -> io::Result<Vec<f32>> {
            let mut file = fs::File::open(file_path)?;
            file.seek(SeekFrom::Start((offset*mem::size_of::<f32>()).try_into().unwrap()))?;
            let mut buffer = vec![0.0; length];
            match self.endianness {
                Endianness::Little => file.read_f32_into::<byteorder::LittleEndian>(&mut buffer)?,
                Endianness::Big => file.read_f32_into::<byteorder::BigEndian>(&mut buffer)?
            };
            Ok(buffer)
        }

        fn get_field_shape(&self) -> io::Result<(usize, usize, usize)> {
            let mx: usize = self.params.get_numerical_param("mx")?;
            let my: usize = self.params.get_numerical_param("my")?;
            let mz: usize = self.params.get_numerical_param("mz")?;
            Ok((mx, my, mz))
        }
    }

    fn read_text_file(file_path: &path::Path) -> io::Result<String> {
        let file = fs::File::open(file_path)?;
        let mut text = String::new();
        let _ = io::BufReader::new(file).read_to_string(&mut text)?;
        Ok(text)
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
                None => Err(io::Error::new(io::ErrorKind::NotFound, format!("parameter {} not found", name)))
            }
        }

        fn get_numerical_param<T>(&self, name: &str) -> io::Result<T>
            where T: num::Num + str::FromStr,
                  T::Err: string::ToString
        {
            let str_value = self.get_str_param(name)?;
            match str_value.parse::<T>() {
                Ok(value) => Ok(value),
                Err(err) => Err(io::Error::new(io::ErrorKind::NotFound, err.to_string()))
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
