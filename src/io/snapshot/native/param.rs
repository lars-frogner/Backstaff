//! Utilities for parameter files in native format.

use super::super::{
    super::{utils, Verbose},
    ParameterValue, SnapshotParameters,
};
use crate::geometry::In3D;
use num;
use regex::{self, Captures, Regex};
use std::{
    borrow::Cow,
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    str, string,
};

#[derive(Clone, Debug)]
/// Representation of parameters for native snapshots.
pub struct NativeSnapshotParameters {
    original_path: PathBuf,
    file_text: ParameterFile,
    parameter_set: ParameterSet,
}

impl NativeSnapshotParameters {
    pub fn new<P: AsRef<Path>>(param_file_path: P, verbose: Verbose) -> io::Result<Self> {
        let original_path = param_file_path.as_ref().to_path_buf();
        let file_text = ParameterFile::new(param_file_path)?;
        if verbose.is_yes() {
            println!(
                "Reading parameters from {}",
                original_path.file_name().unwrap().to_string_lossy()
            );
        }
        let parameter_set = file_text.parse();
        Ok(Self {
            original_path,
            file_text,
            parameter_set,
        })
    }

    pub fn original_path(&self) -> &Path {
        self.original_path.as_path()
    }

    pub fn determine_snap_num(&self) -> io::Result<u32> {
        self.parameter_set.get_numerical_param("isnap")
    }

    pub fn determine_mesh_path(&self) -> io::Result<PathBuf> {
        Ok(self
            .original_path
            .with_file_name(self.parameter_set.get_str_param("meshfile")?))
    }

    pub fn determine_snap_path(&self) -> io::Result<(PathBuf, PathBuf)> {
        let width = super::super::determine_length_of_snap_num_in_file_name(&self.original_path)
            .unwrap_or(3);
        let snap_path = self.original_path.with_file_name(format!(
            "{}_{:0width$}.snap",
            self.parameter_set.get_str_param("snapname")?,
            self.determine_snap_num()?,
            width = width as usize
        ));
        let aux_path = snap_path.with_extension("aux");
        Ok((snap_path, aux_path))
    }

    pub fn determine_grid_periodicity(&self) -> io::Result<In3D<bool>> {
        self.parameter_set.determine_grid_periodicity()
    }

    pub fn determine_if_mhd(&self) -> io::Result<bool> {
        Ok(self.parameter_set.get_numerical_param::<u8>("do_mhd")? > 0)
    }

    pub fn determine_aux_names(&self) -> io::Result<Vec<String>> {
        Ok(self
            .parameter_set
            .get_str_param("aux")?
            .split_whitespace()
            .map(|name| name.to_string())
            .collect())
    }
}

impl SnapshotParameters for NativeSnapshotParameters {
    fn names(&self) -> Vec<&str> {
        self.parameter_set.parameter_names()
    }

    fn get_value(&self, name: &str) -> io::Result<ParameterValue> {
        self.parameter_set
            .get_str_param(name)
            .map(|s| ParameterValue::Str(s.to_owned()))
    }

    fn modify_values(&mut self, modified_values: HashMap<&str, ParameterValue>) {
        for (name, new_value) in modified_values {
            if let Some(old_value) = self.parameter_set.values.get_mut(name) {
                let new_value = new_value.as_string();
                self.file_text
                    .replace_parameter_value(name, new_value.as_str());
                *old_value = new_value;
            }
        }
    }

    fn native_text_representation(&self) -> String {
        self.file_text.text.clone()
    }
}

/// Representation of a parameter file.
#[derive(Clone, Debug)]
struct ParameterFile {
    text: String,
}

impl ParameterFile {
    /// Reads the parameter file at the given path.
    fn new<P: AsRef<Path>>(param_file_path: P) -> io::Result<Self> {
        let text = utils::read_text_file(param_file_path)?;
        Ok(Self::from_text(text))
    }

    /// Parses the parameter file and returns the corresponding parameter set.
    fn parse(&self) -> ParameterSet {
        let regex = Regex::new(r"(?m)^\s*([_\w]+)\s*=\s*(.+?)\s*$").unwrap();
        ParameterSet {
            values: regex
                .captures_iter(&self.text)
                .map(|captures| (captures[1].to_string(), captures[2].to_string()))
                .collect(),
        }
    }

    /// Replaces the value of the given parameter with the given new value.
    ///
    /// This does not modify the original file.
    fn replace_parameter_value(&mut self, name: &str, new_value: &str) {
        let regex = Regex::new(&format!(
            r"(?m)(^\s*{}\s*=\s*)(.+?)(\s*$)",
            regex::escape(name)
        ))
        .unwrap();
        if let Cow::Owned(new_text) = regex.replace_all(&self.text, |caps: &Captures| {
            format!("{}{}{}", &caps[1], new_value, &caps[3])
        }) {
            self.text = new_text;
        }
    }

    fn from_text(text: String) -> Self {
        Self { text }
    }
}

/// Set of parameter names and values associated with a parameter file.
#[derive(Clone, Debug)]
struct ParameterSet {
    values: HashMap<String, String>,
}

impl ParameterSet {
    /// Returns a list of all parameter names in the parameter set.
    fn parameter_names(&self) -> Vec<&str> {
        self.values.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the value of the string parameter with the given name.
    fn get_str_param<'a, 'b>(&'a self, name: &'b str) -> io::Result<&'a str> {
        match self.values.get(name) {
            Some(value) => Ok(value.trim_matches('"')),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Parameter {} not found in parameter file", name),
            )),
        }
    }

    /// Returns the value of the numerical parameter with the given name.
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
                    "Failed parsing string {} in parameter file: {}",
                    str_value,
                    err.to_string()
                ),
            )),
        }
    }

    /// Uses the available parameters to determine the axes for which the snapshot grid is periodic.
    fn determine_grid_periodicity(&self) -> io::Result<In3D<bool>> {
        Ok(In3D::new(
            self.get_numerical_param::<u8>("periodic_x")? == 1,
            self.get_numerical_param::<u8>("periodic_y")? == 1,
            self.get_numerical_param::<u8>("periodic_z")? == 1,
        ))
    }
}

#[cfg(test)]
mod tests {

    use super::{
        super::{
            super::{
                super::{Endianness, Verbose},
                SnapshotReader3,
            },
            NativeSnapshotReader3, NativeSnapshotReaderConfig,
        },
        *,
    };
    use crate::grid::hor_regular::HorRegularGrid3;

    #[test]
    fn param_parsing_works() {
        #![allow(clippy::float_cmp)]
        let text =
            "int = 12 \n file_str=\"file.ext\"\nfloat =  -1.02E-07\ninvalid = number\n;comment";
        let parameter_set = ParameterFile::from_text(text.to_owned()).parse();

        let correct_values: HashMap<_, _> = vec![
            ("int".to_string(), "12".to_string()),
            ("file_str".to_string(), "\"file.ext\"".to_string()),
            ("float".to_string(), "-1.02E-07".to_string()),
            ("invalid".to_string(), "number".to_string()),
        ]
        .into_iter()
        .collect();
        assert_eq!(parameter_set.values, correct_values);

        assert_eq!(parameter_set.get_str_param("file_str").unwrap(), "file.ext");
        assert_eq!(parameter_set.get_numerical_param::<u32>("int").unwrap(), 12);
        assert_eq!(
            parameter_set.get_numerical_param::<f32>("float").unwrap(),
            -1.02e-7
        );
        assert!(parameter_set.get_numerical_param::<f32>("invalid").is_err());
    }

    #[test]
    fn reading_works() {
        let reader =
            NativeSnapshotReader3::<HorRegularGrid3<_>>::new(NativeSnapshotReaderConfig::new(
                "data/en024031_emer3.0sml_ebeam_631.idl",
                Endianness::Little,
                Verbose::No,
            ))
            .unwrap();
        let _field = reader.read_scalar_field("r").unwrap();
    }
}
