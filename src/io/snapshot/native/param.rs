//! Utilities for parameter files in native format.

use super::super::{
    super::{utils, Verbosity},
    MapOfSnapshotParameters, ParameterValue, SnapshotParameters,
};
use crate::geometry::In3D;
use regex::{self, Captures, Regex};
use std::{
    borrow::Cow,
    cell::RefCell,
    io,
    path::{Path, PathBuf},
    str,
};

#[derive(Clone, Debug)]
/// Representation of parameters for native snapshots.
pub struct NativeSnapshotParameters {
    original_path: PathBuf,
    file_text: ParameterFile,
    parameter_set: MapOfSnapshotParameters,
}

impl NativeSnapshotParameters {
    pub fn new(param_file_path: PathBuf, verbosity: &Verbosity) -> io::Result<Self> {
        let file_text = ParameterFile::new(&param_file_path)?;
        let original_path = param_file_path;
        if verbosity.print_messages() {
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

    pub fn determine_snap_num(&self) -> io::Result<i64> {
        self.parameter_set.get_as_int("isnap")
    }

    pub fn determine_mesh_path(&self) -> io::Result<PathBuf> {
        Ok(self.original_path.with_file_name(
            self.parameter_set
                .get_as_unquoted_string("meshfile")?
                .as_ref(),
        ))
    }

    pub fn determine_snap_path(&self) -> io::Result<(PathBuf, PathBuf)> {
        let snap_num = self.determine_snap_num()?;
        let is_scratch = snap_num < 0;
        if is_scratch {
            let snap_name = self.parameter_set.get_as_unquoted_string("snapname")?;
            let snap_path = self
                .original_path
                .with_file_name(format!("{}.snap.scr", snap_name));
            let aux_path = self
                .original_path
                .with_file_name(format!("{}.aux.scr", snap_name));
            Ok((snap_path, aux_path))
        } else {
            let width =
                super::super::determine_length_of_snap_num_in_file_name(&self.original_path)
                    .unwrap_or(3);
            let snap_path = self.original_path.with_file_name(format!(
                "{}_{:0width$}.snap",
                self.parameter_set.get_as_unquoted_string("snapname")?,
                snap_num,
                width = width as usize
            ));
            let aux_path = snap_path.with_extension("aux");
            Ok((snap_path, aux_path))
        }
    }

    pub fn determine_aux_names(&self) -> io::Result<Vec<String>> {
        Ok(self
            .parameter_set
            .get_as_unquoted_string("aux")?
            .split_whitespace()
            .map(|name| name.to_string())
            .collect())
    }

    pub fn determine_grid_periodicity(&self) -> io::Result<In3D<bool>> {
        self.parameter_set.determine_grid_periodicity()
    }

    pub fn determine_if_mhd(&self) -> io::Result<bool> {
        self.parameter_set.determine_if_mhd()
    }
}

impl SnapshotParameters for NativeSnapshotParameters {
    fn heap_clone(&self) -> Box<RefCell<dyn SnapshotParameters>> {
        Box::new(RefCell::new(self.clone()))
    }

    fn n_values(&self) -> usize {
        self.parameter_set.n_values()
    }

    fn names(&self) -> Vec<&str> {
        self.parameter_set.names()
    }

    fn get_value(&self, name: &str) -> io::Result<&ParameterValue> {
        self.parameter_set.get_value(name)
    }

    fn set_value(&mut self, name: &str, value: ParameterValue) {
        self.parameter_set
            .parameters_mut()
            .entry(name.to_string())
            .and_modify(|old_value| {
                self.file_text.replace_parameter_value(name, &value);
                *old_value = value.clone();
            })
            .or_insert_with(|| {
                self.file_text.append_parameter_value(name, &value);
                value
            });
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
    fn new(param_file_path: &Path) -> io::Result<Self> {
        let text = utils::read_text_file(param_file_path)?;
        Ok(Self::from_text(text))
    }

    /// Parses the parameter file and returns the corresponding parameter set.
    fn parse(&self) -> MapOfSnapshotParameters {
        let regex = Regex::new(r"(?m)^\s*([_\w]+)\s*=\s*(.+?)\s*(?:;|$)").unwrap();
        MapOfSnapshotParameters::new(
            regex
                .captures_iter(&self.text)
                .map(|captures| {
                    let name = captures[1].to_string();
                    let value_string = captures[2].to_string();
                    let value = ParameterValue::new_string(value_string); // Always use string representation in order to preserve formatting
                    (name, value)
                })
                .collect(),
        )
    }

    /// Replaces the value of the given parameter with the given new value.
    ///
    /// This does not modify the original file.
    fn replace_parameter_value(&mut self, name: &str, new_value: &ParameterValue) {
        let regex = Regex::new(&format!(
            r"(?m)(^\s*{}\s*=\s*)(.+?)(\s*$)",
            regex::escape(name)
        ))
        .unwrap();
        if let Cow::Owned(new_text) = regex.replace_all(&self.text, |caps: &Captures| {
            format!("{}{}{}", &caps[1], new_value.as_string(), &caps[3])
        }) {
            self.text = new_text;
        }
    }

    fn append_parameter_value(&mut self, name: &str, new_value: &ParameterValue) {
        self.text = format!("{}\n{} = {}", self.text, name, new_value.as_string());
    }

    fn from_text(text: String) -> Self {
        Self { text }
    }
}
