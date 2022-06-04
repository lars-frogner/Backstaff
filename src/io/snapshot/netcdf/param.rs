//! Utilities for parameters in NetCDF format.

use super::super::{super::Verbose, fpa, ParameterValue, SnapshotParameters};
use crate::{geometry::In3D, io_result};
use netcdf_rs::{AttrValue, File, Group, GroupMut};
use std::{collections::HashMap, io};

#[cfg(feature = "comparison")]
use crate::{
    impl_abs_diff_eq_for_parameters, impl_partial_eq_for_parameters,
    impl_relative_eq_for_parameters,
};

#[derive(Clone, Debug)]
/// Representation of parameters for NetCDF snapshots.
pub struct NetCDFSnapshotParameters {
    parameters: HashMap<String, ParameterValue>,
}

impl NetCDFSnapshotParameters {
    pub fn new(file: &File, verbose: Verbose) -> io::Result<Self> {
        if verbose.is_yes() {
            println!(
                "Reading parameters from {}",
                file.path().unwrap().file_name().unwrap().to_string_lossy()
            );
        }
        let root_group = file.root().unwrap();
        let parameters = read_all_parameter_names(&root_group)
            .into_iter()
            .map(|name| read_snapshot_parameter(&root_group, &name).map(|value| (name, value)))
            .collect::<io::Result<_>>()?;
        Ok(Self { parameters })
    }

    /// Uses the available parameters to determine the axes for which the snapshot grid is periodic.
    pub fn determine_grid_periodicity(&self) -> io::Result<In3D<bool>> {
        Ok(In3D::new(
            self.get_value("periodic_x")?.try_as_int()? == 1,
            self.get_value("periodic_y")?.try_as_int()? == 1,
            self.get_value("periodic_z")?.try_as_int()? == 1,
        ))
    }
}

impl SnapshotParameters for NetCDFSnapshotParameters {
    fn n_values(&self) -> usize {
        self.parameters.len()
    }

    fn names(&self) -> Vec<&str> {
        self.parameters.keys().map(|s| s.as_str()).collect()
    }

    fn get_value(&self, name: &str) -> io::Result<ParameterValue> {
        self.parameters.get(name).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parameter {} not found in NetCDF file", name),
            )
        })
    }

    fn modify_values(&mut self, modified_values: HashMap<&str, ParameterValue>) {
        for (name, new_value) in modified_values {
            if let Some(old_value) = self.parameters.get_mut(name) {
                *old_value = new_value;
            }
        }
    }

    fn native_text_representation(&self) -> String {
        let mut text = String::new();
        for (name, value) in &self.parameters {
            text = format!("{}{} = {}\n", &text, name, value.as_string());
        }
        text
    }
}

#[cfg(feature = "comparison")]
impl_partial_eq_for_parameters!(NetCDFSnapshotParameters);

#[cfg(feature = "comparison")]
impl_abs_diff_eq_for_parameters!(NetCDFSnapshotParameters);

#[cfg(feature = "comparison")]
impl_relative_eq_for_parameters!(NetCDFSnapshotParameters);

/// Returns a list of all parameters in the given NetCDF group.
fn read_all_parameter_names(group: &Group) -> Vec<String> {
    group
        .attributes()
        .map(|attr| attr.name().to_string())
        .collect()
}

/// Attempts to read the given snapshot parameter from given NetCDF group.
fn read_snapshot_parameter(group: &Group, name: &str) -> io::Result<ParameterValue> {
    Ok(
        match io_result!(group
            .attribute(name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Parameter {} not found in NetCDF file", name),
                )
            })?
            .value())?
        {
            AttrValue::Uchar(i) => ParameterValue::Int(i as i64),
            AttrValue::Schar(i) => ParameterValue::Int(i as i64),
            AttrValue::Ushort(i) => ParameterValue::Int(i as i64),
            AttrValue::Short(i) => ParameterValue::Int(i as i64),
            AttrValue::Uint(i) => ParameterValue::Int(i as i64),
            AttrValue::Int(i) => ParameterValue::Int(i as i64),
            AttrValue::Ulonglong(i) => ParameterValue::Int(i as i64),
            AttrValue::Longlong(i) => ParameterValue::Int(i),
            AttrValue::Float(f) => ParameterValue::Float(f as fpa),
            AttrValue::Double(f) => ParameterValue::Float(f as fpa),
            AttrValue::Str(s) => ParameterValue::Str(s),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Parameter {} in NetCDF file is not a string or scalar",
                        name
                    ),
                ))
            }
        },
    )
}

/// Writes all given snapshot parameters to the given NetCDF group.
pub fn write_snapshot_parameters<P: SnapshotParameters>(
    group: &mut GroupMut,
    parameters: &P,
) -> io::Result<()> {
    for parameter_name in parameters.names() {
        write_single_snapshot_parameter(
            group,
            parameter_name,
            parameters.get_value(parameter_name)?,
        )?;
    }
    Ok(())
}

/// Writes the given new snapshot parameter to the given NetCDF group.
fn write_single_snapshot_parameter(
    group: &mut GroupMut,
    parameter_name: &str,
    parameter_value: ParameterValue,
) -> io::Result<()> {
    match parameter_value {
        ParameterValue::Str(s) => io_result!(group.add_attribute(parameter_name, s.as_str()))?,
        ParameterValue::Int(i) => io_result!(group.add_attribute(parameter_name, i))?,
        ParameterValue::Float(f) => io_result!(group.add_attribute(parameter_name, f))?,
    };
    Ok(())
}
