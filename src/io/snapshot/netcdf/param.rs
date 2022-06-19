//! Utilities for parameters in NetCDF format.

use super::super::{
    super::Verbosity, fpa, MapOfSnapshotParameters, ParameterValue, SnapshotParameters,
};
use crate::io_result;
use netcdf_rs::{AttrValue, File, Group, GroupMut};
use std::io;

pub type NetCDFSnapshotParameters = MapOfSnapshotParameters;

pub fn read_netcdf_snapshot_parameters(
    file: &File,
    verbosity: &Verbosity,
) -> io::Result<NetCDFSnapshotParameters> {
    if verbosity.print_messages() {
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
    Ok(NetCDFSnapshotParameters::new(parameters))
}

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
            AttrValue::Str(s) => ParameterValue::String(s),
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
pub fn write_snapshot_parameters(
    group: &mut GroupMut,
    parameters: std::cell::Ref<dyn SnapshotParameters>,
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
    parameter_value: &ParameterValue,
) -> io::Result<()> {
    match parameter_value {
        ParameterValue::String(s) => io_result!(group.add_attribute(parameter_name, s.as_str()))?,
        &ParameterValue::Int(i) => io_result!(group.add_attribute(parameter_name, i))?,
        &ParameterValue::Float(f) => io_result!(group.add_attribute(parameter_name, f))?,
    };
    Ok(())
}
