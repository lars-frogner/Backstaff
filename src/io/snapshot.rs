//! Reading and writing of Bifrost simulation data.

pub mod native;

#[cfg(feature = "netcdf")]
pub mod netcdf;

pub mod utils;

use super::{Endianness, Verbosity};
use crate::{
    exit_on_false,
    field::FieldGrid3,
    geometry::{
        Dim3::{self, X, Y, Z},
        In3D,
    },
    grid::Grid3,
};
use lazy_static::lazy_static;
use regex::Regex;
use std::{borrow::Cow, cell::RefCell, collections::HashMap, io, path::Path, str};

#[cfg(feature = "for-testing")]
use approx::{AbsDiffEq, RelativeEq};

/// Floating-point precision assumed for snapshot data.
#[allow(non_camel_case_types)]
pub type fdt = f32;

/// Floating-point precision assumed for parameter values.
#[allow(non_camel_case_types)]
pub type fpa = f64;

#[derive(Clone, Copy, Debug)]
pub enum SnapshotFormat {
    Native,
    #[cfg(feature = "netcdf")]
    NetCDF,
}

/// Snapshot number to assume when not inferrable.
pub const FALLBACK_SNAP_NUM: u64 = 0;

/// Standard names of coordinate arrays
pub const COORDINATE_NAMES: [&str; 12] = [
    "xm", "ym", "zm", "xmdn", "ymdn", "zmdn", "dxidxup", "dyidyup", "dzidzup", "dxidxdn",
    "dyidydn", "dzidzdn",
];
/// Standard name of output time step
pub const OUTPUT_TIME_STEP_NAME: &str = "dtsnap";

/// Standard name of mass density variable
pub const MASS_DENSITY_VARIABLE_NAME: &str = "r";
/// Standard base name of momentum variable
pub const MOMENTUM_VARIABLE_NAME: &str = "p";
/// Standard name of energy density variable
pub const ENERGY_DENSITY_VARIABLE_NAME: &str = "e";
/// Standard base name of magnetic field variable
pub const MAGNETIC_FIELD_VARIABLE_NAME: &str = "b";

/// Standard names of primary MHD variables
pub const PRIMARY_VARIABLE_NAMES_MHD: [&str; 8] = [
    MASS_DENSITY_VARIABLE_NAME,
    "px",
    "py",
    "pz",
    ENERGY_DENSITY_VARIABLE_NAME,
    "bx",
    "by",
    "bz",
];
/// Standard names of primary HD variables
pub const PRIMARY_VARIABLE_NAMES_HD: [&str; 5] = [
    MASS_DENSITY_VARIABLE_NAME,
    "px",
    "py",
    "pz",
    ENERGY_DENSITY_VARIABLE_NAME,
];

/// Defines the properties of a collection of metadata associated with a Bifrost snapshot.
pub trait SnapshotMetadata {
    /// Returns a reference to the parameters associated with the snapshot.
    fn parameters(&self) -> &dyn SnapshotParameters;

    /// Returns the assumed endianness of the snapshot.
    fn endianness(&self) -> Endianness;

    /// Given a complete list of variable names, returns lists of the ones that
    /// should be considered primary and auxiliary, and whether the set of
    /// primary variables correspond to full MHD.
    fn classify_variable_names(
        &self,
        variable_names: &[String],
    ) -> (Vec<String>, Vec<String>, bool) {
        let primary_variable_names_mhd: Vec<_> = PRIMARY_VARIABLE_NAMES_MHD
            .iter()
            .cloned()
            .map(String::from)
            .collect();

        let (primary_variable_names, is_mhd) = if primary_variable_names_mhd
            .iter()
            .all(|name| variable_names.contains(name))
        {
            (primary_variable_names_mhd, true)
        } else {
            let primary_variable_names_hd: Vec<_> = PRIMARY_VARIABLE_NAMES_HD
                .iter()
                .cloned()
                .map(String::from)
                .collect();

            if primary_variable_names_hd
                .iter()
                .all(|name| variable_names.contains(name))
            {
                (primary_variable_names_hd, false)
            } else {
                (Vec::new(), false)
            }
        };

        let auxiliary_variable_names = variable_names
            .iter()
            .cloned()
            .filter(|name| !primary_variable_names.contains(name))
            .collect::<Vec<_>>();

        (primary_variable_names, auxiliary_variable_names, is_mhd)
    }

    /// Returns the name of the snapshot.
    fn snap_name(&self) -> &str;

    /// Returns the number of the snapshot (if available).
    fn snap_num(&self) -> Option<u64>;

    /// Returns the set of snapshot parameters, but modified to account for
    /// changes in the grid, snapshot name and number and included set of quantities.
    fn create_updated_parameters(
        &self,
        grid: &FieldGrid3,
        snap_name: &str,
        signed_snap_num: i64,
        included_auxiliary_variable_names: &[String],
        is_mhd: bool,
    ) -> DynSnapshotParameters {
        let shape = grid.shape();
        let average_grid_cell_extents = grid.average_grid_cell_extents();

        let boxed_new_parameters = self.parameters().heap_clone();
        {
            let mut new_parameters = boxed_new_parameters.borrow_mut();

            new_parameters.set_value("snapname", ParameterValue::new_quoted_string(snap_name));
            new_parameters.set_value("isnap", ParameterValue::new_int(signed_snap_num));
            new_parameters.set_value(
                "meshfile",
                ParameterValue::new_quoted_string(&format!("{}.mesh", snap_name)),
            );

            new_parameters.set_value(
                "aux",
                ParameterValue::new_quoted_string(&included_auxiliary_variable_names.join(" ")),
            );

            new_parameters.set_value(
                "do_mhd",
                ParameterValue::new_int(if is_mhd { 1 } else { 0 }),
            );

            new_parameters.set_value(
                "periodic_x",
                ParameterValue::new_int(if grid.is_periodic(X) { 1 } else { 0 }),
            );
            new_parameters.set_value(
                "periodic_y",
                ParameterValue::new_int(if grid.is_periodic(Y) { 1 } else { 0 }),
            );
            new_parameters.set_value(
                "periodic_z",
                ParameterValue::new_int(if grid.is_periodic(Z) { 1 } else { 0 }),
            );

            new_parameters.set_value("mx", ParameterValue::new_int(shape[X] as i64));
            new_parameters.set_value("my", ParameterValue::new_int(shape[Y] as i64));
            new_parameters.set_value("mz", ParameterValue::new_int(shape[Z] as i64));
            new_parameters.set_value(
                "dx",
                ParameterValue::new_float(average_grid_cell_extents[X]),
            );
            new_parameters.set_value(
                "dy",
                ParameterValue::new_float(average_grid_cell_extents[Y]),
            );
            new_parameters.set_value(
                "dz",
                ParameterValue::new_float(average_grid_cell_extents[Z]),
            );
        }
        boxed_new_parameters
    }
}

#[cfg(feature = "for-testing")]
#[macro_export]
macro_rules! snapshots_eq {
    ($self:expr, $other:expr) => {{
        use $crate::io::snapshot::{SnapshotProvider3, SnapshotReader3};

        let all_variable_names_self = $self.all_variable_names();
        let all_variable_names_other = $other.all_variable_names();
        if all_variable_names_self.len() != all_variable_names_other.len() {
            false
        } else {
            if $self.parameters() != $other.parameters() {
                false
            } else {
                all_variable_names_self.iter().all(|name| {
                    match (
                        $self.read_scalar_field(name),
                        $other.read_scalar_field(name),
                    ) {
                        (Ok(a), Ok(b)) => a == b,
                        _ => false,
                    }
                })
            }
        }
    }};
}

pub type DynSnapshotParameters = Box<RefCell<dyn SnapshotParameters>>;

/// Parameters associated with a snapshot.
pub trait SnapshotParameters {
    /// Returns a mutable reference to a clone of this parameter set living on the heap.
    fn heap_clone(&self) -> DynSnapshotParameters;

    /// Returns the number of parameters associated with the snapshot.
    fn n_values(&self) -> usize;

    /// Returns a list of all parameter names associated with the snapshot.
    fn names(&self) -> Vec<&str>;

    /// Provides the value of the given snapshot parameter.
    fn get_value(&self, name: &str) -> io::Result<&ParameterValue>;

    /// Replaces or adds the given value of the given snapshot parameter.
    fn set_value(&mut self, name: &str, value: ParameterValue);

    /// Returns a text representation of the parameters in the native parameter file format.
    fn native_text_representation(&self) -> String;

    /// Provides the value of the given snapshot parameter as a string.
    fn get_as_string(&self, name: &str) -> io::Result<Cow<str>> {
        Ok(self.get_value(name)?.as_string())
    }

    /// Provides the value of the given snapshot parameter as an unquoted string.
    fn get_as_unquoted_string(&self, name: &str) -> io::Result<Cow<str>> {
        Ok(self.get_value(name)?.as_unquoted_string())
    }

    /// Provides the value of the given snapshot parameter as an int if possible.
    fn get_as_int(&self, name: &str) -> io::Result<i64> {
        self.get_value(name)?.try_as_int()
    }

    /// Provides the value of the given snapshot parameter as a float if possible.
    fn get_as_float(&self, name: &str) -> io::Result<f64> {
        self.get_value(name)?.try_as_float()
    }

    #[cfg(feature = "for-testing")]
    fn relative_eq(
        &self,
        other: &dyn SnapshotParameters,
        epsilon: <fpa as AbsDiffEq>::Epsilon,
        max_relative: <fpa as AbsDiffEq>::Epsilon,
    ) -> bool {
        if self.n_values() != other.n_values() {
            #[cfg(debug_assertions)]
            {
                println!("Number of parameter values not equal");
                dbg!(self.n_values(), other.n_values());
            }
            return false;
        }
        self.names()
            .into_iter()
            .all(|name| match (self.get_value(name), other.get_value(name)) {
                (Ok(a), Ok(b)) => {
                    let equal = a.relative_eq(b, epsilon, max_relative);
                    #[cfg(debug_assertions)]
                    if !equal {
                        println!("Parameter {} not equal", name);
                        dbg!(a, b);
                    }
                    equal
                }
                _ => false,
            })
    }
}

#[derive(Clone, Debug)]
/// Value of a snapshot parameter.
pub enum ParameterValue {
    String(String),
    Int(i64),
    Float(fpa),
}

impl ParameterValue {
    /// Creates a parameter value corresponding to the given string.
    pub fn new_string(string: String) -> Self {
        Self::String(string)
    }

    /// Creates a parameter value corresponding to a quoted version
    /// of the given string.
    pub fn new_quoted_string(string: &str) -> Self {
        Self::String(format!("\"{}\"", string))
    }

    /// Creates a parameter value corresponding to the given integer.
    pub fn new_int(integer: i64) -> Self {
        Self::Int(integer)
    }

    /// Creates a parameter value corresponding to the given float.
    pub fn new_float(float: f64) -> Self {
        Self::Float(float)
    }

    /// Returns a string representation of the parameter value.
    pub fn as_string(&self) -> Cow<str> {
        match *self {
            Self::String(ref s) => Cow::from(s),
            Self::Int(i) => Cow::from(Self::format_int(i)),
            Self::Float(f) => Cow::from(Self::format_float(f)),
        }
    }

    /// Returns an unquoted string representation of the parameter value.
    pub fn as_unquoted_string(&self) -> Cow<str> {
        match *self {
            Self::String(ref s) => Cow::from(if Self::string_is_quoted(s) {
                &s[1..s.len() - 1]
            } else {
                s
            }),
            Self::Int(i) => Cow::from(Self::format_int(i)),
            Self::Float(f) => Cow::from(Self::format_float(f)),
        }
    }

    /// Tries interpreting the parameter value as an integer and
    /// returns the integer if successful.
    pub fn try_as_int(&self) -> io::Result<i64> {
        match *self {
            Self::String(ref s) => match s.parse::<i64>() {
                Ok(i) => Ok(i),
                Err(err) => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed parsing parameter string {} as integer: {}", s, err),
                )),
            },
            Self::Int(i) => Ok(i),
            Self::Float(f) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Cannot interpret parameter value {} as integer", f),
            )),
        }
    }

    /// Tries interpreting the parameter value as a float and returns
    /// the float if successful.
    pub fn try_as_float(&self) -> io::Result<fpa> {
        match *self {
            Self::String(ref s) => match s.parse::<fpa>() {
                Ok(f) => Ok(f),
                Err(err) => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed parsing parameter string {} as float: {}", s, err),
                )),
            },
            Self::Int(i) => Ok(i as fpa),
            Self::Float(f) => Ok(f),
        }
    }

    /// Converts the parameter value to a string and turns into
    /// the `Self::String` variant.
    pub fn turn_into_string_variant(&mut self) {
        *self = Self::String(self.as_string().to_string());
    }

    /// Tries to convert the parameter value to an integer and turns into
    /// the `Self::Int` variant if successful.
    pub fn try_turn_into_int_variant(&mut self) -> io::Result<()> {
        *self = Self::Int(self.try_as_int()?);
        Ok(())
    }

    /// Tries to convert the parameter value to a float and turns into
    /// the `Self::Float` variant if successful.
    pub fn try_turn_into_float_variant(&mut self) -> io::Result<()> {
        *self = Self::Float(self.try_as_float()?);
        Ok(())
    }

    /// Whether the given string surrounded by double quotes.
    pub fn string_is_quoted(string: &str) -> bool {
        string.starts_with('"') && string.ends_with('"')
    }

    fn format_int(integer: i64) -> String {
        format!("{}", integer)
    }

    fn format_float(float: f64) -> String {
        format!("{:15.8E}", float)
    }
}

#[cfg(feature = "for-testing")]
macro_rules! compare_parameter_values {
    ($self:expr, $other:expr, |$a:ident, $b:ident| $compare:expr) => {
        match ($self, $other) {
            (ParameterValue::String(a), ParameterValue::String(b)) => a == b,
            (ParameterValue::Int(a), ParameterValue::Int(b)) => a == b,
            (ParameterValue::Float(a), ParameterValue::Float(b)) => {
                #[allow(clippy::needless_borrow)]
                let compare = |$a: &fpa, $b: &fpa| $compare;
                compare(a, b)
            }
            (self_val, other_val) => {
                if let (Ok(a), Ok(b)) = (self_val.try_as_float(), other_val.try_as_float()) {
                    let compare = |$a: fpa, $b: fpa| $compare;
                    compare(a, b)
                } else if let (Ok(a), Ok(b)) = (self_val.try_as_int(), other_val.try_as_int()) {
                    a == b
                } else {
                    self_val.as_string() == other_val.as_string()
                }
            }
        }
    };
}

#[cfg(feature = "for-testing")]
impl PartialEq for ParameterValue {
    fn eq(&self, other: &Self) -> bool {
        compare_parameter_values!(self, other, |a, b| a == b)
    }
}

#[cfg(feature = "for-testing")]
impl AbsDiffEq for ParameterValue {
    type Epsilon = <fpa as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        fpa::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        compare_parameter_values!(self, other, |a, b| a.abs_diff_eq(&b, epsilon))
    }
}

#[cfg(feature = "for-testing")]
impl RelativeEq for ParameterValue {
    fn default_max_relative() -> Self::Epsilon {
        fpa::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        compare_parameter_values!(self, other, |a, b| a.relative_eq(&b, epsilon, max_relative))
    }
}

/// Representation of parameters as a `HashMap` of `ParameterValue`s.
#[derive(Clone, Debug, Default)]
pub struct MapOfSnapshotParameters(HashMap<String, ParameterValue>);

impl MapOfSnapshotParameters {
    pub fn new(parameters: HashMap<String, ParameterValue>) -> Self {
        Self(parameters)
    }

    pub fn parameters(&self) -> &HashMap<String, ParameterValue> {
        &self.0
    }

    pub fn parameters_mut(&mut self) -> &mut HashMap<String, ParameterValue> {
        &mut self.0
    }

    pub fn determine_if_mhd(&self) -> io::Result<bool> {
        Ok(self.get_as_int("do_mhd")? > 0)
    }

    /// Uses the available parameters to determine the axes for which the snapshot grid is periodic.
    pub fn determine_grid_periodicity(&self) -> io::Result<In3D<bool>> {
        Ok(In3D::new(
            self.get_as_int("periodic_x")? > 0,
            self.get_as_int("periodic_y")? > 0,
            self.get_as_int("periodic_z")? > 0,
        ))
    }
}

impl SnapshotParameters for MapOfSnapshotParameters {
    fn heap_clone(&self) -> DynSnapshotParameters {
        Box::new(RefCell::new(self.clone()))
    }

    fn n_values(&self) -> usize {
        self.parameters().len()
    }

    fn names(&self) -> Vec<&str> {
        self.parameters().keys().map(|s| s.as_str()).collect()
    }

    fn get_value(&self, name: &str) -> io::Result<&ParameterValue> {
        self.parameters().get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Parameter {} not found", name),
            )
        })
    }

    fn set_value(&mut self, name: &str, value: ParameterValue) {
        self.parameters_mut()
            .entry(name.to_string())
            .and_modify(|old_value| {
                *old_value = value.clone();
            })
            .or_insert(value);
    }

    fn native_text_representation(&self) -> String {
        let mut text = String::new();
        for (name, value) in self.parameters() {
            text = format!("{}{} = {}\n", &text, name, value.as_string());
        }
        text
    }
}

/// Parses the file name of the given path and returns the interpreted
/// snapshot name and (if detected) number.
pub fn extract_name_and_num_from_snapshot_path(file_path: &Path) -> (String, Option<u64>) {
    let (snap_name, snap_num_string) = parse_snapshot_file_path(file_path);
    (
        snap_name,
        snap_num_string.map(|s| s.parse::<u64>().unwrap()),
    )
}

/// Parses the file name of the given path and returns the number of digits
/// in the snapshot number part of the file name, if present.
pub fn determine_length_of_snap_num_in_file_name(file_path: &Path) -> Option<u32> {
    parse_snapshot_file_path(file_path)
        .1
        .map(|s| s.len() as u32)
}

/// Parses the file name of the given path and returns a corresponding
/// snapshot file name with the given number and extension.
pub fn create_new_snapshot_file_name_from_path(
    file_path: &Path,
    snap_num: u32,
    extension: &str,
    use_snap_num_as_offset: bool,
) -> String {
    match parse_snapshot_file_path(file_path) {
        (orig_snap_name, Some(orig_snap_num_string)) => {
            let orig_snap_num = orig_snap_num_string.parse::<u32>().unwrap();
            let new_snap_num = if use_snap_num_as_offset {
                orig_snap_num + snap_num
            } else {
                snap_num
            };
            if new_snap_num == 0 {
                format!("{}.{}", orig_snap_name, extension)
            } else {
                format!(
                    "{}_{:0width$}.{}",
                    orig_snap_name,
                    new_snap_num,
                    extension,
                    width = orig_snap_num_string.len()
                )
            }
        }
        (orig_snap_name, None) => {
            if snap_num == 0 {
                format!("{}.{}", orig_snap_name, extension)
            } else {
                format!("{}_{:03}.{}", orig_snap_name, snap_num, extension)
            }
        }
    }
}

lazy_static! {
    static ref SNAPSHOT_FILE_STEM_REGEX: Regex = Regex::new(r"^(.+?)_(\d+)$").unwrap();
    static ref VECTOR_COMPONENT_REGEX: Regex = Regex::new(r"^(.+?)([xyz])(c?)$").unwrap();
}

fn parse_snapshot_file_path(file_path: &Path) -> (String, Option<String>) {
    let file_path = match file_path.extension() {
        Some(extension) if extension == "scr" => Path::new(file_path.file_stem().unwrap()),
        _ => file_path,
    };
    let file_stem = file_path.file_stem().unwrap().to_string_lossy().to_string();
    SNAPSHOT_FILE_STEM_REGEX
        .captures(&file_stem)
        .map(|caps| (caps[1].to_string(), Some(caps[2].to_string())))
        .unwrap_or_else(|| (file_stem, None))
}

/// If the given variable name is a vector component, return the name of the
/// vector quantity, the dimension of the component and the name of all the
/// component quantities.
pub fn quantity_is_vector_component(variable_name: &str) -> Option<(String, Dim3, In3D<String>)> {
    VECTOR_COMPONENT_REGEX.captures(variable_name).map(|caps| {
        let vector_name = caps[1].to_string();
        let dim = Dim3::from_char(caps[2].chars().next().unwrap()).unwrap();
        let component_names =
            In3D::with_each_component(|dim| format!("{}{}{}", &vector_name, dim, &caps[3]));
        (vector_name, dim, component_names)
    })
}

/// For input strings of the format |<enclosed substring>|, returns the
/// enclosed substring, otherwise returns None.
pub fn extract_magnitude_name(name: &str) -> Option<&str> {
    if let (Some('|'), Some('|')) = (name.chars().next(), name.chars().last()) {
        if name.len() > 2 {
            return Some(&name[1..name.len() - 1]);
        }
    }
    None
}

/// Adds | at the beginning and end of the given string.
pub fn add_magnitude_pipes(name: &str) -> String {
    format!("|{}|", name)
}

/// Tries to read the given parameter from the parameter set.
/// If successful, the value is converted with the given closure and
/// returned, otherwise a warning is printed and the given default is returned.
pub fn get_converted_numerical_param_or_fallback_to_default_with_warning<T, U>(
    parameters: &dyn SnapshotParameters,
    display_name: &str,
    name_in_param_file: &str,
    conversion_mapping: &(dyn Fn(T) -> U),
    default_value: U,
) -> U
where
    T: From<fpa>,
    U: std::fmt::Display + Copy,
{
    let use_default = |_| {
        eprintln!(
            "Warning: Could not find parameter {}, falling back to default for {}: {}",
            name_in_param_file, display_name, default_value
        );
        default_value
    };
    parameters
        .get_value(name_in_param_file)
        .map_or_else(use_default, |val| {
            val.try_as_float().map_or_else(use_default, |val| {
                exit_on_false!(
                    val.is_finite(),
                    "Error: Parameter {} must be finite",
                    display_name
                );
                conversion_mapping(val.into())
            })
        })
}
