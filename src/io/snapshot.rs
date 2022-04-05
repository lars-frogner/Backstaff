//! Reading and writing of Bifrost simulation data.

pub mod native;

#[cfg(feature = "netcdf")]
pub mod netcdf;

use super::{Endianness, Verbose};
use crate::{
    field::{ScalarField3, ScalarFieldProvider3, VectorField3},
    geometry::In3D,
    grid::Grid3,
};
use regex::Regex;
use std::{
    collections::{hash_map::Entry, HashMap},
    io,
    marker::PhantomData,
    path::Path,
    str,
    sync::Arc,
};

/// Floating-point precision assumed for Bifrost data.
#[allow(non_camel_case_types)]
pub type fdt = f32;

#[derive(Clone, Copy, Debug)]
pub enum SnapshotFormat {
    Native,
    #[cfg(feature = "netcdf")]
    NetCDF,
}

/// Snapshot number to assume when not inferrable.
pub const FALLBACK_SNAP_NUM: u32 = 0;

/// Standard names of coordinate arrays
pub const COORDINATE_NAMES: [&'static str; 12] = [
    "xm", "ym", "zm", "xmdn", "ymdn", "zmdn", "dxidxup", "dyidyup", "dzidzup", "dxidxdn",
    "dyidydn", "dzidzdn",
];
/// Standard name of output time step
pub const OUTPUT_TIME_STEP_NAME: &'static str = "dtsnap";

/// Standard name of mass density variable
pub const MASS_DENSITY_VARIABLE_NAME: &'static str = "r";
/// Standard base name of momentum variable
pub const MOMENTUM_VARIABLE_NAME: &'static str = "p";
/// Standard name of energy density variable
pub const ENERGY_DENSITY_VARIABLE_NAME: &'static str = "e";
/// Standard base name of magnetic field variable
pub const MAGNETIC_FIELD_VARIABLE_NAME: &'static str = "b";

/// Standard names of primary MHD variables
pub const PRIMARY_VARIABLE_NAMES_MHD: [&'static str; 8] = [
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
pub const PRIMARY_VARIABLE_NAMES_HD: [&'static str; 5] = [
    MASS_DENSITY_VARIABLE_NAME,
    "px",
    "py",
    "pz",
    ENERGY_DENSITY_VARIABLE_NAME,
];

/// Defines the properties of a provider of 3D Bifrost snapshot variables.
pub trait SnapshotProvider3<G: Grid3<fdt>>: ScalarFieldProvider3<fdt, G> {
    type Parameters: SnapshotParameters;

    /// Returns a reference to the parameters associated with the snapshot.
    fn parameters(&self) -> &Self::Parameters;

    /// Returns the assumed endianness of the snapshot.
    fn endianness(&self) -> Endianness;

    /// Returns the names of the primary variables of the snapshot.
    fn primary_variable_names(&self) -> Vec<&str>;

    /// Returns the names of the auxiliary variables of the snapshot.
    fn auxiliary_variable_names(&self) -> Vec<&str>;

    /// Returns the names of all the variables that can be provided.
    fn all_variable_names(&self) -> Vec<&str> {
        let mut all_variable_names = self.primary_variable_names();
        all_variable_names.append(&mut self.auxiliary_variable_names());
        all_variable_names
    }

    /// Returns the names of all the variables that can be provided, except the ones
    /// in the given list.
    fn all_variable_names_except(&self, excluded_variable_names: &[&str]) -> Vec<&str> {
        self.all_variable_names()
            .iter()
            .cloned()
            .filter(|name| !excluded_variable_names.contains(name))
            .collect::<Vec<_>>()
    }

    /// Given a list of variable names, returns a list of the ones that are primary
    /// and a list of the ones that are auxiliary.
    fn classify_variable_names<'a>(
        &'a self,
        variable_names: &[&'a str],
    ) -> (Vec<&'a str>, Vec<&'a str>) {
        let all_primary_variable_names = self.primary_variable_names();

        let included_primary_variable_names = all_primary_variable_names
            .iter()
            .cloned()
            .filter(|name| variable_names.contains(name))
            .collect::<Vec<_>>();

        let included_auxiliary_variable_names = variable_names
            .iter()
            .cloned()
            .filter(|name| !included_primary_variable_names.contains(name))
            .collect::<Vec<_>>();

        (
            included_primary_variable_names,
            included_auxiliary_variable_names,
        )
    }

    /// Returns whether the given variable can be provided.
    fn has_variable(&self, variable_name: &str) -> bool {
        self.all_variable_names().contains(&variable_name)
    }

    /// Returns the name and (if available) number of the snapshot.
    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>);

    /// Provides the specified 3D vector variable.
    fn provide_vector_field(&self, variable_name: &str) -> io::Result<VectorField3<fdt, G>> {
        Ok(VectorField3::new(
            variable_name.to_string(),
            self.arc_with_grid(),
            In3D::new(
                self.provide_scalar_field(&format!("{}x", variable_name))?,
                self.provide_scalar_field(&format!("{}y", variable_name))?,
                self.provide_scalar_field(&format!("{}z", variable_name))?,
            ),
        ))
    }
}

/// Parameters associated with a snapshot.
pub trait SnapshotParameters: Clone {
    /// Returns a list of all parameter names associated with the snapshot.
    fn names(&self) -> Vec<&str>;

    /// Provides the value of the given snapshot parameter.
    fn get_value(&self, name: &str) -> io::Result<ParameterValue>;

    /// Replaces the specified parameter values.
    fn modify_values(&mut self, modified_values: HashMap<&str, ParameterValue>);

    /// Returns a text representation of the parameters in the native parameter file format.
    fn native_text_representation(&self) -> String;

    /// Tries to read the given parameter from the parameter file.
    /// If successful, the value is converted with the given closure and
    /// returned, otherwise a warning is printed and the given default is returned.
    fn get_converted_numerical_param_or_fallback_to_default_with_warning<T, U, C>(
        &self,
        display_name: &str,
        name_in_param_file: &str,
        conversion_mapping: &C,
        default_value: U,
    ) -> U
    where
        T: From<fdt>,
        U: std::fmt::Display + Copy,
        C: Fn(T) -> U,
    {
        let use_default = |_| {
            println!(
                "Could not find parameter {}, falling back to default for {}: {}",
                name_in_param_file, display_name, default_value
            );
            default_value
        };
        self.get_value(name_in_param_file)
            .map_or_else(use_default, |val| {
                val.try_as_float()
                    .map_or_else(use_default, |val| conversion_mapping(val.into()))
            })
    }
}

#[derive(Clone, Debug)]
/// Value of a snapshot parameter.
pub enum ParameterValue {
    Str(String),
    Int(i64),
    Float(fdt),
}

impl ParameterValue {
    /// Return string representation of value.
    pub fn as_string(&self) -> String {
        match *self {
            Self::Str(ref s) => s.clone(),
            Self::Int(i) => format!("{}", i),
            Self::Float(f) => format!("{:8.3E}", f),
        }
    }

    /// Try interpreting the parameter value as an integer, or return an error if not possible.
    pub fn try_as_int(&self) -> io::Result<i64> {
        match *self {
            Self::Str(ref s) => match s.parse::<i64>() {
                Ok(i) => Ok(i),
                Err(err) => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Failed parsing parameter string {} as integer: {}",
                        s,
                        err.to_string()
                    ),
                )),
            },
            Self::Int(i) => Ok(i),
            Self::Float(f) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Cannot interpret parameter value {} as integer", f),
            )),
        }
    }

    /// Try interpreting the parameter value as a float, or return an error if not possible.
    pub fn try_as_float(&self) -> io::Result<fdt> {
        match *self {
            Self::Str(ref s) => match s.parse::<fdt>() {
                Ok(f) => Ok(f),
                Err(err) => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Failed parsing parameter string {} as float: {}",
                        s,
                        err.to_string()
                    ),
                )),
            },
            Self::Int(i) => Ok(i as fdt),
            Self::Float(f) => Ok(f),
        }
    }
}

/// Wrapper for `SnapshotProvider3` that reads or computes snapshot variables only on first request and
/// then caches the results.
#[derive(Clone, Debug)]
pub struct SnapshotCacher3<G: Grid3<fdt>, P> {
    provider: P,
    scalar_fields: HashMap<String, ScalarField3<fdt, G>>,
    vector_fields: HashMap<String, VectorField3<fdt, G>>,
}

impl<G: Grid3<fdt>, P: SnapshotProvider3<G>> SnapshotCacher3<G, P> {
    /// Creates a new snapshot cacher from the given provider.
    pub fn new(provider: P) -> Self {
        SnapshotCacher3 {
            provider,
            scalar_fields: HashMap::new(),
            vector_fields: HashMap::new(),
        }
    }

    /// Returns a reference to the provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Returns a mutable reference to the provider.
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.provider.grid()
    }

    /// Returns a `Result` with a reference to the scalar field representing the given variable,
    /// reading it from file or computing it and caching it if has not already been cached.
    pub fn obtain_scalar_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<&ScalarField3<fdt, G>> {
        Ok(match self.scalar_fields.entry(variable_name.to_string()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(self.provider.provide_scalar_field(variable_name)?)
            }
        })
    }

    /// Returns a `Result` with a reference to the vector field representing the given variable,
    /// reading it from file or computing it and caching it if has not already been cached.
    pub fn obtain_vector_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<&VectorField3<fdt, G>> {
        Ok(match self.vector_fields.entry(variable_name.to_string()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(self.provider.provide_vector_field(variable_name)?)
            }
        })
    }

    /// Makes sure the scalar field representing the given variable is cached.
    pub fn cache_scalar_field(&mut self, variable_name: &str) -> io::Result<()> {
        self.obtain_scalar_field(variable_name).map(|_| ())
    }

    /// Makes sure the scalar field representing the given variable is cached.
    pub fn cache_vector_field(&mut self, variable_name: &str) -> io::Result<()> {
        self.obtain_vector_field(variable_name).map(|_| ())
    }

    /// Returns a reference to the scalar field representing the given variable.
    ///
    /// Panics if the field is not cached.
    pub fn cached_scalar_field(&self, variable_name: &str) -> &ScalarField3<fdt, G> {
        self.scalar_fields
            .get(variable_name)
            .expect("Scalar field is not cached")
    }

    /// Returns a reference to the vector field representing the given variable.
    ///
    /// Panics if the field is not cached.
    pub fn cached_vector_field(&self, variable_name: &str) -> &VectorField3<fdt, G> {
        self.vector_fields
            .get(variable_name)
            .expect("Vector field is not cached")
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

// pub struct SnapshotResampler<G, P> {
//     provider: P,
//     computer: Option<SnapshotComputer3<G, P>>,
//     phantom: PhantomData<G>,
// }

// impl<G: Grid3<fdt>, P: SnapshotProvider3<G>> SnapshotResampler<G, P> {}

/// Parses the file name of the given path and returns the interpreted
/// snapshot name and (if detected) number.
pub fn extract_name_and_num_from_snapshot_path<P: AsRef<Path>>(
    file_path: P,
) -> (String, Option<u32>) {
    let (snap_name, snap_num_string) = parse_snapshot_file_path(file_path);
    (
        snap_name,
        snap_num_string.map(|s| s.parse::<u32>().unwrap()),
    )
}

/// Parses the file name of the given path and returns the number of digits
/// in the snapshot number part of the file name, if present.
pub fn determine_length_of_snap_num_in_file_name<P: AsRef<Path>>(file_path: P) -> Option<u32> {
    parse_snapshot_file_path(file_path)
        .1
        .map(|s| s.len() as u32)
}

/// Parses the file name of the given path and returns a corresponding
/// snapshot file name with the given number and extension.
pub fn create_new_snapshot_file_name_from_path<P: AsRef<Path>>(
    file_path: P,
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

fn parse_snapshot_file_path<P: AsRef<Path>>(file_path: P) -> (String, Option<String>) {
    let file_path = file_path.as_ref();
    let file_path = match file_path.extension() {
        Some(extension) if extension == "scr" => Path::new(file_path.file_stem().unwrap()),
        _ => file_path,
    };
    let file_stem = file_path.file_stem().unwrap().to_string_lossy().to_string();
    let regex = Regex::new(r"^(.+?)_(\d+)$").unwrap();
    regex
        .captures(&file_stem)
        .map(|caps| (caps[1].to_string(), Some(caps[2].to_string())))
        .unwrap_or_else(|| (file_stem, None))
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
pub fn add_magnitude_pipes<S: AsRef<str>>(name: S) -> String {
    format!("|{}|", name.as_ref())
}
