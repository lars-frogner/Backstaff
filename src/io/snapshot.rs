//! Reading and writing of Bifrost simulation data.

pub mod native;

#[cfg(feature = "netcdf")]
pub mod netcdf;

use super::{Endianness, Verbose};
use crate::{
    field::{
        CachingScalarFieldProvider3, ResampledCoordLocation, ResamplingMethod, ScalarField3,
        ScalarFieldCacher3, ScalarFieldProvider3,
    },
    geometry::{Idx3, In3D, PointTransformation2},
    grid::{fgr, Grid3},
    interpolation::Interpolator3,
};
use regex::Regex;
use std::{collections::HashMap, io, marker::PhantomData, path::Path, str, sync::Arc};

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
pub const FALLBACK_SNAP_NUM: u32 = 0;

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

/// Defines the properties of a provider of 3D Bifrost snapshot variables.
pub trait SnapshotProvider3<G: Grid3<fgr>>: ScalarFieldProvider3<fdt, G> {
    type Parameters: SnapshotParameters;

    /// Returns a reference to the parameters associated with the snapshot.
    fn parameters(&self) -> &Self::Parameters;

    /// Returns the assumed endianness of the snapshot.
    fn endianness(&self) -> Endianness;

    /// Returns the names of the primary variables of the snapshot.
    fn primary_variable_names(&self) -> &[String];

    /// Returns the names of the auxiliary variables of the snapshot.
    fn auxiliary_variable_names(&self) -> &[String];

    /// Returns the names of all the variables that can be provided.
    fn all_variable_names(&self) -> &[String];

    /// Returns the names of all the variables that can be provided, except the ones
    /// in the given list.
    fn all_variable_names_except(&self, excluded_variable_names: &[String]) -> Vec<String> {
        self.all_variable_names()
            .iter()
            .cloned()
            .filter(|name| !excluded_variable_names.contains(name))
            .collect::<Vec<_>>()
    }

    /// Given a list of variable names, returns a list of the ones that are primary
    /// and a list of the ones that are auxiliary.
    fn classify_variable_names(&self, variable_names: &[String]) -> (Vec<String>, Vec<String>) {
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
    fn has_variable<S: AsRef<str>>(&self, variable_name: S) -> bool;

    /// Returns the name and (if available) number of the snapshot.
    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>);
}

pub trait SnapshotReader3<G: Grid3<fgr>>: SnapshotProvider3<G> {
    /// Reads the field of the specified 3D scalar variable and returns it by value.
    fn read_scalar_field<S: AsRef<str>>(
        &self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>>;
}

/// Parameters associated with a snapshot.
pub trait SnapshotParameters: Clone {
    /// Returns the number of parameters associated with the snapshot.
    fn n_values(&self) -> usize;

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
        T: From<fpa>,
        U: std::fmt::Display + Copy,
        C: Fn(T) -> U,
    {
        let use_default = |_| {
            eprintln!(
                "Warning: Could not find parameter {}, falling back to default for {}: {}",
                name_in_param_file, display_name, default_value
            );
            default_value
        };
        self.get_value(name_in_param_file)
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
}

#[derive(Clone, Debug)]
/// Value of a snapshot parameter.
pub enum ParameterValue {
    Str(String),
    Int(i64),
    Float(fpa),
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

    /// Try interpreting the parameter value as a float, or return an error if not possible.
    pub fn try_as_float(&self) -> io::Result<fpa> {
        match *self {
            Self::Str(ref s) => match s.parse::<fpa>() {
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
}

/// Wrapper for a `SnapshotProvider3` that resamples the provided fields
/// to a given grid.
pub struct ResampledSnapshotProvider3<GOLD, G, P, T, I> {
    provider: P,
    new_grid: Arc<G>,
    transformation: T,
    resampled_locations: In3D<ResampledCoordLocation>,
    interpolator: I,
    resampling_method: ResamplingMethod,
    verbose: Verbose,
    phantom: PhantomData<GOLD>,
}

impl<GOLD, G, P, T, I> ResampledSnapshotProvider3<GOLD, G, P, T, I>
where
    GOLD: Grid3<fgr>,
    G: Grid3<fgr>,
    P: SnapshotProvider3<GOLD>,
    T: PointTransformation2<fgr>,
    I: Interpolator3,
{
    pub fn new(
        provider: P,
        new_grid: Arc<G>,
        transformation: T,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: I,
        resampling_method: ResamplingMethod,
        verbose: Verbose,
    ) -> Self {
        Self {
            provider,
            new_grid,
            transformation,
            resampled_locations,
            interpolator,
            resampling_method,
            verbose,
            phantom: PhantomData,
        }
    }
}

impl<GOLD, G, P, T, I> ScalarFieldProvider3<fdt, G> for ResampledSnapshotProvider3<GOLD, G, P, T, I>
where
    GOLD: Grid3<fgr>,
    G: Grid3<fgr>,
    P: SnapshotProvider3<GOLD>,
    T: PointTransformation2<fgr>,
    I: Interpolator3,
{
    fn grid(&self) -> &G {
        self.new_grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.new_grid)
    }

    fn produce_scalar_field<S: AsRef<str>>(
        &mut self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
        let field = self.provider.provide_scalar_field(variable_name)?;
        if self.verbose.is_yes() {
            println!("Resampling {}", variable_name);
        }
        Ok(if T::IS_IDENTITY {
            field.resampled_to_grid(
                self.arc_with_grid(),
                self.resampled_locations.clone(),
                &self.interpolator,
                self.resampling_method,
            )
        } else {
            field.resampled_to_transformed_grid(
                self.arc_with_grid(),
                &self.transformation,
                self.resampled_locations.clone(),
                &self.interpolator,
                self.resampling_method,
            )
        })
    }
}

impl<GOLD, G, P, T, I> SnapshotProvider3<G> for ResampledSnapshotProvider3<GOLD, G, P, T, I>
where
    GOLD: Grid3<fgr>,
    G: Grid3<fgr>,
    P: SnapshotProvider3<GOLD>,
    T: PointTransformation2<fgr>,
    I: Interpolator3,
{
    type Parameters = P::Parameters;

    fn parameters(&self) -> &Self::Parameters {
        self.provider.parameters()
    }

    fn endianness(&self) -> Endianness {
        self.provider.endianness()
    }

    fn primary_variable_names(&self) -> &[String] {
        self.provider.primary_variable_names()
    }

    fn auxiliary_variable_names(&self) -> &[String] {
        self.provider.auxiliary_variable_names()
    }

    fn all_variable_names(&self) -> &[String] {
        self.provider.all_variable_names()
    }

    fn has_variable<S: AsRef<str>>(&self, variable_name: S) -> bool {
        self.provider.has_variable(variable_name)
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        self.provider.obtain_snap_name_and_num()
    }
}

/// Wrapper for a `SnapshotProvider3` that extracts a subdomain of the
/// provided fields.
pub struct ExtractedSnapshotProvider3<G, P> {
    provider: P,
    new_grid: Arc<G>,
    lower_indices: Idx3<usize>,
    verbose: Verbose,
}

impl<G, P> ExtractedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    pub fn new(
        provider: P,
        lower_indices: Idx3<usize>,
        upper_indices: Idx3<usize>,
        verbose: Verbose,
    ) -> Self {
        let new_grid = Arc::new(provider.grid().subgrid(&lower_indices, &upper_indices));
        Self {
            provider,
            new_grid,
            lower_indices,
            verbose,
        }
    }
}

impl<G, P> ScalarFieldProvider3<fdt, G> for ExtractedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    fn grid(&self) -> &G {
        self.new_grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.new_grid)
    }

    fn produce_scalar_field<S: AsRef<str>>(
        &mut self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
        let field = self.provider.provide_scalar_field(variable_name)?;
        if self.verbose.is_yes() {
            println!("Extracting {} in subgrid", variable_name);
        }
        Ok(field.subfield(self.arc_with_grid(), &self.lower_indices))
    }
}

impl<G, P> SnapshotProvider3<G> for ExtractedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    type Parameters = P::Parameters;

    fn parameters(&self) -> &Self::Parameters {
        self.provider.parameters()
    }

    fn endianness(&self) -> Endianness {
        self.provider.endianness()
    }

    fn primary_variable_names(&self) -> &[String] {
        self.provider.primary_variable_names()
    }

    fn auxiliary_variable_names(&self) -> &[String] {
        self.provider.auxiliary_variable_names()
    }

    fn all_variable_names(&self) -> &[String] {
        self.provider.all_variable_names()
    }

    fn has_variable<S: AsRef<str>>(&self, variable_name: S) -> bool {
        self.provider.has_variable(variable_name)
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        self.provider.obtain_snap_name_and_num()
    }
}

/// A provider of 3D Bifrost snapshot variables that also supports caching.
pub trait CachingSnapshotProvider3<G: Grid3<fgr>>:
    CachingScalarFieldProvider3<fdt, G> + SnapshotProvider3<G>
{
}

impl<G, C> CachingSnapshotProvider3<G> for C
where
    G: Grid3<fgr>,
    C: CachingScalarFieldProvider3<fdt, G> + SnapshotProvider3<G>,
{
}

impl<G, P> SnapshotProvider3<G> for ScalarFieldCacher3<fdt, G, P>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    type Parameters = P::Parameters;

    fn parameters(&self) -> &Self::Parameters {
        self.provider().parameters()
    }

    fn endianness(&self) -> Endianness {
        self.provider().endianness()
    }

    fn primary_variable_names(&self) -> &[String] {
        self.provider().primary_variable_names()
    }

    fn auxiliary_variable_names(&self) -> &[String] {
        self.provider().auxiliary_variable_names()
    }

    fn all_variable_names(&self) -> &[String] {
        self.provider().all_variable_names()
    }

    fn has_variable<S: AsRef<str>>(&self, variable_name: S) -> bool {
        self.provider().has_variable(variable_name)
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        self.provider().obtain_snap_name_and_num()
    }
}

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
