//! Synthesis of spectral lines.

use crate::{
    compute_derived_quantity,
    constants::{AMU, CLIGHT, KBOLTZMANN},
    exit_on_none,
    field::{
        CachingScalarFieldProvider3, ScalarField2, ScalarField3, ScalarFieldProvider3, VectorField3,
    },
    geometry::{Coords2, In2D, In3D, Point2},
    grid::{fgr, regular::RegularGrid2, CoordLocation, Grid2, Grid3},
    interpolation::Interpolator2,
    io::{
        snapshot::{fdt, CachingSnapshotProvider3, SnapshotProvider3},
        Endianness, Verbosity,
    },
    num::BFloat,
    units::solar::U_U,
};
use lazy_static::lazy_static;
use ndarray::{Array2, Array3, ShapeBuilder};
use numpy::{Element, PyArray1, PyArray2};
use pyo3::{
    exceptions::PyValueError, types::IntoPyDict, IntoPy, Py, PyAny, PyErr, PyResult, Python,
};
use rayon::prelude::*;
use regex::Regex;
use roman;
use std::{
    collections::{hash_map::Entry, HashMap},
    env, io,
    marker::PhantomData,
    mem::MaybeUninit,
    process,
    str::FromStr,
    sync::Arc,
};

/// List of names used in CHIANTI for the first atomic elements.
pub const CHIANTI_ELEMENTS: [&str; 36] = [
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl",
    "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as",
    "se", "br", "kr",
];

lazy_static! {
    static ref SYNTHESIZABLE_QUANTITIES: HashMap<&'static str, (&'static str, Vec<&'static str>)> =
        vec![
            (
                "emis",
                (
                    "Emissivity (cell centered)\n\
                     [erg/s/sr/cm³]",
                    vec!["tg", "nel"]
                )
            ),
            (
                "shiftx",
                (
                    "Line profile shift due to bulk velocity in x-direction (cell centered)\n\
                     [cm]",
                    vec!["ux"]
                )
            ),
            (
                "shifty",
                (
                    "Line profile shift due to bulk velocity in y-direction (cell centered)\n\
                     [cm]",
                    vec!["uy"]
                )
            ),
            (
                "shiftz",
                (
                    "Line profile shift due to bulk velocity in z-direction (cell centered)\n\
                     [cm]",
                    vec!["uz"]
                )
            ),
            (
                "vartg",
                (
                    "Line profile variance due to thermal motion (cell centered)\n\
                     [cm²]",
                    vec!["tg"]
                )
            ),
            (
                "vartgshift2x",
                (
                    "vartg + shiftx^2 (cell centered)\n\
                     [cm²]",
                    vec!["tg", "ux"]
                )
            ),
            (
                "vartgshift2y",
                (
                    "vartg + shifty^2 (cell centered)\n\
                     [cm²]",
                    vec!["tg", "uy"]
                )
            ),
            (
                "vartgshift2z",
                (
                    "vartg + shiftz^2 (cell centered)\n\
                     [cm²]",
                    vec!["tg", "uz"]
                )
            ),
            (
                "emis_shiftx",
                (
                    "emis * shiftx (cell centered)\n\
                     [erg/s/sr/cm²]",
                    vec!["emis", "ux"]
                )
            ),
            (
                "emis_shifty",
                (
                    "emis * shifty (cell centered)\n\
                     [erg/s/sr/cm²]",
                    vec!["emis", "uy"]
                )
            ),
            (
                "emis_shiftz",
                (
                    "emis * shiftz (cell centered)\n\
                     [erg/s/sr/cm²]",
                    vec!["emis", "uz"]
                )
            ),
            (
                "emis_vartgshift2x",
                (
                    "emis * vartgshift2x (cell centered)\n\
                     [erg/s/sr/cm]",
                    vec!["emis", "tg", "ux"]
                )
            ),
            (
                "emis_vartgshift2y",
                (
                    "emis * vartgshift2y (cell centered)\n\
                     [erg/s/sr/cm]",
                    vec!["emis", "tg", "uy"]
                )
            ),
            (
                "emis_vartgshift2z",
                (
                    "emis * vartgshift2z (cell centered)\n\
                     [erg/s/sr/cm]",
                    vec!["emis", "tg", "uz"]
                )
            ),
        ]
        .into_iter()
        .collect();
}

lazy_static! {
    /// A string with an overview of synthesizable quantities and their dependencies.
    pub static ref SYNTHESIZABLE_QUANTITY_TABLE_STRING: String =
        create_synthesizable_quantity_table_string();
}

/// Creates a string with an overview of synthesizable quantities and their dependencies.
fn create_synthesizable_quantity_table_string() -> String {
    let mut lines: Vec<_> = SYNTHESIZABLE_QUANTITIES
        .iter()
        .map(|(name, (description, dependencies))| {
            format!(
                "{} - {} (requires: {})",
                name,
                description,
                dependencies.join(", ")
            )
        })
        .collect();
    lines.sort();
    format!(
        "SYNTHESIZABLE QUANTITIES:\n\
        ================================================================================\n\
        Note: For Doppler shifts we are looking along the positive direction of the axis\n\
        ================================================================================\n\
         {}\n\
         ================================================================================",
        lines.join(
            "\n--------------------------------------------------------------------------------\n"
        )
    )
}

lazy_static! {
    static ref EMISSIVITY_QUANTITY_REGEX: Regex =
        Regex::new(r"^([a-zA-Z]+)_([a-zA-Z]+_[0-9ivxlcdmIVXLCDM]+_[0-9]+(:?\.[0-9]*)?)$").unwrap();
}

fn parse_line_quantity_name(line_quantity_name: &str) -> io::Result<(String, String)> {
    if let Some(groups) = EMISSIVITY_QUANTITY_REGEX.captures(line_quantity_name) {
        let quantity_name = groups
            .get(1)
            .expect("Missing capture group")
            .as_str()
            .to_string();
        let line_name = groups
            .get(2)
            .expect("Missing capture group")
            .as_str()
            .to_lowercase();
        Ok((quantity_name, line_name))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Spectral line quantity {} not supported",
                line_quantity_name
            ),
        ))
    }
}

/// Computer of emissivities from Bifrost 3D simulation snapshots.
#[derive(Debug)]
pub struct EmissivitySnapshotProvider3<G, P, I> {
    provider: P,
    interpolator: I,
    all_variable_names: Vec<String>,
    quantity_dependencies: Vec<&'static str>,
    emissivity_tables: Arc<EmissivityTables<fdt>>,
    cached_scalar_fields: HashMap<String, Arc<ScalarField3<fdt, G>>>,
    verbosity: Verbosity,
    phantom: PhantomData<G>,
}

impl<G, P, I> EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
    I: Interpolator2,
{
    /// Creates a computer of emissivities.
    pub fn new(
        provider: P,
        interpolator: I,
        line_names: &[String],
        quantity_names: &[String],
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (fdt, fdt),
        log_electron_density_limits: (fdt, fdt),
        handle_unavailable: &dyn Fn(&str, Option<Vec<&str>>),
        verbosity: Verbosity,
    ) -> Self {
        let quantity_names: Vec<_> = quantity_names
            .iter()
            .filter(|name| Self::verify_quantity_availability(&provider, name, handle_unavailable))
            .collect();

        let emissivity_tables = Arc::new(EmissivityTables::new(
            line_names,
            n_temperature_points,
            n_electron_density_points,
            log_temperature_limits,
            log_electron_density_limits,
            &verbosity,
        ));

        let mut emissivity_quantity_names = Vec::new();
        let mut quantity_dependencies = Vec::new();

        for quantity_name in &quantity_names {
            emissivity_quantity_names.extend(
                line_names
                    .iter()
                    .map(|line_name| format!("{}_{}", quantity_name, line_name)),
            );
            quantity_dependencies
                .append(&mut SYNTHESIZABLE_QUANTITIES[quantity_name.as_str()].1.clone())
        }

        let mut all_variable_names: Vec<_> = provider.all_variable_names().to_vec();
        all_variable_names.append(&mut emissivity_quantity_names);

        Self {
            provider,
            interpolator,
            all_variable_names,
            quantity_dependencies,
            emissivity_tables,
            cached_scalar_fields: HashMap::new(),
            verbosity,
            phantom: PhantomData,
        }
    }

    fn provide_new_scalar_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<Arc<ScalarField3<fdt, G>>> {
        if self.provider.has_variable(variable_name) {
            self.provider.provide_scalar_field(variable_name)
        } else {
            let (quantity_name, line_name) = parse_line_quantity_name(variable_name)?;
            let verbosity = self.verbosity.clone();

            let field = match quantity_name.as_str() {
                "emis" => self.produce_emissivity_field(variable_name, &line_name),
                name if name.starts_with("shift") && name.ends_with(|end| "xyz".contains(end)) => {
                    let doppler_factor = self.compute_doppler_shift_factor(&line_name);
                    match name.chars().last().unwrap() {
                        'x' => compute_derived_quantity!(
                            shiftx,
                            |ux| ux * doppler_factor,
                            self,
                            &verbosity
                        ),
                        'y' => compute_derived_quantity!(
                            shifty,
                            |uy| uy * doppler_factor,
                            self,
                            &verbosity
                        ),
                        'z' => compute_derived_quantity!(
                            shiftz,
                            |uz| uz * doppler_factor,
                            self,
                            &verbosity
                        ),
                        _ => unreachable!(),
                    }
                }
                "vartg" => {
                    let thermal_variance_factor =
                        self.compute_thermal_variance_factor(&line_name)?;
                    compute_derived_quantity!(
                        vartg,
                        |tg| tg * thermal_variance_factor,
                        self,
                        &verbosity
                    )
                }
                name if name.starts_with("emis_shift")
                    && name.ends_with(|end| "xyz".contains(end)) =>
                {
                    let doppler_factor = self.compute_doppler_shift_factor(&line_name);
                    match name.chars().last().unwrap() {
                        'x' => compute_derived_quantity!(
                            emis_shiftx,
                            |emis, ux| emis * ux * doppler_factor,
                            self,
                            &verbosity
                        ),
                        'y' => compute_derived_quantity!(
                            emis_shifty,
                            |emis, uy| emis * uy * doppler_factor,
                            self,
                            &verbosity
                        ),
                        'z' => compute_derived_quantity!(
                            emis_shiftz,
                            |emis, uz| emis * uz * doppler_factor,
                            self,
                            &verbosity
                        ),
                        _ => unreachable!(),
                    }
                }
                name if name.starts_with("vartgshift2")
                    && name.ends_with(|end| "xyz".contains(end)) =>
                {
                    let doppler_factor_squared =
                        self.compute_doppler_shift_factor(&line_name).powi(2);
                    let thermal_variance_factor =
                        self.compute_thermal_variance_factor(&line_name)?;
                    match name.chars().last().unwrap() {
                        'x' => compute_derived_quantity!(
                            vartgshift2x,
                            |tg, ux| tg * thermal_variance_factor
                                + ux * ux * doppler_factor_squared,
                            self,
                            &verbosity
                        ),
                        'y' => compute_derived_quantity!(
                            vartgshift2y,
                            |tg, uy| tg * thermal_variance_factor
                                + uy * uy * doppler_factor_squared,
                            self,
                            &verbosity
                        ),
                        'z' => compute_derived_quantity!(
                            vartgshift2z,
                            |tg, uz| tg * thermal_variance_factor
                                + uz * uz * doppler_factor_squared,
                            self,
                            &verbosity
                        ),
                        _ => unreachable!(),
                    }
                }
                name if name.starts_with("emis_vartgshift2")
                    && name.ends_with(|end| "xyz".contains(end)) =>
                {
                    let doppler_factor_squared =
                        self.compute_doppler_shift_factor(&line_name).powi(2);
                    let thermal_variance_factor =
                        self.compute_thermal_variance_factor(&line_name)?;
                    match name.chars().last().unwrap() {
                        'x' => compute_derived_quantity!(
                            emis_vartgshift2x,
                            |emis, tg, ux| emis
                                * (tg * thermal_variance_factor + ux * ux * doppler_factor_squared),
                            self,
                            &verbosity
                        ),
                        'y' => compute_derived_quantity!(
                            emis_vartgshift2y,
                            |emis, tg, uy| emis
                                * (tg * thermal_variance_factor + uy * uy * doppler_factor_squared),
                            self,
                            &verbosity
                        ),
                        'z' => compute_derived_quantity!(
                            emis_vartgshift2z,
                            |emis, tg, uz| emis
                                * (tg * thermal_variance_factor + uz * uz * doppler_factor_squared),
                            self,
                            &verbosity
                        ),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
            .map(Arc::new)?;

            if self.quantity_dependencies.contains(&quantity_name.as_str()) {
                if self.verbosity.print_messages() {
                    println!("Caching {}", variable_name);
                }
                let existing = self
                    .cached_scalar_fields
                    .insert(variable_name.to_string(), field.clone());
                debug_assert!(existing.is_none());
            }
            Ok(field)
        }
    }

    fn compute_doppler_shift_factor(&self, line_name: &str) -> fdt {
        let central_wavelength = self.emissivity_tables.central_wavelength(line_name);
        central_wavelength * ((U_U / CLIGHT) as fdt)
    }

    fn compute_thermal_variance_factor(&self, line_name: &str) -> io::Result<fdt> {
        let central_wavelength = self.emissivity_tables.central_wavelength(line_name);
        let atomic_mass: fdt = atomic_mass_from_line_name(line_name)?;
        Ok(
            ((KBOLTZMANN as fdt) / atomic_mass)
                * fdt::powi(central_wavelength / (CLIGHT as fdt), 2),
        )
    }

    fn produce_emissivity_field(
        &mut self,
        line_quantity_name: &str,
        line_name: &str,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let temperatures = self.provider.provide_scalar_field("tg")?;
        let electron_densities = self.provider.provide_scalar_field("nel")?;

        if self.verbosity.print_messages() {
            println!("Looking up emissivities for {}", line_name);
        }

        let temperature_buffer = temperatures.values().as_slice_memory_order().unwrap();
        let electron_density_buffer = electron_densities.values().as_slice_memory_order().unwrap();

        let mut emissivities = Array3::uninit(temperatures.values().raw_dim().f());
        let emissivity_buffer = emissivities.as_slice_memory_order_mut().unwrap();

        self.emissivity_tables.evaluate(
            emissivity_buffer,
            &self.interpolator,
            line_name,
            temperature_buffer,
            electron_density_buffer,
        );

        let emissivities = unsafe { emissivities.assume_init() };

        Ok(ScalarField3::new(
            line_quantity_name.to_string(),
            temperatures.arc_with_grid(),
            In3D::same(CoordLocation::Center),
            emissivities,
        ))
    }

    fn quantity_is_available(
        provider: &P,
        quantity_name: &str,
    ) -> (bool, Option<Vec<&'static str>>) {
        if let Some((_, dependencies)) = SYNTHESIZABLE_QUANTITIES.get(quantity_name) {
            let missing_dependencies: Vec<_> = dependencies
                .iter()
                .filter_map(|name| {
                    if SYNTHESIZABLE_QUANTITIES.contains_key(name) || provider.has_variable(name) {
                        None
                    } else {
                        Some(*name)
                    }
                })
                .collect();

            (missing_dependencies.is_empty(), Some(missing_dependencies))
        } else {
            (false, None)
        }
    }

    fn verify_quantity_availability(
        provider: &P,
        quantity_name: &str,
        handle_unavailable: &dyn Fn(&str, Option<Vec<&str>>),
    ) -> bool {
        let (available, missing_dependencies) =
            Self::quantity_is_available(provider, quantity_name);
        if available {
            true
        } else {
            handle_unavailable(quantity_name, missing_dependencies);
            false
        }
    }
}

impl<G, P, I> ScalarFieldProvider3<fdt, G> for EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
    I: Interpolator2,
{
    fn grid(&self) -> &G {
        self.provider.grid()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        self.provider.arc_with_grid()
    }

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<fdt, G>> {
        if self.scalar_field_is_cached(variable_name) {
            Ok(self.cached_scalar_field(variable_name).clone())
        } else {
            match Arc::try_unwrap(self.provide_new_scalar_field(variable_name)?) {
                Ok(owned_field) => Ok(owned_field),
                Err(field_ref) => Ok(field_ref.as_ref().clone()),
            }
        }
    }

    fn provide_scalar_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<Arc<ScalarField3<fdt, G>>> {
        if self.scalar_field_is_cached(variable_name) {
            Ok(self.arc_with_cached_scalar_field(variable_name).clone())
        } else {
            self.provide_new_scalar_field(variable_name)
        }
    }
}

impl<G, P, I> CachingScalarFieldProvider3<fdt, G> for EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
    I: Interpolator2,
{
    fn scalar_field_is_cached(&self, variable_name: &str) -> bool {
        self.cached_scalar_fields.contains_key(variable_name)
            || self.provider.scalar_field_is_cached(variable_name)
    }

    fn vector_field_is_cached(&self, variable_name: &str) -> bool {
        self.provider.vector_field_is_cached(variable_name)
    }

    fn cache_scalar_field(&mut self, variable_name: &str) -> io::Result<()> {
        if self.provider.has_variable(variable_name) {
            self.provider.cache_scalar_field(variable_name)
        } else {
            if !self.cached_scalar_fields.contains_key(variable_name) {
                let field = self.provide_scalar_field(variable_name)?;
                if self.verbosity.print_messages() {
                    println!("Caching {}", variable_name);
                }
                self.cached_scalar_fields
                    .insert(variable_name.to_string(), field);
            }
            Ok(())
        }
    }

    fn cache_vector_field(&mut self, variable_name: &str) -> io::Result<()> {
        self.provider.cache_vector_field(variable_name)
    }

    fn arc_with_cached_scalar_field(&self, variable_name: &str) -> &Arc<ScalarField3<fdt, G>> {
        if let Some(field) = self.cached_scalar_fields.get(variable_name) {
            if self.verbosity.print_messages() {
                println!("Using cached {}", variable_name);
            }
            field
        } else {
            self.provider.arc_with_cached_scalar_field(variable_name)
        }
    }

    fn arc_with_cached_vector_field(&self, variable_name: &str) -> &Arc<VectorField3<fdt, G>> {
        self.provider.arc_with_cached_vector_field(variable_name)
    }

    fn drop_scalar_field(&mut self, variable_name: &str) {
        if self.cached_scalar_fields.contains_key(variable_name) {
            if self.verbosity.print_messages() {
                println!("Dropping {} from cache", variable_name);
            }
            self.cached_scalar_fields.remove(variable_name);
        } else {
            self.provider.drop_scalar_field(variable_name)
        }
    }

    fn drop_vector_field(&mut self, variable_name: &str) {
        self.provider.drop_vector_field(variable_name)
    }

    fn drop_all_fields(&mut self) {
        self.cached_scalar_fields.clear();
        self.provider.drop_all_fields()
    }
}

impl<G, P, I> SnapshotProvider3<G> for EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
    I: Interpolator2,
{
    type Parameters = P::Parameters;

    fn parameters(&self) -> &Self::Parameters {
        self.provider.parameters()
    }

    fn endianness(&self) -> Endianness {
        self.provider.endianness()
    }

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn has_variable(&self, variable_name: &str) -> bool {
        self.all_variable_names.contains(&variable_name.to_string())
            || self.provider.has_variable(variable_name)
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u64>) {
        self.provider.obtain_snap_name_and_num()
    }
}

fn run_python_with_result<C, R>(command: C) -> R
where
    C: FnOnce(Python) -> PyResult<R>,
{
    Python::with_gil(|py| match set_pythonpaths(py).and_then(|_| command(py)) {
        Ok(result) => result,
        Err(err) => {
            err.print(py);
            process::exit(1)
        }
    })
}

fn set_pythonpaths(py: Python) -> PyResult<()> {
    let pythonpaths = py.import("sys")?.getattr("path")?;
    for pythonpath in env!("PYTHONPATH").split(':') {
        pythonpaths.getattr("insert")?.call1((0, pythonpath))?;
    }
    Ok(())
}

macro_rules! with_py_error {
    ($expr:expr, $err_type:ty) => {
        $expr.map_err(|err| PyErr::new::<$err_type, _>(format!("{}", err)))
    };
}

/// Map containing the tuple of central wavelength \[cm\] and table of emissivities \[erg/s/sr/cm³\]
/// associated with each spectral line name.
type EmissivityTableMap<F> = HashMap<String, (F, ScalarField2<F, RegularGrid2<fgr>>)>;

type EmissivityTableArrMap<F> = HashMap<String, (F, Array2<F>)>;

type IonLineNameMap = HashMap<String, Vec<String>>;
type IonLineWavelengthMap<F> = HashMap<String, Vec<F>>;

lazy_static! {
    static ref ION_LINE_REGEX: Regex =
        Regex::new(r"^([a-zA-Z]+)_([0-9ivxlcdmIVXLCDM]+)_([0-9]+(:?\.[0-9]*)?)$").unwrap();
}

/// Holds tables of emissivity as function of temperature and electron density
/// for optically thin spectral lines.
#[derive(Clone, Debug)]
pub struct EmissivityTables<F: BFloat> {
    emissivity_tables: EmissivityTableMap<F>,
}

impl<F> EmissivityTables<F>
where
    F: BFloat + FromStr + Element + IntoPy<Py<PyAny>>,
    HashMap<String, Vec<F>>: IntoPyDict,
    <F as FromStr>::Err: std::fmt::Display,
{
    /// Computes emissivity tables with the given shape and bounds for the given spectral lines,
    /// and stores them in a new `EmissivityTables` object.
    pub fn new(
        line_names: &[String],
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (F, F),
        log_electron_density_limits: (F, F),
        verbosity: &Verbosity,
    ) -> Self {
        assert!(n_temperature_points > 1);
        assert!(n_electron_density_points > 1);

        let (log_table_temperatures, log_table_electron_densities, emissivity_tables) =
            run_python_with_result(|py| {
                Self::compute_emissivity_tables_py(
                    py,
                    line_names,
                    n_temperature_points,
                    n_electron_density_points,
                    log_temperature_limits,
                    log_electron_density_limits,
                    verbosity,
                )
            });

        let half_log_temperature_interval =
            0.5 * (log_table_temperatures[1] - log_table_temperatures[0]);
        let half_log_electron_density_interval =
            0.5 * (log_table_electron_densities[1] - log_table_electron_densities[0]);
        let lower_log_temperatures = log_table_temperatures
            .iter()
            .map(|&val| val - half_log_temperature_interval)
            .collect();
        let lower_log_electron_densities = log_table_electron_densities
            .iter()
            .map(|&val| val - half_log_electron_density_interval)
            .collect();
        let table_lower_coords = Coords2::new(lower_log_temperatures, lower_log_electron_densities);
        let table_center_coords =
            Coords2::new(log_table_temperatures, log_table_electron_densities);

        let table_grid = Arc::new(RegularGrid2::from_coords(
            table_center_coords,
            table_lower_coords,
            In2D::same(false),
        ));

        let emissivity_tables = emissivity_tables
            .into_iter()
            .map(|(line_name, (wavelength, emissivities))| {
                (
                    line_name.clone(),
                    (
                        wavelength,
                        ScalarField2::new(
                            line_name,
                            Arc::clone(&table_grid),
                            In2D::same(CoordLocation::Center),
                            emissivities,
                        ),
                    ),
                )
            })
            .collect();

        Self { emissivity_tables }
    }

    /// Returns the central wavelength [cm] for the given spectral line.
    fn central_wavelength(&self, line_name: &str) -> F {
        exit_on_none!(
            self.emissivity_tables.get(line_name),
            "Error: Invalid line name {}",
            line_name
        )
        .0
    }

    /// Provides sampled emissivities for the given spectral line, temperatures and electron densities.
    fn evaluate<I: Interpolator2>(
        &self,
        emissivity_buffer: &mut [MaybeUninit<F>],
        interpolator: &I,
        line_name: &str,
        temperatures: &[F],
        electron_densities: &[F],
    ) {
        let n_samples = temperatures.len();
        assert_eq!(electron_densities.len(), n_samples);

        let emissivity_table = &exit_on_none!(
            self.emissivity_tables.get(line_name),
            "Error: Invalid line name {}",
            line_name
        )
        .1;

        emissivity_buffer
            .par_iter_mut()
            .zip(temperatures)
            .zip(electron_densities)
            .for_each(|((emissivity, &temperature), &electron_density)| {
                let point =
                    Point2::new(temperature.into().log10(), electron_density.into().log10());
                emissivity.write(
                    F::from(
                        interpolator
                            .interp_scalar_field(emissivity_table, &point)
                            .inside_or_moved_or_default(0.0),
                    )
                    .unwrap(),
                );
            });
    }

    fn compute_emissivity_tables_py(
        py: Python,
        line_names: &[String],
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (F, F),
        log_electron_density_limits: (F, F),
        verbosity: &Verbosity,
    ) -> PyResult<(Vec<fgr>, Vec<fgr>, EmissivityTableArrMap<F>)> {
        let kwargs = [
            ("dtype", F::get_dtype(py).into_py(py)),
            ("n_temperature_points", n_temperature_points.into_py(py)),
            (
                "n_electron_density_points",
                n_electron_density_points.into_py(py),
            ),
            ("log_temperature_limits", log_temperature_limits.into_py(py)),
            (
                "log_electron_density_limits",
                log_electron_density_limits.into_py(py),
            ),
            ("verbose", verbosity.print_messages().into_py(py)),
        ]
        .into_py_dict(py);

        let syn = py.import("backstaff.chianti_synthesis")?;

        let (ion_line_name_map, ion_line_wavelength_map) =
            with_py_error!(Self::line_names_to_ion_maps(line_names), PyValueError)?;

        let (
            log_table_temperatures,
            log_table_electron_densities,
            line_names,
            wavelengths,
            emissivity_tables,
        ): (&PyAny, &PyAny, &PyAny, &PyAny, &PyAny) = syn
            .getattr("compute_emissivity_tables")?
            .call(
                (
                    ion_line_name_map.into_py_dict(py),
                    ion_line_wavelength_map.into_py_dict(py),
                ),
                Some(kwargs),
            )?
            .extract()?;

        let log_table_temperatures: &PyArray1<fgr> = log_table_temperatures.extract()?;
        let log_table_electron_densities: &PyArray1<fgr> =
            log_table_electron_densities.extract()?;
        let line_names: Vec<String> = line_names.extract()?;
        let wavelengths: &PyArray1<F> = wavelengths.extract()?;
        let emissivity_tables: Vec<&PyArray2<F>> = emissivity_tables.extract()?;

        let emissivity_tables = line_names
            .into_iter()
            .zip(wavelengths.to_owned_array().into_iter())
            .zip(emissivity_tables.into_iter())
            .map(|((line_name, wavelength), table)| {
                (line_name, (wavelength, table.to_owned_array()))
            })
            .collect();

        Ok((
            log_table_temperatures
                .readonly()
                .to_vec()
                .expect("Array not contiguous"),
            log_table_electron_densities
                .readonly()
                .to_vec()
                .expect("Array not contiguous"),
            emissivity_tables,
        ))
    }

    fn line_names_to_ion_maps(
        line_names: &[String],
    ) -> io::Result<(IonLineNameMap, IonLineWavelengthMap<F>)> {
        let mut ion_line_name_map: IonLineNameMap = HashMap::new();
        let mut ion_line_wavelength_map: IonLineWavelengthMap<F> = HashMap::new();

        for line_name in line_names {
            let (ion_name, _, _, wavelength) = parse_spectral_line_name(line_name)?;
            let line_name = line_name.to_string();

            match ion_line_name_map.entry(ion_name.clone()) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().push(line_name);
                }
                Entry::Vacant(entry) => {
                    entry.insert(vec![line_name]);
                }
            }
            match ion_line_wavelength_map.entry(ion_name) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().push(wavelength);
                }
                Entry::Vacant(entry) => {
                    entry.insert(vec![wavelength]);
                }
            }
        }
        Ok((ion_line_name_map, ion_line_wavelength_map))
    }
}

/// Parses a spectral line name in the format \<element>\_\<ionization stage>\_\<wavelength>
/// and returns a tuple of the ion name (\<element>\_\<ionization stage>), nuclear charge,
/// ionization stage and wavelength.
fn parse_spectral_line_name<F>(line_name: &str) -> io::Result<(String, u32, u32, F)>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    if let Some(groups) = ION_LINE_REGEX.captures(line_name) {
        let element_name = groups
            .get(1)
            .expect("Missing capture group")
            .as_str()
            .to_lowercase();

        let nuclear_charge = if let Some(pos) = CHIANTI_ELEMENTS
            .iter()
            .position(|&valid_element_name| valid_element_name == element_name.as_str())
        {
            (pos + 1) as u32
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Invalid element {} for spectral line {}, supported elements are [{}]",
                    &element_name,
                    line_name,
                    CHIANTI_ELEMENTS.join(", ")
                ),
            ));
        };

        let ionization_stage = groups.get(2).expect("Missing capture group").as_str();
        let ionization_stage_uppercase = ionization_stage.to_uppercase();

        let ionization_stage_value = match roman::from(ionization_stage_uppercase.as_str()) {
            Some(value) => value as u32,
            None => match ionization_stage.parse() {
                Ok(value) => value,
                Err(err) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "Invalid ionization stage for spectral line {}: {}",
                            line_name, err
                        ),
                    ))
                }
            },
        };

        if ionization_stage_value > nuclear_charge {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Ionization stage {} for element {} exceeds maximum of {}",
                    ionization_stage, &element_name, nuclear_charge
                ),
            ));
        }

        let ion_name = format!("{}_{}", element_name, ionization_stage.to_lowercase());

        let wavelength: F = match groups
            .get(3)
            .expect("Missing capture group")
            .as_str()
            .parse()
        {
            Ok(wavelength) => wavelength,
            Err(err) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Invalid wavelength for spectral line {}: {}",
                        line_name, err
                    ),
                ))
            }
        };
        Ok((ion_name, nuclear_charge, ionization_stage_value, wavelength))
    } else {
        return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid format for spectral line {}, must be <element>_<ionization stage>_<wavelength>, e.g. si_iv_1393.755", line_name),
            ));
    }
}

/// Computes atomic mass [g] of the ion for the given spectral line.
fn atomic_mass_from_line_name<F>(line_name: &str) -> io::Result<F>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
{
    let (_, nuclear_charge, _, _) = parse_spectral_line_name::<f32>(line_name)?;
    Ok(F::from_f64(2.0 * AMU as f64 * nuclear_charge as f64).unwrap())
}
