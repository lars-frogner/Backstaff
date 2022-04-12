//! Synthesis of spectral lines.

use super::{ScalarField3, ScalarFieldProvider3};
use crate::{
    field::ScalarField2,
    geometry::{Coords2, In2D, In3D, Point2},
    grid::{regular::RegularGrid2, CoordLocation, Grid2, Grid3},
    interpolation::Interpolator2,
    io::{
        snapshot::{fdt, SnapshotProvider3},
        Endianness, Verbose,
    },
    num::BFloat,
};
use lazy_static::lazy_static;
use ndarray::{Array2, Array3, ShapeBuilder};
use numpy::{Element, PyArray1, PyArray2};
use pyo3::{
    exceptions::PyValueError, types::IntoPyDict, IntoPy, Py, PyAny, PyErr, PyResult, Python,
};
use rayon::prelude::*;
use regex::Regex;
use std::{
    collections::{hash_map::Entry, HashMap},
    io,
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

/// Moment of a spectral line profile.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum LineProfileMoment {
    Zeroth = 0,
    First = 1,
    Second = 2,
}

/// Computer of emissivities from Bifrost 3D simulation snapshots.
#[derive(Debug)]
pub struct EmissivitySnapshotProvider3<G, P, I> {
    provider: P,
    interpolator: I,
    auxiliary_variable_names: Vec<String>,
    all_variable_names: Vec<String>,
    emissivity_tables: Arc<EmissivityTables<fdt>>,
    verbose: Verbose,
    phantom: PhantomData<G>,
}

impl<G, P, I> EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
    I: Interpolator2,
{
    /// Creates a computer of emissivities.
    pub fn new(
        provider: P,
        interpolator: I,
        line_names: &[String],
        highest_moment: LineProfileMoment,
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (fdt, fdt),
        log_electron_density_limits: (fdt, fdt),
        verbose: Verbose,
    ) -> Self {
        let emissivity_tables = Arc::new(EmissivityTables::new(
            line_names,
            n_temperature_points,
            n_electron_density_points,
            log_temperature_limits,
            log_electron_density_limits,
            verbose,
        ));

        match highest_moment {
            LineProfileMoment::Zeroth => Self::new_for_moment(
                provider,
                interpolator,
                line_names,
                LineProfileMoment::Zeroth,
                emissivity_tables,
                verbose,
            ),
            LineProfileMoment::First => {
                todo!()
            }
            LineProfileMoment::Second => {
                todo!()
            }
        }
    }

    fn new_for_moment(
        provider: P,
        interpolator: I,
        line_names: &[String],
        moment: LineProfileMoment,
        emissivity_tables: Arc<EmissivityTables<fdt>>,
        verbose: Verbose,
    ) -> Self {
        let mut emissivity_quantity_names: Vec<_> = line_names
            .iter()
            .map(|line_name| {
                format!(
                    "{}_{}",
                    match moment {
                        LineProfileMoment::Zeroth => "emis",
                        LineProfileMoment::First => "shift",
                        LineProfileMoment::Second => "var",
                    },
                    line_name
                )
            })
            .collect();

        let mut auxiliary_variable_names: Vec<_> = provider.auxiliary_variable_names().to_vec();
        auxiliary_variable_names.append(&mut emissivity_quantity_names);

        let mut all_variable_names = provider.primary_variable_names().to_vec();
        all_variable_names.append(&mut auxiliary_variable_names.clone());

        Self {
            provider,
            interpolator,
            auxiliary_variable_names,
            all_variable_names,
            emissivity_tables,
            verbose: verbose,
            phantom: PhantomData,
        }
    }

    fn evaluate_emissivities(&mut self, line_name: &str) -> io::Result<ScalarField3<fdt, G>> {
        if self.verbose.is_yes() {
            println!("Synthesizing {}", line_name);
        }

        let temperatures = self.provider.provide_scalar_field("tg")?;
        let electron_densities = self.provider.provide_scalar_field("nel")?;

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
            line_name.to_string(),
            temperatures.arc_with_grid(),
            In3D::same(CoordLocation::Center),
            emissivities,
        ))
    }
}

impl<G, P, I> ScalarFieldProvider3<fdt, G> for EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
    I: Interpolator2,
{
    fn grid(&self) -> &G {
        self.provider.grid()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        self.provider.arc_with_grid()
    }

    fn produce_scalar_field<S: AsRef<str>>(
        &mut self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
        if self.provider.has_variable(variable_name) {
            self.provider.produce_scalar_field(variable_name)
        } else {
            self.evaluate_emissivities(variable_name)
        }
    }
}

impl<G, P, I> SnapshotProvider3<G> for EmissivitySnapshotProvider3<G, P, I>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
    I: Interpolator2,
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
        &self.auxiliary_variable_names
    }

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        self.provider.obtain_snap_name_and_num()
    }
}

fn run_python_with_result<C, R>(command: C) -> R
where
    C: FnOnce(Python) -> PyResult<R>,
{
    Python::with_gil(|py| match command(py) {
        Ok(result) => result,
        Err(err) => {
            err.print(py);
            process::exit(1)
        }
    })
}

macro_rules! with_py_error {
    ($expr:expr, $err_type:ty) => {
        $expr.map_err(|err| PyErr::new::<$err_type, _>(format!("{}", err)))
    };
}

type EmissivityTableMap<F> = HashMap<String, ScalarField2<F, RegularGrid2<F>>>;

lazy_static! {
    static ref ION_LINE_REGEX: Regex =
        Regex::new(r"^([a-zA-Z]+)_([0-9]+)_([0-9]+(:?\.[0-9]*)?)$").unwrap();
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
        verbose: Verbose,
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
                    verbose,
                )
            });

        let half_log_temperature_interval =
            F::from_f32(0.5).unwrap() * (log_table_temperatures[1] - log_table_temperatures[0]);
        let half_log_electron_density_interval = F::from_f32(0.5).unwrap()
            * (log_table_electron_densities[1] - log_table_electron_densities[0]);
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
            .map(|(k, v)| {
                (
                    k.clone(),
                    ScalarField2::new(
                        k,
                        Arc::clone(&table_grid),
                        In2D::same(CoordLocation::Center),
                        v,
                    ),
                )
            })
            .collect();

        Self { emissivity_tables }
    }

    /// Provides sampled emissivities for the given spectral line, temperatures and electron densities.
    pub fn evaluate<I: Interpolator2>(
        &self,
        emissivity_buffer: &mut [MaybeUninit<F>],
        interpolator: &I,
        line_name: &str,
        temperatures: &[F],
        electron_densities: &[F],
    ) {
        let n_samples = temperatures.len();
        assert_eq!(electron_densities.len(), n_samples);

        let emissivity_table = exit_on_none!(
            self.emissivity_tables.get(line_name),
            "Error: Invalid line name {}",
            line_name
        );

        emissivity_buffer
            .par_iter_mut()
            .zip(temperatures)
            .zip(electron_densities)
            .for_each(|((emissivity, &temperature), &electron_density)| {
                let point = Point2::new(temperature.log10(), electron_density.log10());
                emissivity.write(
                    interpolator
                        .interp_extrap_scalar_field(emissivity_table, &point)
                        .expect_inside(),
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
        verbose: Verbose,
    ) -> PyResult<(Vec<F>, Vec<F>, HashMap<String, Array2<F>>)> {
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
            ("verbose", verbose.is_yes().into_py(py)),
        ]
        .into_py_dict(py);

        let syn = py.import("backstaff.chianti_synthesis")?;

        let (ion_line_name_map, ion_line_wavelength_map) =
            with_py_error!(Self::line_names_to_ion_maps(line_names), PyValueError)?;

        let result = syn.getattr("compute_emissivity_tables")?.call(
            (
                ion_line_name_map.into_py_dict(py),
                ion_line_wavelength_map.into_py_dict(py),
            ),
            Some(kwargs),
        )?;

        let (log_table_temperatures, log_table_electron_densities, emissivity_tables): (
            &PyArray1<F>,
            &PyArray1<F>,
            HashMap<String, &PyArray2<F>>,
        ) = result.extract()?;

        let emissivity_tables = emissivity_tables
            .into_iter()
            .map(|(k, v)| (k, v.to_owned_array()))
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
    ) -> io::Result<(HashMap<String, Vec<String>>, HashMap<String, Vec<F>>)> {
        let mut ion_line_name_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut ion_line_wavelength_map: HashMap<String, Vec<F>> = HashMap::new();

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
pub fn parse_spectral_line_name<F, S>(line_name: S) -> io::Result<(String, u32, u32, F)>
where
    F: BFloat + FromStr,
    <F as FromStr>::Err: std::fmt::Display,
    S: AsRef<str>,
{
    let line_name = line_name.as_ref();

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

        let ionization_stage: u32 = match groups
            .get(2)
            .expect("Missing capture group")
            .as_str()
            .parse()
        {
            Ok(ionization_stage) => ionization_stage,
            Err(err) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Invalid ionization stage for spectral line {}: {}",
                        line_name, err
                    ),
                ))
            }
        };

        if ionization_stage > nuclear_charge {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Ionization stage {} for element {} exceeds maximum of {}",
                    ionization_stage, &element_name, nuclear_charge
                ),
            ));
        }

        let ion_name = format!("{}_{}", element_name, ionization_stage);

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
        Ok((ion_name, nuclear_charge, ionization_stage, wavelength))
    } else {
        return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid format for spectral line {}, must be <element>_<ionization stage>_<wavelength>, e.g. si_4_1393.755", line_name),
            ));
    }
}
