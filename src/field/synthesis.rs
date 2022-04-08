//! Synthesis of spectral lines.

use std::{
    collections::{hash_map::Keys, HashMap},
    process,
    sync::Arc,
    time::Instant,
};

use ndarray::{Array1, Array2, Array3};
use numpy::{Element, PyArray1, PyArray2, PyArray3};
use pyo3::{types::IntoPyDict, IntoPy, Py, PyAny, PyResult, Python};
use rayon::prelude::*;

use crate::{
    field::ScalarField2,
    geometry::{Coords2, In2D, Point2},
    grid::{regular::RegularGrid2, CoordLocation, Grid2},
    interpolation::Interpolator2,
    io::Verbose,
    num::BFloat,
};

type EmissivityTableMap<F> = HashMap<String, ScalarField2<F, RegularGrid2<F>>>;

pub struct EmissivityTables<F: BFloat> {
    emissivity_tables: EmissivityTableMap<F>,
    table_grid: Arc<RegularGrid2<F>>,
    verbose: Verbose,
}

impl<'a, F> EmissivityTables<F>
where
    F: BFloat + Element + IntoPy<Py<PyAny>> + 'a,
    &'a HashMap<String, Vec<F>>: IntoPyDict,
{
    pub fn new(
        ion_lines: &'a HashMap<String, Vec<F>>,
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (F, F),
        log_electron_density_limits: (F, F),
        verbose: Verbose,
    ) -> Self {
        assert!(n_temperature_points > 1);
        assert!(n_electron_density_points > 1);

        let (log_table_temperatures, log_table_electron_densities, emissivity_tables) =
            Self::compute_emissivity_tables(
                ion_lines,
                n_temperature_points,
                n_electron_density_points,
                log_temperature_limits,
                log_electron_density_limits,
                verbose,
            );

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

        Self {
            emissivity_tables,
            table_grid,
            verbose,
        }
    }

    pub fn evaluate<I: Interpolator2>(
        &self,
        interpolator: &I,
        line_name: &str,
        temperatures: &[F],
        electron_densities: &[F],
    ) -> Array1<F> {
        let n_samples = temperatures.len();
        assert_eq!(electron_densities.len(), n_samples);

        let emissivity_table = exit_on_none!(
            self.emissivity_tables.get(line_name),
            "Error: Invalid line name {}",
            line_name
        );

        if self.verbose.is_yes() {
            println!("Looking up {} emissivities for {}", n_samples, line_name);
        }

        let mut emissivities = Array1::uninit(temperatures.len());
        let emissivities_buffer = emissivities.as_slice_memory_order_mut().unwrap();

        let start_instant = Instant::now();

        emissivities_buffer
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

        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());

        unsafe { emissivities.assume_init() }
    }

    fn compute_emissivity_tables(
        ion_lines: &'a HashMap<String, Vec<F>>,
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (F, F),
        log_electron_density_limits: (F, F),
        verbose: Verbose,
    ) -> (Vec<F>, Vec<F>, HashMap<String, Array2<F>>) {
        Python::with_gil(|py| {
            match Self::compute_emissivity_tables_py(
                py,
                ion_lines,
                n_temperature_points,
                n_electron_density_points,
                log_temperature_limits,
                log_electron_density_limits,
                verbose,
            ) {
                Ok(result) => result,
                Err(err) => {
                    err.print(py);
                    process::exit(1)
                }
            }
        })
    }

    fn compute_emissivity_tables_py(
        py: Python,
        ion_lines: &'a HashMap<String, Vec<F>>,
        n_temperature_points: usize,
        n_electron_density_points: usize,
        log_temperature_limits: (F, F),
        log_electron_density_limits: (F, F),
        verbose: Verbose,
    ) -> PyResult<(Vec<F>, Vec<F>, HashMap<String, Array2<F>>)> {
        let ion_lines_py = ion_lines.into_py_dict(py);

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
        let result = syn
            .getattr("compute_emissivity_tables")?
            .call((ion_lines_py,), Some(kwargs))?;

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
}
