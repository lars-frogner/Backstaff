//! Computation of various derived physical quantities.

use crate::{
    field::{
        self, CachingScalarFieldProvider3, DynCachingScalarFieldProvider3, FieldGrid3,
        ResampledCoordLocation, ResamplingMethod, ScalarField3, ScalarFieldProvider3, VectorField3,
    },
    geometry::{Dim3::Z, Idx3, In3D},
    grid::{CoordLocation, Grid3},
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        InterpGridVerifier3,
    },
    io::{snapshot::fdt, Verbosity},
    io_result,
    units::solar::{U_B, U_E, U_L, U_L3, U_P, U_R, U_T, U_U},
};
use lazy_static::lazy_static;
use ndarray::Axis;
use rayon::prelude::*;
use regex::Regex;
use std::{collections::HashMap, io, sync::Arc};

lazy_static! {
    static ref DERIVABLE_QUANTITIES: HashMap<&'static str, (&'static str, Vec<&'static str>)> =
        vec![
            (
                "ux",
                (
                    "Velocity in x-direction (cell centered)\n\
                 [10 km/s]",
                    vec!["r", "px"]
                )
            ),
            (
                "uy",
                (
                    "Velocity in y-direction (cell centered)\n\
                 [10 km/s]",
                    vec!["r", "py"]
                )
            ),
            (
                "uz",
                (
                    "Velocity in z-direction (cell centered)\n\
                 [10 km/s]",
                    vec!["r", "pz"]
                )
            ),
            (
                "ubeam",
                (
                    "Volume integrated beam heating\n\
                 [energy/time in Bifrost units]",
                    vec!["qbeam"]
                )
            ),
            (
                "coldepth",
                (
                    "Vertical column depth\n\
                 [mass density * length in Bifrost units]",
                    vec!["r"]
                )
            ),
        ]
        .into_iter()
        .collect();
}

lazy_static! {
    static ref QUANTITY_CGS_SCALES: HashMap<&'static str, fdt> = vec![
        ("r", U_R as fdt),
        ("e", U_E as fdt),
        ("px", (U_R * U_U) as fdt),
        ("py", (U_R * U_U) as fdt),
        ("pz", (U_R * U_U) as fdt),
        ("bx", (*U_B as fdt)),
        ("by", (*U_B as fdt)),
        ("bz", (*U_B as fdt)),
        ("p", (U_P as fdt)),
        ("tg", 1.0),
        ("cs", (U_P as fdt)),
        ("nel", 1.0),
        ("ux", (U_U as fdt)),
        ("uy", (U_U as fdt)),
        ("uz", (U_U as fdt)),
        ("dedt", ((U_E / U_T) as fdt)),
        ("qjoule", ((U_E / U_T) as fdt)),
        ("qspitz", ((U_E / U_T) as fdt)),
        ("qvisc", ((U_E / U_T) as fdt)),
        ("qtot", ((U_E / U_T) as fdt)),
        ("qthin", ((U_E / U_T) as fdt)),
        ("qgenrad", ((U_E / U_T) as fdt)),
        ("qpdv", ((U_E / U_T) as fdt)),
        ("qbeam", ((U_E / U_T) as fdt)),
        ("ubeam", ((U_E * U_L3 / U_T) as fdt)),
        ("beam_en", ((U_E / U_T) as fdt)),
        ("coldepth", ((U_R * U_L) as fdt))
    ]
    .into_iter()
    .collect();
}

lazy_static! {
    /// A string with an overview of available quantities and their dependencies.
    pub static ref AVAILABLE_QUANTITY_TABLE_STRING: String =
        create_available_quantity_table_string();
}

lazy_static! {
    static ref CGS_REGEX: Regex = Regex::new(r"^(\w+)_cgs$").unwrap();
    static ref CENTER_REGEX: Regex = Regex::new(r"^(\w+[xyz])c$").unwrap();
    static ref MOD_REGEX: Regex = Regex::new(r"^mod(\w+)$").unwrap();
}

fn cgs_base_name(quantity_name: &str) -> Option<&str> {
    CGS_REGEX
        .captures(quantity_name)
        .map(|groups| groups.get(1).map_or("", |m| m.as_str()))
}

fn centered_base_name(quantity_name: &str) -> Option<&str> {
    CENTER_REGEX
        .captures(quantity_name)
        .map(|groups| groups.get(1).map_or("", |m| m.as_str()))
}

fn mod_vec_component_names(quantity_name: &str) -> Option<(String, String, String)> {
    MOD_REGEX.captures(quantity_name).map(|groups| {
        let vector_name = groups.get(1).map_or("", |m| m.as_str());
        let x_comp_name = format!("{}x", vector_name);
        let y_comp_name = format!("{}y", vector_name);
        let z_comp_name = format!("{}z", vector_name);
        (x_comp_name, y_comp_name, z_comp_name)
    })
}

/// Creates a string with an overview of available quantities and their dependencies.
fn create_available_quantity_table_string() -> String {
    let mut lines: Vec<_> = DERIVABLE_QUANTITIES
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
        "DERIVABLE QUANTITIES:\n\
        ================================================================================\n\
         {}\n\
         ================================================================================\n\
         mod<vector quantity> - Magnitude of any vector quantity (cell centered)\n\
         E.g., write modb to derive magnetic field strength from components bx, by and bz\n\
         --------------------------------------------------------------------------------\n\
         <quantity><component>c - Cell centered version of a vector component quantity\n\
         E.g., write bzc to get bz interpolated to cell centers\n\
         --------------------------------------------------------------------------------\n\
         <quantity>_cgs - Values of a quantity converted to CGS units\n\
         (not available for all quantities)\n\
         ================================================================================",
        lines.join(
            "\n--------------------------------------------------------------------------------\n"
        )
    )
}

/// Computer of derived quantities from Bifrost 3D simulation snapshots.
pub struct DerivedScalarFieldProvider3 {
    provider: DynCachingScalarFieldProvider3<fdt>,
    derived_quantity_names: Vec<String>,
    all_variable_names: Vec<String>,
    cached_scalar_fields: HashMap<String, Arc<ScalarField3<fdt>>>,
    verbosity: Verbosity,
}

impl DerivedScalarFieldProvider3 {
    /// Creates a computer of derived 3D quantities.
    pub fn new(
        provider: DynCachingScalarFieldProvider3<fdt>,
        derived_quantity_names: Vec<String>,
        handle_unavailable: &dyn Fn(&str, Option<Vec<&str>>),
        verbosity: Verbosity,
    ) -> Self {
        let derived_quantity_names: Vec<_> = derived_quantity_names
            .into_iter()
            .filter(|name| Self::verify_variable_availability(&*provider, name, handle_unavailable))
            .collect();

        let mut all_variable_names = provider.all_variable_names().to_vec();
        all_variable_names.append(&mut derived_quantity_names.clone());

        Self {
            provider,
            derived_quantity_names,
            all_variable_names,
            cached_scalar_fields: HashMap::new(),
            verbosity,
        }
    }

    /// Returns a reference to the wrapped provider.
    pub fn provider(&self) -> &dyn CachingScalarFieldProvider3<fdt> {
        &*self.provider
    }

    /// Returns a mutable reference to the wrapped provider.
    pub fn provider_mut(&mut self) -> &mut dyn CachingScalarFieldProvider3<fdt> {
        &mut *self.provider
    }

    /// Returns the names of the derived quantities that this computer will provide as auxiliary variables.
    pub fn derived_quantity_names(&self) -> &[String] {
        &self.derived_quantity_names
    }

    fn produce_uncached_scalar_field(
        &mut self,
        variable_name: &str,
    ) -> io::Result<ScalarField3<fdt>> {
        if self.provider().has_variable(variable_name) {
            self.provider_mut().produce_scalar_field(variable_name)
        } else {
            let verbosity = self.verbosity.clone();
            compute_quantity(self, variable_name, &verbosity)
        }
    }

    fn basic_variable_is_available(
        provider: &dyn CachingScalarFieldProvider3<fdt>,
        variable_name: &str,
    ) -> (bool, Option<Vec<&'static str>>) {
        if provider.has_variable(variable_name) {
            (true, None)
        } else if let Some((_, dependencies)) = DERIVABLE_QUANTITIES.get(variable_name) {
            let missing_dependencies: Vec<_> = dependencies
                .iter()
                .filter_map(|name| {
                    if provider.has_variable(name) || DERIVABLE_QUANTITIES.contains_key(name) {
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

    fn verify_variable_availability(
        provider: &dyn CachingScalarFieldProvider3<fdt>,
        variable_name: &str,
        handle_unavailable: &dyn Fn(&str, Option<Vec<&str>>),
    ) -> bool {
        let (available, missing_dependencies) =
            Self::basic_variable_is_available(provider, variable_name);
        if available {
            true
        } else if let Some(cgs_base_name) = cgs_base_name(variable_name) {
            if let Some((x_comp_name, y_comp_name, z_comp_name)) =
                mod_vec_component_names(cgs_base_name)
            {
                let (available_x, missing_dependencies_x) =
                    Self::basic_variable_is_available(provider, &x_comp_name);
                let (available_y, missing_dependencies_y) =
                    Self::basic_variable_is_available(provider, &y_comp_name);
                let (available_z, missing_dependencies_z) =
                    Self::basic_variable_is_available(provider, &z_comp_name);
                if available_x && available_y && available_z {
                    true
                } else {
                    let mut missing_dependencies = Vec::new();
                    if let Some(m) = missing_dependencies_x {
                        missing_dependencies.extend(m);
                    }
                    if let Some(m) = missing_dependencies_y {
                        missing_dependencies.extend(m);
                    }
                    if let Some(m) = missing_dependencies_z {
                        missing_dependencies.extend(m);
                    }
                    handle_unavailable(
                        variable_name,
                        if missing_dependencies.is_empty() {
                            None
                        } else {
                            Some(missing_dependencies)
                        },
                    );
                    false
                }
            } else if let Some(centered_base_name) = centered_base_name(cgs_base_name) {
                let (available, missing_dependencies) =
                    Self::basic_variable_is_available(provider, centered_base_name);
                if available {
                    true
                } else {
                    handle_unavailable(variable_name, missing_dependencies);
                    false
                }
            } else {
                let (available, missing_dependencies) =
                    Self::basic_variable_is_available(provider, cgs_base_name);
                if available {
                    true
                } else {
                    handle_unavailable(variable_name, missing_dependencies);
                    false
                }
            }
        } else if let Some((x_comp_name, y_comp_name, z_comp_name)) =
            mod_vec_component_names(variable_name)
        {
            let (available_x, missing_dependencies_x) =
                Self::basic_variable_is_available(provider, &x_comp_name);
            let (available_y, missing_dependencies_y) =
                Self::basic_variable_is_available(provider, &y_comp_name);
            let (available_z, missing_dependencies_z) =
                Self::basic_variable_is_available(provider, &z_comp_name);
            if available_x && available_y && available_z {
                true
            } else {
                let mut missing_dependencies = Vec::new();
                if let Some(m) = missing_dependencies_x {
                    missing_dependencies.extend(m);
                }
                if let Some(m) = missing_dependencies_y {
                    missing_dependencies.extend(m);
                }
                if let Some(m) = missing_dependencies_z {
                    missing_dependencies.extend(m);
                }
                handle_unavailable(
                    variable_name,
                    if missing_dependencies.is_empty() {
                        None
                    } else {
                        Some(missing_dependencies)
                    },
                );
                false
            }
        } else if let Some(centered_base_name) = centered_base_name(variable_name) {
            let (available, missing_dependencies) =
                Self::basic_variable_is_available(provider, centered_base_name);
            if available {
                true
            } else {
                handle_unavailable(variable_name, missing_dependencies);
                false
            }
        } else {
            handle_unavailable(variable_name, missing_dependencies);
            false
        }
    }
}

impl ScalarFieldProvider3<fdt> for DerivedScalarFieldProvider3 {
    fn grid(&self) -> &FieldGrid3 {
        self.provider().grid()
    }

    fn arc_with_grid(&self) -> Arc<FieldGrid3> {
        self.provider().arc_with_grid()
    }

    fn all_variable_names(&self) -> &[String] {
        &self.all_variable_names
    }

    fn has_variable(&self, variable_name: &str) -> bool {
        Self::verify_variable_availability(self.provider(), variable_name, &|_, _| {})
    }

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<fdt>> {
        if self.scalar_field_is_cached(variable_name) {
            Ok(self.cached_scalar_field(variable_name).clone())
        } else {
            self.produce_uncached_scalar_field(variable_name)
        }
    }

    fn provide_scalar_field(&mut self, variable_name: &str) -> io::Result<Arc<ScalarField3<fdt>>> {
        if self.scalar_field_is_cached(variable_name) {
            Ok(self.arc_with_cached_scalar_field(variable_name).clone())
        } else {
            Ok(Arc::new(self.produce_uncached_scalar_field(variable_name)?))
        }
    }
}

impl CachingScalarFieldProvider3<fdt> for DerivedScalarFieldProvider3 {
    fn scalar_field_is_cached(&self, variable_name: &str) -> bool {
        self.cached_scalar_fields.contains_key(variable_name)
            || self.provider().scalar_field_is_cached(variable_name)
    }

    fn vector_field_is_cached(&self, variable_name: &str) -> bool {
        self.provider().vector_field_is_cached(variable_name)
    }

    fn cache_scalar_field(&mut self, variable_name: &str) -> io::Result<()> {
        if self.provider().has_variable(variable_name) {
            self.provider_mut().cache_scalar_field(variable_name)
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
        self.provider_mut().cache_vector_field(variable_name)
    }

    fn arc_with_cached_scalar_field(&self, variable_name: &str) -> &Arc<ScalarField3<fdt>> {
        if let Some(field) = self.cached_scalar_fields.get(variable_name) {
            if self.verbosity.print_messages() {
                println!("Using cached {}", variable_name);
            }
            field
        } else {
            self.provider().arc_with_cached_scalar_field(variable_name)
        }
    }

    fn arc_with_cached_vector_field(&self, variable_name: &str) -> &Arc<VectorField3<fdt>> {
        self.provider().arc_with_cached_vector_field(variable_name)
    }

    fn drop_scalar_field(&mut self, variable_name: &str) {
        if self.cached_scalar_fields.contains_key(variable_name) {
            if self.verbosity.print_messages() {
                println!("Dropping {} from cache", variable_name);
            }
            self.cached_scalar_fields.remove(variable_name);
        } else {
            self.provider_mut().drop_scalar_field(variable_name)
        }
    }

    fn drop_vector_field(&mut self, variable_name: &str) {
        self.provider_mut().drop_vector_field(variable_name)
    }

    fn drop_all_fields(&mut self) {
        self.cached_scalar_fields.clear();
        self.provider_mut().drop_all_fields()
    }
}

/// Computes a derived quantity field using a compute closure on fields from the given provider .
macro_rules! compute_derived_quantity {
    ($name:ident, |$dep_name:ident| $computer:expr, $provider:expr, $verbosity:expr) => {
        $crate::field::quantities::compute_general_single_dep_quantity(
            stringify!($name),
            $provider,
            stringify!($dep_name),
            |$dep_name| $computer,
            $verbosity,
        )
    };
    ($name:ident, with indices |$indices:ident, $dep_name:ident| $computer:expr, $provider:expr, $verbosity:expr) => {
        $crate::field::quantities::compute_general_single_dep_quantity_with_indices(
            stringify!($name),
            $provider,
            stringify!($dep_name),
            |$indices, $dep_name| $computer,
            $verbosity,
        )
    };
    ($name:ident, |$dep_name_1:ident, $dep_name_2:ident| $computer:expr, $provider:expr, $verbosity:expr) => {
        $crate::field::quantities::compute_general_double_dep_quantity(
            stringify!($name),
            $provider,
            stringify!($dep_name_1),
            stringify!($dep_name_2),
            |$dep_name_1, $dep_name_2| $computer,
            $verbosity,
        )
    };
    ($name:ident, |$dep_name_1:ident, $dep_name_2:ident, $dep_name_3:ident| $computer:expr, $provider:expr, $verbosity:expr) => {
        $crate::field::quantities::compute_general_triple_dep_quantity(
            stringify!($name),
            $provider,
            stringify!($dep_name_1),
            stringify!($dep_name_2),
            stringify!($dep_name_3),
            |$dep_name_1, $dep_name_2, $dep_name_3| $computer,
            $verbosity,
        )
    };
}

/// Computes the derived quantity field with the given name.
fn compute_quantity(
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    quantity_name: &str,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    let grid = provider.arc_with_grid();

    if DERIVABLE_QUANTITIES.contains_key(quantity_name) {
        match quantity_name {
            "ux" => compute_quantity_quotient("ux", provider, "px", "r", 1.0, verbosity),
            "uy" => compute_quantity_quotient("uy", provider, "py", "r", 1.0, verbosity),
            "uz" => compute_quantity_quotient("uz", provider, "pz", "r", 1.0, verbosity),
            "ubeam" => compute_derived_quantity!(ubeam,
                with indices |indices, qbeam| qbeam * grid.grid_cell_volume(indices) as fdt,
                provider, verbosity
            ),
            "coldepth" => compute_column_depth(provider, verbosity),
            _ => unreachable!(),
        }
    } else if let Some(cgs_base_name) = cgs_base_name(quantity_name) {
        if let Some((x_comp_name, y_comp_name, z_comp_name)) =
            mod_vec_component_names(cgs_base_name)
        {
            if let Some(&scale) = QUANTITY_CGS_SCALES.get(x_comp_name.as_str()) {
                compute_vector_quantity_norm(
                    quantity_name,
                    provider,
                    &x_comp_name,
                    &y_comp_name,
                    &z_comp_name,
                    scale,
                    verbosity,
                )
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("CGS version of quantity {} not available", cgs_base_name),
                ))
            }
        } else if let Some(centered_base_name) = centered_base_name(cgs_base_name) {
            if let Some(&scale) = QUANTITY_CGS_SCALES.get(centered_base_name) {
                compute_centered_quantity(
                    quantity_name,
                    provider,
                    centered_base_name,
                    scale,
                    verbosity,
                )
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "CGS version of quantity {} not available",
                        centered_base_name
                    ),
                ))
            }
        } else if let Some(&scale) = QUANTITY_CGS_SCALES.get(cgs_base_name) {
            compute_scaled_quantity(quantity_name, provider, cgs_base_name, scale, verbosity)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CGS version of quantity {} not available", cgs_base_name),
            ))
        }
    } else if let Some((x_comp_name, y_comp_name, z_comp_name)) =
        mod_vec_component_names(quantity_name)
    {
        compute_vector_quantity_norm(
            quantity_name,
            provider,
            &x_comp_name,
            &y_comp_name,
            &z_comp_name,
            1.0,
            verbosity,
        )
    } else if let Some(centered_base_name) = centered_base_name(quantity_name) {
        compute_centered_quantity(quantity_name, provider, centered_base_name, 1.0, verbosity)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Quantity {} unavailable and can not be derived",
                quantity_name
            ),
        ))
    }
}

pub fn compute_scaled_quantity(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name: &str,
    scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    if scale == 1.0 {
        provider
            .produce_scalar_field(dep_name)
            .map(|field| field.with_name(quantity_name.to_string()))
    } else {
        compute_general_single_dep_quantity(
            quantity_name,
            provider,
            dep_name,
            |value| scale * value,
            verbosity,
        )
    }
}

pub fn compute_centered_quantity(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name: &str,
    scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    if scale == 1.0 {
        compute_centered_general_single_dep_quantity(
            quantity_name,
            provider,
            dep_name,
            |_| {},
            verbosity,
        )
    } else {
        compute_centered_general_single_dep_quantity(
            quantity_name,
            provider,
            dep_name,
            |value| *value *= scale,
            verbosity,
        )
    }
}

pub fn compute_quantity_product(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    factor_1_name: &str,
    factor_2_name: &str,
    scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    compute_general_double_dep_quantity(
        quantity_name,
        provider,
        factor_1_name,
        factor_2_name,
        |factor_1, factor_2| scale * (factor_1 * factor_2),
        verbosity,
    )
}

pub fn compute_quantity_quotient(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dividend_name: &str,
    divisor_name: &str,
    scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    compute_general_double_dep_quantity(
        quantity_name,
        provider,
        dividend_name,
        divisor_name,
        |dividend, divisor| scale * (dividend / divisor),
        verbosity,
    )
}

pub fn compute_vector_quantity_norm(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    x_component_name: &str,
    y_component_name: &str,
    z_component_name: &str,
    scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    compute_general_triple_dep_quantity(
        quantity_name,
        provider,
        x_component_name,
        y_component_name,
        z_component_name,
        |x_comp, y_comp, z_comp| {
            scale * (x_comp * x_comp + y_comp * y_comp + z_comp * z_comp).sqrt()
        },
        verbosity,
    )
}

pub fn compute_sum_of_single_and_squared_term_quantity(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    single_name: &str,
    squared_name: &str,
    single_scale: fdt,
    squared_scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    compute_general_double_dep_quantity(
        quantity_name,
        provider,
        single_name,
        squared_name,
        |single, squared| single_scale * single + squared_scale * squared * squared,
        verbosity,
    )
}

pub fn compute_sum_of_single_and_squared_term_quantity_product(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    factor_name: &str,
    single_name: &str,
    squared_name: &str,
    single_scale: fdt,
    squared_scale: fdt,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    compute_general_triple_dep_quantity(
        quantity_name,
        provider,
        factor_name,
        single_name,
        squared_name,
        |factor, single, squared| {
            factor * (single_scale * single + squared_scale * squared * squared)
        },
        verbosity,
    )
}

pub fn compute_general_single_dep_quantity<C>(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name: &str,
    compute: C,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>>
where
    C: Fn(fdt) -> fdt + Sync,
{
    let field = provider.produce_scalar_field(dep_name)?;

    if verbosity.print_messages() {
        println!("Computing {}", quantity_name);
    }

    let locations = field.locations().clone();
    let mut values = field.into_values();
    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer
        .par_iter_mut()
        .for_each(|value| *value = compute(*value));

    Ok(ScalarField3::new(
        quantity_name.to_string(),
        provider.arc_with_grid(),
        locations,
        values,
    ))
}

pub fn compute_centered_general_single_dep_quantity<C>(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name: &str,
    compute: C,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>>
where
    C: Fn(&mut fdt) + Sync + Send,
{
    let center_locations = In3D::same(CoordLocation::Center);

    let field = provider.provide_scalar_field(dep_name)?;

    let mut centered_field = if field.locations() == &center_locations {
        field.as_ref().clone()
    } else {
        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());

        io_result!(interpolator.verify_grid(provider.grid()))?;

        if verbosity.print_messages() {
            println!("Resampling {} to grid cell centers", field.name());
        }
        field.resampled_to_grid(
            provider.arc_with_grid(),
            In3D::same(ResampledCoordLocation::center()),
            &interpolator,
            ResamplingMethod::DirectSampling,
            verbosity,
        )
    }
    .with_name(quantity_name.to_string());

    let values = centered_field.values_mut();
    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer.par_iter_mut().for_each(compute);

    Ok(centered_field)
}

pub fn compute_general_single_dep_quantity_with_indices<C>(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name: &str,
    compute: C,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>>
where
    C: Fn(&Idx3<usize>, fdt) -> fdt + Sync,
{
    let field = provider.produce_scalar_field(dep_name)?;

    if verbosity.print_messages() {
        println!("Computing {}", quantity_name);
    }

    let grid = provider.grid();
    let grid_shape = grid.shape();

    let locations = field.locations().clone();
    let mut values = field.into_values();
    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| {
            let indices = field::compute_3d_array_indices_from_flat_idx(grid_shape, idx);
            *value = compute(&indices, *value);
        });

    Ok(ScalarField3::new(
        quantity_name.to_string(),
        provider.arc_with_grid(),
        locations,
        values,
    ))
}

pub fn compute_general_double_dep_quantity<C>(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name_1: &str,
    dep_name_2: &str,
    compute: C,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>>
where
    C: Fn(fdt, fdt) -> fdt + Sync,
{
    let mut field_1 = provider.produce_scalar_field(dep_name_1)?;
    let field_2 = provider.provide_scalar_field(dep_name_2)?;

    let center_locations = In3D::same(CoordLocation::Center);
    let mut locations = field_1.locations().clone();

    if field_1.locations() != field_2.locations() {
        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());

        io_result!(interpolator.verify_grid(provider.grid()))?;

        let resample_to_center = |field: &mut ScalarField3<_>| {
            if field.locations() != &center_locations {
                if verbosity.print_messages() {
                    println!("Resampling {} to grid cell centers", field.name());
                }
                *field = field.resampled_to_grid(
                    provider.arc_with_grid(),
                    In3D::same(ResampledCoordLocation::center()),
                    &interpolator,
                    ResamplingMethod::DirectSampling,
                    verbosity,
                );
            };
        };
        let mut field_2 = field_2.as_ref().clone();
        resample_to_center(&mut field_1);
        resample_to_center(&mut field_2);
        locations = center_locations;
    }

    if verbosity.print_messages() {
        println!("Computing {}", quantity_name);
    }

    let mut values_1 = field_1.into_values();
    let values_1_buffer = values_1.as_slice_memory_order_mut().unwrap();

    let values_2 = field_2.values();
    let values_2_buffer = values_2.as_slice_memory_order().unwrap();

    values_1_buffer
        .par_iter_mut()
        .zip(values_2_buffer)
        .for_each(|(value_1, &value_2)| *value_1 = compute(*value_1, value_2));

    Ok(ScalarField3::new(
        quantity_name.to_string(),
        provider.arc_with_grid(),
        locations,
        values_1,
    ))
}

pub fn compute_general_triple_dep_quantity<C>(
    quantity_name: &str,
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    dep_name_1: &str,
    dep_name_2: &str,
    dep_name_3: &str,
    compute: C,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>>
where
    C: Fn(fdt, fdt, fdt) -> fdt + Sync,
{
    let mut field_1 = provider.produce_scalar_field(dep_name_1)?;
    let field_2 = provider.provide_scalar_field(dep_name_2)?;
    let field_3 = provider.provide_scalar_field(dep_name_3)?;

    let center_locations = In3D::same(CoordLocation::Center);
    let mut locations = field_1.locations().clone();

    if field_1.locations() != field_2.locations() || field_1.locations() != field_3.locations() {
        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());

        io_result!(interpolator.verify_grid(provider.grid()))?;

        let resample_to_center = |field: &mut ScalarField3<_>| {
            if field.locations() != &center_locations {
                if verbosity.print_messages() {
                    println!("Resampling {} to grid cell centers", field.name());
                }
                *field = field.resampled_to_grid(
                    provider.arc_with_grid(),
                    In3D::same(ResampledCoordLocation::center()),
                    &interpolator,
                    ResamplingMethod::DirectSampling,
                    verbosity,
                );
            };
        };
        let mut field_2 = field_2.as_ref().clone();
        let mut field_3 = field_3.as_ref().clone();
        resample_to_center(&mut field_1);
        resample_to_center(&mut field_2);
        resample_to_center(&mut field_3);
        locations = center_locations;
    }

    if verbosity.print_messages() {
        println!("Computing {}", quantity_name);
    }

    let mut values_1 = field_1.into_values();
    let values_1_buffer = values_1.as_slice_memory_order_mut().unwrap();

    let values_2 = field_2.values();
    let values_2_buffer = values_2.as_slice_memory_order().unwrap();

    let values_3 = field_3.values();
    let values_3_buffer = values_3.as_slice_memory_order().unwrap();

    values_1_buffer
        .par_iter_mut()
        .zip(values_2_buffer)
        .zip(values_3_buffer)
        .for_each(|((value_1, &value_2), &value_3)| *value_1 = compute(*value_1, value_2, value_3));

    Ok(ScalarField3::new(
        quantity_name.to_string(),
        provider.arc_with_grid(),
        locations,
        values_1,
    ))
}

pub fn compute_column_depth(
    provider: &mut dyn ScalarFieldProvider3<fdt>,
    verbosity: &Verbosity,
) -> io::Result<ScalarField3<fdt>> {
    let quantity_name = "coldepth";

    let field = provider.produce_scalar_field("r")?;

    if verbosity.print_messages() {
        println!("Computing {}", quantity_name);
    }

    let grid = provider.grid();
    let grid_shape = grid.shape();

    let locations = field.locations().clone();
    let mut values = field.into_values();

    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| {
            let indices = field::compute_3d_array_indices_from_flat_idx(grid_shape, idx);
            *value *= grid.grid_cell_extents(&indices)[Z] as fdt;
        });

    values.accumulate_axis_inplace(Axis(2), |&prev, curr| *curr += prev);

    Ok(ScalarField3::new(
        quantity_name.to_string(),
        provider.arc_with_grid(),
        locations,
        values,
    ))
}
