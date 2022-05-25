//! Computation of various derived physical quantities.

use crate::{
    field::{
        self, CachingScalarFieldProvider3, ResampledCoordLocation, ResamplingMethod, ScalarField3,
        ScalarFieldProvider3, VectorField3,
    },
    geometry::{Idx3, In3D},
    grid::{fgr, CoordLocation, Grid3},
    interpolation::{
        poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig},
        Interpolator3,
    },
    io::{
        snapshot::{fdt, CachingSnapshotProvider3, SnapshotProvider3},
        Endianness, Verbose,
    },
    io_result,
    units::solar::{U_B, U_E, U_L3, U_P, U_R, U_T, U_U},
};
use lazy_static::lazy_static;
use rayon::prelude::*;
use regex::Regex;
use std::{collections::HashMap, io, marker::PhantomData, sync::Arc};

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
        ("ux", U_U as fdt),
        ("uy", U_U as fdt),
        ("uz", U_U as fdt),
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
        ("beam_en", ((U_E / U_T) as fdt))
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
#[derive(Clone, Debug)]
pub struct DerivedSnapshotProvider3<G, P> {
    provider: P,
    derived_quantity_names: Vec<String>,
    auxiliary_variable_names: Vec<String>,
    all_variable_names: Vec<String>,
    cached_scalar_fields: HashMap<String, Arc<ScalarField3<fdt, G>>>,
    verbose: Verbose,
    phantom: PhantomData<G>,
}

impl<G, P> DerivedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
{
    /// Creates a computer of derived 3D quantities.
    pub fn new<H>(
        provider: P,
        derived_quantity_names: Vec<String>,
        handle_unavailable: H,
        verbose: Verbose,
    ) -> Self
    where
        H: Fn(&str, Option<Vec<&str>>) + Copy,
    {
        let derived_quantity_names: Vec<_> = derived_quantity_names
            .into_iter()
            .filter(|name| Self::verify_variable_availability(&provider, name, handle_unavailable))
            .collect();

        let mut auxiliary_variable_names: Vec<_> = provider.auxiliary_variable_names().to_vec();
        auxiliary_variable_names.append(&mut derived_quantity_names.clone());

        let mut all_variable_names = provider.primary_variable_names().to_vec();
        all_variable_names.append(&mut auxiliary_variable_names.clone());

        Self {
            provider,
            derived_quantity_names,
            auxiliary_variable_names,
            all_variable_names,
            cached_scalar_fields: HashMap::new(),
            verbose,
            phantom: PhantomData,
        }
    }

    /// Returns the names of the derived quantities that this computer will provide as auxiliary variables.
    pub fn derived_quantity_names(&self) -> &[String] {
        &self.derived_quantity_names
    }

    fn produce_uncached_scalar_field<S: AsRef<str>>(
        &mut self,
        variable_name: S,
    ) -> io::Result<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
        if self.provider.has_variable(variable_name) {
            self.provider.produce_scalar_field(variable_name)
        } else {
            compute_quantity(self, variable_name, self.verbose)
        }
    }

    fn basic_variable_is_available<S: AsRef<str>>(
        provider: &P,
        variable_name: S,
    ) -> (bool, Option<Vec<&str>>) {
        if provider.has_variable(&variable_name) {
            (true, None)
        } else if let Some((_, dependencies)) = DERIVABLE_QUANTITIES.get(variable_name.as_ref()) {
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

    fn verify_variable_availability<S, H>(
        provider: &P,
        variable_name: S,
        handle_unavailable: H,
    ) -> bool
    where
        S: AsRef<str>,
        H: Fn(&str, Option<Vec<&str>>),
    {
        let variable_name = variable_name.as_ref();
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

impl<G, P> ScalarFieldProvider3<fdt, G> for DerivedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
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
        if self.scalar_field_is_cached(variable_name) {
            Ok(self.cached_scalar_field(variable_name).clone())
        } else {
            self.produce_uncached_scalar_field(variable_name)
        }
    }

    fn provide_scalar_field<S: AsRef<str>>(
        &mut self,
        variable_name: S,
    ) -> io::Result<Arc<ScalarField3<fdt, G>>> {
        let variable_name = variable_name.as_ref();
        if self.scalar_field_is_cached(variable_name) {
            Ok(self.arc_with_cached_scalar_field(variable_name).clone())
        } else {
            Ok(Arc::new(self.produce_uncached_scalar_field(variable_name)?))
        }
    }
}

impl<G, P> CachingScalarFieldProvider3<fdt, G> for DerivedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
{
    fn scalar_field_is_cached<S: AsRef<str>>(&self, variable_name: S) -> bool {
        let variable_name = variable_name.as_ref();
        self.cached_scalar_fields.contains_key(variable_name)
            || self.provider.scalar_field_is_cached(variable_name)
    }

    fn vector_field_is_cached<S: AsRef<str>>(&self, variable_name: S) -> bool {
        self.provider.vector_field_is_cached(variable_name)
    }

    fn cache_scalar_field<S: AsRef<str>>(&mut self, variable_name: S) -> io::Result<()> {
        let variable_name = variable_name.as_ref();
        if self.provider.has_variable(variable_name) {
            self.provider.cache_scalar_field(variable_name)
        } else {
            if !self.cached_scalar_fields.contains_key(variable_name) {
                let field = self.provide_scalar_field(variable_name)?;
                if self.verbose.is_yes() {
                    println!("Caching {}", variable_name);
                }
                self.cached_scalar_fields
                    .insert(variable_name.to_string(), field);
            }
            Ok(())
        }
    }

    fn cache_vector_field<S: AsRef<str>>(&mut self, variable_name: S) -> io::Result<()> {
        self.provider.cache_vector_field(variable_name)
    }

    fn arc_with_cached_scalar_field<S: AsRef<str>>(
        &self,
        variable_name: S,
    ) -> &Arc<ScalarField3<fdt, G>> {
        let variable_name = variable_name.as_ref();
        if let Some(field) = self.cached_scalar_fields.get(variable_name) {
            if self.verbose.is_yes() {
                println!("Using cached {}", variable_name);
            }
            field
        } else {
            self.provider.arc_with_cached_scalar_field(variable_name)
        }
    }

    fn arc_with_cached_vector_field<S: AsRef<str>>(
        &self,
        variable_name: S,
    ) -> &Arc<VectorField3<fdt, G>> {
        self.provider.arc_with_cached_vector_field(variable_name)
    }

    fn drop_scalar_field<S: AsRef<str>>(&mut self, variable_name: S) {
        let variable_name = variable_name.as_ref();
        if self.cached_scalar_fields.contains_key(variable_name) {
            if self.verbose.is_yes() {
                println!("Dropping {} from cache", variable_name);
            }
            self.cached_scalar_fields.remove(variable_name);
        } else {
            self.provider.drop_scalar_field(variable_name)
        }
    }

    fn drop_vector_field<S: AsRef<str>>(&mut self, variable_name: S) {
        self.provider.drop_vector_field(variable_name)
    }

    fn drop_all_fields(&mut self) {
        self.cached_scalar_fields.clear();
        self.provider.drop_all_fields()
    }
}

impl<G, P> SnapshotProvider3<G> for DerivedSnapshotProvider3<G, P>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
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

    fn has_variable<S: AsRef<str>>(&self, variable_name: S) -> bool {
        Self::verify_variable_availability(&self.provider, variable_name.as_ref(), |_, _| {})
    }

    fn obtain_snap_name_and_num(&self) -> (String, Option<u32>) {
        self.provider.obtain_snap_name_and_num()
    }
}

/// Computes a derived quantity field using a compute closure on fields from the given provider .
#[macro_export]
macro_rules! compute_derived_quantity {
    ($name:ident, |$dep_name:ident| $computer:expr, $provider:expr, $verbose:expr) => {
        crate::field::quantities::compute_quantity_unary(
            stringify!($name),
            $provider,
            stringify!($dep_name),
            |$dep_name| $computer,
            $verbose,
        )
    };
    ($name:ident, with indices |$indices:ident, $dep_name:ident| $computer:expr, $provider:expr, $verbose:expr) => {
        crate::field::quantities::compute_quantity_unary_with_indices(
            stringify!($name),
            $provider,
            stringify!($dep_name),
            |$indices, $dep_name| $computer,
            $verbose,
        )
    };
    ($name:ident, |$dep_name_1:ident, $dep_name_2:ident| $computer:expr, $provider:expr, $verbose:expr) => {
        crate::field::quantities::compute_quantity_binary(
            stringify!($name),
            $provider,
            stringify!($dep_name_1),
            stringify!($dep_name_2),
            |$dep_name_1, $dep_name_2| $computer,
            $verbose,
        )
    };
    ($name:ident, |$dep_name_1:ident, $dep_name_2:ident, $dep_name_3:ident| $computer:expr, $provider:expr, $verbose:expr) => {
        crate::field::quantities::compute_quantity_tertiary(
            stringify!($name),
            $provider,
            stringify!($dep_name_1),
            stringify!($dep_name_2),
            stringify!($dep_name_3),
            |$dep_name_1, $dep_name_2, $dep_name_3| $computer,
            $verbose,
        )
    };
}

/// Computes the derived quantity field with the given name.
fn compute_quantity<G, P>(
    provider: &mut P,
    quantity_name: &str,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let grid = provider.arc_with_grid();

    if DERIVABLE_QUANTITIES.contains_key(quantity_name) {
        match quantity_name {
            "ux" => compute_derived_quantity!(ux, |px, r| px / r, provider, verbose),
            "uy" => compute_derived_quantity!(uy, |py, r| py / r, provider, verbose),
            "uz" => compute_derived_quantity!(uz, |pz, r| pz / r, provider, verbose),
            "ubeam" => compute_derived_quantity!(ubeam,
                with indices |indices, qbeam| qbeam * grid.grid_cell_volume(indices) as fdt,
                provider, verbose
            ),
            _ => unreachable!(),
        }
    } else if let Some(cgs_base_name) = cgs_base_name(quantity_name) {
        if let Some((x_comp_name, y_comp_name, z_comp_name)) =
            mod_vec_component_names(cgs_base_name)
        {
            if let Some(&scale) = QUANTITY_CGS_SCALES.get(x_comp_name.as_str()) {
                compute_quantity_tertiary(
                    quantity_name,
                    provider,
                    &x_comp_name,
                    &y_comp_name,
                    &z_comp_name,
                    |x_comp, y_comp, z_comp| {
                        (x_comp * x_comp + y_comp * y_comp + z_comp * z_comp).sqrt() * scale
                    },
                    verbose,
                )
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("CGS version of quantity {} not available", cgs_base_name),
                ))
            }
        } else if let Some(centered_base_name) = centered_base_name(cgs_base_name) {
            if let Some(&scale) = QUANTITY_CGS_SCALES.get(centered_base_name) {
                if scale == 1.0 {
                    compute_centered_quantity_unary(
                        quantity_name,
                        provider,
                        centered_base_name,
                        |_| {},
                        verbose,
                    )
                } else {
                    compute_centered_quantity_unary(
                        quantity_name,
                        provider,
                        centered_base_name,
                        |val| *val *= scale,
                        verbose,
                    )
                }
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
            if scale == 1.0 {
                provider
                    .produce_scalar_field(cgs_base_name)
                    .map(|field| field.with_name(quantity_name.to_string()))
            } else {
                compute_quantity_unary(
                    quantity_name,
                    provider,
                    cgs_base_name,
                    |val| val * scale,
                    verbose,
                )
            }
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CGS version of quantity {} not available", cgs_base_name),
            ))
        }
    } else if let Some((x_comp_name, y_comp_name, z_comp_name)) =
        mod_vec_component_names(quantity_name)
    {
        compute_quantity_tertiary(
            quantity_name,
            provider,
            &x_comp_name,
            &y_comp_name,
            &z_comp_name,
            |x_comp, y_comp, z_comp| (x_comp * x_comp + y_comp * y_comp + z_comp * z_comp).sqrt(),
            verbose,
        )
    } else if let Some(centered_base_name) = centered_base_name(quantity_name) {
        compute_centered_quantity_unary(
            quantity_name,
            provider,
            centered_base_name,
            |_| {},
            verbose,
        )
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Quantity {} not supported", quantity_name),
        ))
    }
}

pub fn compute_quantity_unary<G, P, C>(
    quantity_name: &str,
    provider: &mut P,
    dep_name: &str,
    compute: C,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<fdt, G>,
    C: Fn(fdt) -> fdt + Sync,
{
    let field = provider.produce_scalar_field(dep_name)?;

    if verbose.is_yes() {
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

pub fn compute_centered_quantity_unary<G, P, C>(
    quantity_name: &str,
    provider: &mut P,
    dep_name: &str,
    compute: C,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<fdt, G>,
    C: Fn(&mut fdt) + Sync + Send,
{
    let center_locations = In3D::same(CoordLocation::Center);

    let field = provider.provide_scalar_field(dep_name)?;

    let mut centered_field = if field.locations() == &center_locations {
        field.as_ref().clone()
    } else {
        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());

        io_result!(interpolator.verify_grid(provider.grid()))?;

        if verbose.is_yes() {
            println!("Resampling {} to grid cell centers", field.name());
        }
        field.resampled_to_grid(
            provider.arc_with_grid(),
            In3D::same(ResampledCoordLocation::center()),
            &interpolator,
            ResamplingMethod::DirectSampling,
        )
    }
    .with_name(quantity_name.to_string());

    let values = centered_field.values_mut();
    let values_buffer = values.as_slice_memory_order_mut().unwrap();

    values_buffer.par_iter_mut().for_each(compute);

    Ok(centered_field)
}

pub fn compute_quantity_unary_with_indices<G, P, C>(
    quantity_name: &str,
    provider: &mut P,
    dep_name: &str,
    compute: C,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<fdt, G>,
    C: Fn(&Idx3<usize>, fdt) -> fdt + Sync,
{
    let field = provider.produce_scalar_field(dep_name)?;

    if verbose.is_yes() {
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

pub fn compute_quantity_binary<G, P, C>(
    quantity_name: &str,
    provider: &mut P,
    dep_name_1: &str,
    dep_name_2: &str,
    compute: C,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<fdt, G>,
    C: Fn(fdt, fdt) -> fdt + Sync,
{
    let mut field_1 = provider.produce_scalar_field(dep_name_1)?;
    let field_2 = provider.provide_scalar_field(dep_name_2)?;

    let center_locations = In3D::same(CoordLocation::Center);
    let mut locations = field_1.locations().clone();

    if field_1.locations() != field_2.locations() {
        let interpolator = PolyFitInterpolator3::new(PolyFitInterpolatorConfig::default());

        io_result!(interpolator.verify_grid(provider.grid()))?;

        let resample_to_center = |field: &mut ScalarField3<_, _>| {
            if field.locations() != &center_locations {
                if verbose.is_yes() {
                    println!("Resampling {} to grid cell centers", field.name());
                }
                *field = field.resampled_to_grid(
                    provider.arc_with_grid(),
                    In3D::same(ResampledCoordLocation::center()),
                    &interpolator,
                    ResamplingMethod::DirectSampling,
                );
            };
        };
        let mut field_2 = field_2.as_ref().clone();
        resample_to_center(&mut field_1);
        resample_to_center(&mut field_2);
        locations = center_locations;
    }

    if verbose.is_yes() {
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

pub fn compute_quantity_tertiary<G, P, C>(
    quantity_name: &str,
    provider: &mut P,
    dep_name_1: &str,
    dep_name_2: &str,
    dep_name_3: &str,
    compute: C,
    verbose: Verbose,
) -> io::Result<ScalarField3<fdt, G>>
where
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<fdt, G>,
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

        let resample_to_center = |field: &mut ScalarField3<_, _>| {
            if field.locations() != &center_locations {
                if verbose.is_yes() {
                    println!("Resampling {} to grid cell centers", field.name());
                }
                *field = field.resampled_to_grid(
                    provider.arc_with_grid(),
                    In3D::same(ResampledCoordLocation::center()),
                    &interpolator,
                    ResamplingMethod::DirectSampling,
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

    if verbose.is_yes() {
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
