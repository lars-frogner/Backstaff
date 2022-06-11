//! Scalar and vector fields.

pub mod quantities;

#[cfg(feature = "synthesis")]
pub mod synthesis;

use crate::{
    geometry::{
        CoordRefs2, CoordRefs3, Coords2, Dim2,
        Dim3::{self, X, Y, Z},
        Idx2, Idx3, In2D, In3D, Point3, PointTransformation2, SimplePolygon2, Vec2, Vec3,
    },
    grid::{fgr, regular::RegularGrid2, CoordLocation, Grid1, Grid2, Grid3, GridPointQuery3},
    interpolation::Interpolator3,
    io::Verbosity,
    num::{BFloat, KeyValueOrderableByValue},
};
use ieee754::Ieee754;
use indicatif::ParallelProgressIterator;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::{
    collections::{hash_map::Entry, HashMap},
    io, iter,
    path::Path,
    sync::Arc,
};
use sysinfo::{RefreshKind, System, SystemExt};

#[cfg(feature = "serialization")]
use serde::Serialize;

#[cfg(feature = "pickle")]
use crate::io::utils::save_data_as_pickle;

#[cfg(feature = "for-testing")]
use approx::{AbsDiffEq, RelativeEq};

#[cfg(feature = "for-testing")]
use crate::num::ComparableSlice;

/// Defines the properties of a provider of 3D scalar fields.
pub trait ScalarFieldProvider3<F: BFloat, G: Grid3<fgr>>: Sync {
    /// Returns a reference to the grid.
    fn grid(&self) -> &G;

    /// Returns a new atomic reference counted pointer to the grid.
    fn arc_with_grid(&self) -> Arc<G>;

    /// Produces the field of the specified 3D scalar variable and returns it by value.
    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<F, G>>;

    /// Provides a reference to the field of the specified 3D scalar variable.
    fn provide_scalar_field(&mut self, variable_name: &str) -> io::Result<Arc<ScalarField3<F, G>>> {
        Ok(Arc::new(self.produce_scalar_field(variable_name)?))
    }

    /// Produces the field of the specified 3D vector variable and returns it by value.
    fn produce_vector_field(&mut self, variable_name: &str) -> io::Result<VectorField3<F, G>> {
        Ok(VectorField3::new(
            variable_name.to_string(),
            self.arc_with_grid(),
            In3D::new(
                self.produce_scalar_field(&format!("{}x", variable_name))
                    .or_else(|_| self.produce_scalar_field(&format!("{}xc", variable_name)))?,
                self.produce_scalar_field(&format!("{}y", variable_name))
                    .or_else(|_| self.produce_scalar_field(&format!("{}yc", variable_name)))?,
                self.produce_scalar_field(&format!("{}z", variable_name))
                    .or_else(|_| self.produce_scalar_field(&format!("{}zc", variable_name)))?,
            ),
        ))
    }

    /// Provides a reference to the field of the specified 3D vector variable.
    fn provide_vector_field(&mut self, variable_name: &str) -> io::Result<Arc<VectorField3<F, G>>> {
        Ok(Arc::new(self.produce_vector_field(variable_name)?))
    }
}

/// Defines the properties of a `ScalarFieldProvider3` wrapper that can
/// be used to cache provided fields.
pub trait CachingScalarFieldProvider3<F, G>: ScalarFieldProvider3<F, G>
where
    F: BFloat,
    G: Grid3<fgr>,
{
    /// Whether the scalar field representing the given variable is cached.
    fn scalar_field_is_cached(&self, variable_name: &str) -> bool;

    /// Whether the vector field representing the given variable is cached.
    fn vector_field_is_cached(&self, variable_name: &str) -> bool;

    /// Makes sure the scalar field representing the given variable is cached.
    fn cache_scalar_field(&mut self, variable_name: &str) -> io::Result<()>;

    /// Makes sure the vector field representing the given variable is cached.
    fn cache_vector_field(&mut self, variable_name: &str) -> io::Result<()>;

    /// Returns a reference to an `Arc` with the scalar field representing the given variable.
    ///
    /// Panics if the field is not cached.
    fn arc_with_cached_scalar_field(&self, variable_name: &str) -> &Arc<ScalarField3<F, G>>;

    /// Returns a reference to the scalar field representing the given variable.
    ///
    /// Panics if the field is not cached.
    fn cached_scalar_field(&self, variable_name: &str) -> &ScalarField3<F, G> {
        self.arc_with_cached_scalar_field(variable_name).as_ref()
    }

    /// Returns a reference to an `Arc` with the vector field representing the given variable.
    ///
    /// Panics if the field is not cached.
    fn arc_with_cached_vector_field(&self, variable_name: &str) -> &Arc<VectorField3<F, G>>;

    /// Returns a reference to the vector field representing the given variable.
    ///
    /// Panics if the field is not cached.
    fn cached_vector_field(&self, variable_name: &str) -> &VectorField3<F, G> {
        self.arc_with_cached_vector_field(variable_name).as_ref()
    }

    /// Removes the scalar field representing the given variable from the cache.
    fn drop_scalar_field(&mut self, variable_name: &str);

    /// Removes the vector field representing the given variable from the cache.
    fn drop_vector_field(&mut self, variable_name: &str);

    /// Removes all cached scalar and vector fields.
    fn drop_all_fields(&mut self);
}

#[derive(Debug)]
enum CachedField<T> {
    ManuallyCached(Arc<T>),
    AutomaticallyCached(Arc<T>),
}

impl<T> CachedField<T> {
    fn field(&self) -> &Arc<T> {
        match self {
            Self::ManuallyCached(field) => field,
            Self::AutomaticallyCached(field) => field,
        }
    }

    fn was_automatically_cached(&self) -> bool {
        match self {
            Self::ManuallyCached(_) => false,
            Self::AutomaticallyCached(_) => true,
        }
    }
}

/// Wrapper for `ScalarFieldProvider3` that automatically caches provided variables.
#[derive(Debug)]
pub struct ScalarFieldCacher3<F, G, P> {
    provider: P,
    max_memory_usage_fraction: f32,
    verbosity: Verbosity,
    system: System,
    scalar_fields: HashMap<String, CachedField<ScalarField3<F, G>>>,
    vector_fields: HashMap<String, Arc<VectorField3<F, G>>>,
    request_counts: HashMap<String, u64>,
}

impl<F, G, P> ScalarFieldCacher3<F, G, P>
where
    F: BFloat,
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<F, G>,
{
    /// Creates a new snapshot cacher from the given provider.
    pub fn new_manual_cacher(provider: P, verbosity: Verbosity) -> Self {
        Self::new_automatic_cacher(provider, 0.0, verbosity)
    }

    /// Creates a new snapshot cacher from the given provider.
    pub fn new_automatic_cacher(provider: P, max_memory_usage: f32, verbosity: Verbosity) -> Self {
        assert!(max_memory_usage >= 0.0);
        Self {
            provider,
            max_memory_usage_fraction: max_memory_usage * 1e-2,
            verbosity,
            system: System::new_with_specifics(RefreshKind::new().with_memory()),
            scalar_fields: HashMap::new(),
            vector_fields: HashMap::new(),
            request_counts: HashMap::new(),
        }
    }

    /// Returns a reference to the wrapped provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Returns a mutable reference to the wrapped provider.
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    fn increment_request_count(&mut self, variable_name: &str) -> u64 {
        match self.request_counts.entry(variable_name.to_string()) {
            Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                *entry += 1;
                *entry
            }
            Entry::Vacant(entry) => {
                entry.insert(1);
                1
            }
        }
    }

    fn least_requested_cached_variable(&self) -> Option<(String, u64)> {
        Iterator::min(self.request_counts.iter().filter_map(|(name, &count)| {
            if let Some(entry) = self.scalar_fields.get(name) {
                if entry.was_automatically_cached() {
                    Some(KeyValueOrderableByValue(name, count))
                } else {
                    None
                }
            } else {
                None
            }
        }))
        .map(|KeyValueOrderableByValue(name, count)| (name.clone(), count))
    }
}

impl<F, G, P> ScalarFieldProvider3<F, G> for ScalarFieldCacher3<F, G, P>
where
    F: BFloat,
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<F, G>,
{
    fn grid(&self) -> &G {
        self.provider.grid()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        self.provider.arc_with_grid()
    }

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<F, G>> {
        Ok(self.provide_scalar_field(variable_name)?.as_ref().clone())
    }

    fn provide_scalar_field(&mut self, variable_name: &str) -> io::Result<Arc<ScalarField3<F, G>>> {
        let request_count = self.increment_request_count(variable_name);

        let (field, variable_to_replace) = match self.scalar_fields.entry(variable_name.to_string())
        {
            Entry::Occupied(entry) => {
                if self.verbosity.print_messages() {
                    println!("Using cached {}", variable_name);
                }
                (Arc::clone(entry.into_mut().field()), None)
            }
            Entry::Vacant(entry) => {
                let field = self.provider.provide_scalar_field(variable_name)?;

                self.system.refresh_memory();

                let available_memory = self.system.available_memory() as f32;
                let total_memory = self.system.total_memory() as f32;
                let used_memory = total_memory - available_memory;
                let used_memory_fraction = used_memory / total_memory;

                let max_memory_exceeded = used_memory_fraction > self.max_memory_usage_fraction;

                if max_memory_exceeded {
                    match self.least_requested_cached_variable() {
                        Some((least_requested_variable, least_requested_count))
                            if request_count > least_requested_count =>
                        {
                            if self.verbosity.print_messages()
                                && self.max_memory_usage_fraction > 0.0
                            {
                                println!(
                                    "Replacing {} with {} in cache ({:.0}% of {:.1} GB memory in use)",
                                    &least_requested_variable,
                                    variable_name,
                                    used_memory_fraction * 1e2,
                                    total_memory * 1e-6,
                                );
                            }
                            (field, Some(least_requested_variable))
                        }
                        _ => {
                            if self.verbosity.print_messages()
                                && self.max_memory_usage_fraction > 0.0
                            {
                                println!(
                                    "Not caching {} ({:.0}% of {:.1} GB memory in use)",
                                    variable_name,
                                    used_memory_fraction * 1e2,
                                    total_memory * 1e-6,
                                );
                            }
                            (field, None)
                        }
                    }
                } else {
                    if self.verbosity.print_messages() {
                        println!("Caching {}", variable_name);
                    }
                    (
                        Arc::clone(
                            entry
                                .insert(CachedField::AutomaticallyCached(field))
                                .field(),
                        ),
                        None,
                    )
                }
            }
        };
        if let Some(variable_to_replace) = variable_to_replace {
            self.drop_scalar_field(&variable_to_replace);

            if self.verbosity.print_messages() {
                println!("Caching {}", variable_name);
            }
            self.scalar_fields.insert(
                variable_name.to_string(),
                CachedField::AutomaticallyCached(field.clone()),
            );
        }
        Ok(field)
    }
}

impl<F, G, P> CachingScalarFieldProvider3<F, G> for ScalarFieldCacher3<F, G, P>
where
    F: BFloat,
    G: Grid3<fgr>,
    P: ScalarFieldProvider3<F, G>,
{
    fn scalar_field_is_cached(&self, variable_name: &str) -> bool {
        self.scalar_fields.contains_key(variable_name)
    }

    fn vector_field_is_cached(&self, variable_name: &str) -> bool {
        self.vector_fields.contains_key(variable_name)
    }

    fn cache_scalar_field(&mut self, variable_name: &str) -> io::Result<()> {
        if !self.scalar_field_is_cached(variable_name) {
            let field = self.provider.provide_scalar_field(variable_name)?;
            if self.verbosity.print_messages() {
                println!("Caching {}", variable_name);
            }
            self.scalar_fields.insert(
                variable_name.to_string(),
                CachedField::ManuallyCached(field),
            );
        }
        Ok(())
    }

    fn cache_vector_field(&mut self, variable_name: &str) -> io::Result<()> {
        if !self.vector_field_is_cached(variable_name) {
            let field = self.provider.provide_vector_field(variable_name)?;
            if self.verbosity.print_messages() {
                println!("Caching {}", variable_name);
            }
            self.vector_fields.insert(variable_name.to_string(), field);
        }
        Ok(())
    }

    fn arc_with_cached_scalar_field(&self, variable_name: &str) -> &Arc<ScalarField3<F, G>> {
        let field = self
            .scalar_fields
            .get(variable_name)
            .expect("Scalar field is not cached")
            .field();
        if self.verbosity.print_messages() {
            println!("Using cached {}", variable_name);
        }
        field
    }

    fn arc_with_cached_vector_field(&self, variable_name: &str) -> &Arc<VectorField3<F, G>> {
        let field = self
            .vector_fields
            .get(variable_name)
            .expect("Vector field is not cached");
        if self.verbosity.print_messages() {
            println!("Using cached {}", variable_name);
        }
        field
    }

    fn drop_scalar_field(&mut self, variable_name: &str) {
        if self.scalar_field_is_cached(variable_name) {
            if self.verbosity.print_messages() {
                println!("Dropping {} from cache", variable_name);
            }
            self.scalar_fields.remove(variable_name);
        }
    }

    fn drop_vector_field(&mut self, variable_name: &str) {
        if self.vector_field_is_cached(variable_name) {
            if self.verbosity.print_messages() {
                println!("Dropping {} from cache", variable_name);
            }
            self.vector_fields.remove(variable_name);
        }
    }

    fn drop_all_fields(&mut self) {
        if self.verbosity.print_messages() {
            println!("Dropping cache");
        }
        self.scalar_fields.clear();
        self.vector_fields.clear();
    }
}

pub type FieldValueComputer<F> = Box<dyn Fn(fgr, fgr, fgr) -> F + Sync>;

/// Object for generating general 3D scalar fields by computing values
/// using provided closures.
pub struct CustomScalarFieldGenerator3<F, G> {
    grid: Arc<G>,
    variable_computers: HashMap<String, (FieldValueComputer<F>, In3D<CoordLocation>)>,
    verbosity: Verbosity,
}

impl<F, G> CustomScalarFieldGenerator3<F, G>
where
    F: BFloat,
    G: Grid3<fgr>,
{
    /// Creates a new generator of 3D scalar fields.
    ///
    /// The fields are generated on the given grid. For each field to
    /// compute, a variable name, closure and coordinate location must
    /// be provided. Each closure computes a value given a point on the
    /// grid.
    pub fn new_with_variables(
        grid: Arc<G>,
        variable_computers: HashMap<String, (FieldValueComputer<F>, In3D<CoordLocation>)>,
        verbosity: Verbosity,
    ) -> Self {
        Self {
            grid,
            variable_computers,
            verbosity,
        }
    }

    pub fn new(grid: Arc<G>, verbosity: Verbosity) -> Self {
        Self::new_with_variables(grid, HashMap::new(), verbosity)
    }

    /// Adds a new cell-centered variable with the given computation closure,
    /// overwriting any existing variable with the same name.
    pub fn with_variable(self, name: String, computer: FieldValueComputer<F>) -> Self {
        self.with_variable_at_locations(name, computer, In3D::same(CoordLocation::Center))
    }

    /// Adds a new variable with the given computation closure and coordinate
    /// locations, overwriting any existing variable with the same name.
    pub fn with_variable_at_locations(
        mut self,
        name: String,
        computer: FieldValueComputer<F>,
        locations: In3D<CoordLocation>,
    ) -> Self {
        self.variable_computers.insert(name, (computer, locations));
        self
    }

    /// Creates a vector with the names of all variables that can be computed.
    pub fn all_variable_names(&self) -> Vec<String> {
        self.variable_computers.keys().cloned().collect()
    }

    fn compute_field(&self, variable_name: &str) -> io::Result<ScalarField3<F, G>> {
        let variable_name = variable_name.to_string();

        if self.verbosity.print_messages() {
            println!("Computing {}", &variable_name);
        }

        let (computer, locations) =
            self.variable_computers.get(&variable_name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Invalid variable name {}", &variable_name),
                )
            })?;

        let grid = self.grid();
        let grid_shape = grid.shape();

        let coords = ScalarField3::<F, G>::coords_from_grid(grid, locations);

        let mut values = Array3::uninit(grid_shape.to_tuple().f());
        let values_buffer = values.as_slice_memory_order_mut().unwrap();

        values_buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let indices = compute_3d_array_indices_from_flat_idx(grid_shape, idx);
                let point = coords.point(&indices);
                value.write(computer(point[X], point[Y], point[Z]));
            });
        let values = unsafe { values.assume_init() };
        Ok(ScalarField3::new(
            variable_name,
            self.arc_with_grid(),
            locations.clone(),
            values,
        ))
    }
}

impl<F, G> ScalarFieldProvider3<F, G> for CustomScalarFieldGenerator3<F, G>
where
    F: BFloat,
    G: Grid3<fgr>,
    FieldValueComputer<F>: Sync,
{
    fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    fn produce_scalar_field(&mut self, variable_name: &str) -> io::Result<ScalarField3<F, G>> {
        self.compute_field(variable_name)
    }
}

#[macro_export]
macro_rules! field_value_computer {
    (constant = $c:expr; ($float_type:ty)) => {
        Box::new(|_, _, _| $c as $float_type)
    };
    (
        $( constant  = $c:expr; )?
        $( slope     = ($ax:expr, $ay:expr, $az:expr); )?
        $( curvature = ($ax2:expr, $axy:expr, $axz:expr,
                        $ayx:expr, $ay2:expr, $ayz:expr,
                        $azx:expr, $azy:expr, $az2:expr); )?
        ($float_type:ty)
    ) => {
        Box::new(|x, y, z| (0.0 $(+ $c)?
                               $(+ $ax * x + $ay * y + $az * z)?
                               $(+ $ax2 * x * x + $axy * x * y + $axz * x * z
                                 + $ayx * y * x + $ay2 * y * y + $ayz * y * z
                                 + $azx * z * x + $azy * z * y + $az2 * z * z)?)
                            as $float_type)
    };
    (|$x:ident, $y:ident, $z:ident| $compute:expr) => {
        Box::new(|$x, $y, $z| $compute)
    };
}

/// Location in the grid cell for resampled field values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResampledCoordLocation {
    Original,
    Specific(CoordLocation),
}

impl ResampledCoordLocation {
    pub fn center() -> Self {
        Self::Specific(CoordLocation::Center)
    }
    pub fn lower_edge() -> Self {
        Self::Specific(CoordLocation::LowerEdge)
    }
    pub fn into_location(self, original: CoordLocation) -> CoordLocation {
        match self {
            Self::Original => original,
            Self::Specific(location) => location,
        }
    }
    pub fn convert_to_locations_3d(
        resampled: In3D<Self>,
        original: &In3D<CoordLocation>,
    ) -> In3D<CoordLocation> {
        In3D::with_each_component(|dim| resampled[dim].into_location(original[dim]))
    }

    pub fn from_locations_3d(locations: &In3D<CoordLocation>) -> In3D<Self> {
        In3D::with_each_component(|dim| Self::Specific(locations[dim]))
    }
}

/// Method for resampling a field.
#[derive(Clone, Copy, Debug)]
pub enum ResamplingMethod {
    SampleAveraging,
    CellAveraging,
    DirectSampling,
}

/// A 3D scalar field.
///
/// Holds the grid and values of a 3D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct ScalarField3<F, G> {
    name: String,
    grid: Arc<G>,
    locations: In3D<CoordLocation>,
    values: Array3<F>,
}

macro_rules! sliding_window {
    ($iter:expr) => {
        $iter.zip($iter.skip(1))
    };
}

impl<F, G> ScalarField3<F, G>
where
    F: BFloat,
    G: Grid3<fgr>,
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate locations specifying where in the grid cell the values are defined.
    pub fn new(
        name: String,
        grid: Arc<G>,
        locations: In3D<CoordLocation>,
        values: Array3<F>,
    ) -> Self {
        let grid_shape = grid.shape();
        let values_shape = values.shape();
        assert!(
            grid_shape[X] == values_shape[0]
                && grid_shape[Y] == values_shape[1]
                && grid_shape[Z] == values_shape[2],
            "Shape of grid does not match shape of array of values."
        );
        Self {
            name,
            grid,
            locations,
            values,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs3<fgr> {
        CoordRefs3::new(
            &self.grid.coords_by_type(self.locations[X])[X],
            &self.grid.coords_by_type(self.locations[Y])[Y],
            &self.grid.coords_by_type(self.locations[Z])[Z],
        )
    }

    /// Returns a reference to the 3D array of field values.
    pub fn values(&self) -> &Array3<F> {
        &self.values
    }

    /// Returns a mutable reference to the 3D array of field values.
    pub fn values_mut(&mut self) -> &mut Array3<F> {
        &mut self.values
    }

    /// Returns the field value at the given 3D index.
    pub fn value(&self, indices: &Idx3<usize>) -> F {
        self.values[[indices[X], indices[Y], indices[Z]]]
    }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> {
        self.grid.shape()
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values are defined.
    pub fn locations(&self) -> &In3D<CoordLocation> {
        &self.locations
    }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array3<F> {
        self.values
    }

    /// Consumes the scalar field and returns a version with the given name.
    pub fn with_name(self, name: String) -> Self {
        Self {
            name,
            grid: self.grid,
            locations: self.locations,
            values: self.values,
        }
    }

    /// Creates a new scalar field restricted to slices of the coordinate arrays of
    /// the original field.
    pub fn subfield(&self, subgrid: Arc<G>, start_indices: &Idx3<usize>) -> Self {
        let subgrid_shape = subgrid.shape();

        let buffer = self.values.slice(s![
            start_indices[X]..(start_indices[X] + subgrid_shape[X]),
            start_indices[Y]..(start_indices[Y] + subgrid_shape[Y]),
            start_indices[Z]..(start_indices[Z] + subgrid_shape[Z])
        ]);
        let values =
            Array::from_shape_vec(buffer.raw_dim().f(), buffer.t().iter().cloned().collect())
                .unwrap();
        Self::new(self.name.clone(), subgrid, self.locations.clone(), values)
    }

    /// Computes the 3D indices and value of the minimum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_minimum(&self) -> Option<(Idx3<usize>, F)> {
        ParallelIterator::min(
            self.values
                .as_slice_memory_order()
                .unwrap()
                .par_iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some(KeyValueOrderableByValue(idx, value))
                    }
                }),
        )
        .map(|KeyValueOrderableByValue(idx_of_min_value, min_value)| {
            (
                compute_3d_array_indices_from_flat_idx(self.shape(), idx_of_min_value),
                min_value,
            )
        })
    }

    /// Computes the 3D indices and value of the maximum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_maximum(&self) -> Option<(Idx3<usize>, F)> {
        ParallelIterator::max(
            self.values
                .as_slice_memory_order()
                .unwrap()
                .par_iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some(KeyValueOrderableByValue(idx, value))
                    }
                }),
        )
        .map(|KeyValueOrderableByValue(idx_of_max_value, max_value)| {
            (
                compute_3d_array_indices_from_flat_idx(self.shape(), idx_of_max_value),
                max_value,
            )
        })
    }

    /// Resamples the scalar field onto the given grid using the given method and
    /// returns the resampled field.
    pub fn resampled_to_grid<H, I>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        method: ResamplingMethod,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        I: Interpolator3,
    {
        match method {
            ResamplingMethod::SampleAveraging => self.resampled_to_grid_with_sample_averaging(
                grid,
                resampled_locations,
                interpolator,
                verbosity,
            ),
            ResamplingMethod::CellAveraging => {
                self.resampled_to_grid_with_cell_averaging(grid, resampled_locations, verbosity)
            }
            ResamplingMethod::DirectSampling => self.resampled_to_grid_with_direct_sampling(
                grid,
                resampled_locations,
                interpolator,
                verbosity,
            ),
        }
    }

    /// Resamples the scalar field onto the given grid using the given method and
    /// returns the resampled field.
    ///
    /// The horizontal components of the grid are transformed with respect to the original
    /// grid using the given point transformation prior to resampling.
    pub fn resampled_to_transformed_grid<H, T, I>(
        &self,
        grid: Arc<H>,
        transformation: &T,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        method: ResamplingMethod,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        T: PointTransformation2<fgr>,
        I: Interpolator3,
    {
        match method {
            ResamplingMethod::SampleAveraging => self
                .resampled_to_transformed_grid_with_sample_averaging(
                    grid,
                    transformation,
                    resampled_locations,
                    interpolator,
                    verbosity,
                ),
            ResamplingMethod::CellAveraging => self
                .resampled_to_transformed_grid_with_cell_averaging(
                    grid,
                    transformation,
                    resampled_locations,
                    verbosity,
                ),
            ResamplingMethod::DirectSampling => self
                .resampled_to_transformed_grid_with_direct_sampling(
                    grid,
                    transformation,
                    resampled_locations,
                    interpolator,
                    verbosity,
                ),
        }
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, values are interpolated from all overlapped original
    /// grid cells and averaged with weights according to the intersected volumes.
    /// If the new grid cell is contained within an original grid cell, this reduces
    /// to a single interpolation.
    ///
    /// This method gives robust results for arbitrary resampling grids, but is slower
    /// than direct sampling or weighted cell averaging.
    pub fn resampled_to_grid_with_sample_averaging<H, I>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        I: Interpolator3,
    {
        let overlying_grid = grid;
        let overlying_locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let mut overlying_values = Array3::uninit(overlying_grid.shape().to_tuple().f());
        let overlying_values_buffer = overlying_values.as_slice_memory_order_mut().unwrap();
        let n_overlying_values = overlying_values_buffer.len();

        overlying_values_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(verbosity.create_progress_bar(n_overlying_values))
            .for_each(|(overlying_idx, overlying_value)| {
                let (mut lower_overlying_corner, mut upper_overlying_corner) =
                    Self::compute_overlying_grid_cell_corners_for_resampling(
                        overlying_grid.as_ref(),
                        overlying_idx,
                    );
                self.shift_overlying_grid_cell_corners_for_sample_averaging(
                    &overlying_locations,
                    &mut lower_overlying_corner,
                    &mut upper_overlying_corner,
                );

                // Get indices and edges of all grid cells fully or partially within by the overlying grid cell,
                // making sure that none of the coordinates are wrapped.

                let (lower_underlying_indices, lower_offsets) =
                    self.determine_underlying_indices_and_offsets(&lower_overlying_corner);
                let (upper_underlying_indices, _) =
                    self.determine_underlying_indices_and_offsets(&upper_overlying_corner);

                let (underlying_indices, underlying_edges) = self
                    .obtain_indexed_monotonic_grid_cell_edges(
                        &lower_underlying_indices,
                        &upper_underlying_indices,
                        &lower_offsets,
                    );

                let compute_overlap_centers_and_lengths_along_dim = |dim| {
                    sliding_window!(
                        iter::once(&lower_overlying_corner[dim]) // First edge is lower edge of overlying cell
                            .chain(
                                // Next are the lower edges of the underlying cells completely inside the overlying cell
                                underlying_edges[dim][1..underlying_edges[dim].len() - 1].iter(),
                            )
                            .chain(iter::once(&upper_overlying_corner[dim]))
                    ) // Create sliding window iterator over edge pairs
                    .map(|(&lower_coord, &upper_coord)| {
                        let overlap_length = upper_coord - lower_coord;
                        let overlap_center = lower_coord + overlap_length * 0.5;
                        (overlap_center, overlap_length)
                    })
                    .collect::<Vec<_>>()
                };

                // Compute the center points and extents of the "sub grid cells" found
                // by intersecting the underlying grid with the overlying grid.
                let overlap_centers_and_lengths =
                    In3D::with_each_component(compute_overlap_centers_and_lengths_along_dim);

                let mut accum_value = 0.0;
                let mut accum_weight = 0.0;

                // Accumulate the interpolated value from each sub grid cell center,
                // weighted with the relative volume of the sub grid cell.
                for (&underlying_k, &(overlap_center_z, overlap_length_z)) in underlying_indices[Z]
                    .iter()
                    .zip(overlap_centers_and_lengths[Z].iter())
                {
                    for (&underlying_j, &(overlap_center_y, overlap_length_y)) in underlying_indices
                        [Y]
                        .iter()
                        .zip(overlap_centers_and_lengths[Y].iter())
                    {
                        for (&underlying_i, &(overlap_center_x, overlap_length_x)) in
                            underlying_indices[X]
                                .iter()
                                .zip(overlap_centers_and_lengths[X].iter())
                        {
                            let weight = overlap_length_x * overlap_length_y * overlap_length_z;
                            let point =
                                Point3::new(overlap_center_x, overlap_center_y, overlap_center_z);
                            let indices = Idx3::new(underlying_i, underlying_j, underlying_k);

                            let wrapped_point = self.grid().wrap_point(&point).unwrap();

                            accum_value += interpolator.interp_scalar_field_known_cell(
                                self,
                                &wrapped_point,
                                &indices,
                            ) as fgr
                                * weight;

                            accum_weight += weight;
                        }
                    }
                }

                assert_ne!(
                    accum_weight, 0.0,
                    "Overlying grid cell is fully outside non-periodic boundary."
                );

                overlying_value.write(F::from(accum_value / accum_weight).unwrap());
            });
        let overlying_values = unsafe { overlying_values.assume_init() };

        ScalarField3::new(
            self.name.clone(),
            overlying_grid,
            overlying_locations,
            overlying_values,
        )
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    /// The horizontal components of the grid are transformed with respect to the original
    /// grid using the given point transformation prior to resampling.
    ///
    /// For each new grid cell, values are interpolated from all overlapped original
    /// grid cells and averaged with weights according to the intersected volumes.
    /// If the new grid cell is contained within an original grid cell, this reduces
    /// to a single interpolation.
    ///
    /// This method gives robust results for arbitrary resampling grids, but is slower
    /// than direct sampling or weighted cell averaging.
    pub fn resampled_to_transformed_grid_with_sample_averaging<H, T, I>(
        &self,
        grid: Arc<H>,
        transformation: &T,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        T: PointTransformation2<fgr>,
        I: Interpolator3,
    {
        const MIN_INTERSECTION_AREA: fgr = 1e-6;

        let overlying_grid = grid;
        let overlying_locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let mut overlying_values = Array3::uninit(overlying_grid.shape().to_tuple().f());
        let overlying_values_buffer = overlying_values.as_slice_memory_order_mut().unwrap();
        let n_overlying_values = overlying_values_buffer.len();

        overlying_values_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(verbosity.create_progress_bar(n_overlying_values))
            .for_each(|(overlying_idx, overlying_value)| {
                let (mut lower_overlying_corner, mut upper_overlying_corner) =
                    Self::compute_overlying_grid_cell_corners_for_resampling(
                        overlying_grid.as_ref(),
                        overlying_idx,
                    );
                self.shift_overlying_grid_cell_corners_for_sample_averaging(
                    &overlying_locations,
                    &mut lower_overlying_corner,
                    &mut upper_overlying_corner,
                );

                // Create polygon corresponding to overlying grid cell in the
                // coordinate system of the underlying grid
                let hor_overlying_grid_cell_polygon =
                    SimplePolygon2::rectangle_from_horizontal_bounds(
                        &lower_overlying_corner.to_vec3(),
                        &upper_overlying_corner.to_vec3(),
                    )
                    .transformed(transformation);

                // Determine axis-aligned bounding box for the overlying grid cell in the
                // coordinate system of the underlying grid
                let (hor_bounding_box_lower_corner, hor_bounding_box_upper_corner) =
                    hor_overlying_grid_cell_polygon.bounds().unwrap();

                let bounding_box_lower_corner = Point3::new(
                    hor_bounding_box_lower_corner[Dim2::X],
                    hor_bounding_box_lower_corner[Dim2::Y],
                    lower_overlying_corner[Z],
                );
                let bounding_box_upper_corner = Point3::new(
                    hor_bounding_box_upper_corner[Dim2::X],
                    hor_bounding_box_upper_corner[Dim2::Y],
                    upper_overlying_corner[Z],
                );

                // Get indices and edges of all grid cells fully or partially within by the bounding box,
                // making sure that none of the coordinates are wrapped.

                let (lower_underlying_indices, lower_offsets) =
                    self.determine_underlying_indices_and_offsets(&bounding_box_lower_corner);
                let (upper_underlying_indices, _) =
                    self.determine_underlying_indices_and_offsets(&bounding_box_upper_corner);

                let (underlying_indices, underlying_edges) = self
                    .obtain_indexed_monotonic_grid_cell_edges(
                        &lower_underlying_indices,
                        &upper_underlying_indices,
                        &lower_offsets,
                    );

                // Compute the center points and areas of the polynomials found by
                // horizontally intersecting the underlying grid with the overlying grid.
                let hor_overlap_indices_areas_and_centers: Vec<_> = underlying_indices[Y]
                    .iter()
                    .zip(sliding_window!(underlying_edges[Y].iter()))
                    .flat_map(|(underlying_j, (lower_edge_y, upper_edge_y))| {
                        underlying_indices[X]
                            .iter()
                            .zip(sliding_window!(underlying_edges[X].iter()))
                            .filter_map(|(underlying_i, (lower_edge_x, upper_edge_x))| {
                                SimplePolygon2::rectangle_from_bounds(
                                    &Vec2::new(*lower_edge_x, *lower_edge_y),
                                    &Vec2::new(*upper_edge_x, *upper_edge_y),
                                )
                                .intersection(&hor_overlying_grid_cell_polygon)
                                .and_then(|intersection_polygon| {
                                    intersection_polygon.area_and_centroid().and_then(
                                        |(area, centroid)| {
                                            // Skip unimportant intersections with very small area,
                                            // which also avoids using possibly unreliable centroids
                                            if area >= MIN_INTERSECTION_AREA {
                                                Some((*underlying_i, *underlying_j, area, centroid))
                                            } else {
                                                None
                                            }
                                        },
                                    )
                                })
                            })
                    })
                    .collect();

                // Compute the center coordinates and extents of the segments found by
                // intersecting the z-components of the underlying grid and overlying grid.
                let overlap_lengths_and_centers_z = sliding_window!(
                    iter::once(&lower_overlying_corner[Z]) // First edge is lower edge of overlying cell
                        .chain(
                            // Next are the lower edges of the underlying cells completely inside the overlying cell
                            underlying_edges[Z][1..underlying_edges[Z].len() - 1].iter(),
                        )
                        .chain(iter::once(&upper_overlying_corner[Z])) // Last edge is the upper edge of the overlying cell
                ) // Create sliding window iterator over edge pairs
                .map(|(&lower_coord, &upper_coord)| {
                    let overlap_length = upper_coord - lower_coord;
                    let overlap_center = lower_coord + overlap_length * 0.5;
                    (overlap_length, overlap_center)
                })
                .collect::<Vec<_>>();

                let mut accum_value = 0.0;
                let mut accum_weight = 0.0;

                // Accumulate the interpolated value from each sub grid cell center,
                // weighted with the relative volume of the sub grid cell.
                for (underlying_k, &(overlap_z_length, overlap_z_center)) in underlying_indices[Z]
                    .iter()
                    .zip(overlap_lengths_and_centers_z.iter())
                {
                    for (underlying_i, underlying_j, hor_overlap_area, hor_overlap_center) in
                        &hor_overlap_indices_areas_and_centers
                    {
                        let weight = *hor_overlap_area * overlap_z_length;
                        let point = Point3::new(
                            hor_overlap_center[Dim2::X],
                            hor_overlap_center[Dim2::Y],
                            overlap_z_center,
                        );
                        let indices = Idx3::new(*underlying_i, *underlying_j, *underlying_k);

                        let wrapped_point = self.grid().wrap_point(&point).unwrap();

                        accum_value += interpolator.interp_scalar_field_known_cell(
                            self,
                            &wrapped_point,
                            &indices,
                        ) as fgr
                            * weight;

                        accum_weight += weight;
                    }
                }

                assert_ne!(
                    accum_weight, 0.0,
                    "Overlying grid cell is fully outside non-periodic boundary."
                );

                overlying_value.write(F::from(accum_value / accum_weight).unwrap());
            });
        let overlying_values = unsafe { overlying_values.assume_init() };

        ScalarField3::new(
            self.name.clone(),
            overlying_grid,
            overlying_locations,
            overlying_values,
        )
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, the values of all overlapped original grid cells are
    /// averaged with weights according to the intersected volumes.
    ///
    /// This method is suited for downsampling. It is faster than weighted sample
    /// averaging, but slightly less accurate.
    pub fn resampled_to_grid_with_cell_averaging<H: Grid3<fgr>>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H> {
        let overlying_grid = grid;
        let overlying_locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let mut overlying_values = Array3::uninit(overlying_grid.shape().to_tuple().f());
        let overlying_values_buffer = overlying_values.as_slice_memory_order_mut().unwrap();
        let n_overlying_values = overlying_values_buffer.len();

        overlying_values_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(verbosity.create_progress_bar(n_overlying_values))
            .for_each(|(overlying_idx, overlying_value)| {
                let (mut lower_overlying_corner, mut upper_overlying_corner) =
                    Self::compute_overlying_grid_cell_corners_for_resampling(
                        overlying_grid.as_ref(),
                        overlying_idx,
                    );
                self.shift_overlying_grid_cell_corners_for_cell_averaging(
                    &overlying_locations,
                    &mut lower_overlying_corner,
                    &mut upper_overlying_corner,
                );

                // Get indices and edges of all grid cells fully or partially within by the overlying grid cell,
                // making sure that none of the coordinates are wrapped.

                let (lower_underlying_indices, lower_offsets) =
                    self.determine_underlying_indices_and_offsets(&lower_overlying_corner);
                let (upper_underlying_indices, _) =
                    self.determine_underlying_indices_and_offsets(&upper_overlying_corner);

                let (underlying_indices, underlying_edges) = self
                    .obtain_indexed_monotonic_grid_cell_edges(
                        &lower_underlying_indices,
                        &upper_underlying_indices,
                        &lower_offsets,
                    );

                let compute_overlap_lengths_along_dim = |dim| {
                    sliding_window!(
                        iter::once(&lower_overlying_corner[dim]) // First edge is lower edge of overlying cell
                            .chain(
                                // Next are the lower edges of the underlying cells completely inside the overlying cell
                                underlying_edges[dim][1..underlying_edges[dim].len() - 1].iter(),
                            )
                            .chain(iter::once(&upper_overlying_corner[dim]))
                    ) // Create sliding window iterator over edge pairs
                    .map(|(&lower_coord, &upper_coord)| upper_coord - lower_coord)
                    .collect::<Vec<_>>()
                };

                // Compute the extents of the "sub grid cells" found by intersecting
                // the underlying grid with the overlying grid
                let overlap_lengths = In3D::with_each_component(compute_overlap_lengths_along_dim);

                let mut accum_value = 0.0;
                let mut accum_weight = 0.0;

                // Accumulate the value from each sub grid cell, weighted with the
                // relative volume of the sub grid cell.
                for (&underlying_k, &overlap_length_z) in
                    underlying_indices[Z].iter().zip(overlap_lengths[Z].iter())
                {
                    for (&underlying_j, &overlap_length_y) in
                        underlying_indices[Y].iter().zip(overlap_lengths[Y].iter())
                    {
                        for (&underlying_i, &overlap_length_x) in
                            underlying_indices[X].iter().zip(overlap_lengths[X].iter())
                        {
                            let weight = overlap_length_x * overlap_length_y * overlap_length_z;
                            let indices = Idx3::new(underlying_i, underlying_j, underlying_k);

                            accum_value += self.value(&indices).into() * weight;
                            accum_weight += weight;
                        }
                    }
                }

                assert_ne!(
                    accum_weight, 0.0,
                    "Overlying grid cell is fully outside non-periodic boundary."
                );

                overlying_value.write(F::from(accum_value / accum_weight).unwrap());
            });
        let overlying_values = unsafe { overlying_values.assume_init() };
        ScalarField3::new(
            self.name.clone(),
            overlying_grid,
            overlying_locations,
            overlying_values,
        )
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    /// The horizontal components of the grid are transformed with respect to the original
    /// grid using the given point transformation prior to resampling.
    ///
    /// For each new grid cell, the values of all overlapped original grid cells are
    /// averaged with weights according to the intersected volumes.
    ///
    /// This method is suited for downsampling. It is faster than weighted sample
    /// averaging, but slightly less accurate.
    pub fn resampled_to_transformed_grid_with_cell_averaging<H, T>(
        &self,
        grid: Arc<H>,
        transformation: &T,
        resampled_locations: In3D<ResampledCoordLocation>,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        T: PointTransformation2<fgr>,
    {
        const MIN_INTERSECTION_AREA: fgr = 1e-6;

        let overlying_grid = grid;
        let overlying_locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let mut overlying_values = Array3::uninit(overlying_grid.shape().to_tuple().f());
        let overlying_values_buffer = overlying_values.as_slice_memory_order_mut().unwrap();
        let n_overlying_values = overlying_values_buffer.len();

        overlying_values_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(verbosity.create_progress_bar(n_overlying_values))
            .for_each(|(overlying_idx, overlying_value)| {
                let (mut lower_overlying_corner, mut upper_overlying_corner) =
                    Self::compute_overlying_grid_cell_corners_for_resampling(
                        overlying_grid.as_ref(),
                        overlying_idx,
                    );
                self.shift_overlying_grid_cell_corners_for_cell_averaging(
                    &overlying_locations,
                    &mut lower_overlying_corner,
                    &mut upper_overlying_corner,
                );

                // Create polygon corresponding to overlying grid cell in the
                // coordinate system of the underlying grid
                let hor_overlying_grid_cell_polygon =
                    SimplePolygon2::rectangle_from_horizontal_bounds(
                        &lower_overlying_corner.to_vec3(),
                        &upper_overlying_corner.to_vec3(),
                    )
                    .transformed(transformation);

                // Determine axis-aligned bounding box for the overlying grid cell in the
                // coordinate system of the underlying grid
                let (hor_bounding_box_lower_corner, hor_bounding_box_upper_corner) =
                    hor_overlying_grid_cell_polygon.bounds().unwrap();

                let bounding_box_lower_corner = Point3::new(
                    hor_bounding_box_lower_corner[Dim2::X],
                    hor_bounding_box_lower_corner[Dim2::Y],
                    lower_overlying_corner[Z],
                );
                let bounding_box_upper_corner = Point3::new(
                    hor_bounding_box_upper_corner[Dim2::X],
                    hor_bounding_box_upper_corner[Dim2::Y],
                    upper_overlying_corner[Z],
                );

                // Get indices and edges of all grid cells fully or partially within by the overlying grid cell,
                // making sure that none of the coordinates are wrapped.

                let (lower_underlying_indices, lower_offsets) =
                    self.determine_underlying_indices_and_offsets(&bounding_box_lower_corner);
                let (upper_underlying_indices, _) =
                    self.determine_underlying_indices_and_offsets(&bounding_box_upper_corner);

                let (underlying_indices, underlying_edges) = self
                    .obtain_indexed_monotonic_grid_cell_edges(
                        &lower_underlying_indices,
                        &upper_underlying_indices,
                        &lower_offsets,
                    );

                // Compute the areas of the polynomials found by  horizontally intersecting
                // the underlying grid with the overlying grid.
                let hor_overlap_indices_and_areas: Vec<_> = underlying_indices[Y]
                    .iter()
                    .zip(sliding_window!(underlying_edges[Y].iter()))
                    .flat_map(|(underlying_j, (lower_edge_y, upper_edge_y))| {
                        underlying_indices[X]
                            .iter()
                            .zip(sliding_window!(underlying_edges[X].iter()))
                            .filter_map(|(underlying_i, (lower_edge_x, upper_edge_x))| {
                                SimplePolygon2::rectangle_from_bounds(
                                    &Vec2::new(*lower_edge_x, *lower_edge_y),
                                    &Vec2::new(*upper_edge_x, *upper_edge_y),
                                )
                                .intersection(&hor_overlying_grid_cell_polygon)
                                .and_then(|intersection_polygon| {
                                    intersection_polygon.area().and_then(|area| {
                                        // Skip unimportant intersections with very small area
                                        if area >= MIN_INTERSECTION_AREA {
                                            Some((*underlying_i, *underlying_j, area))
                                        } else {
                                            None
                                        }
                                    })
                                })
                            })
                    })
                    .collect();

                // Compute the extents of the segments found by intersecting the
                // z-components of the underlying grid and overlying grid.
                let overlap_lengths_z = sliding_window!(
                    iter::once(&lower_overlying_corner[Z]) // First edge is lower edge of overlying cell
                        .chain(
                            // Next are the lower edges of the underlying cells completely inside the overlying cell
                            underlying_edges[Z][1..underlying_edges[Z].len() - 1].iter(),
                        )
                        .chain(iter::once(&upper_overlying_corner[Z]))
                ) // Create sliding window iterator over edge pairs
                .map(|(&lower_coord, &upper_coord)| upper_coord - lower_coord)
                .collect::<Vec<_>>();

                let mut accum_value = 0.0;
                let mut accum_weight = 0.0;

                // Accumulate the value from each sub grid cell, weighted with the
                // relative volume of the sub grid cell.
                for (underlying_k, &overlap_length_z) in
                    underlying_indices[Z].iter().zip(overlap_lengths_z.iter())
                {
                    for (underlying_i, underlying_j, hor_overlap_area) in
                        &hor_overlap_indices_and_areas
                    {
                        let weight = *hor_overlap_area * overlap_length_z;
                        let indices = Idx3::new(*underlying_i, *underlying_j, *underlying_k);

                        accum_value += self.value(&indices).into() * weight;
                        accum_weight += weight;
                    }
                }

                assert_ne!(
                    accum_weight, 0.0,
                    "Overlying grid cell is fully outside non-periodic boundary."
                );

                overlying_value.write(F::from(accum_value / accum_weight).unwrap());
            });
        let overlying_values = unsafe { overlying_values.assume_init() };
        ScalarField3::new(
            self.name.clone(),
            overlying_grid,
            overlying_locations,
            overlying_values,
        )
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    ///
    /// Each value on the new grid is found by interpolation of the values on the old grid
    /// at the new coordinate location.
    ///
    /// This is the preferred method for upsampling. For heavy downsampling it yields a
    /// more noisy result than weighted averaging.
    pub fn resampled_to_grid_with_direct_sampling<H, I>(
        &self,
        grid: Arc<H>,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        I: Interpolator3,
    {
        let locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let new_coords = Self::coords_from_grid(grid.as_ref(), &locations);

        let grid_shape = grid.shape();
        let mut new_values = Array3::uninit(grid_shape.to_tuple().f());
        let values_buffer = new_values.as_slice_memory_order_mut().unwrap();
        let n_values = values_buffer.len();

        values_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(verbosity.create_progress_bar(n_values))
            .for_each(|(idx, value)| {
                let indices = compute_3d_array_indices_from_flat_idx(grid_shape, idx);
                let point = new_coords.point(&indices);
                value.write(
                    F::from(
                        interpolator
                            .interp_scalar_field(self, &point)
                            .expect_inside_or_moved(),
                    )
                    .unwrap(),
                );
            });
        let new_values = unsafe { new_values.assume_init() };
        ScalarField3::new(self.name.clone(), grid, locations, new_values)
    }

    /// Resamples the scalar field onto the given grid and returns the resampled field.
    /// The horizontal components of the grid are transformed with respect to the original
    /// grid using the given point transformation prior to resampling.
    ///
    /// Each value on the new grid is found by interpolation of the values on the old grid
    /// at the new coordinate location.
    ///
    /// This is the preferred method for upsampling. For heavy downsampling it yields a
    /// more noisy result than weighted averaging.
    pub fn resampled_to_transformed_grid_with_direct_sampling<H, T, I>(
        &self,
        grid: Arc<H>,
        transformation: &T,
        resampled_locations: In3D<ResampledCoordLocation>,
        interpolator: &I,
        verbosity: &Verbosity,
    ) -> ScalarField3<F, H>
    where
        H: Grid3<fgr>,
        T: PointTransformation2<fgr>,
        I: Interpolator3,
    {
        let locations =
            ResampledCoordLocation::convert_to_locations_3d(resampled_locations, self.locations());
        let new_coords = Self::coords_from_grid(grid.as_ref(), &locations);

        let grid_shape = grid.shape();
        let mut new_values = Array3::uninit(grid_shape.to_tuple().f());
        let values_buffer = new_values.as_slice_memory_order_mut().unwrap();
        let n_values = values_buffer.len();

        values_buffer
            .par_iter_mut()
            .enumerate()
            .progress_with(verbosity.create_progress_bar(n_values))
            .for_each(|(idx, value)| {
                let indices = compute_3d_array_indices_from_flat_idx(grid_shape, idx);
                let point = new_coords.point(&indices);
                let transformed_point = transformation.transform_horizontally(&point);
                value.write(
                    F::from(
                        interpolator
                            .interp_scalar_field(self, &transformed_point)
                            .expect_inside_or_moved(),
                    )
                    .unwrap(),
                );
            });
        let new_values = unsafe { new_values.assume_init() };
        ScalarField3::new(self.name.clone(), grid, locations, new_values)
    }

    /// Returns a view of the 2D slice of the field located at the given index along the given axis.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> ArrayView2<F> {
        self.values.index_axis(Axis(axis as usize), idx)
    }

    /// Returns a 2D scalar field corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(
        &self,
        interpolator: &I,
        x_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::XSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.slice_across_x();
        self.create_slice_across_x(
            Arc::new(slice_grid),
            interpolator,
            x_coord,
            resampled_location,
        )
    }

    /// Returns a 2D scalar field corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(
        &self,
        interpolator: &I,
        y_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::YSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.slice_across_y();
        self.create_slice_across_y(
            Arc::new(slice_grid),
            interpolator,
            y_coord,
            resampled_location,
        )
    }

    /// Returns a 2D scalar field corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(
        &self,
        interpolator: &I,
        z_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::ZSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.slice_across_z();
        self.create_slice_across_z(
            Arc::new(slice_grid),
            interpolator,
            z_coord,
            resampled_location,
        )
    }

    /// Returns a 2D scalar field corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(
        &self,
        interpolator: &I,
        axis: Dim3,
        coord: fgr,
        location: CoordLocation,
    ) -> ScalarField2<F, RegularGrid2<fgr>>
    where
        I: Interpolator3,
    {
        let slice_grid = self.grid.regular_slice_across_axis(axis);
        self.create_regular_slice_across_axis(
            Arc::new(slice_grid),
            interpolator,
            axis,
            coord,
            location,
        )
    }

    fn compute_overlying_grid_cell_corners_for_resampling<H: Grid3<fgr>>(
        overlying_grid: &H,
        overlying_grid_cell_idx: usize,
    ) -> (Point3<fgr>, Point3<fgr>) {
        let overlying_indices =
            compute_3d_array_indices_from_flat_idx(overlying_grid.shape(), overlying_grid_cell_idx);
        overlying_grid.grid_cell_extremal_corners(&overlying_indices)
    }

    fn shift_overlying_grid_cell_corners_for_sample_averaging(
        &self,
        overlying_locations: &In3D<CoordLocation>,
        lower_overlying_corner: &mut Point3<fgr>,
        upper_overlying_corner: &mut Point3<fgr>,
    ) {
        let lower_bounds = self.grid().lower_bounds();
        let upper_bounds = self.grid().upper_bounds();

        for &dim in &Dim3::slice() {
            if let CoordLocation::LowerEdge = overlying_locations[dim] {
                // Shift the overlying grid cell half a cell down to be centered around the
                // location of the value to estimate
                let shift = -(upper_overlying_corner[dim] - lower_overlying_corner[dim]) * 0.5;
                lower_overlying_corner[dim] += shift;
                upper_overlying_corner[dim] += shift;

                // Correct any shift outside a non-periodic boundary
                if !self.grid().is_periodic(dim) {
                    if lower_overlying_corner[dim] < lower_bounds[dim] {
                        lower_overlying_corner[dim] = lower_bounds[dim];
                    }
                    if upper_overlying_corner[dim] >= upper_bounds[dim] {
                        upper_overlying_corner[dim] = upper_bounds[dim].prev();
                    }
                }
            }
        }
    }

    fn shift_overlying_grid_cell_corners_for_cell_averaging(
        &self,
        overlying_locations: &In3D<CoordLocation>,
        lower_overlying_corner: &mut Point3<fgr>,
        upper_overlying_corner: &mut Point3<fgr>,
    ) {
        let underlying_locations = self.locations();
        let average_underlying_cell_extents = self.grid().average_grid_cell_extents();

        let lower_bounds = self.grid().lower_bounds();
        let upper_bounds = self.grid().upper_bounds();

        for &dim in &Dim3::slice() {
            let mut shift = 0.0;
            if let CoordLocation::LowerEdge = underlying_locations[dim] {
                // Shift overlying grid cell half an underlying grid cell up to compensate
                // for downward bias due the underlying values being located on lower edges
                shift += average_underlying_cell_extents[dim];
            }
            if let CoordLocation::LowerEdge = overlying_locations[dim] {
                // Shift the overlying grid cell half a cell down to be centered around the
                // location of the value to estimate
                shift -= upper_overlying_corner[dim] - lower_overlying_corner[dim];
            }
            shift *= 0.5;
            lower_overlying_corner[dim] += shift;
            upper_overlying_corner[dim] += shift;

            // Correct any shift outside a non-periodic boundary
            if !self.grid().is_periodic(dim) {
                if lower_overlying_corner[dim] < lower_bounds[dim] {
                    lower_overlying_corner[dim] = lower_bounds[dim];
                }
                if upper_overlying_corner[dim] >= upper_bounds[dim] {
                    upper_overlying_corner[dim] = upper_bounds[dim].prev();
                }
            }
        }
    }

    fn determine_underlying_indices_and_offsets(
        &self,
        overlying_corner: &Point3<fgr>,
    ) -> (Idx3<usize>, Vec3<fgr>) {
        match self.grid().find_closest_grid_cell(overlying_corner) {
            GridPointQuery3::Inside(indices) => (indices, Vec3::zero()),
            GridPointQuery3::MovedInside((indices, moved_point)) => {
                let offsets = overlying_corner - &moved_point;
                (indices, offsets)
            }
            _ => unreachable!(),
        }
    }

    fn obtain_indexed_monotonic_grid_cell_edges(
        &self,
        lower_indices: &Idx3<usize>,
        upper_indices: &Idx3<usize>,
        lower_offsets: &Vec3<fgr>,
    ) -> (In3D<Vec<usize>>, In3D<Vec<fgr>>) {
        let determine_underlying_indices_and_edges = |dim| {
            self.grid
                .determine_indexed_monotonic_grid_cell_edges_between(
                    dim,
                    lower_indices[dim],
                    upper_indices[dim],
                    lower_offsets[dim],
                )
        };

        let (underlying_indices_x, underlying_edges_x) = determine_underlying_indices_and_edges(X);
        let (underlying_indices_y, underlying_edges_y) = determine_underlying_indices_and_edges(Y);
        let (underlying_indices_z, underlying_edges_z) = determine_underlying_indices_and_edges(Z);

        (
            In3D::new(
                underlying_indices_x,
                underlying_indices_y,
                underlying_indices_z,
            ),
            In3D::new(underlying_edges_x, underlying_edges_y, underlying_edges_z),
        )
    }

    fn coords_from_grid<'a, 'b, H: Grid3<fgr>>(
        grid: &'a H,
        locations: &'b In3D<CoordLocation>,
    ) -> CoordRefs3<'a, fgr> {
        CoordRefs3::new(
            &grid.coords_by_type(locations[X])[X],
            &grid.coords_by_type(locations[Y])[Y],
            &grid.coords_by_type(locations[Z])[Z],
        )
    }

    fn set_grid(&mut self, new_grid: Arc<G>) {
        let grid_shape = new_grid.shape();
        let values_shape = self.values.shape();
        assert!(
            grid_shape[X] == values_shape[0]
                && grid_shape[Y] == values_shape[1]
                && grid_shape[Z] == values_shape[2],
            "Shape of new grid does not match shape of array of values."
        );
        self.grid = new_grid;
    }

    fn create_slice_across_x<I>(
        &self,
        slice_grid: Arc<G::XSliceGrid>,
        interpolator: &I,
        x_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::XSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_locations = self.select_slice_locations([Y, Z], resampled_location);
        let slice_values =
            self.compute_slice_values(interpolator, X, x_coord, resampled_location, false);
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn create_slice_across_y<I>(
        &self,
        slice_grid: Arc<G::YSliceGrid>,
        interpolator: &I,
        y_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::YSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_locations = self.select_slice_locations([X, Z], resampled_location);
        let slice_values =
            self.compute_slice_values(interpolator, Y, y_coord, resampled_location, false);
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn create_slice_across_z<I>(
        &self,
        slice_grid: Arc<G::ZSliceGrid>,
        interpolator: &I,
        z_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> ScalarField2<F, G::ZSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_locations = self.select_slice_locations([X, Y], resampled_location);
        let slice_values =
            self.compute_slice_values(interpolator, Z, z_coord, resampled_location, false);
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn create_regular_slice_across_axis<I>(
        &self,
        slice_grid: Arc<RegularGrid2<fgr>>,
        interpolator: &I,
        axis: Dim3,
        coord: fgr,
        location: CoordLocation,
    ) -> ScalarField2<F, RegularGrid2<fgr>>
    where
        I: Interpolator3,
    {
        let slice_locations = In2D::same(location);
        let slice_values = self.compute_slice_values(
            interpolator,
            axis,
            coord,
            ResampledCoordLocation::Specific(location),
            true,
        );
        ScalarField2::new(
            self.name.to_string(),
            slice_grid,
            slice_locations,
            slice_values,
        )
    }

    fn compute_slice_indices_from_flat_idx(&self, axes: [Dim3; 2], idx: usize) -> [usize; 2] {
        let shape = self.shape();
        let indices = compute_2d_array_indices_from_flat_idx(
            &In2D::with_each_component(|dim| shape[axes[dim.num()]]),
            idx,
        );
        [indices[Dim2::X], indices[Dim2::Y]]
    }

    fn select_slice_locations(
        &self,
        axes: [Dim3; 2],
        resampled_location: ResampledCoordLocation,
    ) -> In2D<CoordLocation> {
        match resampled_location {
            ResampledCoordLocation::Original => {
                In2D::with_each_component(|dim| self.locations[axes[dim.num()]])
            }
            ResampledCoordLocation::Specific(location) => In2D::same(location),
        }
    }

    fn select_slice_coords(
        &self,
        axes: [Dim3; 2],
        resampled_location: ResampledCoordLocation,
    ) -> [&[fgr]; 2] {
        match resampled_location {
            ResampledCoordLocation::Original => {
                let coords = self.coords();
                [coords[axes[0]], coords[axes[1]]]
            }
            ResampledCoordLocation::Specific(CoordLocation::Center) => {
                let centers = self.grid.centers();
                [&centers[axes[0]], &centers[axes[1]]]
            }
            ResampledCoordLocation::Specific(CoordLocation::LowerEdge) => {
                let lower_edges = self.grid.lower_edges();
                [&lower_edges[axes[0]], &lower_edges[axes[1]]]
            }
        }
    }

    fn select_regular_slice_coords(&self, axes: [Dim3; 2], location: CoordLocation) -> [&[fgr]; 2] {
        match location {
            CoordLocation::Center => {
                let regular_centers = self.grid.regular_centers();
                [regular_centers[axes[0]], regular_centers[axes[1]]]
            }
            CoordLocation::LowerEdge => {
                let regular_lower_edges = self.grid.regular_lower_edges();
                [regular_lower_edges[axes[0]], regular_lower_edges[axes[1]]]
            }
        }
    }

    fn compute_slice_values<I>(
        &self,
        interpolator: &I,
        axis: Dim3,
        coord: fgr,
        resampled_location: ResampledCoordLocation,
        regular: bool,
    ) -> Array2<F>
    where
        I: Interpolator3,
    {
        let lower_bound = self.grid.lower_bounds()[axis];
        let upper_bound = self.grid.upper_bounds()[axis];
        if coord < lower_bound || coord >= upper_bound {
            panic!("Slicing coordinate is outside the grid bounds.");
        }

        let axes = Dim3::slice_except(axis);

        let coords = if regular {
            if let ResampledCoordLocation::Specific(location) = resampled_location {
                self.select_regular_slice_coords(axes, location)
            } else {
                panic!("Original coord locations not supported for regular slice.")
            }
        } else {
            self.select_slice_coords(axes, resampled_location)
        };

        self.interpolate_slice_values(interpolator, axes, &coords, coord)
    }

    fn interpolate_slice_values<I>(
        &self,
        interpolator: &I,
        axes: [Dim3; 2],
        coords: &[&[fgr]; 2],
        slice_axis_coord: fgr,
    ) -> Array2<F>
    where
        I: Interpolator3,
    {
        let slice_shape = (coords[0].len(), coords[1].len());
        let mut slice_values = Array2::uninit(slice_shape.f());
        let values_buffer = slice_values.as_slice_memory_order_mut().unwrap();

        values_buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let [idx_0, idx_1] = self.compute_slice_indices_from_flat_idx(axes, idx);
                let mut point_in_slice = Point3::equal_components(slice_axis_coord);
                point_in_slice[axes[0]] = coords[0][idx_0];
                point_in_slice[axes[1]] = coords[1][idx_1];
                value.write(
                    F::from(
                        interpolator
                            .interp_scalar_field(self, &point_in_slice)
                            .expect_inside(),
                    )
                    .unwrap(),
                );
            });
        unsafe { slice_values.assume_init() }
    }
}

#[cfg(feature = "for-testing")]
macro_rules! impl_partial_eq_for_field {
    ($T:ident <$F:ident, $G:ident>, $H:ident, $GT:ident <$GTF:ident>) => {
        impl<$F, $G, $H> PartialEq<$T<$F, $H>> for $T<$F, $G>
        where
            $F: BFloat,
            $G: $GT<$GTF> + PartialEq<$H>,
            $H: $GT<$GTF> + PartialEq<$G>,
        {
            fn eq(&self, other: &$T<$F, $H>) -> bool {
                self.locations() == other.locations()
                    && self.grid() == other.grid()
                    && self.values() == other.values()
            }
        }
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_abs_diff_eq_for_field {
    ($T:ident <$F:ident, $G:ident>, $H:ident, $GT:ident <$GTF:ident>) => {
        impl<$F, $G, $H> AbsDiffEq<$T<$F, $H>> for $T<$F, $G>
        where
            $F: BFloat + AbsDiffEq,
            $F::Epsilon: Copy,
            $G: $GT<$GTF> + AbsDiffEq<$H>,
            $H: $GT<$GTF> + AbsDiffEq<$G>,
            <$G as AbsDiffEq<$H>>::Epsilon: ::std::convert::From<<$F as AbsDiffEq>::Epsilon>,
        {
            type Epsilon = <$F as AbsDiffEq>::Epsilon;

            fn default_epsilon() -> Self::Epsilon {
                $F::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &$T<$F, $H>, epsilon: Self::Epsilon) -> bool {
                let self_values = ComparableSlice(self.values().as_slice_memory_order().unwrap());
                let other_values = ComparableSlice(other.values().as_slice_memory_order().unwrap());
                self.locations() == other.locations()
                    && self.grid().abs_diff_eq(other.grid(), epsilon.into())
                    && self_values.abs_diff_eq(&other_values, epsilon)
            }
        }
    };
}

#[cfg(feature = "for-testing")]
macro_rules! impl_relative_eq_for_field {
    ($T:ident <$F:ident, $G:ident>, $H:ident, $GT:ident <$GTF:ident>) => {
        impl<$F, $G, $H> RelativeEq<$T<$F, $H>> for $T<$F, $G>
        where
            $F: BFloat + RelativeEq,
            $F::Epsilon: Copy,
            $G: $GT<$GTF> + RelativeEq<$H>,
            $H: $GT<$GTF> + RelativeEq<$G>,
            <$G as AbsDiffEq<$H>>::Epsilon: ::std::convert::From<<$F as AbsDiffEq>::Epsilon>,
        {
            fn default_max_relative() -> Self::Epsilon {
                $F::default_max_relative()
            }

            fn relative_eq(
                &self,
                other: &$T<$F, $H>,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                let self_values = ComparableSlice(self.values().as_slice_memory_order().unwrap());
                let other_values = ComparableSlice(other.values().as_slice_memory_order().unwrap());
                if self.locations() != other.locations() {
                    #[cfg(debug_assertions)]
                    {
                        println!("Locations for {} not equal", self.name());
                        dbg!(self.locations(), other.locations());
                    }
                    return false;
                }
                if self
                    .grid()
                    .relative_ne(other.grid(), epsilon.into(), max_relative.into())
                {
                    #[cfg(debug_assertions)]
                    println!("Grids for {} not equal", self.name());
                    return false;
                }
                if self_values.relative_ne(&other_values, epsilon, max_relative) {
                    #[cfg(debug_assertions)]
                    println!("Values for {} not equal", self.name());
                    dbg!(self_values, other_values);
                    return false;
                }
                true
            }
        }
    };
}

#[cfg(feature = "for-testing")]
impl_partial_eq_for_field!(ScalarField3<F, G>, H, Grid3<fgr>);

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_for_field!(ScalarField3<F, G>, H, Grid3<fgr>);

#[cfg(feature = "for-testing")]
impl_relative_eq_for_field!(ScalarField3<F, G>, H, Grid3<fgr>);

/// A 3D vector field.
///
/// Holds the grid and values of the three components of a 3D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct VectorField3<F, G> {
    name: String,
    grid: Arc<G>,
    components: In3D<ScalarField3<F, G>>,
}

impl<F, G> VectorField3<F, G>
where
    F: BFloat,
    G: Grid3<fgr>,
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In3D<ScalarField3<F, G>>) -> Self {
        components[X].set_grid(Arc::clone(&grid));
        components[Y].set_grid(Arc::clone(&grid));
        components[Z].set_grid(Arc::clone(&grid));
        Self {
            name,
            grid,
            components,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim3) -> &ScalarField3<F, G> {
        &self.components[dim]
    }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim3) -> CoordRefs3<fgr> {
        self.components[dim].coords()
    }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In3D<CoordRefs3<fgr>> {
        In3D::with_each_component(|dim| self.coords(dim))
    }

    /// Returns a reference to the 3D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim3) -> &Array3<F> {
        self.components[dim].values()
    }

    /// Returns a mutable reference to the 3D array of field values
    /// for the specified component.
    pub fn values_mut(&mut self, dim: Dim3) -> &mut Array3<F> {
        self.components[dim].values_mut()
    }

    /// Returns a reference to the 3D array of field values for each component.
    pub fn all_values(&self) -> In3D<&Array3<F>> {
        In3D::with_each_component(|dim| self.values(dim))
    }

    /// Returns the field vector at the given 3D index.
    pub fn vector(&self, indices: &Idx3<usize>) -> Vec3<F> {
        Vec3::new(
            self.values(X)[[indices[X], indices[Y], indices[Z]]],
            self.values(Y)[[indices[X], indices[Y], indices[Z]]],
            self.values(Z)[[indices[X], indices[Y], indices[Z]]],
        )
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim3) -> &In3D<CoordLocation> {
        self.components[dim].locations()
    }

    /// Returns the 3D shape of the grid.
    pub fn shape(&self) -> &In3D<usize> {
        self.grid.shape()
    }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, values are interpolated from all overlapped original
    /// grid cells and averaged with weights according to the intersected volumes.
    /// If the new grid cell is contained within an original grid cell, this reduces
    /// to a single interpolation.
    ///
    /// This method gives robust results for arbitrary resampling grids, but is slower
    /// than direct sampling or weighted cell averaging.
    pub fn resampled_to_grid_with_sample_averaging<H, I>(
        &self,
        grid: Arc<H>,
        interpolator: &I,
        verbosity: &Verbosity,
    ) -> VectorField3<F, H>
    where
        H: Grid3<fgr>,
        I: Interpolator3,
    {
        let components = In3D::with_each_component(|dim| {
            self.components[dim].resampled_to_grid_with_sample_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
                verbosity,
            )
        });
        VectorField3::new(self.name.clone(), grid, components)
    }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    ///
    /// For each new grid cell, the values of all overlapped original grid cells are
    /// averaged with weights according to the intersected volumes.
    ///
    /// This method is suited for downsampling. It is faster than weighted sample
    /// averaging, but slightly less accurate.
    pub fn resampled_to_grid_with_cell_averaging<H: Grid3<fgr>>(
        &self,
        grid: Arc<H>,
        verbosity: &Verbosity,
    ) -> VectorField3<F, H> {
        let components = In3D::with_each_component(|dim| {
            self.components[dim].resampled_to_grid_with_cell_averaging(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                verbosity,
            )
        });
        VectorField3::new(self.name.clone(), grid, components)
    }

    /// Resamples the vector field onto the given grid and returns the resampled field.
    ///
    /// Each value on the new grid is found by interpolation of the values on the old grid
    /// at the new coordinate location.
    ///
    /// This is the preferred method for upsampling. For heavy downsampling it yields a
    /// more noisy result than weighted averaging.
    pub fn resampled_to_grid_with_direct_sampling<H, I>(
        &self,
        grid: Arc<H>,
        interpolator: &I,
        verbosity: &Verbosity,
    ) -> VectorField3<F, H>
    where
        H: Grid3<fgr>,
        I: Interpolator3,
    {
        let components = In3D::with_each_component(|dim| {
            self.components[dim].resampled_to_grid_with_direct_sampling(
                Arc::clone(&grid),
                In3D::same(ResampledCoordLocation::Original),
                interpolator,
                verbosity,
            )
        });
        VectorField3::new(self.name.clone(), grid, components)
    }

    /// Returns a view of the 2D slice located at the given index along the given dimension,
    /// for each component of the field.
    pub fn slice_across_axis_at_idx(&self, axis: Dim3, idx: usize) -> In3D<ArrayView2<F>> {
        In3D::with_each_component(|dim| self.components[dim].slice_across_axis_at_idx(axis, idx))
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the x-axis at the given coordinate.
    pub fn slice_across_x<I>(
        &self,
        interpolator: &I,
        x_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> PlaneVectorField3<F, G::XSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.slice_across_x());
        let slice_field_components = In3D::with_each_component(|dim| {
            self.components[dim].create_slice_across_x(
                Arc::clone(&slice_grid),
                interpolator,
                x_coord,
                resampled_location,
            )
        });
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the y-axis at the given coordinate.
    pub fn slice_across_y<I>(
        &self,
        interpolator: &I,
        y_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> PlaneVectorField3<F, G::YSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.slice_across_y());
        let slice_field_components = In3D::with_each_component(|dim| {
            self.components[dim].create_slice_across_y(
                Arc::clone(&slice_grid),
                interpolator,
                y_coord,
                resampled_location,
            )
        });
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a slice through the z-axis at the given coordinate.
    pub fn slice_across_z<I>(
        &self,
        interpolator: &I,
        z_coord: fgr,
        resampled_location: ResampledCoordLocation,
    ) -> PlaneVectorField3<F, G::ZSliceGrid>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.slice_across_z());
        let slice_field_components = In3D::with_each_component(|dim| {
            self.components[dim].create_slice_across_z(
                Arc::clone(&slice_grid),
                interpolator,
                z_coord,
                resampled_location,
            )
        });
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }

    /// Returns a field of 3D vectors in a 2D plane corresponding to a regular slice through the given axis at the given coordinate.
    pub fn regular_slice_across_axis<I>(
        &self,
        interpolator: &I,
        axis: Dim3,
        coord: fgr,
        location: CoordLocation,
    ) -> PlaneVectorField3<F, RegularGrid2<fgr>>
    where
        I: Interpolator3,
    {
        let slice_grid = Arc::new(self.grid.regular_slice_across_axis(axis));
        let slice_field_components = In3D::with_each_component(|dim| {
            self.components[dim].create_regular_slice_across_axis(
                Arc::clone(&slice_grid),
                interpolator,
                axis,
                coord,
                location,
            )
        });
        PlaneVectorField3::new(self.name.to_string(), slice_grid, slice_field_components)
    }
}

/// A 2D scalar field.
///
/// Holds the grid and values of a 2D scalar field, as well as the
/// specific coordinates where the values are defined.
/// The array of values is laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct ScalarField2<F, G> {
    name: String,
    grid: Arc<G>,
    locations: In2D<CoordLocation>,
    values: Array2<F>,
}

#[cfg_attr(feature = "serialization", derive(Serialize))]
struct ScalarFieldSerializeData2<F> {
    coords: Coords2<fgr>,
    values: Array2<F>,
}

impl<F, G> ScalarField2<F, G>
where
    F: BFloat,
    G: Grid2<fgr>,
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate locations specifying where in the grid cell the values are defined.
    pub fn new(
        name: String,
        grid: Arc<G>,
        locations: In2D<CoordLocation>,
        values: Array2<F>,
    ) -> Self {
        Self {
            name,
            grid,
            locations,
            values,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> CoordRefs2<fgr> {
        CoordRefs2::new(
            &self.grid.coords_by_type(self.locations[Dim2::X])[Dim2::X],
            &self.grid.coords_by_type(self.locations[Dim2::Y])[Dim2::Y],
        )
    }

    /// Returns a reference to the 2D array of field values.
    pub fn values(&self) -> &Array2<F> {
        &self.values
    }

    /// Returns a mutable reference to the 2D array of field values.
    pub fn values_mut(&mut self) -> &mut Array2<F> {
        &mut self.values
    }

    /// Returns the field value at the given 2D index.
    pub fn value(&self, indices: &Idx2<usize>) -> F {
        self.values[[indices[Dim2::X], indices[Dim2::Y]]]
    }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> {
        self.grid.shape()
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values are defined.
    pub fn locations(&self) -> &In2D<CoordLocation> {
        &self.locations
    }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array2<F> {
        self.values
    }

    /// Consumes the scalar field and returns a version with the given name.
    pub fn with_name(self, name: String) -> Self {
        Self {
            name,
            grid: self.grid,
            locations: self.locations,
            values: self.values,
        }
    }

    /// Computes the 2D indices and value of the minimum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_minimum(&self) -> Option<(Idx2<usize>, F)> {
        ParallelIterator::min(
            self.values
                .as_slice_memory_order()
                .unwrap()
                .par_iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some(KeyValueOrderableByValue(idx, value))
                    }
                }),
        )
        .map(|KeyValueOrderableByValue(idx_of_min_value, min_value)| {
            (
                compute_2d_array_indices_from_flat_idx(self.shape(), idx_of_min_value),
                min_value,
            )
        })
    }

    /// Computes the 2D indices and value of the maximum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_maximum(&self) -> Option<(Idx2<usize>, F)> {
        ParallelIterator::max(
            self.values
                .as_slice_memory_order()
                .unwrap()
                .par_iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some(KeyValueOrderableByValue(idx, value))
                    }
                }),
        )
        .map(|KeyValueOrderableByValue(idx_of_max_value, max_value)| {
            (
                compute_2d_array_indices_from_flat_idx(self.shape(), idx_of_max_value),
                max_value,
            )
        })
    }

    /// Serializes the field data into pickle format and save at the given path.
    #[cfg(feature = "pickle")]
    pub fn save_as_pickle(&self, output_file_path: &Path) -> io::Result<()>
    where
        F: Serialize,
    {
        let data = ScalarFieldSerializeData2 {
            coords: self.coords().into_owned(),
            values: self.values().clone(),
        };
        save_data_as_pickle(output_file_path, &data)
    }

    fn set_grid(&mut self, new_grid: Arc<G>) {
        let grid_shape = new_grid.shape();
        let values_shape = self.values.shape();
        assert!(
            grid_shape[Dim2::X] == values_shape[0] && grid_shape[Dim2::Y] == values_shape[1],
            "Shape of new grid does not match shape of array of values."
        );
        self.grid = new_grid;
    }
}

#[cfg(feature = "for-testing")]
impl_partial_eq_for_field!(ScalarField2<F, G>, H, Grid2<fgr>);

#[cfg(feature = "for-testing")]
impl_abs_diff_eq_for_field!(ScalarField2<F, G>, H, Grid2<fgr>);

#[cfg(feature = "for-testing")]
impl_relative_eq_for_field!(ScalarField2<F, G>, H, Grid2<fgr>);

/// A 2D vector field.
///
/// Holds the grid and values of the two components of a 2D vector field,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct VectorField2<F, G> {
    name: String,
    grid: Arc<G>,
    components: In2D<ScalarField2<F, G>>,
}

impl<F, G> VectorField2<F, G>
where
    F: BFloat,
    G: Grid2<fgr>,
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In2D<ScalarField2<F, G>>) -> Self {
        components[Dim2::X].set_grid(Arc::clone(&grid));
        components[Dim2::Y].set_grid(Arc::clone(&grid));
        Self {
            name,
            grid,
            components,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim2) -> &ScalarField2<F, G> {
        &self.components[dim]
    }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim2) -> CoordRefs2<fgr> {
        self.components[dim].coords()
    }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In2D<CoordRefs2<fgr>> {
        In2D::with_each_component(|dim| self.coords(dim))
    }

    /// Returns a reference to the 2D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim2) -> &Array2<F> {
        self.components[dim].values()
    }

    /// Returns a mutable reference to the 2D array of field values.
    pub fn values_mut(&mut self, dim: Dim2) -> &mut Array2<F> {
        self.components[dim].values_mut()
    }

    /// Returns a reference to the 2D array of field values for each component.
    pub fn all_values(&self) -> In2D<&Array2<F>> {
        In2D::with_each_component(|dim| self.values(dim))
    }

    /// Returns the field vector at the given 2D index.
    pub fn vector(&self, indices: &Idx2<usize>) -> Vec2<F> {
        Vec2::new(
            self.values(Dim2::X)[[indices[Dim2::X], indices[Dim2::Y]]],
            self.values(Dim2::Y)[[indices[Dim2::X], indices[Dim2::Y]]],
        )
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim2) -> &In2D<CoordLocation> {
        self.components[dim].locations()
    }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> {
        self.grid.shape()
    }
}

/// A field of 3D vectors in a 2D plane.
///
/// Holds the grid and values of the three components of a vector field in a 2D plane,
/// as well as the specific coordinates where the component values are defined.
/// The arrays of component values are laid out in column-major order in memory.
#[derive(Clone, Debug)]
pub struct PlaneVectorField3<F, G> {
    name: String,
    grid: Arc<G>,
    components: In3D<ScalarField2<F, G>>,
}

impl<F, G> PlaneVectorField3<F, G>
where
    F: BFloat,
    G: Grid2<fgr>,
{
    /// Creates a new vector field given a name, a grid, and the scalar fields
    /// representing the component values.
    pub fn new(name: String, grid: Arc<G>, mut components: In3D<ScalarField2<F, G>>) -> Self {
        components[X].set_grid(Arc::clone(&grid));
        components[Y].set_grid(Arc::clone(&grid));
        components[Z].set_grid(Arc::clone(&grid));
        Self {
            name,
            grid,
            components,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a reference to the scalar field representing the specified
    /// vector field component.
    pub fn component(&self, dim: Dim3) -> &ScalarField2<F, G> {
        &self.components[dim]
    }

    /// Returns a set of references to the coordinates where the field
    /// values of the specified component are defined.
    pub fn coords(&self, dim: Dim3) -> CoordRefs2<fgr> {
        self.components[dim].coords()
    }

    /// Returns a set of references to the coordinates where the field
    /// values of each component are defined.
    pub fn all_coords(&self) -> In3D<CoordRefs2<fgr>> {
        In3D::with_each_component(|dim| self.coords(dim))
    }

    /// Returns a reference to the 3D array of field values for each component.
    pub fn all_values(&self) -> In3D<&Array2<F>> {
        In3D::with_each_component(|dim| self.values(dim))
    }

    /// Returns the field vector at the given 3D index.
    pub fn vector(&self, indices: &Idx2<usize>) -> Vec3<F> {
        Vec3::new(
            self.values(X)[[indices[Dim2::X], indices[Dim2::Y]]],
            self.values(Y)[[indices[Dim2::X], indices[Dim2::Y]]],
            self.values(Y)[[indices[Dim2::X], indices[Dim2::Y]]],
        )
    }

    /// Returns a reference to the 2D array of field values for the
    /// specified component.
    pub fn values(&self, dim: Dim3) -> &Array2<F> {
        self.components[dim].values()
    }

    /// Returns a mutable reference to the 2D array of field values for the
    /// specified component.
    pub fn values_mut(&mut self, dim: Dim3) -> &mut Array2<F> {
        self.components[dim].values_mut()
    }

    /// Returns a reference to the coordinate locations specifying
    /// where in the grid cell the values of the given component are defined.
    pub fn locations(&self, dim: Dim3) -> &In2D<CoordLocation> {
        self.components[dim].locations()
    }

    /// Returns the 2D shape of the grid.
    pub fn shape(&self) -> &In2D<usize> {
        self.grid.shape()
    }
}

/// A 1D scalar field.
///
/// Holds the grid and values of a 1D scalar field, as well as the
/// specific coordinates where the values are defined.
#[derive(Clone, Debug)]
pub struct ScalarField1<F, G> {
    name: String,
    grid: Arc<G>,
    location: CoordLocation,
    values: Array1<F>,
}

#[cfg_attr(feature = "serialization", derive(Serialize))]
struct ScalarFieldSerializeData1<F> {
    coords: Vec<fgr>,
    values: Array1<F>,
}

impl<F, G> ScalarField1<F, G>
where
    F: BFloat,
    G: Grid1<fgr>,
{
    /// Creates a new scalar field given a name, a grid, the values and
    /// coordinate location specifying where in the grid cell the values are defined.
    pub fn new(name: String, grid: Arc<G>, location: CoordLocation, values: Array1<F>) -> Self {
        Self {
            name,
            grid,
            location,
            values,
        }
    }

    /// Returns a reference to the name of the field.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &G {
        self.grid.as_ref()
    }

    /// Returns a new atomic reference counted pointer to the grid.
    pub fn arc_with_grid(&self) -> Arc<G> {
        Arc::clone(&self.grid)
    }

    /// Returns a set of references to the coordinates where the field
    /// values are defined.
    pub fn coords(&self) -> &[fgr] {
        self.grid.coords_by_type(self.location)
    }

    /// Returns a reference to the 1D array of field values.
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    /// Returns a mutable reference to the 1D array of field values.
    pub fn values_mut(&mut self) -> &mut Array1<F> {
        &mut self.values
    }

    /// Returns the field value at the given index.
    pub fn value(&self, index: usize) -> F {
        self.values[index]
    }

    /// Returns the size of the grid.
    pub fn size(&self) -> usize {
        self.grid.size()
    }

    /// Returns the coordinate location specifying where in the grid cell
    /// the values are defined.
    pub fn location(&self) -> CoordLocation {
        self.location
    }

    /// Consumes the scalar field and returns the owned array of field values.
    pub fn into_values(self) -> Array1<F> {
        self.values
    }

    /// Consumes the scalar field and returns a version with the given name.
    pub fn with_name(self, name: String) -> Self {
        Self {
            name,
            grid: self.grid,
            location: self.location,
            values: self.values,
        }
    }

    /// Computes the index and value of the minimum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_minimum(&self) -> Option<(usize, F)> {
        ParallelIterator::min(
            self.values
                .as_slice_memory_order()
                .unwrap()
                .par_iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some(KeyValueOrderableByValue(idx, value))
                    }
                }),
        )
        .map(|KeyValueOrderableByValue(idx_of_min_value, min_value)| (idx_of_min_value, min_value))
    }

    /// Computes the index and value of the maximum of the field.
    ///
    /// NaN values are ignored. Returns `None` if there are no finite values.
    pub fn find_maximum(&self) -> Option<(usize, F)> {
        ParallelIterator::max(
            self.values
                .as_slice_memory_order()
                .unwrap()
                .par_iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some(KeyValueOrderableByValue(idx, value))
                    }
                }),
        )
        .map(|KeyValueOrderableByValue(idx_of_max_value, max_value)| (idx_of_max_value, max_value))
    }

    /// Serializes the field data into pickle format and save at the given path.
    #[cfg(feature = "pickle")]
    pub fn save_as_pickle(&self, output_file_path: &Path) -> io::Result<()>
    where
        F: Serialize,
    {
        let data = ScalarFieldSerializeData1 {
            coords: self.coords().to_vec(),
            values: self.values().clone(),
        };
        save_data_as_pickle(output_file_path, &data)
    }
}

/// Computes the 3D array indices corresponding to a given index into the flattened version of the array,
/// assuming the array is laid out in column-major order.
pub fn compute_3d_array_indices_from_flat_idx(shape: &In3D<usize>, idx: usize) -> Idx3<usize> {
    let i = idx % shape[X];
    let j = idx / shape[X] % shape[Y];
    let k = idx / (shape[X] * shape[Y]);
    Idx3::new(i, j, k)
}

/// Computes the 2D array indices corresponding to a given index into the flattened version of the array,
/// assuming the array is laid out in column-major order.
pub fn compute_2d_array_indices_from_flat_idx(shape: &In2D<usize>, idx: usize) -> Idx2<usize> {
    let i = idx % shape[Dim2::X];
    let j = idx / shape[Dim2::X];
    Idx2::new(i, j)
}
