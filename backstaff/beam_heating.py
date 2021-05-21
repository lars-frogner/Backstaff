import numpy as np
import scipy.special as special
import scipy.integrate as integrate
import scipy.interpolate as interpolate
try:
    import backstaff.units as units
except ModuleNotFoundError:
    import units

SAHA_SCALE = (units.HPLANCK*units.HPLANCK/
              (2.0*np.pi*units.M_ELECTRON*units.KBOLTZMANN))**1.5

# Fraction of a mass of plasma assumed to be made up of hydrogen.
HYDROGEN_MASS_FRACTION = 0.735

COLLISION_SCALE = 2.0*np.pi*(units.Q_ELECTRON*units.Q_ELECTRON/
                             units.KEV_TO_ERG)**2

ELECTRON_COULOMB_OFFSET = 0.5*np.log(units.KEV_TO_ERG**3/
                                     (2*np.pi*units.Q_ELECTRON**6))

NEUTRAL_HYDROGEN_COULOMB_OFFSET = np.log(2/(1.105*units.XI_H*1e-3))

MIN_ELECTRON_ENERGY_FOR_COULOMB_LOG = 0.1  # [keV]


def compute_hydrogen_level_energy(n):
    return units.XI_H*1e-3*units.KEV_TO_ERG/n**2  # [erg]


def compute_hydrogen_level_degeneracy(n):
    return n**2


def compute_relative_hydrogen_level_populations(temperature,
                                                highest_energy_level=5):
    P = np.ones(highest_energy_level)
    for na in range(1, highest_energy_level):
        nb = na + 1
        ga = compute_hydrogen_level_degeneracy(na)
        gb = compute_hydrogen_level_degeneracy(nb)
        Ea = compute_hydrogen_level_energy(na)
        Eb = compute_hydrogen_level_energy(nb)
        P[na] = P[na - 1]*(gb/ga)*np.exp(-(Eb - Ea)/
                                         (units.KBOLTZMANN*temperature))
    return P


def compute_equilibrium_hydrogen_populations(mass_density,
                                             temperature,
                                             electron_density,
                                             highest_energy_level=5):
    neutral_hydrogen_density = compute_equilibrium_neutral_hydrogen_density(
        mass_density,
        temperature,
        electron_density,
    )
    total_hydrogen_density = compute_total_hydrogen_density(mass_density)
    populations = compute_relative_hydrogen_level_populations(
        temperature, highest_energy_level)
    neutral_population_densities = populations*neutral_hydrogen_density/np.sum(
        populations)
    proton_density = total_hydrogen_density - neutral_hydrogen_density
    return neutral_population_densities, proton_density


# Evaluates the beta function B(a, b) = int t^(a-1)*(1-t)^(b-1) dt from t=0 to t=1.
def compute_beta(a, b):
    return special.beta(a, b)


# Evaluates the unregularized incomplete beta function
# B(x; a, b) = int t^(a-1)*(1-t)^(b-1) dt from t=0 to t=x.
def compute_incomplete_beta(x, a, b):
    return special.betainc(a, b, x)*special.beta(a, b)


def compute_equilibrium_hydrogen_ionization_fraction(
    temperature,
    electron_density,
):
    tmp = electron_density*SAHA_SCALE/temperature**1.5
    return 1.0/(1.0 + tmp*np.exp(units.XI_H*units.EV_TO_ERG/
                                 (units.KBOLTZMANN*temperature)))


def compute_equilibrium_neutral_hydrogen_density(
    mass_density,
    temperature,
    electron_density,
):
    tmp = electron_density*SAHA_SCALE/temperature**1.5
    return mass_density*HYDROGEN_MASS_FRACTION*tmp/(
        units.M_H*(tmp + np.exp(-units.XI_H*units.EV_TO_ERG/
                                (units.KBOLTZMANN*temperature))))


def compute_mean_energy(delta, lower_cutoff_energy):
    return lower_cutoff_energy*(delta - 0.5)/(delta - 1.5)


def compute_total_hydrogen_density(mass_density):
    return (HYDROGEN_MASS_FRACTION/units.M_H)*mass_density  # [hydrogen/cm^3]


def compute_total_helium_density_no_metals(mass_density):
    return (
        (1 - HYDROGEN_MASS_FRACTION)/units.M_HE)*mass_density  # [helium/cm^3]


def compute_electron_coulomb_logarithm(electron_density, electron_energy):
    return ELECTRON_COULOMB_OFFSET + 0.5*np.log(
        np.maximum(electron_energy, MIN_ELECTRON_ENERGY_FOR_COULOMB_LOG)**3/
        electron_density)


def compute_neutral_hydrogen_coulomb_logarithm(electron_energy):
    return NEUTRAL_HYDROGEN_COULOMB_OFFSET + np.log(
        np.maximum(electron_energy, MIN_ELECTRON_ENERGY_FOR_COULOMB_LOG))


def compute_effective_coulomb_logarithm(
    ionization_fraction,
    electron_coulomb_logarithm,
    neutral_hydrogen_coulomb_logarithm,
):
    return ionization_fraction * electron_coulomb_logarithm \
        + (1.0 - ionization_fraction) * neutral_hydrogen_coulomb_logarithm


def compute_stopping_column_depth(
    pitch_angle_cosine,
    electron_energy,
    coulomb_logarithm,
):
    return np.abs(pitch_angle_cosine) * electron_energy**2 \
        / (3.0 * COLLISION_SCALE * coulomb_logarithm)


def compute_heating_scale(
    total_power,
    delta,
    pitch_angle_cosine,
    lower_cutoff_energy,
):
    return COLLISION_SCALE * total_power * (delta - 2.0) \
        / (2.0 * np.abs(pitch_angle_cosine) * lower_cutoff_energy**2)


def compute_cumulative_integral_over_distance(distances, values, initial=0):
    return integrate.cumtrapz(values, x=distances, initial=initial)


def compute_cumulative_heat_power(distances, beam_heating):
    return compute_cumulative_integral_over_distance(distances, beam_heating)


def compute_collisional_coef_SFP(
    electron_density,
    neutral_hydrogen_density,
    electron_energy,
):
    return 4.989_344e-25*(
        electron_density*compute_electron_coulomb_logarithm(
            electron_density, electron_energy) + neutral_hydrogen_density*
        compute_neutral_hydrogen_coulomb_logarithm(electron_energy))


def compute_collisional_depth_derivative_SFP(
    electron_density,
    neutral_hydrogen_density,
    pitch_angle_factor,
    mean_energy,
):
    return pitch_angle_factor*compute_collisional_coef_SFP(
        electron_density, neutral_hydrogen_density, mean_energy)


def compute_beam_heating_SFP(delta, total_power, lower_cutoff_energy,
                             collisional_depth, collisional_depth_derivative):
    return total_power*collisional_depth_derivative*(
        (delta - 2.0)/
        (2*(lower_cutoff_energy*units.KEV_TO_ERG/units.MC2_ELECTRON)**2)
    )*(1.0 + collisional_depth/
       (lower_cutoff_energy*units.KEV_TO_ERG/units.MC2_ELECTRON)**2)**(-0.5*
                                                                       delta)


def compute_remaining_power_SFP(
    delta,
    total_power,
    lower_cutoff_energy,
    collisional_depth,
):
    return total_power*(1.0 + collisional_depth/
                        (lower_cutoff_energy*units.KEV_TO_ERG/
                         units.MC2_ELECTRON)**2)**(1.0 - 0.5*delta)


class Atmosphere:
    @staticmethod
    def hor_avg_from_bifrost_data(bifrost_data, start_depth=None):
        depths = bifrost_data.z*units.U_L
        distances = depths - depths[0]
        mass_densities = np.mean(bifrost_data.get_var('r'),
                                 axis=(0, 1))*units.U_R
        temperatures = np.mean(bifrost_data.get_var('tg'), axis=(0, 1))
        electron_densities = np.mean(bifrost_data.get_var('nel'), axis=(0, 1))

        return Atmosphere(depths,
                          distances,
                          mass_densities,
                          temperatures,
                          electron_densities,
                          start_depth=start_depth)

    @staticmethod
    def column_from_bifrost_data(bifrost_data,
                                 x=12.5e8,
                                 y=12.5e8,
                                 start_depth=None):
        i = np.searchsorted(bifrost_data.x, x/units.U_L)
        j = np.searchsorted(bifrost_data.y, y/units.U_L)
        depths = bifrost_data.z*units.U_L
        distances = depths - depths[0]
        mass_densities = bifrost_data.get_var('r')[i, j, :]*units.U_R
        temperatures = bifrost_data.get_var('tg')[i, j, :]
        electron_densities = bifrost_data.get_var('nel')[i, j, :]

        return Atmosphere(depths,
                          distances,
                          mass_densities,
                          temperatures,
                          electron_densities,
                          start_depth=start_depth)

    @staticmethod
    def from_electron_beam_swarm(electron_beam_swarm, start_depth=None):
        depths = electron_beam_swarm.get_varying_scalar_values(
            'z')[0]*units.U_L
        distances = electron_beam_swarm.get_varying_scalar_values(
            's')[0]*units.U_L
        mass_densities = electron_beam_swarm.get_varying_scalar_values(
            'r')[0]*units.U_R
        temperatures = electron_beam_swarm.get_varying_scalar_values('tg')[0]
        electron_densities = electron_beam_swarm.get_varying_scalar_values(
            'nel')[0]

        return Atmosphere(depths,
                          distances,
                          mass_densities,
                          temperatures,
                          electron_densities,
                          start_depth=start_depth)

    @staticmethod
    def from_VAL3C_atmosphere(atmosphere_path, number_of_points=500):

        with open(atmosphere_path, 'r') as f:
            names = f.readline().split()
            _units = f.readline().split()
            values_arr = np.loadtxt(f)

        values = {name: values_arr[i, :] for i, name in enumerate(names)}

        depths = -values['h']*1e5
        mass_densities = values['sigma']
        temperatures = values['T']
        electron_densities = values['n_e']

        resample = lambda arr: 10**interpolate.interp1d(
            np.linspace(0, 1, arr.size), np.log10(arr), kind='linear')(
                np.linspace(0, 1, number_of_points))

        new_depths = resample(depths)

        return Atmosphere(new_depths,
                          new_depths - new_depths[0],
                          resample(mass_densities),
                          resample(temperatures),
                          resample(electron_densities),
                          start_depth=None)

    @staticmethod
    def from_FALC_atmosphere(atmosphere_path,
                             number_of_points=500,
                             atmosphere_start_depth=None):

        with open(atmosphere_path, 'r') as f:
            values_arr = np.loadtxt(f)

        depths = -values_arr[:, 0]*1e5
        mass_densities = values_arr[:, 10]
        temperatures = values_arr[:, 3]
        electron_densities = values_arr[:, 7]

        new_depths = np.linspace(
            depths[0] if atmosphere_start_depth is None else
            atmosphere_start_depth*units.U_L, depths[-1], number_of_points)

        resample = lambda arr: 10**interpolate.interp1d(
            depths, np.log10(arr), kind='linear', fill_value='extrapolate')(
                new_depths)

        return Atmosphere(new_depths,
                          new_depths - new_depths[0],
                          resample(mass_densities),
                          resample(temperatures),
                          resample(electron_densities),
                          start_depth=None)

    def __init__(self,
                 depths,
                 distances,
                 mass_densities,
                 temperatures,
                 electron_densities,
                 start_depth=None):
        self.__depths = depths
        self.__distances = distances
        self.__mass_densities = mass_densities
        self.__temperatures = temperatures
        self.__electron_densities = electron_densities
        self.start_depth = start_depth

    @property
    def start_depth(self):
        return self.__start_depth

    @start_depth.setter
    def start_depth(self, start_depth):
        self.__start_depth = start_depth
        self.__start_idx = 0 if start_depth is None else np.searchsorted(
            self.__depths, start_depth)

    @property
    def full_depths(self):
        return self.__depths

    @property
    def full_distances(self):
        return self.__distances

    @property
    def full_mass_densities(self):
        return self.__mass_densities

    @property
    def full_temperatures(self):
        return self.__temperatures

    @property
    def full_electron_densities(self):
        return self.__electron_densities

    @property
    def depths(self):
        return self.__depths[self.__start_idx:]

    @property
    def distances(self):
        return self.__distances[self.__start_idx:]

    @property
    def mass_densities(self):
        return self.__mass_densities[self.__start_idx:]

    @property
    def temperatures(self):
        return self.__temperatures[self.__start_idx:]

    @property
    def electron_densities(self):
        return self.__electron_densities[self.__start_idx:]


class Distribution:
    def __init__(self, total_power, delta, pitch_angle, lower_cutoff_energy):
        self.total_power = total_power
        self.__delta = delta
        self.pitch_angle = pitch_angle
        self.lower_cutoff_energy = lower_cutoff_energy

    @property
    def pitch_angle(self):
        return self.__pitch_angle

    @pitch_angle.setter
    def pitch_angle(self, pitch_angle):
        self.__pitch_angle = pitch_angle
        self.__pitch_angle_cosine = np.cos(pitch_angle*np.pi/180.0)

    @property
    def pitch_angle_cosine(self):
        return self.__pitch_angle_cosine

    @property
    def lower_cutoff_energy(self):
        return self.__lower_cutoff_energy

    @lower_cutoff_energy.setter
    def lower_cutoff_energy(self, lower_cutoff_energy):
        self.__lower_cutoff_energy = lower_cutoff_energy

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, delta):
        self.__delta = delta


class HeatedAtmosphere(Atmosphere):
    def __init__(self, atmosphere, distribution):
        super().__init__(atmosphere.full_depths,
                         atmosphere.full_distances,
                         atmosphere.full_mass_densities,
                         atmosphere.full_temperatures,
                         atmosphere.full_electron_densities,
                         start_depth=atmosphere.start_depth)

        self.compute_beam_heating(distribution)
        self.compute_conductive_heating()

    def compute_beam_heating(self, distribution):

        mean_energy = compute_mean_energy(distribution.delta,
                                          distribution.lower_cutoff_energy)

        self.electron_coulomb_logarithm = compute_electron_coulomb_logarithm(
            self.electron_densities[0], mean_energy)

        self.neutral_hydrogen_coulomb_logarithm = compute_neutral_hydrogen_coulomb_logarithm(
            mean_energy)

        self.stopping_ionized_column_depth = compute_stopping_column_depth(
            distribution.pitch_angle_cosine, distribution.lower_cutoff_energy,
            self.electron_coulomb_logarithm)

        self.total_hydrogen_densities = compute_total_hydrogen_density(
            self.mass_densities)

        self.ionization_fractions = compute_equilibrium_hydrogen_ionization_fraction(
            self.temperatures,
            self.electron_densities,
        )
        self.effective_coulomb_logarithms = compute_effective_coulomb_logarithm(
            self.ionization_fractions,
            self.electron_coulomb_logarithm,
            self.neutral_hydrogen_coulomb_logarithm,
        )

        coulomb_logarithm_ratios = self.effective_coulomb_logarithms/self.electron_coulomb_logarithm

        self.hydrogen_column_depths = compute_cumulative_integral_over_distance(
            self.distances, self.total_hydrogen_densities)

        self.equivalent_ionized_column_depths = compute_cumulative_integral_over_distance(
            self.distances, self.total_hydrogen_densities*
            self.effective_coulomb_logarithms/self.electron_coulomb_logarithm)

        self.column_depth_ratios = self.hydrogen_column_depths*coulomb_logarithm_ratios/self.stopping_ionized_column_depth
        self.betas = np.asfarray([
            compute_incomplete_beta(column_depth_ratio, 0.5*
                                    distribution.delta, 1.0/
                                    3.0) if column_depth_ratio < 1.0 else
            compute_beta(0.5*distribution.delta, 1.0/3.0)
            for column_depth_ratio in self.column_depth_ratios
        ])

        self.equivalent_ionized_column_depth_ratios = self.equivalent_ionized_column_depths/self.stopping_ionized_column_depth

        heating_scale = compute_heating_scale(distribution.total_power,
                                              distribution.delta,
                                              distribution.pitch_angle_cosine,
                                              distribution.lower_cutoff_energy)

        self.heat_fraction = self.equivalent_ionized_column_depth_ratios**(
            -0.5*distribution.delta)

        self.beam_heating = heating_scale \
            * self.betas \
            * self.total_hydrogen_densities \
            * self.effective_coulomb_logarithms \
            * self.heat_fraction

        self.beam_heating[0] = 0.0

        self.cumulative_heat_power = compute_cumulative_heat_power(
            self.distances, self.beam_heating)

        self.remaining_beam_powers = distribution.total_power - self.cumulative_heat_power

    def compute_conductive_heating(self):

        kappa_0 = 4.6e13*1e8**(-5.0/2.0)*(40.0/self.electron_coulomb_logarithm)

        self.dT_ds = np.gradient(self.temperatures, self.distances)
        self.d2T_ds2 = np.gradient(self.dT_ds, self.distances)
        self.conductive_heating_gradient_term = kappa_0*self.temperatures**(
            5.0/2.0)*5*self.dT_ds**2/(2*self.temperatures)
        self.conductive_heating_curvature_term = kappa_0*self.temperatures**(
            5.0/2.0)*self.d2T_ds2
        self.conductive_heating = self.conductive_heating_gradient_term + self.conductive_heating_curvature_term


class HeatedAtmosphereSFP(Atmosphere):
    def __init__(self, atmosphere, distribution):
        super().__init__(atmosphere.full_depths,
                         atmosphere.full_distances,
                         atmosphere.full_mass_densities,
                         atmosphere.full_temperatures,
                         atmosphere.full_electron_densities,
                         start_depth=atmosphere.start_depth)

        mean_energy = compute_mean_energy(distribution.delta,
                                          distribution.lower_cutoff_energy)

        self.neutral_hydrogen_densities = compute_equilibrium_neutral_hydrogen_density(
            self.mass_densities, self.temperatures, self.electron_densities)

        self.collisional_depth_derivatives = compute_collisional_depth_derivative_SFP(
            self.electron_densities, self.neutral_hydrogen_densities, 2.0,
            mean_energy)
        self.collisional_depths = compute_cumulative_integral_over_distance(
            self.distances, self.collisional_depth_derivatives)

        self.beam_heating = compute_beam_heating_SFP(
            distribution.delta, distribution.total_power,
            distribution.lower_cutoff_energy, self.collisional_depths,
            self.collisional_depth_derivatives)

        self.remaining_beam_powers = compute_remaining_power_SFP(
            distribution.delta, distribution.total_power,
            distribution.lower_cutoff_energy, self.collisional_depths)
