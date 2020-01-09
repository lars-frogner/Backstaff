import numpy as np
import scipy.special as special
import scipy.integrate as integrate
try:
    import backstaff.units as units
except ModuleNotFoundError:
    import units

SAHA_SCALE = (units.HPLANCK*units.HPLANCK/
              (2.0*np.pi*units.M_ELECTRON*units.KBOLTZMANN))**1.5

# Fraction of a mass of plasma assumed to be made up of hydrogen.
HYDROGEN_MASS_FRACTION = 0.735

# 2*pi*(electron charge [esu])^4/(electron rest energy [erg])^2
COLLISION_SCALE = 2.0*np.pi*(units.Q_ELECTRON*units.Q_ELECTRON/
                             units.MC2_ELECTRON)**2

# 1/2*ln( (2*pi*me*c/h)^3/(pi*alpha) [1/cm^3] )
ELECTRON_COULOMB_OFFSET = 37.853_791

# -ln( I_H [m_e*c^2] )
NEUTRAL_HYDROGEN_COULOMB_OFFSET = 10.53422


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
    return lower_cutoff_energy*(delta - 1.0)/(delta - 2.0)


def compute_total_hydrogen_density(mass_density):
    return (HYDROGEN_MASS_FRACTION/units.M_H)*mass_density  # [hydrogen/cm^3]


def compute_electron_coulomb_logarithm(electron_density, electron_energy):
    return np.maximum(
        0.0, ELECTRON_COULOMB_OFFSET + 0.5*np.log(
            (electron_energy*(electron_energy + 2.0))**2/electron_density))


def compute_neutral_hydrogen_coulomb_logarithm(electron_energy):
    return np.maximum(
        0.0, NEUTRAL_HYDROGEN_COULOMB_OFFSET +
        0.5*np.log(electron_energy*electron_energy*(electron_energy + 2.0)))


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
        lower_cutoff_energy_mec2,
):
    return COLLISION_SCALE * total_power * (delta - 2.0) \
        / (2.0 * np.abs(pitch_angle_cosine) * lower_cutoff_energy_mec2**2)


def compute_cumulative_integral_over_distance(distances, values, initial=0):
    return integrate.cumtrapz(values, x=distances, initial=initial)


def compute_remaining_beam_power(total_power, distances, beam_heating):
    return total_power - compute_cumulative_integral_over_distance(
        distances, beam_heating)


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
        (delta - 2.0)/(2*lower_cutoff_energy**2))*(
            1.0 + collisional_depth/lower_cutoff_energy**2)**(-0.5*delta)


def compute_remaining_power_SFP(
        delta,
        total_power,
        lower_cutoff_energy,
        collisional_depth,
):
    return total_power*(1.0 + collisional_depth/lower_cutoff_energy**2)**(
        1.0 - 0.5*delta)


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

    @property
    def lower_cutoff_energy_mec2(self):
        return self.__lower_cutoff_energy_mec2

    @property
    def mean_energy_mec2(self):
        return self.__mean_energy_mec2

    @lower_cutoff_energy.setter
    def lower_cutoff_energy(self, lower_cutoff_energy):
        self.__lower_cutoff_energy = lower_cutoff_energy
        self.__lower_cutoff_energy_mec2 = lower_cutoff_energy*units.KEV_TO_ERG/units.MC2_ELECTRON
        self.__mean_energy_mec2 = compute_mean_energy(
            self.__delta, self.__lower_cutoff_energy_mec2)

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, delta):
        self.__delta = delta
        self.__mean_energy_mec2 = compute_mean_energy(
            self.__delta, self.__lower_cutoff_energy_mec2)


class HeatedAtmosphere(Atmosphere):
    def __init__(self, atmosphere, distribution):
        super().__init__(atmosphere.full_depths,
                         atmosphere.full_distances,
                         atmosphere.full_mass_densities,
                         atmosphere.full_temperatures,
                         atmosphere.full_electron_densities,
                         start_depth=atmosphere.start_depth)

        global_electron_energy = 2.0*units.KEV_TO_ERG/units.MC2_ELECTRON

        self.electron_coulomb_logarithm = compute_electron_coulomb_logarithm(
            self.electron_densities[0],
            global_electron_energy)  #distribution.mean_energy_mec2)

        self.neutral_hydrogen_coulomb_logarithm = compute_neutral_hydrogen_coulomb_logarithm(
            global_electron_energy)

        self.stopping_ionized_column_depth = compute_stopping_column_depth(
            distribution.pitch_angle_cosine,
            distribution.lower_cutoff_energy_mec2,
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

        heating_scale = compute_heating_scale(
            distribution.total_power, distribution.delta,
            distribution.pitch_angle_cosine,
            distribution.lower_cutoff_energy_mec2)

        self.beam_heating = heating_scale \
            * self.betas \
            * self.total_hydrogen_densities \
            * self.effective_coulomb_logarithms \
            * self.equivalent_ionized_column_depth_ratios**(-0.5 * distribution.delta)

        self.beam_heating[0] = 0.0

        self.remaining_beam_powers = compute_remaining_beam_power(
            distribution.total_power, self.distances, self.beam_heating)


class HeatedAtmosphereSFP(Atmosphere):
    def __init__(self, atmosphere, distribution):
        super().__init__(atmosphere.full_depths,
                         atmosphere.full_distances,
                         atmosphere.full_mass_densities,
                         atmosphere.full_temperatures,
                         atmosphere.full_electron_densities,
                         start_depth=atmosphere.start_depth)

        global_electron_energy = 2.0*units.KEV_TO_ERG/units.MC2_ELECTRON

        self.neutral_hydrogen_densities = compute_equilibrium_neutral_hydrogen_density(
            self.mass_densities, self.temperatures, self.electron_densities)

        self.collisional_depth_derivatives = compute_collisional_depth_derivative_SFP(
            self.electron_densities, self.neutral_hydrogen_densities, 2.0,
            global_electron_energy)
        self.collisional_depths = compute_cumulative_integral_over_distance(
            self.distances, self.collisional_depth_derivatives)

        self.beam_heating = compute_beam_heating_SFP(
            distribution.delta, distribution.total_power,
            distribution.lower_cutoff_energy_mec2, self.collisional_depths,
            self.collisional_depth_derivatives)

        self.remaining_beam_powers = compute_remaining_power_SFP(
            distribution.delta, distribution.total_power,
            distribution.lower_cutoff_energy_mec2, self.collisional_depths)
