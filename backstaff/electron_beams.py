import collections
import functools
import numpy as np
from pathlib import Path
try:
    import backstaff.units as units
    import backstaff.plotting as plotting
    import backstaff.field_lines as field_lines
except ModuleNotFoundError:
    import units
    import plotting
    import field_lines


class ElectronBeamSwarm(field_lines.FieldLineSet3):

    VALUE_DESCRIPTIONS = {
        'x': r'$x$ [Mm]',
        'y': r'$y$ [Mm]',
        'z': 'Height [Mm]',
        's': 'Distance [Mm]',
        'total_power': 'Total power [erg/s]',
        'total_power_density': r'Total power density [erg/(cm$^3\;$s)]',
        'lower_cutoff_energy': 'Lower cut-off energy [keV]',
        'acceleration_volume': r'Acceleration site volume [cm$^3$]',
        'estimated_thermalization_distance':
        'estimated thermalization distance [Mm]',
        'total_propagation_distance': 'Total propagation distance [Mm]',
        'acceleration_height': 'Acceleration site height [Mm]',
        'depletion_height': 'Depletion height [Mm]',
        'beam_electron_fraction': 'Beam electrons relative to total electrons',
        'return_current_speed_fraction': 'Speed relative to speed of light',
        'estimated_electron_density': r'Electron density [electrons/cm$^3$]',
        'deposited_power': 'Deposited power [erg/s]',
        'power_change': 'Power change [erg/s]',
        'remaining_power': 'Remaining power [erg/s]',
        'r': r'Mass density [g/cm$^3$]',
        'tg': 'Temperature [K]',
        'nel': r'Electron density [electrons/cm$^3$]',
        'krec': 'Reconnection factor [Bifrost units]',
        'qspitz': r'Power density change [erg/(cm$^3\;$s)]',
        'r0': r'Mass density [g/cm$^3$]',
        'tg0': 'Temperature [K]',
    }

    VALUE_UNIT_CONVERTERS = {
        'r': lambda f: f*units.U_R,
        'qspitz': lambda f: f*units.U_E/units.U_T,
        'r0': lambda f: f*units.U_R,
        'z': lambda f: -f,
        'z0': lambda f: -f,
    }

    @staticmethod
    def from_file(file_path,
                  acceleration_data_type=None,
                  params={},
                  derived_quantities=[],
                  verbose=False):
        import backstaff.reading as reading
        file_path = Path(file_path)
        extension = file_path.suffix
        if extension == '.pickle':
            electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(
                file_path,
                acceleration_data_type=acceleration_data_type,
                params=params,
                derived_quantities=derived_quantities,
                verbose=verbose)
        elif extension == '.fl':
            electron_beam_swarm = reading.read_electron_beam_swarm_from_custom_binary_file(
                file_path,
                acceleration_data_type=acceleration_data_type,
                params=params,
                derived_quantities=derived_quantities,
                verbose=verbose)
        else:
            raise ValueError(
                'Invalid file extension {} for electron beam data.'.format(
                    extension))

        return electron_beam_swarm

    def __init__(self,
                 domain_bounds,
                 number_of_beams,
                 fixed_scalar_values,
                 fixed_vector_values,
                 varying_scalar_values,
                 varying_vector_values,
                 acceleration_data,
                 params={},
                 derived_quantities=[],
                 verbose=False):
        assert isinstance(acceleration_data, dict)
        self.number_of_beams = number_of_beams
        self.acceleration_data = acceleration_data
        super().__init__(domain_bounds,
                         number_of_beams,
                         fixed_scalar_values,
                         fixed_vector_values,
                         varying_scalar_values,
                         varying_vector_values,
                         params=params,
                         derived_quantities=derived_quantities,
                         verbose=verbose)

        if self.verbose:
            print('Acceleration data:\n    {}'.format('\n    '.join(
                self.acceleration_data.keys())))

    def get_number_of_beams(self):
        return self.number_of_beams

    def get_acceleration_data(self, acceleration_data_type):
        return self.acceleration_data[acceleration_data_type]

    def get_acceleration_sites(self):
        return self.get_acceleration_data('acceleration_sites')

    def _derive_quantities(self, derived_quantities):

        for value_name in filter(
                lambda name: name[-1] == '0' and self.
                has_varying_scalar_values(name[:-1]) and not self.
                has_fixed_scalar_values(name[:-1]), derived_quantities):

            self.fixed_scalar_values[value_name] = np.asfarray([
                values[0]
                for values in self.get_varying_scalar_values(value_name[:-1])
            ])

        if 'total_power_density' in derived_quantities:
            self.fixed_scalar_values[
                'total_power_density'] = self.get_fixed_scalar_values(
                    'total_power')*self.get_fixed_scalar_values(
                        'acceleration_volume')

        if 'mean_electron_energy' in derived_quantities:
            self._obtain_mean_electron_energies()

        if 'acceleration_height' in derived_quantities:
            self.fixed_scalar_values['acceleration_height'] = np.asfarray(
                [-z[0] for z in self.get_varying_scalar_values('z')])

        if 'depletion_height' in derived_quantities:
            self.fixed_scalar_values['depletion_height'] = np.asfarray(
                [-z[-1] for z in self.get_varying_scalar_values('z')])

        if 'acceleration_site_electron_density' in derived_quantities:
            self._obtain_acceleration_site_electron_densities()

        if 'beam_electron_fraction' in derived_quantities:
            self._obtain_beam_electron_fractions()

        if 'return_current_speed_fraction' in derived_quantities:
            mean_electron_energies = self._obtain_mean_electron_energies(
            )*units.KEV_TO_ERG
            mean_electron_speed_fractions = np.sqrt(
                1.0 - 1.0/(1.0 + mean_electron_energies/units.MC2_ELECTRON)**2)
            beam_electron_fractions = self._obtain_beam_electron_fractions()
            self.fixed_scalar_values[
                'return_current_speed_fraction'] = beam_electron_fractions*mean_electron_speed_fractions

        if 'estimated_electron_density' in derived_quantities:
            assert self.has_varying_scalar_values('r')
            self.varying_scalar_values['estimated_electron_density'] = [
                self.varying_scalar_values['r'][i]*units.U_R*
                units.MASS_DENSITY_TO_ELECTRON_DENSITY
                for i in range(self.get_number_of_beams())
            ]

        if 's' in derived_quantities:
            assert self.has_param('dense_step_length')
            ds = self.get_param('dense_step_length')
            self.varying_scalar_values['s'] = [
                np.arange(len(x))*ds
                for x in self.get_varying_scalar_values('x')
            ]

        if 'power_change' in derived_quantities:
            self.varying_scalar_values['power_change'] = [
                arr.copy()
                for arr in self.varying_scalar_values['deposited_power']
            ]
            for i in range(self.get_number_of_beams()):
                self.varying_scalar_values['power_change'][i][
                    0] -= self.fixed_scalar_values['total_power'][i]

        if 'remaining_power' in derived_quantities:
            self.varying_scalar_values['remaining_power'] = [
                self.fixed_scalar_values['total_power'][i] -
                np.cumsum(self.varying_scalar_values['deposited_power'][i])
                for i in range(self.get_number_of_beams())
            ]

    def _obtain_mean_electron_energies(self):
        if not self.has_fixed_scalar_values('mean_electron_energy'):
            assert self.has_param('power_law_delta')
            delta = self.get_param('power_law_delta')
            self.fixed_scalar_values['mean_electron_energy'] = (
                (delta - 1.0)/(delta - 2.0)
            )*self.get_fixed_scalar_values('lower_cutoff_energy')

        return self.get_fixed_scalar_values('mean_electron_energy')

    def _obtain_acceleration_site_electron_densities(self):
        if not self.has_fixed_scalar_values(
                'acceleration_site_electron_density'):
            assert self.has_fixed_scalar_values(
                'r0') or self.has_fixed_scalar_values('r')
            self.fixed_scalar_values[
                'acceleration_site_electron_density'] = self.get_fixed_scalar_values(
                    'r0' if self.has_fixed_scalar_values('r0') else 'r'
                )*units.U_R*units.MASS_DENSITY_TO_ELECTRON_DENSITY

        return self.get_fixed_scalar_values(
            'acceleration_site_electron_density')

    def _obtain_beam_electron_fractions(self):
        if not self.has_fixed_scalar_values('beam_electron_fraction'):
            assert self.has_param('particle_energy_fraction')
            assert self.has_fixed_scalar_values(
                'bx') and self.has_fixed_scalar_values(
                    'by') and self.has_fixed_scalar_values('bz')
            assert self.has_fixed_scalar_values(
                'ix') and self.has_fixed_scalar_values(
                    'iy') and self.has_fixed_scalar_values('iz')

            bx = self.get_fixed_scalar_values('bx')*units.U_B
            by = self.get_fixed_scalar_values('by')*units.U_B
            bz = self.get_fixed_scalar_values('bz')*units.U_B
            ix = self.get_fixed_scalar_values('ix')
            iy = self.get_fixed_scalar_values('iy')
            iz = self.get_fixed_scalar_values('iz')
            free_energy = (bx*bx + by*by + bz*bz - (bx*ix + by*iy + bz*iz)**2/
                           (ix*ix + iy*iy + iz*iz))/(8.0*np.pi)
            mean_electron_energies = self._obtain_mean_electron_energies(
            )*units.KEV_TO_ERG
            electron_densities = self._obtain_acceleration_site_electron_densities(
            )
            self.fixed_scalar_values[
                'beam_electron_fraction'] = self.get_param(
                    'particle_energy_fraction')*free_energy/(
                        mean_electron_energies*electron_densities)

        return self.get_fixed_scalar_values('beam_electron_fraction')


class AccelerationSites(field_lines.FieldLineSet3):

    VALUE_DESCRIPTIONS = ElectronBeamSwarm.VALUE_DESCRIPTIONS
    VALUE_UNIT_CONVERTERS = ElectronBeamSwarm.VALUE_UNIT_CONVERTERS

    def __init__(self,
                 domain_bounds,
                 number_of_sites,
                 fixed_scalar_values,
                 fixed_vector_values,
                 varying_scalar_values,
                 varying_vector_values,
                 params={},
                 derived_quantities=[],
                 verbose=False):
        self.number_of_sites = number_of_sites
        super().__init__(domain_bounds,
                         number_of_sites,
                         fixed_scalar_values,
                         fixed_vector_values,
                         varying_scalar_values,
                         varying_vector_values,
                         params=params,
                         derived_quantities=derived_quantities,
                         verbose=verbose)

    def get_number_of_sites(self):
        return self.number_of_sites


def find_beams_propagating_longer_than_distance(min_distance,
                                                fixed_scalar_values):
    return list(
        np.nonzero(
            fixed_scalar_values['total_propagation_distance'] > min_distance)
        [0])


def plot_electron_beams(*args, **kwargs):
    field_lines.plot_field_lines(*args, **kwargs)


def plot_electron_beam_properties(*args, **kwargs):
    field_lines.plot_field_line_properties(*args, **kwargs)


def plot_beam_value_histogram(*args, **kwargs):
    field_lines.plot_field_line_value_histogram(*args, **kwargs)


def plot_beam_value_histogram_difference(*args, **kwargs):
    field_lines.plot_field_line_value_histogram_difference(*args, **kwargs)


def plot_beam_value_2d_histogram(*args, **kwargs):
    field_lines.plot_field_line_value_2d_histogram(*args, **kwargs)


def plot_beam_value_2d_histogram_difference(*args, **kwargs):
    field_lines.plot_field_line_value_2d_histogram_difference(*args, **kwargs)


def plot_beam_value_2d_histogram_comparison(*args, **kwargs):
    field_lines.plot_field_line_value_2d_histogram_comparison(*args, **kwargs)
