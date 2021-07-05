import numpy as np
from pathlib import Path
try:
    import backstaff.reading as reading
    import backstaff.units as units
    import backstaff.plotting as plotting
except ModuleNotFoundError:
    import reading
    import units
    import plotting


class Cork:
    def __init__(self, positions, velocities, scalar_field_values,
                 vector_field_values, first_time_idx):
        assert isinstance(positions, np.ndarray)
        assert isinstance(velocities, np.ndarray)
        assert isinstance(scalar_field_values, list)
        assert isinstance(vector_field_values, list)
        assert isinstance(first_time_idx, int)
        self.positions = positions
        self.velocities = velocities
        self.scalar_field_values = scalar_field_values
        self.vector_field_values = vector_field_values
        self.first_time_idx = first_time_idx

    def get_number_of_times(self):
        return self.positions.shape[1]

    def exists_at(self, time_idx):
        return time_idx >= self.first_time_idx and (
            time_idx - self.first_time_idx) < self.get_number_of_times()

    def get_time_slice(self):
        return slice(self.first_time_idx,
                     self.first_time_idx + self.get_number_of_times())


class CorkSet:

    VALUE_DESCRIPTIONS = {
        'x': r'$x$ [Mm]',
        'y': r'$y$ [Mm]',
        'z': 'Height [Mm]',
        'r': r'Mass density [g/cm$^3$]',
        'tg': 'Temperature [K]',
        'p': r'Gas pressure [dyn/cm$^2$]',
    }

    VALUE_UNIT_CONVERTERS = {
        'r': lambda f: f*units.U_R,
        'p': lambda f: f*units.U_P,
        'z': lambda f: -f,
        'bx': lambda f: f*units.U_B,
        'by': lambda f: f*units.U_B,
        'bz': lambda f: f*units.U_B,
    }

    @staticmethod
    def from_file(file_path, params={}, derived_quantities=[], verbose=False):
        file_path = Path(file_path)
        extension = file_path.suffix
        if extension == '.pickle':
            cork_set = reading.read_cork_set_from_pickle(
                file_path,
                params=params,
                derived_quantities=derived_quantities,
                verbose=verbose)
        else:
            raise ValueError(
                'Invalid file extension {} for cork data.'.format(extension))

        return cork_set

    def __init__(self,
                 corks,
                 times,
                 domain_bounds,
                 scalar_quantity_names,
                 vector_magnitude_names,
                 vector_quantity_names,
                 params={},
                 derived_quantities=[],
                 verbose=False):
        assert all([upper >= lower for lower, upper in domain_bounds])
        assert isinstance(corks, list)
        assert isinstance(times, np.ndarray)
        assert isinstance(derived_quantities, list)
        assert isinstance(params, dict)
        self.bounds_x, self.bounds_y, self.bounds_z = tuple(domain_bounds)
        self.bounds_z = (-self.bounds_z[1], -self.bounds_z[0]
                         )  # Use height instead of depth
        self.corks = corks
        self.times = times
        self.scalar_quantity_names = scalar_quantity_names
        self.vector_magnitude_names = vector_magnitude_names
        self.vector_quantity_names = vector_quantity_names
        self.params = params
        self._derive_quantities(derived_quantities)
        self.verbose = bool(verbose)

        if self.verbose:
            print('Scalar quantity names:\n    {}'.format('\n    '.join(
                self.scalar_quantity_names)))
            print('Vector magnitude names:\n    {}'.format('\n    '.join(
                self.vector_magnitude_names)))
            print('Vector quantity names:\n    {}'.format('\n    '.join(
                self.vector_quantity_names)))
            print('Parameters:\n    {}'.format('\n    '.join(
                self.params.keys())))

    def _derive_quantities(self, derived_quantities):
        pass

    def get_cork_times(self, cork_idx):
        return self.times[self.corks[cork_idx].get_time_slice()]

    def get_cork_positions(self, cork_idx):
        return self.corks[cork_idx].positions

    def get_cork_velocities(self, cork_idx):
        return self.corks[cork_idx].velocities

    def get_scalar_quantity_idx(self, quantity_name):
        return self.vector_magnitude_names.index(
            quantity_name
        ) if quantity_name in self.vector_magnitude_names else self.scalar_quantity_names.index(
            quantity_name)

    def get_vector_quantity_idx(self, quantity_name):
        return self.vector_quantity_names.index(quantity_name)

    def get_cork_scalar_quantity_evolution(self, quantity_name, cork_idx):
        quantity_idx = self.get_scalar_quantity_idx(quantity_name)
        return self.corks[cork_idx].scalar_field_values[quantity_idx]

    def get_cork_vector_quantity_evolution(self, quantity_name, cork_idx):
        quantity_idx = self.get_vector_quantity_idx(quantity_name)
        return self.corks[cork_idx].vector_field_values[quantity_idx]

    def get_time(self, time_idx):
        return self.times[time_idx]

    def get_positions_at_time(self, time_idx):
        return np.stack([
            cork.positions[:, time_idx] for cork in self.corks
            if cork.exists_at(time_idx)
        ])

    def get_velocities_at_time(self, time_idx):
        return np.stack([
            cork.velocities[:, time_idx] for cork in self.corks
            if cork.exists_at(time_idx)
        ])

    def get_scalar_quantity_at_time(self, quantity_name, time_idx):
        quantity_idx = self.get_scalar_quantity_idx(quantity_name)
        return np.stack([
            cork.scalar_field_values[quantity_idx, time_idx]
            for cork in self.corks if cork.exists_at(time_idx)
        ])

    def get_vector_quantity_at_time(self, quantity_name, time_idx):
        quantity_idx = self.get_vector_quantity_idx(quantity_name)
        return np.stack([
            cork.vector_field_values[quantity_idx, :, time_idx]
            for cork in self.corks if cork.exists_at(time_idx)
        ])


if __name__ == '__main__':
    corks = CorkSet.from_file('corks.pickle', verbose=True)
    print(corks.get_positions_at_time(0).shape)
