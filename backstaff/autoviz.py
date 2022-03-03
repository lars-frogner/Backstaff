#!/usr/bin/env python
import os
import sys
import re
import pathlib
import logging
import shutil
import csv
import yaml
import numpy as np
from matplotlib.offsetbox import AnchoredText
from helita.sim.bifrost import BifrostData

try:
    import backstaff.units as units
    import backstaff.running as running
    import backstaff.fields as fields
    import backstaff.plotting as plotting
except ModuleNotFoundError:
    import units
    import running
    import fields
    import plotting


def update_dict_nested(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict_nested(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def error(logger, *args, **kwargs):
    logger.error(*args, **kwargs)
    sys.exit(1)


class Quantity:
    def __init__(self, name, unit_scale, description, cmap_name):
        self.name = name
        self.unit_scale = float(unit_scale)
        self.description = description
        self.cmap_name = cmap_name

    def get_plot_kwargs(self):
        return dict(clabel=self.description, cmap_name=self.cmap_name)

    @property
    def tag(self):
        return self.name

    def is_available(self, bifrost_data):
        if hasattr(bifrost_data, self.name):
            return True
        try:
            bifrost_data.get_var(self.name)
            return True
        except:
            return False

    @classmethod
    def parse_file(cls, file_path, logger=logging):
        def decomment(csvfile):
            for row in csvfile:
                raw = row.split('#')[0].strip()
                if raw:
                    yield raw

        logger.debug(f'Parsing {file_path}')

        file_path = pathlib.Path(file_path)

        quantities = {}
        with open(file_path, newline='') as f:
            reader = csv.reader(decomment(f))
            for row in reader:
                if len(row) == 3:
                    row.append('')

                if len(row) != 4:
                    error(
                        logger,
                        f'Invalid number of entries in CSV line in {file_path}: {", ".join(row)}'
                    )

                name, unit, description, cmap_name = row

                name = name.strip()

                unit = unit.strip()
                try:
                    unit = float(unit)
                except ValueError:
                    try:
                        unit = getattr(units, unit)
                    except AttributeError:
                        if len(unit) != 0:
                            logger.warning(
                                f'Unit {unit} for quantity {name} in {file_path} not recognized, using 1.0'
                            )
                        unit = 1.0

                description = description.strip()

                cmap_name = cmap_name.strip()
                if len(cmap_name) == 0:
                    cmap_name = 'viridis'

                logger.debug(
                    f'Using quantity info: {name}, {unit:g}, {description}, {cmap_name}'
                )

                quantities[name] = cls(name, unit, description, cmap_name)

        return quantities


class Reduction:
    def __init__(self, axis):
        self.axis = int(axis)
        self.axis_names = ['x', 'y', 'z']

    @staticmethod
    def parse(reduction_config, logger=logging):
        if not isinstance(reduction_config, dict):
            error(
                logger,
                f'reduction entry must be dict, is {type(reduction_config)}')

        classes = dict(accumulation=Accumulation, slice=Slice)
        reduction = None
        for name, cls in classes.items():
            if name in reduction_config:
                logger.debug(f'Found reduction {name}')
                reduction = cls(**reduction_config[name])
        if reduction is None:
            error('Missing reduction entry')

        return reduction

    def get_plot_kwargs(self, field):
        plot_axis_names = list(self.axis_names)
        plot_axis_names.pop(self.axis)
        hor_coords = field.get_horizontal_coords()
        vert_coords = field.get_vertical_coords()
        aspect_ratio = (hor_coords[-1] - hor_coords[0]) / (
            vert_coords[-1] - vert_coords[0]) + 0.25
        return dict(xlabel=f'{plot_axis_names[0]} [Mm]',
                    ylabel=f'{plot_axis_names[1]} [Mm]',
                    fig_kwargs=dict(aspect_ratio=aspect_ratio))


class Accumulation(Reduction):
    @property
    def tag(self):
        return f'accum_{self.axis_names[self.axis]}'

    def __call__(self, bifrost_data, quantity):
        return fields.ScalarField2.accumulated_from_bifrost_data(
            bifrost_data,
            quantity.name,
            accum_axis=self.axis,
            scale=quantity.unit_scale)


class Slice(Reduction):
    def __init__(self, axis=0, coord=0.0):
        super().__init__(axis)
        self.coord = float(coord)

    @property
    def tag(self):
        return f'slice_{self.axis_names[self.axis]}_{self.coord:g}'

    def __call__(self, bifrost_data, quantity):
        return fields.ScalarField2.slice_from_bifrost_data(
            bifrost_data,
            quantity.name,
            slice_axis=self.axis,
            slice_coord=self.coord,
            scale=quantity.unit_scale)


class Scaling:
    def __init__(self, vmin=None, vmax=None) -> None:
        self.vmin = vmin
        self.vmax = vmax

    @staticmethod
    def parse(scaling_config, logger=logging):
        classes = dict(linear=LinearScaling,
                       log=LogScaling,
                       symlog=SymlogScaling)
        if isinstance(scaling_config, dict):
            scaling = None
            for name, cls in classes.items():
                if name in scaling_config:
                    logger.debug(f'Found scaling {name}')
                    scaling = cls(**scaling_config[name])
            if scaling is None:
                error('Missing reduction entry')
        elif isinstance(scaling_config, str):
            logger.debug(f'Found scaling type {scaling_config}')
            scaling = classes[scaling_config]()
        else:
            error(
                f'scaling entry must be dict or str, is {type(scaling_config)}'
            )

        return scaling


class SymlogScaling(Scaling):
    def __init__(self,
                 linthresh=None,
                 vmax=None,
                 linthresh_quantile=0.2) -> None:
        self.linthresh = linthresh
        self.vmax = vmax
        self.linthresh_quantile = float(linthresh_quantile)

    @property
    def tag(self):
        return 'symlog'

    def get_plot_kwargs(self, field):
        linthresh = self.linthresh
        vmax = self.vmax
        if linthresh is None or vmax is None:
            values = np.abs(field.get_values())
            if linthresh is None:
                linthresh = np.quantile(values, self.linthresh_quantile)
            if vmax is None:
                vmax = values.max()

        return dict(symlog=True, vmin=-vmax, vmax=vmax, linthresh=linthresh)


class LinearScaling(Scaling):
    @property
    def tag(self):
        return 'linear'

    def get_plot_kwargs(self, *args):
        return dict(log=False, vmin=self.vmin, vmax=self.vmax)


class LogScaling(LinearScaling):
    @property
    def tag(self):
        return 'log'

    def get_plot_kwargs(self, *args):
        plot_kwargs = super().get_plot_kwargs(*args)
        plot_kwargs['log'] = True
        return plot_kwargs


class PlotDescription:
    def __init__(self, quantity, reduction, scaling, **extra_plot_kwargs):
        self.quantity = quantity
        self.reduction = reduction
        self.scaling = scaling
        self.extra_plot_kwargs = extra_plot_kwargs

    @classmethod
    def parse(cls, quantities, plot_config, logger=logging):
        if not isinstance(plot_config, dict):
            error(logger,
                  f'plots list entry must be dict, is {type(plot_config)}')

        if 'quantity' not in plot_config:
            error(logger, f'Missing entry quantity')

        quantity = plot_config.pop('quantity')

        if not isinstance(quantity, str):
            error(logger, f'quantity entry must be str, is {type(quantity)}')

        logger.debug(f'Found quantity {quantity}')

        if quantity not in quantities:
            error(logger, f'Quantity {quantity} not present in quantity file')

        quantity = quantities[quantity]

        if 'reduction' not in plot_config:
            error(logger, f'Missing reduction entry')

        reduction = Reduction.parse(plot_config.pop('reduction'),
                                    logger=logger)

        if 'scaling' not in plot_config:
            error(logger, f'Missing scaling entry')

        scaling = Scaling.parse(plot_config.pop('scaling'), logger=logger)

        return cls(quantity, reduction, scaling, **plot_config)

    @property
    def tag(self):
        return f'{self.scaling.tag}_{self.quantity.tag}_{self.reduction.tag}'

    def get_plot_kwargs(self, field):
        return update_dict_nested(
            update_dict_nested(
                update_dict_nested(self.quantity.get_plot_kwargs(),
                                   self.reduction.get_plot_kwargs(field)),
                self.scaling.get_plot_kwargs(field)), self.extra_plot_kwargs)

    def get_field(self, bifrost_data):
        return self.reduction(bifrost_data, self.quantity)


class SimulationRun:
    def __init__(self,
                 name,
                 data_dir,
                 start_snap_num=None,
                 end_snap_num=None,
                 logger=logging):
        self._name = name
        self._data_dir = data_dir
        self._start_snap_num = start_snap_num
        self._end_snap_num = end_snap_num
        self._logger = logger

        self._active_data_dir = data_dir

        self._snap_nums = self._find_snap_nums()

    @classmethod
    def parse(cls, simulation_run_config, logger=logging):

        simulation_name = simulation_run_config.get(
            'name', simulation_run_config.get('simulation_name', None))

        if simulation_name is None:
            error(logger, f'Missing simulation_name entry')

        if not isinstance(simulation_name, str):
            error(
                logger,
                f'simulation_name entry must be str, is {type(simulation_name)}'
            )

        simulation_name = simulation_name.strip()

        logger.debug(f'Using simulation_name {simulation_name}')

        simulation_dir = simulation_run_config.get(
            'dir', simulation_run_config.get('simulation_dir', None))

        if simulation_dir is None:
            error(logger, f'Missing entry simulation_dir')

        if simulation_dir is None:
            error(logger, 'Missing simulation_dir entry')

        if not isinstance(simulation_dir, str):
            error(
                logger,
                f'simulation_dir entry must be str, is {type(simulation_dir)}')

        simulation_dir = pathlib.Path(simulation_dir)

        if not simulation_dir.is_absolute():
            error(
                logger,
                f'simulation_dir entry {simulation_dir} must be an absolute path'
            )

        if not simulation_dir.is_dir():
            error(logger,
                  f'Could not find simulation_dir directory {simulation_dir}')

        simulation_dir = simulation_dir.resolve()

        logger.debug(f'Using simulation_dir {simulation_dir}')

        if 'start_snap_num' in simulation_run_config and simulation_run_config[
                'start_snap_num'] is not None:
            start_snap_num = simulation_run_config['start_snap_num']

            if not isinstance(start_snap_num, int):
                error(
                    logger,
                    f'start_snap_num entry must be int, is {type(start_snap_num)}'
                )

            start_snap_num = int(start_snap_num)
        else:
            start_snap_num = None

        logger.debug(f'Using start_snap_num {start_snap_num}')

        if 'end_snap_num' in simulation_run_config and simulation_run_config[
                'end_snap_num'] is not None:
            end_snap_num = simulation_run_config['end_snap_num']

            if not isinstance(end_snap_num, int):
                error(
                    logger,
                    f'end_snap_num entry must be int, is {type(end_snap_num)}')

            end_snap_num = int(end_snap_num)
        else:
            end_snap_num = None

        logger.debug(f'Using end_snap_num {end_snap_num}')

        return cls(simulation_name,
                   simulation_dir,
                   start_snap_num=start_snap_num,
                   end_snap_num=end_snap_num,
                   logger=logger)

    @property
    def logger(self):
        return self._logger

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def data_available(self):
        return len(self._snap_nums) > 0

    def __iter__(self):
        return iter(self._snap_nums)

    def __call__(self, snap_num):
        return self._get_bifrost_data(snap_num)

    def ensure_data_is_ready(self, prepared_data_dir, plot_descriptions):
        assert self.data_available
        self._active_data_dir = self._data_dir

        bifrost_data = self._get_bifrost_data(self._snap_nums[0])
        available_quantities, unavailable_quantities = self._check_quantity_availability(
            bifrost_data, plot_descriptions)
        if len(unavailable_quantities) == 0:
            return plot_descriptions

        if prepared_data_dir.is_dir():
            try:
                prepared_bifrost_data = self._get_bifrost_data(
                    self._snap_nums[0], other_data_dir=prepared_data_dir)
            except:
                prepared_bifrost_data = None

            if prepared_bifrost_data is not None:
                _, unavailable_prepared_quantities = self._check_quantity_availability(
                    prepared_bifrost_data, plot_descriptions)
                if len(unavailable_prepared_quantities) == 0:
                    self._active_data_dir = prepared_data_dir
                    prepared_snap_nums = self._find_snap_nums(
                        other_data_dir=prepared_data_dir)
                    missing_snap_nums = [
                        snap_num for snap_num in self._snap_nums
                        if snap_num not in prepared_snap_nums
                    ]
                    if len(missing_snap_nums) > 0:
                        self._prepare_data(prepared_data_dir,
                                           available_quantities,
                                           unavailable_quantities,
                                           snap_nums=missing_snap_nums)
                    return plot_descriptions

        self._prepare_data(prepared_data_dir, available_quantities,
                           unavailable_quantities)

        prepared_bifrost_data = self._get_bifrost_data(
            self._snap_nums[0], other_data_dir=prepared_data_dir)
        return self._filter_out_unavailable_plot_descriptions(
            prepared_bifrost_data, plot_descriptions)

    def _filter_out_unavailable_plot_descriptions(self, bifrost_data,
                                                  plot_descriptions):
        available_quantities, _ = self._check_quantity_availability(
            bifrost_data, plot_descriptions)

        def available(plot_description):
            quantity_name = plot_description.quantity.name
            is_available = quantity_name in available_quantities
            if not is_available:
                self.logger.warning(
                    f'Could not obtain quantity {quantity_name} for simulation {self.name}, skipping'
                )
            return is_available

        return list(filter(available, plot_descriptions))

    def _get_bifrost_data(self, snap_num, other_data_dir=None):
        fdir = self._active_data_dir if other_data_dir is None else other_data_dir
        self.logger.debug(f'Reading snap {snap_num} of {self.name} in {fdir}')
        assert snap_num in self._snap_nums
        return BifrostData(self.name, fdir=fdir, snap=snap_num, verbose=False)

    def _find_snap_nums(self, other_data_dir=None):
        input_dir = self._active_data_dir if other_data_dir is None else other_data_dir
        snap_nums = self._find_all_snap_nums(input_dir)
        if self._start_snap_num is not None:
            snap_nums = [n for n in snap_nums if n >= self._start_snap_num]
        if self._end_snap_num is not None:
            snap_nums = [n for n in snap_nums if n <= self._end_snap_num]
        self.logger.debug(
            f'Found snaps {", ".join(map(str, snap_nums))} in {input_dir}')
        return snap_nums

    def _find_all_snap_nums(self, input_dir):
        p = re.compile('{}_(\d\d\d)\.idl$'.format(self.name))
        snap_nums = []
        for name in os.listdir(input_dir):
            match = p.match(name)
            if match:
                snap_nums.append(int(match.group(1)))
        return sorted(snap_nums)

    def _check_quantity_availability(self, bifrost_data, plot_descriptions):
        available_quantities = []
        unavailable_quantities = []
        for plot_description in plot_descriptions:
            quantity_name = plot_description.quantity.name
            if quantity_name not in available_quantities + unavailable_quantities:
                if plot_description.quantity.is_available(bifrost_data):
                    self.logger.debug(
                        f'Quantity {quantity_name} available for {bifrost_data.file_root}'
                    )
                    available_quantities.append(quantity_name)
                else:
                    self.logger.debug(
                        f'Quantity {quantity_name} not available for {bifrost_data.file_root}'
                    )
                    unavailable_quantities.append(quantity_name)
        return available_quantities, unavailable_quantities

    def _prepare_data(self,
                      prepared_data_dir,
                      available_quantities,
                      unavailable_quantities,
                      snap_nums=None):
        self._active_data_dir = prepared_data_dir
        os.makedirs(prepared_data_dir, exist_ok=True)

        if snap_nums is None:
            snap_nums = self._snap_nums

        param_file_name = f'{self.name}_{self._snap_nums[0]}.idl'

        quantity_specification = []
        if len(available_quantities) > 0:
            quantity_specification.append('--included-quantities=' +
                                          ','.join(available_quantities))
        if len(unavailable_quantities) > 0:
            quantity_specification.append('--derived-quantities=' +
                                          ','.join(unavailable_quantities))

        snap_range_specification = [
            f'--snap-range={snap_nums[0]},{snap_nums[-1]}'
        ] if len(snap_nums) > 1 else []

        return_code = running.run_command([
            'backstaff', '--protected-file-types=', 'snapshot',
            *snap_range_specification, param_file_name, 'write',
            '--ignore-warnings', '--overwrite', *quantity_specification,
            str((prepared_data_dir / param_file_name).resolve())
        ],
                                          cwd=self._data_dir,
                                          logger=self.logger.debug,
                                          error_logger=self.logger.error)
        if return_code != 0:
            error(self.logger, 'Non-zero return code')

        for snap_num in snap_nums:
            snap_path = self._data_dir / f'{self.name}_{snap_num:03}.snap'

            return_code = running.run_command(
                ['ln', '-s',
                 str(snap_path),
                 str(prepared_data_dir)],
                logger=self.logger.debug,
                error_logger=self.logger.error)

            if return_code != 0:
                error(self.logger, 'Non-zero return code')


class Visualizer:
    def __init__(self, simulation_run, output_dir_name='autoviz'):
        self._simulation_run = simulation_run
        self._logger = simulation_run.logger

        self._output_dir = self._simulation_run.data_dir / output_dir_name
        self._prepared_data_dir = self._output_dir / 'data'

        self._logger.debug(f'Using output directory {self._output_dir}')

    @property
    def logger(self):
        return self._logger

    @property
    def simulation_name(self):
        return self._simulation_run.name

    def clean(self):
        if not self._output_dir.is_dir():
            print('No data to clean')
            return

        print(
            f'The directory {self._output_dir} and all its content will be removed'
        )
        while True:
            answer = input('Continue? [y/N] ').strip().lower()
            if answer in ('', 'n'):
                print('Aborted')
                break
            if answer == 'y':
                shutil.rmtree(self._output_dir)
                self.logger.debug(f'Removed {self._output_dir}')
                break

    def visualize(self, *plot_descriptions, overwrite=False):
        if not self._simulation_run.data_available:
            self.logger.warning(
                f'No data for simulation {self._simulation_run.name} in {self._simulation_run.data_dir}, aborting'
            )
            return

        plot_descriptions = self._simulation_run.ensure_data_is_ready(
            self._prepared_data_dir, plot_descriptions)

        for snap_num in self._simulation_run:
            for plot_description in plot_descriptions:
                frame_dir = self._output_dir / plot_description.tag
                os.makedirs(frame_dir, exist_ok=True)

                output_path = frame_dir / f'{snap_num}.png'

                if output_path.exists() and not overwrite:
                    self.logger.debug(
                        f'{output_path} already exists, skipping')
                    continue

                bifrost_data = self._simulation_run(snap_num)
                var = bifrost_data.get_var('ubeam')
                self.logger.debug(
                    f'Plotting frame for snap {snap_num} of {plot_description.tag}'
                )
                self._plot_frame(bifrost_data, plot_description, output_path)

    def _plot_frame(self, bifrost_data, plot_description, output_path):

        time = float(bifrost_data.params['t']) * units.U_T
        time_text = AnchoredText(f'{time:.1f} s', 'upper left', frameon=False)

        field = plot_description.get_field(bifrost_data)

        field.plot(output_path=output_path,
                   extra_artists=[time_text],
                   **plot_description.get_plot_kwargs(field))


class Visualization:
    def __init__(self, visualizer, *plot_descriptions):
        self._visualizer = visualizer
        self._plot_descriptions = plot_descriptions

    @property
    def visualizer(self):
        return self._visualizer

    def clean(self, *args, **kwargs):
        self.visualizer.clean(*args, **kwargs)

    def visualize(self, **kwargs):
        self.visualizer.visualize(*self._plot_descriptions, **kwargs)


def parse_config_file(file_path, logger=logging):
    logger.debug(f'Parsing {file_path}')

    file_path = pathlib.Path(file_path)

    if not file_path.exists():
        error(logger, f'Could not find config file {file_path}')

    with open(file_path, 'r') as f:
        try:
            entries = yaml.safe_load(f)
        except yaml.YAMLError as e:
            error(logger, e)

    if isinstance(entries, list):
        entries = dict(simulations=entries)

    global_quantity_path = entries.get('quantity_path', None)
    if global_quantity_path is not None:
        global_quantity_path = pathlib.Path(global_quantity_path)

    simulations = entries.get('simulations', [])

    if not isinstance(simulations, list):
        simulations = [simulations]

    visualizations = []

    for simulation in simulations:
        simulation_run = SimulationRun.parse(simulation, logger=logger)

        if 'quantity_file' not in simulation:
            if global_quantity_path is None:
                quantity_file = pathlib.Path('quantities.csv')
            else:
                if global_quantity_path.is_dir():
                    quantity_file = global_quantity_path / 'quantities.csv'
                else:
                    quantity_file = global_quantity_path
        else:
            quantity_file = pathlib.Path(simulation['quantity_file'])

        if not quantity_file.is_absolute():
            if global_quantity_path is None or not global_quantity_path.is_dir(
            ):
                quantity_file = simulation_run.data_dir / quantity_file
            else:
                quantity_file = global_quantity_path / quantity_file

        quantity_file = quantity_file.resolve()

        if not quantity_file.exists():
            error(logger, f'Could not find quantity_file {quantity_file}')

        quantities = Quantity.parse_file(quantity_file, logger=logger)

        global_plots = entries.get('plots', [])
        if not isinstance(global_plots, list):
            global_plots = [global_plots]

        local_plots = simulation.get('plots', [])
        if not isinstance(local_plots, list):
            local_plots = [local_plots]

        plots = global_plots + local_plots

        plot_descriptions = [
            PlotDescription.parse(quantities, dict(plot_config), logger=logger)
            for plot_config in plots
        ]

        visualizer = Visualizer(simulation_run)
        visualizations.append(Visualization(visualizer, *plot_descriptions))

    return visualizations


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize Bifrost simulation runs.')
    parser.add_argument(
        'config_file',
        help=
        'Path to visualization config file (in YAML format) in simulation directory'
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Whether to overwrite existing visualizations')
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help='Use DEBUG log level')
    parser.add_argument(
        '-s',
        '--simulations',
        metavar='SIMULATIONS',
        help=
        'Subset of simulations in config file to operate on (comma-separated)')
    parser.add_argument('-c',
                        '--clean',
                        action='store_true',
                        help='Clean visualization data')

    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger('autoviz')
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    all_visualizations = parse_config_file(args.config_file, logger=logger)

    if args.simulations is None:
        visualizations = all_visualizations
    else:
        simulation_names = args.simulations.split(',')
        visualizations = [
            v for v in all_visualizations
            if v.visualizer.simulation_name in simulation_names
        ]

    for visualization in visualizations:
        if args.clean:
            visualization.clean()
        else:
            visualization.visualize(overwrite=args.overwrite)
