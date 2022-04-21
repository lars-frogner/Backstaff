#!/usr/bin/env python
import os
import sys
import re
import pathlib
import logging
import shutil
import csv
import warnings
import numpy as np
from tqdm import tqdm
from ruamel.yaml import YAML
from joblib import Parallel, delayed
from matplotlib.offsetbox import AnchoredText

try:
    import backstaff.units as units
    import backstaff.running as running
    import backstaff.fields as fields
    import backstaff.helita_utils as helita_utils
except ModuleNotFoundError:
    import units
    import running
    import fields
    import helita_utils


def update_dict_nested(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict_nested(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def abort(logger, *args, **kwargs):
    logger.critical(*args, **kwargs)
    sys.exit(1)


def bifrost_data_has_variable(bifrost_data, variable_name, try_fetch=False):
    if hasattr(bifrost_data, variable_name) or variable_name in bifrost_data.compvars:
        return True
    if try_fetch:
        try:
            bifrost_data.get_var(variable_name)
            return True
        except:
            return False
    else:
        return False


def bifrost_data_has_variables(bifrost_data, *variable_names, **kwargs):
    result = True
    for variable_name in variable_names:
        if not bifrost_data_has_variable(bifrost_data, variable_name, **kwargs):
            result = False
    return result


class Quantity:
    @staticmethod
    def for_reduction(name, unit_scale=1):
        return Quantity(name, unit_scale, None, None)

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

    @property
    def dependency_name(self):
        return self.name

    @property
    def dependency_type(self):
        return "derived"

    def set_name(self, name):
        self.name = name

    def is_available(self, bifrost_data):
        return bifrost_data_has_variable(bifrost_data, self.name, try_fetch=True)

    @classmethod
    def parse_file(cls, file_path, logger=logging):
        def decomment(csvfile):
            for row in csvfile:
                raw = row.split("#")[0].strip()
                if raw:
                    yield raw

        logger.debug(f"Parsing {file_path}")

        file_path = pathlib.Path(file_path)

        quantities = {}
        with open(file_path, newline="") as f:
            reader = csv.reader(decomment(f))
            for row in reader:
                if len(row) == 3:
                    row.append("")

                if len(row) != 4:
                    abort(
                        logger,
                        f'Invalid number of entries in CSV line in {file_path}: {", ".join(row)}',
                    )

                name, unit, description, cmap_name = row

                name = name.strip()

                unit = unit.strip()

                if len(unit) > 0 and unit[0] == "-":
                    unit_sign = -1
                    unit = unit[1:].strip()
                else:
                    unit_sign = 1

                try:
                    unit = float(unit)
                except ValueError:
                    try:
                        unit = getattr(units, unit)
                    except AttributeError:
                        if len(unit) != 0:
                            logger.warning(
                                f"Unit {unit} for quantity {name} in {file_path} not recognized, using 1.0"
                            )
                        unit = 1.0
                unit *= unit_sign

                description = description.strip()

                cmap_name = cmap_name.strip()
                if len(cmap_name) == 0:
                    cmap_name = "viridis"

                logger.debug(
                    f"Using quantity info: {name}, {unit:g}, {description}, {cmap_name}"
                )

                quantities[name] = cls(name, unit, description, cmap_name)

        return quantities


class SynthesizedQuantity(Quantity):
    @staticmethod
    def from_quantity(quantity, line_name, axis_name, logger=logging):
        try:
            SynthesizedQuantity.get_central_wavelength(line_name)
        except:
            abort(
                logger,
                f"Invalid format for spectral line name {line_name} , must be <element>_<ionization stage>_<central wavelength in Å> (e.g. si_4_1393.755)",
            )

        subtypes = [LineIntensityQuantity, LineShiftQuantity, LineVarianceQuantity]

        constructor = None
        for subtype in subtypes:
            if quantity.name in subtype.supported_quantity_names:
                constructor = subtype
                break

        if constructor is None:
            all_names = sum(
                [subtype.supported_quantity_names for subtype in subtypes], []
            )
            abort(
                logger,
                f'Invalid name {quantity.name} for spectral line quantity (valid quantities are {",".join(all_names)})',
            )

        return constructor(
            quantity.name,
            line_name,
            axis_name,
            quantity.unit_scale,
            quantity.description,
            quantity.cmap_name,
        )

    @staticmethod
    def is_synthesized_quantity(quantity):
        subtypes = [LineIntensityQuantity, LineShiftQuantity, LineVarianceQuantity]
        all_names = sum([subtype.supported_quantity_names for subtype in subtypes], [])
        return quantity.name in all_names

    @staticmethod
    def get_central_wavelength(line_name):
        return float(line_name.split("_")[-1]) * 1e-8  # [cm]

    @staticmethod
    def get_command_args(quantities):
        if len(quantities) == 0:
            return []
        line_names = list(set((quantity.line_name for quantity in quantities)))
        quantity_names = list(set((quantity.quantity_name for quantity in quantities)))
        return [
            "synthesize",
            f'--spectral-lines={",".join(line_names)}',
            f'--quantities={",".join(quantity_names)}',
            "-v",
            "--ignore-warnings",
        ]

    def __init__(self, quantity_name, line_name, axis_name, *args, **kwargs):
        super().__init__(f"{quantity_name}_{line_name}", *args, **kwargs)
        self.quantity_name = quantity_name
        self.line_name = line_name
        self.axis_name = axis_name

        self.central_wavelength = self.get_central_wavelength(line_name)

        self.emis_name = f"emis_{self.line_name}"
        self.shift_name = f"shift{self.axis_name}_{self.line_name}"
        self.emis_shift_name = f"emis_{self.shift_name}"
        self.vartg_name = f"vartg_{self.line_name}"
        self.vartgshift2_name = f"vartgshift2{self.axis_name}_{self.shift_name}"
        self.emis_vartgshift2_name = f"emis_{self.vartgshift2_name}"

    @property
    def dependency_name(self):
        raise NotImplementedError()

    @property
    def dependency_type(self):
        return "synthesized"

    def is_available(self, bifrost_data):
        raise NotImplementedError()

    def process_base_quantity(self, field):
        return field if self.unit_scale == 1 else field * self.unit_scale


class LineIntensityQuantity(SynthesizedQuantity):
    base_quantity_name = "intens"
    supported_quantity_names = [base_quantity_name]

    @property
    def dependency_name(self):
        return self.emis_name

    def is_available(self, bifrost_data):
        return bifrost_data_has_variable(bifrost_data, self.emis_name)

    def process_base_quantity(self, field):
        return super().process_base_quantity(field)


class LineShiftQuantity(SynthesizedQuantity):
    base_quantity_name = "profshift"
    supported_quantity_names = [base_quantity_name, "dopvel"]

    @property
    def dependency_name(self):
        return self.emis_shift_name

    def corresponding_intensity_quantity(self):
        return LineIntensityQuantity(
            LineIntensityQuantity.base_quantity_name,
            self.line_name,
            self.axis_name,
            1,
            None,
            None,
        )

    def is_available(self, bifrost_data):
        return bifrost_data_has_variable(
            bifrost_data, self.emis_shift_name
        ) or bifrost_data_has_variables(bifrost_data, self.emis_name, self.shift_name)

    def process_base_quantity(self, field):
        if self.quantity_name == "dopvel":
            field = field.with_values(
                units.CLIGHT * field.get_values() / self.central_wavelength
            )  # [cm/s]
        return super().process_base_quantity(field)


class LineVarianceQuantity(SynthesizedQuantity):
    base_quantity_name = "profvar"
    supported_quantity_names = [base_quantity_name, "profwidth", "widthvel"]

    @property
    def dependency_name(self):
        return self.emis_vartgshift2_name

    def corresponding_intensity_quantity(self):
        return LineIntensityQuantity(
            LineIntensityQuantity.base_quantity_name,
            self.line_name,
            self.axis_name,
            1,
            None,
            None,
        )

    def corresponding_shift_quantity(self):
        return LineShiftQuantity(
            LineShiftQuantity.base_quantity_name,
            self.line_name,
            self.axis_name,
            1,
            None,
            None,
        )

    def is_available(self, bifrost_data):
        return (
            bifrost_data_has_variable(bifrost_data, self.emis_vartgshift2_name)
            or bifrost_data_has_variables(
                bifrost_data, self.emis_name, self.vartgshift2_name
            )
            or bifrost_data_has_variables(
                bifrost_data, self.emis_name, self.vartg_name, self.shift_name
            )
        )

    def process_base_quantity(self, field):
        if self.quantity_name in ("profwidth", "widthvel"):
            field = field.with_values(
                (2 * np.sqrt(2 * np.log(2))) * np.sqrt(field.get_values())
            )  # [cm]
            if self.quantity_name == "widthvel":
                field = field.with_values(
                    units.CLIGHT * field.get_values() / self.central_wavelength
                )  # [cm/s]
        return super().process_base_quantity(field)


class Reduction:
    AXIS_NAMES = ["x", "y", "z"]

    def __init__(self, axis):
        self.axis = int(axis)

    @property
    def axis_name(self):
        return self.AXIS_NAMES[self.axis]

    @property
    def yields_multiple_fields(self):
        return False

    @staticmethod
    def parse(reduction_config, logger=logging):
        if isinstance(reduction_config, str):
            reduction_config = {reduction_config: {}}

        if not isinstance(reduction_config, dict):
            abort(logger, f"Reduction entry must be dict, is {type(reduction_config)}")

        classes = dict(
            scan=Scan,
            sum=Sum,
            mean=Mean,
            slice=Slice,
            integral=Integral,
            synthesis=Synthesis,
        )
        reductions = None
        for name, cls in classes.items():
            if name in reduction_config:
                logger.debug(f"Found reduction {name}")
                reduction_config = dict(reduction_config[name])
                axes = reduction_config.pop("axes", "x")
                if not isinstance(axes, list):
                    axes = [axes]
                reductions = [
                    cls(axis=Reduction.AXIS_NAMES.index(axis_name), **reduction_config)
                    for axis_name in axes
                ]
        if reductions is None:
            abort(logger, "Missing valid reduction entry")

        return reductions

    def get_plot_kwargs(self, field):
        plot_axis_names = list(self.AXIS_NAMES)
        plot_axis_names.pop(self.axis)
        return dict(
            xlabel=f"${plot_axis_names[0]}$ [Mm]", ylabel=f"${plot_axis_names[1]}$ [Mm]"
        )

    def _get_axis_size(self, bifrost_data):
        return getattr(bifrost_data, self.axis_name).size

    def _parse_slice_coord_or_idx_to_idx(self, bifrost_data, coord_or_idx):
        return max(
            0,
            min(
                self._get_axis_size(bifrost_data) - 1,
                self.__class__._parse_float_or_int_input_to_int(
                    coord_or_idx,
                    lambda coord: fields.ScalarField2.slice_coord_to_idx(
                        bifrost_data, self.axis, coord
                    ),
                ),
            ),
        )

    def _parse_distance_or_stride_to_stride(self, bifrost_data, distance_or_stride):
        return max(
            1,
            self.__class__._parse_float_or_int_input_to_int(
                distance_or_stride,
                lambda distance: int(
                    distance / getattr(bifrost_data, f"d{self.axis_name}")
                ),
            ),
        )

    def _parse_coord_or_idx_range_to_slice(self, bifrost_data, coord_or_idx_range):
        assert (
            isinstance(coord_or_idx_range, (tuple, list))
            and len(coord_or_idx_range) == 2
        )

        coord_or_idx_range_copy = list(coord_or_idx_range)
        for i in range(2):
            if isinstance(coord_or_idx_range[i], int):
                coord_or_idx_range_copy[i] = None

        if self.axis == 2:
            coords = helita_utils.inverted_zdn(bifrost_data)
        else:
            coords = [bifrost_data.xdn, bifrost_data.ydn][self.axis]

        coord_slice = helita_utils.inclusive_coord_slice(
            coords, coord_or_idx_range_copy
        )
        coord_slice = [coord_slice.start, coord_slice.stop]

        if isinstance(coord_or_idx_range[0], int):
            coord_slice[0] = coord_or_idx_range[0]
        if isinstance(coord_or_idx_range[1], int):
            coord_slice[1] = coord_or_idx_range[1] + 1

        return slice(*coord_slice)

    @staticmethod
    def _parse_float_or_int_input_to_int(float_or_int, converter):
        is_float = False
        if isinstance(float_or_int, str) and len(float_or_int) > 0:
            if float_or_int[0] == "i":
                float_or_int = int(float_or_int[1:])
                is_float = False
            elif float_or_int[0] == "c":
                float_or_int = float(float_or_int[1:])
                is_float = True
            else:
                float_or_int = float(float_or_int)
                is_float = True
        elif isinstance(float_or_int, int):
            float_or_int = float_or_int
            is_float = False
        else:
            float_or_int = float(float_or_int)
            is_float = True

        if is_float:
            result = converter(float_or_int)
        else:
            result = float_or_int

        return int(result)


class Scan(Reduction):
    def __init__(self, axis=0, start=None, end=None, step="i1"):
        super().__init__(axis)
        self.start = start
        self.end = end
        self.step = step

    @property
    def tag(self):
        return f'scan_{self.axis_name}_{"" if self.start is None else self.start}:{"" if self.end is None else self.end}{"" if self.step == "i1" else f":{self.step}"}'

    @property
    def yields_multiple_fields(self):
        return True

    def __call__(self, bifrost_data, quantity):
        start_idx = (
            0
            if self.start is None
            else self._parse_slice_coord_or_idx_to_idx(bifrost_data, self.start)
        )
        end_idx = (
            (self._get_axis_size(bifrost_data) - 1)
            if self.end is None
            else self._parse_slice_coord_or_idx_to_idx(bifrost_data, self.end)
        )
        stride = self._parse_distance_or_stride_to_stride(bifrost_data, self.step)

        return ScanSlices(self, bifrost_data, quantity, start_idx, end_idx, stride)

    def get_slice_label(self, bifrost_data, slice_idx):
        coords = getattr(bifrost_data, self.axis_name)
        if self.axis == 2:
            coords = coords[::-1]
        return f"${self.axis_name} = {coords[slice_idx]:.2f}$ Mm"


class ScanSlices:
    def __init__(self, scan, bifrost_data, quantity, start_idx, end_idx, stride):
        self._scan = scan
        self._bifrost_data = bifrost_data
        self._quantity = quantity
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._stride = stride

    def get_ids(self):
        return list(range(self._start_idx, self._end_idx + 1, self._stride))

    def __call__(self, slice_idx):
        return ScanSlice(
            fields.ScalarField2.slice_from_bifrost_data(
                self._bifrost_data,
                self._quantity.name,
                slice_axis=self._scan.axis,
                slice_idx=slice_idx,
                scale=self._quantity.unit_scale,
            ),
            self._scan.get_slice_label(self._bifrost_data, slice_idx),
        )


class ScanSlice:
    def __init__(self, field, label):
        self._field = field
        self._label = label

    @property
    def field(self):
        return self._field

    @property
    def label(self):
        return self._label


class Sum(Reduction):
    @property
    def tag(self):
        return f"sum_{self.axis_name}"

    def __call__(self, bifrost_data, quantity):
        return fields.ScalarField2.accumulated_from_bifrost_data(
            bifrost_data,
            quantity.name,
            accum_axis=self.axis,
            accum_operator=np.nansum,
            scale=quantity.unit_scale,
        )


class Mean(Reduction):
    def __init__(self, axis=0, ignore_val=None):
        super().__init__(axis)
        self.ignore_val = ignore_val

    @property
    def tag(self):
        return f"mean_{self.axis_name}"

    def __call__(self, bifrost_data, quantity):
        def value_processor(field):
            if self.ignore_val is None:
                return field
            else:
                field = field.copy()
                ignore_val = self.ignore_val
                if not isinstance(self.ignore_val, list):
                    ignore_val = [ignore_val]
                for val in ignore_val:
                    field[field == val] = np.nan
                return field

        def mean(*args, **kwargs):
            with warnings.catch_warnings():  # Suppress "mean of empty slice" warning
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = np.nanmean(*args, **kwargs)
            return result

        return fields.ScalarField2.accumulated_from_bifrost_data(
            bifrost_data,
            quantity.name,
            accum_axis=self.axis,
            value_processor=value_processor,
            accum_operator=mean,
            scale=quantity.unit_scale,
        )


class Integral(Reduction):
    def __init__(self, axis=0, start=None, end=None):
        super().__init__(axis)
        self.start = start
        self.end = end

    @property
    def tag(self):
        return f'integral_{self.axis_name}{"" if self.start is None else self.start}:{"" if self.end is None else self.end}'

    def __call__(
        self, bifrost_data, quantity, *extra_quantity_names, combine_fields=lambda x: x
    ):
        coord_slice = self._parse_coord_or_idx_range_to_slice(
            bifrost_data, (self.start, self.end)
        )

        if self.axis == 0:

            def value_processor(*fields):
                return combine_fields(*[field[coord_slice, :, :] for field in fields])

            dx = bifrost_data.dx * units.U_L
            accum_operator = lambda field, axis=None: np.sum(field, axis=axis) * dx
        elif self.axis == 1:

            def value_processor(*fields):
                return combine_fields(*[field[:, coord_slice, :] for field in fields])

            dy = bifrost_data.dy * units.U_L
            accum_operator = lambda field, axis=None: np.sum(field, axis=axis) * dy
        elif self.axis == 2:
            dz = fields.ScalarField1.dz_in_bifrost_data(
                bifrost_data, scale=units.U_L
            ).get_values()[np.newaxis, np.newaxis, coord_slice]

            def value_processor(*fields):
                return (
                    combine_fields(*[field[:, :, coord_slice] for field in fields]) * dz
                )

            accum_operator = np.sum

        return fields.ScalarField2.accumulated_from_bifrost_data(
            bifrost_data,
            [quantity.name, *extra_quantity_names],
            accum_axis=self.axis,
            value_processor=value_processor,
            accum_operator=accum_operator,
            scale=quantity.unit_scale,
        )


class Synthesis(Integral):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = {}

    @property
    def tag(self):
        return f'syn_{self.axis_name}{"" if self.start is None else self.start}:{"" if self.end is None else self.end}'

    def __call__(self, bifrost_data, quantity):
        if isinstance(quantity, LineIntensityQuantity):
            return self._compute_intensity(bifrost_data, quantity)
        elif isinstance(quantity, LineShiftQuantity):
            return self._compute_shift(bifrost_data, quantity)
        elif isinstance(quantity, LineVarianceQuantity):
            return self._compute_variance(bifrost_data, quantity)
        else:
            raise ValueError(f"Invalid quantity {quantity.name} for synthesis")

    @staticmethod
    def _create_cache_entry(bifrost_data, quantity):
        return (
            bifrost_data.root_name,
            bifrost_data.snap,
            quantity.base_quantity_name,
            quantity.line_name,
        )

    def _compute_intensity(self, bifrost_data, quantity, processed=True):
        cache_entry = self.__class__._create_cache_entry(bifrost_data, quantity)
        if cache_entry in self.cache:
            return (
                quantity.process_base_quantity(self.cache[cache_entry])
                if processed
                else self.cache[cache_entry]
            )

        if not bifrost_data_has_variable(bifrost_data, quantity.emis_name):
            raise IOError(
                f"Missing quantity {quantity.emis_name} for computing spectral line intensity"
            )

        intensity = super().__call__(
            bifrost_data, Quantity.for_reduction(quantity.emis_name)
        )

        self.cache[cache_entry] = intensity
        return quantity.process_base_quantity(intensity) if processed else intensity

    def _compute_shift(self, bifrost_data, quantity, processed=True):
        cache_entry = self.__class__._create_cache_entry(bifrost_data, quantity)
        if cache_entry in self.cache:
            return (
                quantity.process_base_quantity(self.cache[cache_entry])
                if processed
                else self.cache[cache_entry]
            )

        intensity = self._compute_intensity(
            bifrost_data, quantity.corresponding_intensity_quantity(), processed=False
        )

        if bifrost_data_has_variable(bifrost_data, quantity.emis_shift_name):
            weighted_shift = super().__call__(
                bifrost_data, Quantity.for_reduction(quantity.emis_shift_name)
            )
        elif bifrost_data_has_variables(
            bifrost_data, quantity.emis_name, quantity.shift_name
        ):
            weighted_shift = super().__call__(
                bifrost_data,
                Quantity.for_reduction(quantity.shift_name),
                quantity.emis_name,
                combine_fields=lambda shift, emis: emis * shift,
            )
        else:
            raise IOError(
                f"Missing quantities {quantity.emis_shift_name} or {quantity.emis_name} and {quantity.shift_name} for computing spectral line shift"
            )

        shift = weighted_shift / intensity

        self.cache[cache_entry] = shift
        return quantity.process_base_quantity(shift) if processed else shift

    def _compute_variance(self, bifrost_data, quantity, processed=True):
        cache_entry = self.__class__._create_cache_entry(bifrost_data, quantity)
        if cache_entry in self.cache:
            return (
                quantity.process_base_quantity(self.cache[cache_entry])
                if processed
                else self.cache[cache_entry]
            )

        intensity = self._compute_intensity(
            bifrost_data, quantity.corresponding_intensity_quantity(), processed=False
        )
        shift = self._compute_shift(
            bifrost_data, quantity.corresponding_shift_quantity(), processed=False
        )

        if bifrost_data_has_variable(bifrost_data, quantity.emis_vartgshift2_name):
            weighted_variance = super().__call__(
                bifrost_data, Quantity.for_reduction(quantity.emis_vartgshift2_name)
            )
        elif bifrost_data_has_variables(
            bifrost_data, quantity.emis_name, quantity.vartgshift2_name
        ):
            weighted_variance = super().__call__(
                bifrost_data,
                Quantity.for_reduction(quantity.vartgshift2_name),
                quantity.emis_name,
                combine_fields=lambda vartgshift2, emis: emis * vartgshift2,
            )
        elif bifrost_data_has_variables(
            bifrost_data, quantity.vartg_name, quantity.emis_name, quantity.shift_name
        ):
            weighted_variance = super().__call__(
                bifrost_data,
                Quantity.for_reduction(quantity.vartg_name),
                quantity.emis_name,
                quantity.shift_name,
                combine_fields=lambda vartg, emis, shift: emis * (vartg + shift**2),
            )
        else:
            raise IOError(
                f"Missing quantities {quantity.emis_vartgshift2_name} or {quantity.emis_name} and {quantity.vartgshift2_name} or {quantity.emis_name}, {quantity.vartg_name} and {quantity.shift_name} for computing spectral line variance"
            )

        variance = weighted_variance / intensity - shift * shift

        self.cache[cache_entry] = variance
        return quantity.process_base_quantity(variance) if processed else variance


class Slice(Reduction):
    def __init__(self, axis=0, pos="i0"):
        super().__init__(axis)
        self.pos = pos

    @property
    def tag(self):
        return f"slice_{self.AXIS_NAMES[self.axis]}_{self.pos}"

    def __call__(self, bifrost_data, quantity):
        slice_idx = self._parse_slice_coord_or_idx_to_idx(bifrost_data, self.pos)
        return fields.ScalarField2.slice_from_bifrost_data(
            bifrost_data,
            quantity.name,
            slice_axis=self.axis,
            slice_idx=slice_idx,
            scale=quantity.unit_scale,
        )


class Scaling:
    def __init__(self, vmin=None, vmax=None) -> None:
        self.vmin = vmin
        self.vmax = vmax

    @staticmethod
    def parse(scaling_config, logger=logging):
        classes = dict(linear=LinearScaling, log=LogScaling, symlog=SymlogScaling)
        if isinstance(scaling_config, dict):
            scaling = None
            for name, cls in classes.items():
                if name in scaling_config:
                    logger.debug(f"Found scaling {name}")
                    scaling = cls(**scaling_config[name])
            if scaling is None:
                abort(logger, "Missing reduction entry")
        elif isinstance(scaling_config, str):
            logger.debug(f"Found scaling type {scaling_config}")
            scaling = classes[scaling_config]()
        else:
            abort(
                logger, f"scaling entry must be dict or str, is {type(scaling_config)}"
            )

        return scaling


class SymlogScaling(Scaling):
    def __init__(self, linthresh=None, vmax=None, linthresh_quantile=0.2) -> None:
        self.linthresh = linthresh
        self.vmax = vmax
        self.linthresh_quantile = float(linthresh_quantile)

    @property
    def tag(self):
        return "symlog"

    def get_plot_kwargs(self, field):
        linthresh = self.linthresh
        vmax = self.vmax
        if linthresh is None or vmax is None:
            values = np.abs(field.get_values())
            if linthresh is None:
                linthresh = np.quantile(values[values > 0], self.linthresh_quantile)
            if vmax is None:
                vmax = values.max()
        return dict(symlog=True, vmin=-vmax, vmax=vmax, linthresh=linthresh)


class LinearScaling(Scaling):
    @property
    def tag(self):
        return "linear"

    def get_plot_kwargs(self, *args):
        return dict(log=False, vmin=self.vmin, vmax=self.vmax)


class LogScaling(LinearScaling):
    @property
    def tag(self):
        return "log"

    def get_plot_kwargs(self, *args):
        plot_kwargs = super().get_plot_kwargs(*args)
        plot_kwargs["log"] = True
        return plot_kwargs


class PlotDescription:
    def __init__(self, quantity, reduction, scaling, name=None, **extra_plot_kwargs):
        self.quantity = quantity
        self.reduction = reduction
        self.scaling = scaling
        self.name = name
        self.extra_plot_kwargs = extra_plot_kwargs

    @classmethod
    def parse(cls, quantities, plot_config, allow_reference=True, logger=logging):
        try:
            plot_config = dict(plot_config)
        except ValueError:
            if allow_reference and isinstance(plot_config, str):
                name = plot_config
                return [name]
            else:
                abort(
                    logger,
                    f'plots list entry must be dict{", or str referring to plot" if allow_reference else ""}, is {type(plot_config)}',
                )

        return cls._parse_dict(quantities, plot_config, logger=logger)

    @classmethod
    def _parse_dict(cls, quantities, plot_config, logger=logging):
        name = plot_config.pop("name", None)

        if "quantity" not in plot_config:
            abort(logger, f"Missing entry quantity")

        quantity = plot_config.pop("quantity")

        if not isinstance(quantity, str):
            abort(logger, f"quantity entry must be str, is {type(quantity)}")

        logger.debug(f"Found quantity {quantity}")

        if quantity not in quantities:
            abort(logger, f"Quantity {quantity} not present in quantity file")

        quantity = quantities[quantity]

        if "reduction" not in plot_config:
            abort(logger, f"Missing reduction entry")

        reductions = Reduction.parse(plot_config.pop("reduction"), logger=logger)
        reduction_is_synthesis = isinstance(reductions[0], Synthesis)

        spectral_line_name = plot_config.pop("spectral_line", None)

        quantity_is_synthesized = SynthesizedQuantity.is_synthesized_quantity(quantity)

        if quantity_is_synthesized:
            if spectral_line_name is None:
                abort(
                    logger, f"Quantity {quantity.name} requires a spectral_line entry"
                )
            if not reduction_is_synthesis:
                abort(
                    logger,
                    f"Quantity {quantity.name} requires reduction to be synthesis",
                )
        else:
            if reduction_is_synthesis:
                abort(
                    logger,
                    f"Quantity {quantity.name} not compatible with synthesis reduction",
                )

        if spectral_line_name is not None:
            if reduction_is_synthesis:
                quantity = SynthesizedQuantity.from_quantity(
                    quantity, spectral_line_name, reductions[0].axis_name, logger=logger
                )
            else:
                quantity.set_name(f"{quantity.name}_{spectral_line_name}")

        if "scaling" not in plot_config:
            abort(logger, "Missing scaling entry")

        scaling = Scaling.parse(plot_config.pop("scaling"), logger=logger)

        return [
            cls(quantity, reduction, scaling, name=name, **plot_config)
            for reduction in reductions
        ]

    @property
    def tag(self):
        return f"{self.scaling.tag}_{self.quantity.tag}_{self.reduction.tag}"

    @property
    def has_multiple_fields(self):
        return self.reduction.yields_multiple_fields

    def get_plot_kwargs(self, field):
        return update_dict_nested(
            update_dict_nested(
                update_dict_nested(
                    self.quantity.get_plot_kwargs(),
                    self.reduction.get_plot_kwargs(field),
                ),
                self.scaling.get_plot_kwargs(field),
            ),
            self.extra_plot_kwargs,
        )

    def get_field(self, bifrost_data):
        return self.reduction(bifrost_data, self.quantity)


class VideoDescription:
    def __init__(self, fps=15):
        self.fps = fps

    @classmethod
    def parse(cls, video_config, logger=logging):
        if isinstance(video_config, dict):
            return cls(**video_config)
        elif video_config:
            return cls()
        else:
            return None

    @property
    def config(self):
        return dict(fps=self.fps)


class SimulationRun:
    def __init__(
        self,
        name,
        data_dir,
        start_snap_num=None,
        end_snap_num=None,
        video_description=None,
        logger=logging,
    ):
        self._name = name
        self._data_dir = data_dir
        self._start_snap_num = start_snap_num
        self._end_snap_num = end_snap_num
        self._video_description = video_description
        self._logger = logger

        self._snap_nums = self._find_snap_nums()

    @classmethod
    def parse(cls, simulation_run_config, logger=logging):

        simulation_name = simulation_run_config.get(
            "name", simulation_run_config.get("simulation_name", None)
        )

        if simulation_name is None:
            abort(logger, f"Missing simulation_name entry")

        if not isinstance(simulation_name, str):
            abort(
                logger, f"simulation_name entry must be str, is {type(simulation_name)}"
            )

        simulation_name = simulation_name.strip()

        logger.debug(f"Using simulation_name {simulation_name}")

        simulation_dir = simulation_run_config.get(
            "dir", simulation_run_config.get("simulation_dir", None)
        )

        if simulation_dir is None:
            abort(logger, f"Missing entry simulation_dir")

        if simulation_dir is None:
            abort(logger, "Missing simulation_dir entry")

        if not isinstance(simulation_dir, str):
            abort(
                logger, f"simulation_dir entry must be str, is {type(simulation_dir)}"
            )

        simulation_dir = pathlib.Path(simulation_dir)

        if not simulation_dir.is_absolute():
            abort(
                logger,
                f"simulation_dir entry {simulation_dir} must be an absolute path",
            )

        if not simulation_dir.is_dir():
            abort(logger, f"Could not find simulation_dir directory {simulation_dir}")

        simulation_dir = simulation_dir.resolve()

        logger.debug(f"Using simulation_dir {simulation_dir}")

        start_snap_num = None
        end_snap_num = None

        snap_nums = simulation_run_config.get("snap_nums", None)
        if isinstance(snap_nums, str):
            parts = snap_nums.split(":")
            if len(parts) < 2:
                start_snap_num = int(parts[0])
                end_snap_num = start_snap_num
            elif len(parts) == 2:
                start_snap_num = None if len(parts[0].strip()) == 0 else int(parts[0])
                end_snap_num = None if len(parts[1].strip()) == 0 else int(parts[1])
            else:
                abort(logger, f"Invalid format for snap_nums: {snap_nums}")
        elif snap_nums is not None:
            start_snap_num = int(snap_nums)
            end_snap_num = start_snap_num

        logger.debug(f"Using start_snap_num {start_snap_num}")
        logger.debug(f"Using end_snap_num {end_snap_num}")

        video_description = simulation_run_config.get("video", None)
        if video_description is not None:
            video_description = VideoDescription.parse(video_description, logger=logger)
        return cls(
            simulation_name,
            simulation_dir,
            start_snap_num=start_snap_num,
            end_snap_num=end_snap_num,
            video_description=video_description,
            logger=logger,
        )

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

    @property
    def snap_nums(self):
        return list(self._snap_nums)

    @property
    def video_config(self):
        return (
            None if self._video_description is None else self._video_description.config
        )

    def set_logger(self, logger):
        self._logger = logger

    def ensure_data_is_ready(self, prepared_data_dir, plot_descriptions):
        self.logger.info(f"Preparing data for {self.name}")

        assert self.data_available

        plot_data_locations = {}

        bifrost_data = self.get_bifrost_data(self._snap_nums[0])

        (
            _,
            unavailable_quantities,
            available_plots,
            unavailable_plots,
        ) = self._check_quantity_availability(bifrost_data, plot_descriptions)

        if len(available_plots) > 0:
            plot_data_locations[self.data_dir] = available_plots

        if len(unavailable_plots) == 0:
            return plot_data_locations

        if prepared_data_dir.is_dir():
            try:
                prepared_bifrost_data = self.get_bifrost_data(
                    self._snap_nums[0], other_data_dir=prepared_data_dir
                )
            except:
                prepared_bifrost_data = None

            if prepared_bifrost_data is not None:
                (
                    available_quantities_prepared,
                    unavailable_quantities_prepared,
                    available_plots_prepared,
                    _,
                ) = self._check_quantity_availability(
                    prepared_bifrost_data, unavailable_plots
                )

                if len(unavailable_quantities_prepared) == 0:
                    if len(available_plots_prepared) > 0:
                        plot_data_locations[
                            prepared_data_dir
                        ] = available_plots_prepared

                    prepared_snap_nums = self._find_snap_nums(
                        other_data_dir=prepared_data_dir
                    )
                    missing_snap_nums = [
                        snap_num
                        for snap_num in self._snap_nums
                        if snap_num not in prepared_snap_nums
                    ]
                    if len(missing_snap_nums) > 0:
                        prepared_bifrost_data = None
                        self._prepare_derived_data(
                            prepared_data_dir,
                            available_quantities_prepared,
                            snap_nums=missing_snap_nums,
                        )

                    return plot_data_locations

        prepared_bifrost_data = None
        self._prepare_derived_data(prepared_data_dir, unavailable_quantities)

        prepared_bifrost_data = self.get_bifrost_data(
            self._snap_nums[0], other_data_dir=prepared_data_dir
        )
        (
            _,
            unavailable_quantities_prepared,
            available_plots_prepared,
            _,
        ) = self._check_quantity_availability(prepared_bifrost_data, unavailable_plots)

        if len(available_plots_prepared) > 0:
            plot_data_locations[prepared_data_dir] = available_plots_prepared

        for quantity in unavailable_quantities_prepared:
            self.logger.error(
                f"Could not obtain quantity {quantity.name} for simulation {self.name}, skipping"
            )

        return plot_data_locations

    def get_bifrost_data(self, snap_num, other_data_dir=None):
        fdir = self.data_dir if other_data_dir is None else other_data_dir
        self.logger.debug(f"Reading snap {snap_num} of {self.name} in {fdir}")
        assert snap_num in self._snap_nums
        return helita_utils.CachingBifrostData(
            self.name, fdir=fdir, snap=snap_num, verbose=False
        )

    def _find_snap_nums(self, other_data_dir=None):
        input_dir = self.data_dir if other_data_dir is None else other_data_dir
        snap_nums = self._find_all_snap_nums(input_dir)
        if self._start_snap_num is not None:
            snap_nums = [n for n in snap_nums if n >= self._start_snap_num]
        if self._end_snap_num is not None:
            snap_nums = [n for n in snap_nums if n <= self._end_snap_num]
        self.logger.debug(
            f'Found snaps {", ".join(map(str, snap_nums))} in {input_dir}'
        )
        return snap_nums

    def _find_all_snap_nums(self, input_dir):
        p = re.compile("{}_(\d\d\d)\.idl$".format(self.name))
        snap_nums = []
        for name in os.listdir(input_dir):
            match = p.match(name)
            if match:
                snap_nums.append(int(match.group(1)))
        return sorted(snap_nums)

    def _check_quantity_availability(self, bifrost_data, plot_descriptions):
        available_quantities = []
        unavailable_quantities = []
        available_plots = []
        unavailable_plots = []

        for plot_description in plot_descriptions:
            quantity = plot_description.quantity

            if quantity in available_quantities:
                available_plots.append(plot_description)
            elif quantity in unavailable_quantities:
                unavailable_plots.append(plot_description)
            else:
                if plot_description.quantity.is_available(bifrost_data):
                    self.logger.debug(
                        f"Quantity {quantity.name} available for {bifrost_data.file_root}"
                    )
                    available_quantities.append(quantity)
                    available_plots.append(plot_description)
                else:
                    self.logger.debug(
                        f"Quantity {quantity.name} not available for {bifrost_data.file_root}"
                    )
                    unavailable_quantities.append(quantity)
                    unavailable_plots.append(plot_description)

        return (
            available_quantities,
            unavailable_quantities,
            available_plots,
            unavailable_plots,
        )

    def _prepare_derived_data(self, prepared_data_dir, quantities, snap_nums=None):
        if len(quantities) == 0:
            return

        os.makedirs(prepared_data_dir, exist_ok=True)

        if snap_nums is None:
            snap_nums = self._snap_nums

        param_file_name = f"{self.name}_{snap_nums[0]}.idl"

        snap_range_specification = (
            [f"--snap-range={snap_nums[0]},{snap_nums[-1]}"]
            if len(snap_nums) > 1
            else []
        )

        derived_dependency_names = []
        synthesized_quantities = []
        for quantity in quantities:
            if quantity.dependency_type == "derived":
                derived_dependency_names.append(quantity.dependency_name)
            elif quantity.dependency_type == "synthesized":
                synthesized_quantities.append(quantity)
            else:
                raise ValueError(f"Invalid dependency type {quantity.dependency_type}")

        synthesize_command_args = SynthesizedQuantity.get_command_args(
            synthesized_quantities
        )
        all_dependency_names = derived_dependency_names + [
            quantity.dependency_name for quantity in synthesized_quantities
        ]

        return_code = running.run_command(
            [
                "backstaff",
                "--protected-file-types=",
                "snapshot",
                "-v",
                *snap_range_specification,
                param_file_name,
                "derive",
                "-v",
                "--ignore-warnings",
                *synthesize_command_args,
                "write",
                "-v",
                "--ignore-warnings",
                "--overwrite",
                f'--included-quantities={",".join(all_dependency_names)}',
                str((prepared_data_dir / param_file_name).resolve()),
            ],
            cwd=self.data_dir,
            logger=self.logger.debug,
            error_logger=self.logger.error,
        )
        if return_code != 0:
            abort(self.logger, "Non-zero return code")

        for snap_num in snap_nums:
            snap_name = f"{self.name}_{snap_num:03}.snap"
            snap_path = self.data_dir / snap_name
            linked_snap_path = prepared_data_dir / snap_name

            if (
                linked_snap_path.with_suffix(".idl").is_file()
                and not linked_snap_path.is_file()
            ):
                os.symlink(snap_path, linked_snap_path)

            if return_code != 0:
                abort(self.logger, "Non-zero return code")


class Visualizer:
    def __init__(self, simulation_run, output_dir_name="autoviz"):
        self._simulation_run = simulation_run
        self._logger = simulation_run.logger

        self._output_dir = self._simulation_run.data_dir / output_dir_name
        self._prepared_data_dir = self._output_dir / "data"

        self._logger.debug(f"Using output directory {self._output_dir}")

    @property
    def logger(self):
        return self._logger

    @property
    def simulation_name(self):
        return self._simulation_run.name

    def set_logger(self, logger):
        self._simulation_run.set_logger(logger)
        self._logger = logger

    def clean(self):
        if not self._output_dir.is_dir():
            print(f"No data to clean for {self.simulation_name}")
            return

        print(f"The directory {self._output_dir} and all its content will be removed")
        while True:
            answer = input("Continue? [y/N] ").strip().lower()
            if answer in ("", "n"):
                print("Aborted")
                break
            if answer == "y":
                shutil.rmtree(self._output_dir)
                self.logger.debug(f"Removed {self._output_dir}")
                break

    def create_videos_only(self, *plot_descriptions):
        video_config = self._simulation_run.video_config
        if video_config is not None:

            snap_nums = self._simulation_run.snap_nums
            if len(snap_nums) == 0:
                return

            for plot_description in plot_descriptions:
                frame_dir = self._output_dir / plot_description.tag

                if plot_description.has_multiple_fields:
                    bifrost_data = self._simulation_run.get_bifrost_data(snap_nums[0])

                    for snap_num in snap_nums:
                        fields = plot_description.get_field(bifrost_data)
                        field_ids = fields.get_ids()

                        output_dir = frame_dir / f"{snap_num}"
                        self._create_video_from_frames(
                            output_dir,
                            field_ids,
                            frame_dir.with_name(f"{frame_dir.stem}_{snap_num}.mp4"),
                            **video_config,
                        )
                else:
                    self._create_video_from_frames(
                        frame_dir,
                        snap_nums,
                        frame_dir.with_suffix(".mp4"),
                        **video_config,
                    )

    def visualize(
        self,
        *plot_descriptions,
        overwrite=False,
        job_idx=0,
        show_progress=True,
        new_logger_builder=None,
    ):
        if new_logger_builder is not None:
            self.set_logger(new_logger_builder())

        if not self._simulation_run.data_available:
            self.logger.error(
                f"No data for simulation {self.simulation_name} in {self._simulation_run.data_dir}, aborting"
            )
            return

        def add_progress_bar(iterable, extra_desc=None):
            if not show_progress:
                return iterable

            return tqdm(
                iterable,
                desc=f"{self.simulation_name} {plot_description.tag}"
                + ("" if extra_desc is None else f" {extra_desc}"),
                position=job_idx,
                ascii=True,
            )

        plot_data_locations = self._simulation_run.ensure_data_is_ready(
            self._prepared_data_dir, plot_descriptions
        )

        plot_data_locations_inverted = {}
        for data_dir, plot_descriptions in plot_data_locations.items():
            for plot_description in plot_descriptions:
                plot_data_locations_inverted[plot_description] = data_dir

        snap_nums = self._simulation_run.snap_nums

        for plot_description, data_dir in plot_data_locations_inverted.items():

            frame_dir = self._output_dir / plot_description.tag
            os.makedirs(frame_dir, exist_ok=True)

            self.logger.info(
                f"Plotting frames for {plot_description.tag} in {self.simulation_name}"
            )

            bifrost_data = self._simulation_run.get_bifrost_data(snap_nums[0], data_dir)

            if plot_description.has_multiple_fields:
                for snap_num in snap_nums:
                    output_dir = frame_dir / f"{snap_num}"
                    os.makedirs(output_dir, exist_ok=True)

                    bifrost_data.set_snap(snap_num)

                    fields = plot_description.get_field(bifrost_data)
                    field_ids = fields.get_ids()

                    for field_id in add_progress_bar(
                        field_ids, extra_desc=f"(snap {snap_num})"
                    ):
                        output_path = output_dir / f"{field_id}.png"

                        if output_path.exists() and not overwrite:
                            self.logger.debug(f"{output_path} already exists, skipping")
                            continue

                        field_wrapper = fields(field_id)
                        self._plot_frame(
                            bifrost_data,
                            plot_description,
                            field_wrapper.field,
                            output_path,
                            label=field_wrapper.label,
                        )

                    if self._simulation_run.video_config is not None:
                        self._create_video_from_frames(
                            output_dir,
                            field_ids,
                            frame_dir.with_name(f"{frame_dir.stem}_{snap_num}.mp4"),
                            **self._simulation_run.video_config,
                        )

            else:
                for snap_num in add_progress_bar(snap_nums):
                    output_path = frame_dir / f"{snap_num}.png"

                    if output_path.exists() and not overwrite:
                        self.logger.debug(f"{output_path} already exists, skipping")
                        continue

                    bifrost_data.set_snap(snap_num)

                    field = plot_description.get_field(bifrost_data)
                    self._plot_frame(bifrost_data, plot_description, field, output_path)

                if self._simulation_run.video_config is not None:
                    self._create_video_from_frames(
                        frame_dir,
                        snap_nums,
                        frame_dir.with_suffix(".mp4"),
                        **self._simulation_run.video_config,
                    )

    def _plot_frame(
        self, bifrost_data, plot_description, field, output_path, label=None
    ):
        time = float(bifrost_data.params["t"]) * units.U_T
        text = f"{time:.1f} s"

        if label is not None:
            text = f"{text}\n{label}"

        field.plot(
            output_path=output_path,
            extra_artists=[AnchoredText(text, "upper left", frameon=False)],
            **plot_description.get_plot_kwargs(field),
        )

    def _create_video_from_frames(self, frame_dir, frame_indices, output_path, fps=15):
        self.logger.info(
            f"Creating video {output_path.name} from {self.simulation_name}"
        )

        tempdir = frame_dir / ".ffmpeg_tmp"
        if tempdir.exists():
            shutil.rmtree(tempdir)

        os.makedirs(tempdir)
        frame_num = 0
        for frame_idx in frame_indices:
            frame_path = frame_dir / f"{frame_idx:d}.png"
            linked_frame_path = tempdir / f"{frame_num:d}.png"
            if frame_path.is_file():
                os.symlink(frame_path, linked_frame_path)
                frame_num += 1

        frame_path_template = tempdir / "%d.png"

        return_code = running.run_command(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-r",
                "{:d}".format(fps),
                "-start_number",
                "0",
                "-i",
                str(frame_path_template),
                "-vf",
                "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            logger=self.logger.debug,
            error_logger=self.logger.error,
        )

        shutil.rmtree(tempdir)

        if return_code != 0:
            self.logger.error("Could not create video, skipping")


class Visualization:
    def __init__(self, visualizer, *plot_descriptions):
        self._visualizer = visualizer
        self._plot_descriptions = plot_descriptions

    @property
    def visualizer(self):
        return self._visualizer

    def create_videos_only(self, **kwargs):
        self.visualizer.create_videos_only(*self._plot_descriptions, **kwargs)

    def visualize(self, **kwargs):
        self.visualizer.visualize(*self._plot_descriptions, **kwargs)


def parse_config_file(file_path, logger=logging):
    logger.debug(f"Parsing {file_path}")

    file_path = pathlib.Path(file_path)

    if not file_path.exists():
        abort(logger, f"Could not find config file {file_path}")

    yaml = YAML()
    with open(file_path, "r") as f:
        try:
            entries = yaml.load(f)
        except yaml.YAMLError as e:
            abort(logger, e)

    if isinstance(entries, list):
        entries = dict(simulations=entries)

    global_quantity_path = entries.get("quantity_path", None)
    if global_quantity_path is not None:
        global_quantity_path = pathlib.Path(global_quantity_path)

    simulations = entries.get("simulations", [])

    if not isinstance(simulations, list):
        simulations = [simulations]

    visualizations = []

    for simulation in simulations:
        simulation_run = SimulationRun.parse(simulation, logger=logger)

        if "quantity_file" not in simulation:
            if global_quantity_path is None:
                quantity_file = pathlib.Path("quantities.csv")
            else:
                if global_quantity_path.is_dir():
                    quantity_file = global_quantity_path / "quantities.csv"
                else:
                    quantity_file = global_quantity_path
        else:
            quantity_file = pathlib.Path(simulation["quantity_file"])

        if not quantity_file.is_absolute():
            if global_quantity_path is None or not global_quantity_path.is_dir():
                quantity_file = simulation_run.data_dir / quantity_file
            else:
                quantity_file = global_quantity_path / quantity_file

        quantity_file = quantity_file.resolve()

        if not quantity_file.exists():
            abort(logger, f"Could not find quantity_file {quantity_file}")

        quantities = Quantity.parse_file(quantity_file, logger=logger)

        global_plots = entries.get("plots", [])
        if not isinstance(global_plots, list):
            global_plots = [global_plots]

        local_plots = simulation.get("plots", [])
        if not isinstance(local_plots, list):
            local_plots = [local_plots]

        global_plot_descriptions = [
            plot_description
            for plot_config in global_plots
            for plot_description in PlotDescription.parse(
                quantities, plot_config, allow_reference=False, logger=logger
            )
        ]

        references, plot_descriptions = [], []
        for plot_config in local_plots:
            for p in PlotDescription.parse(
                quantities, plot_config, allow_reference=True, logger=logger
            ):
                (references, plot_descriptions)[isinstance(p, PlotDescription)].append(
                    p
                )

        global_plot_descriptions_with_name = []
        for plot_description in global_plot_descriptions:
            (global_plot_descriptions_with_name, plot_descriptions)[
                plot_description.name is None
            ].append(plot_description)

        for name in references:
            found_plot = False
            for plot_description in global_plot_descriptions_with_name:
                if name == plot_description.name:
                    plot_descriptions.append(plot_description)
                    found_plot = True
            if not found_plot:
                logger.warning(f"No plots found with name {name}, skipping")

        visualizer = Visualizer(simulation_run)
        visualizations.append(Visualization(visualizer, *plot_descriptions))

    return visualizations


class LoggerBuilder:
    def __init__(self, name="autoviz", level=logging.INFO, log_file=None):
        self.name = name
        self.level = level
        self.log_file = log_file

    def __call__(self):
        logger = logging.getLogger(self.name)
        if len(logger.handlers) == 0:
            logger.setLevel(self.level)
            logger.propagate = False
            if self.log_file is None:
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
                logger.addHandler(sh)
            else:
                fh = logging.FileHandler(self.log_file, mode="a")
                fh.setFormatter(
                    logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
                )
                logger.addHandler(fh)

            def handle_exception(exc_type, exc_value, exc_traceback):
                if issubclass(exc_type, KeyboardInterrupt):
                    sys.__excepthook__(exc_type, exc_value, exc_traceback)
                    return

                logger.critical(
                    "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
                )

            sys.excepthook = handle_exception

        return logger


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Bifrost simulation runs.")

    parser.add_argument(
        "config_file", help="path to visualization config file (in YAML format)"
    )
    parser.add_argument(
        "-c", "--clean", action="store_true", help="clean visualization data"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="use debug log level"
    )
    parser.add_argument(
        "-v",
        "--video-only",
        action="store_true",
        help="only generate videos from existing frames",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing video frames",
    )
    parser.add_argument(
        "--hide-progress", action="store_true", help="whether to hide progress bars"
    )
    parser.add_argument(
        "-l",
        "--log-file",
        metavar="PATH",
        help="where to write log (prints to terminal by default)",
    )
    parser.add_argument(
        "-s",
        "--simulations",
        metavar="NAMES",
        help="subset of simulations in config file to operate on (comma-separated)",
    )
    parser.add_argument(
        "-n",
        "--n-threads",
        type=int,
        default=1,
        metavar="NUM",
        help="max number of threads to use for visualization",
    )

    args = parser.parse_args()

    logger_builder = LoggerBuilder(
        level=(logging.DEBUG if args.debug else logging.INFO), log_file=args.log_file
    )
    logger = logger_builder()

    all_visualizations = parse_config_file(args.config_file, logger=logger)

    if args.simulations is None:
        visualizations = all_visualizations
    else:
        simulation_names = args.simulations.split(",")
        visualizations = [
            v
            for v in all_visualizations
            if v.visualizer.simulation_name in simulation_names
        ]

    if len(visualizations) == 0:
        logger.info("Nothing to visualize")
        sys.exit()

    if args.clean:
        for visualization in visualizations:
            visualization.visualizer.clean()
    elif args.video_only:
        for visualization in visualizations:
            visualization.create_videos_only()
    else:
        n_jobs = min(args.n_threads, len(visualizations))
        Parallel(n_jobs=n_jobs)(
            delayed(
                lambda idx, v: v.visualize(
                    overwrite=args.overwrite,
                    job_idx=idx,
                    show_progress=(not args.hide_progress),
                    new_logger_builder=(None if n_jobs == 1 else logger_builder),
                )
            )(idx, v)
            for idx, v in enumerate(visualizations)
        )
