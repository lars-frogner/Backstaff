import numpy as np

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain


class BifrostDataFilter(VTKPythonAlgorithmBase):
    def __init__(self, n_inputs=1, propagate_input=None, select_quantities=False):
        super().__init__(
            nInputPorts=n_inputs,
            nOutputPorts=1,
            inputType="vtkDataSet",
            outputType="vtkDataSet",
        )
        self._propagate_input = propagate_input
        self._select_quantities = select_quantities
        self._quantity_names = None

    def RequestDataObject(self, request, input_info, output_info):
        input_VTK_data = self.GetInputData(input_info, 0, 0)
        output_VTK_data = self.GetOutputData(output_info, 0)
        assert input_VTK_data is not None
        if output_VTK_data is None or (
            not output_VTK_data.IsA(input_VTK_data.GetClassName())
        ):
            output_VTK_data = input_VTK_data.NewInstance()
            output_info.GetInformationObject(0).Set(
                output_VTK_data.DATA_OBJECT(), output_VTK_data
            )
        return super().RequestDataObject(request, input_info, output_info)

    def RequestData(self, request, input_info, output_info):
        input_VTK_data = [
            self.GetInputData(input_info, idx, 0)
            for idx in range(self.GetNumberOfInputPorts())
        ]
        output_VTK_data = self.GetOutputData(output_info, 0)

        if self._propagate_input is not None:
            output_VTK_data.ShallowCopy(input_VTK_data[self._propagate_input])

        input_datasets = list(map(dsa.WrapDataObject, input_VTK_data))
        output_dataset = dsa.WrapDataObject(output_VTK_data)

        input_point_data = [data.PointData for data in input_datasets]

        if self._select_quantities:
            quantity_names = (
                list(input_datasets[0].PointData.keys())
                if self._quantity_names is None
                else list(self._quantity_names)
            )
            output_point_data = self.ComputeOutputData(
                *input_point_data, quantity_names
            )
        else:
            output_point_data = self.ComputeOutputData(*input_point_data)

        output_dataset.SetDimensions(input_datasets[0].GetDimensions())

        dataset_type = output_VTK_data.GetClassName()

        if dataset_type == "vtkImageData":
            output_dataset.SetOrigin(input_datasets[0].GetOrigin())
            output_dataset.SetSpacing(input_datasets[0].GetSpacing())
        elif dataset_type == "vtkRectilinearGrid":
            output_dataset.SetXCoordinates(input_datasets[0].GetXCoordinates())
            output_dataset.SetYCoordinates(input_datasets[0].GetYCoordinates())
            output_dataset.SetZCoordinates(input_datasets[0].GetZCoordinates())
        else:
            raise ValueError(f"Dataset type {dataset_type} not supported")

        for name, data in output_point_data.items():
            output_dataset.PointData.append(data, name)
            output_dataset.PointData.SetActiveScalars(name)

        return 1

    def SetPropagateInput(self, propagate_input):
        if propagate_input != self._propagate_input:
            self.Modified()
        self._propagate_input = propagate_input

    def SetQuantityNames(self, quantity_names_str):
        if quantity_names_str is None or len(quantity_names_str.strip()) == 0:
            if self._quantity_names is not None:
                self.Modified()
            self._quantity_names = None
        else:
            quantity_names = [name.strip() for name in quantity_names_str.split(",")]
            if quantity_names != self._quantity_names:
                self.Modified()
            self._quantity_names = quantity_names


@smproxy.filter(label="Bifrost Difference")
@smproperty.input(name="Input 2", port_index=1)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
@smproperty.input(name="Input 1", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class BifrostDifferenceFilter(BifrostDataFilter):
    def __init__(self):
        super().__init__(n_inputs=2, select_quantities=True, propagate_input=None)
        self._difference_types = ["abs"]
        self._propagate_input_map = {"None": None, "Input 1": 0, "Input 2": 1}
        self._difference_computers = dict(
            abs=self.ComputeAbsDiff,
            rel=self.ComputeRelDiff,
            hybrid=self.ComputeHybridDiff,
        )

    def ComputeAbsDiff(self, arr_1, arr_2):
        return arr_2 - arr_1

    def ComputeRelDiff(self, arr_1, arr_2):
        return (arr_2 - arr_1) / arr_1

    def ComputeHybridDiff(self, arr_1, arr_2):
        abs_diff = self.ComputeAbsDiff(arr_1, arr_2)
        normed_abs_diff = abs_diff / np.mean(arr_1)
        rel_diff = self.ComputeRelDiff(arr_1, arr_2)
        signs = np.sign(abs_diff)
        return signs * np.sqrt(normed_abs_diff * rel_diff + 1e-9)

    def ComputeOutputData(self, input_data_1, input_data_2, quantity_names):
        data = {}
        for difference_type in ["abs", "rel", "hybrid"]:
            if difference_type in self._difference_types:
                for name in quantity_names:
                    data[f"{name}_{difference_type}_diff"] = self._difference_computers[
                        difference_type
                    ](input_data_1[name], input_data_2[name])
        return data

    @smproperty.stringvector(name="PropagateInput", default_values="")
    def PropagateInput(self, input):
        value = "" if input is None else input.strip().lower()
        if len(value) == 0 or value in ("no", "none"):
            propagate_input = None
        elif value in ("1", "input 1"):
            propagate_input = 0
        elif value in ("2", "input 2"):
            propagate_input = 1
        else:
            raise ValueError(
                f'Invalid input for PropagateInput: {input} (use "", "1" or "2")'
            )
        super().SetPropagateInput(propagate_input)

    @smproperty.stringvector(name="QuantityNames", default_values="")
    def SetQuantityNames(self, quantity_names_str):
        super().SetQuantityNames(quantity_names_str)

    @smproperty.stringvector(name="DifferenceTypes", default_values="abs")
    def SetDifferenceTypes(self, difference_types_str):
        if difference_types_str is None or len(difference_types_str.strip()) == 0:
            if len(self._difference_types) > 0:
                self.Modified()
            self._difference_types = []
        else:
            difference_types = [
                name.strip() for name in difference_types_str.split(",")
            ]
            if difference_types != self._difference_types:
                self.Modified()
            self._difference_types = difference_types
