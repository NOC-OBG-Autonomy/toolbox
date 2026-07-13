import pytest
import json
from pelagos_py.steps.input_output.export import ExportStep
from unittest.mock import MagicMock, patch
import xarray as xr
import numpy as np
import pandas as pd

#   Most of export unit testing is handled second-hand
#   Might be nice to implement an actual test of the export functionality
#   As a number of lines remain untested
#   src/pelagos_py/steps/input_output/export.py                         41     15    63%   87-88, 96, 100, 103, 111, 115, 118-123, 131-133

#   Fake data borrowed from write_report
@pytest.fixture
def qc_dataset():
    """A test dataset to stash in the step context"""
    ds = xr.Dataset()

    attrs = {
        "range_test_flag_cts": json.dumps({"1": 10, "4": 2}),
        "range_test_stats": json.dumps({"mean": 1.2}),
        "range_test_params": json.dumps({"threshold": [-2.5, 40]}),
        "gross_range_test_flag_cts": json.dumps({"1": 8}),
        "gross_range_test_params": json.dumps({"fail": [0, 20]}),
    }

    ds["TEMP_QC"] = xr.DataArray(np.zeros(5), attrs=attrs)

    #   Non-QC variable for reference
    ds["TEMP"] = xr.DataArray(np.zeros(5))
    ds["LONGITUDE"] = xr.DataArray(np.linspace(10, 11, 5))  #   Near gburg
    ds["LATITUDE"] = xr.DataArray(np.linspace(57, 58, 5))
    ds.attrs["dataset_id"] = "glider_test_run"

    return ds


@pytest.fixture
def step(qc_dataset):
    s = ExportStep(
        name="Data Export",
        parameters={
            "output_path": "output.nc",
            "export_format": "netcdf",
            "compression": 4,
            },
        )

    s.log = MagicMock()
    s.context = {"data": qc_dataset}
    return s

@pytest.mark.parametrize(
        "export_format,extension,compression",
            [
                ("netcdf", "nc", 1),
                ("netcdf", "nc", 9),
                ("hdf5", "h5", 4),
                ("csv", "csv", 4),
                ("parquet", "parquet", 4),
            ],
        )
def test_export_formats(step, export_format, extension, compression, tmp_path):
    outfile = tmp_path / f"test.{extension}"

    step.parameters["export_format"] = export_format
    step.parameters["output_path"] = str(outfile)
    step.parameters["compression"] = compression

    step.run()

    assert outfile.exists()

    if export_format in ["netcdf", "hdf5"]:
        ds = xr.open_dataset(outfile)

        assert "TEMP" in ds
        assert "TEMP_QC" in ds
        assert ds["TEMP"].encoding["zlib"] is True
        assert ds["TEMP"].encoding["complevel"] == compression

        ds.close()
    
    elif export_format == "csv":
        df = pd.read_csv(outfile)
        assert "TEMP" in df.columns
        assert "TEMP_QC" in df.columns
    
    elif export_format == "parquet":
        df = pd.read_parquet(outfile)
        assert "TEMP" in df.columns
        assert "TEMP_QC" in df.columns

def test_qc_history_added(step, qc_dataset):
    qc = {"temperature": ["range check"]}   #   Can be whatever
    step.context["qc_history"] = qc
    step.run()

    assert qc_dataset.attrs["delayed_qc_history"] == json.dumps(qc)

#   Test the valueerrors
def test_invalid_export_format(step):
    step.parameters["export_format"] = "xlsx"

    with pytest.raises(ValueError, match="Unsupported export format"):
        step.run()

def test_blank_output_path(step):
    step.parameters["output_path"] = None

    with pytest.raises(ValueError, match="Output path must be specified for data export"):
        step.run()

def test_nonstr_output_path(step):
    step.parameters["output_path"] = 42

    with pytest.raises(ValueError, match="Output path must be a string."):
        step.run()
