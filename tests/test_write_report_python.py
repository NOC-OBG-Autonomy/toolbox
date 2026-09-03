"""Tests the step 'Write Data Report (Python)' (fpdf2-based report)."""

#   Test module import
from pelagos_py.steps.input_output import write_report_python as wrp
import pytest
from unittest.mock import patch, MagicMock

#   Other imports
from datetime import datetime, timezone
import xarray as xr
import json
import numpy as np
from importlib.metadata import PackageNotFoundError


@pytest.fixture
def qc_dataset():
    ds = xr.Dataset()

    attrs = {
        "range_test_flag_cts": json.dumps({"1": 10, "4": 2}),
        "range_test_stats": json.dumps({"mean": 1.2}),
        "range_test_params": json.dumps({"threshold": [-2.5, 40]}),
        "gross_range_test_flag_cts": json.dumps({"1": 8}),
        # intentionally omit stats to test default {}
        "gross_range_test_params": json.dumps({"fail": [0, 20]}),
        #   Glossary is read from the first QC var carrying these.
        "flag_values": [0, 1, 2],
        "flag_meanings": "NO_QC GOOD PROB_GOOD",
        "long_name": "Sea temperature QC",
        "units": "1",
    }

    ds["TEMP_QC"] = xr.DataArray(np.zeros(5), attrs=attrs)

    #   Non-QC variable for reference
    ds["TEMP"] = xr.DataArray(
        np.zeros(5), attrs={"long_name": "Sea temperature", "units": "degC"}
    )
    ds["LONGITUDE"] = xr.DataArray(np.linspace(10, 11, 5))  #   Near gburg
    ds["LATITUDE"] = xr.DataArray(np.linspace(57, 58, 5))
    ds.attrs["dataset_id"] = "glider_test_run"

    return ds


### Small pure helpers


def test_sanitize_replaces_unicode():
    #   Known symbols swap to readable equivalents, anything else is replaced
    #   rather than raising on latin-1 output.
    assert wrp.sanitize("α β γ") == "alpha beta gamma"
    assert wrp.sanitize("“quote”") == '"quote"'
    #   An out-of-range char survives (replaced, not raised).
    out = wrp.sanitize("emoji 😀")
    assert isinstance(out, str)
    out.encode("latin-1")  #   Must not raise


def test_long_date_suffixes():
    assert wrp.long_date(datetime(2026, 6, 23, 22, 49)) == "23rd June 2026, 22:49 UTC"
    assert wrp.long_date(datetime(2026, 6, 1, 0, 0)).startswith("1st June")
    assert wrp.long_date(datetime(2026, 6, 2, 0, 0)).startswith("2nd June")
    #   The teens are all "th" regardless of last digit.
    assert wrp.long_date(datetime(2026, 6, 11, 0, 0)).startswith("11th June")
    assert wrp.long_date(datetime(2026, 6, 12, 0, 0)).startswith("12th June")


def test_pelagos_version_unknown():
    with patch.object(wrp, "version", side_effect=PackageNotFoundError):
        assert wrp.pelagos_version() == "unknown"

    with patch.object(wrp, "version", return_value="1.2.3"):
        assert wrp.pelagos_version() == "1.2.3"


def test_current_info():
    #   Mirrors the rst report's test: mock the OS / environment.
    fixed_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with (
        patch.object(wrp, "datetime") as mock_datetime,
        patch.object(wrp.getpass, "getuser", return_value="aaron-mau"),
        patch.object(wrp, "version", return_value="0.1.dev318+gdaacfb0d8"),
        patch.object(wrp.platform, "python_version", return_value="3.14.2"),
        patch.object(wrp.platform, "system", return_value="Linux"),
        patch.object(wrp.platform, "release", return_value="6.17.0-8-generic"),
    ):
        mock_datetime.now.return_value = fixed_now
        result = wrp.current_info()

    assert result == {
        "timestamp_utc": fixed_now.isoformat(),
        "user": "aaron-mau",
        "version": "0.1.dev318+gdaacfb0d8",
        "python_version": "3.14.2",
        "system": "Linux: 6.17.0-8-generic",
    }


### QC dictionary builders


def test_build_qc_dict(qc_dataset):
    result = wrp.build_qc_dict(qc_dataset)

    assert "TEMP_QC" in result
    assert "range_test" in result["TEMP_QC"]
    assert "gross_range_test" in result["TEMP_QC"]

    range_test = result["TEMP_QC"]["range_test"]
    assert range_test["params"] == {"threshold": [-2.5, 40]}
    assert range_test["flag_counts"] == {"1": 10, "4": 2}
    assert range_test["stats"] == {"mean": 1.2}

    #   stats was intentionally omitted -> defaults to {}
    assert result["TEMP_QC"]["gross_range_test"]["stats"] == {}


def test_flatten_qc_dict():
    qc_dict = {
        "TEMP_QC": {
            "range_test": {
                "flag_counts": {1: 2212921, 3: 2500, 4: 0},  #   0 dropped
            }
        },
        "CNDC_QC": {},  #   empty tests -> skipped entirely
    }

    result = wrp.flatten_qc_dict(qc_dict)

    assert result == [
        ["TEMP_QC", "range_test", 1, "2,212,921"],
        ["TEMP_QC", "range_test", 3, "2,500"],
    ]


### Index / glossary rows


def test_qc_flag_glossary_rows_from_dataset(qc_dataset):
    rows = wrp.qc_flag_glossary_rows(qc_dataset)
    #   Read from TEMP_QC's flag_values/flag_meanings (3 entries here).
    assert rows == [
        ["0", "NO_QC", "No QC performed"],
        ["1", "GOOD", "Good data"],
        ["2", "PROB_GOOD", "Probably good data"],
    ]


def test_qc_flag_glossary_rows_fallback():
    #   No QC var carries flag_values/meanings -> Argo default table.
    ds = xr.Dataset({"TEMP": xr.DataArray(np.zeros(3))})
    rows = wrp.qc_flag_glossary_rows(ds)
    assert len(rows) == len(wrp._DEFAULT_QC_FLAGS)
    assert rows[0] == ["0", "NO_QC", "No QC performed"]


def test_variable_index_rows(qc_dataset):
    rows = wrp.variable_index_rows(qc_dataset)
    by_var = {r[0]: r for r in rows}

    assert by_var["TEMP"] == ["TEMP", "Sea temperature", "degC"]
    #   "1" is kept; LONGITUDE has no attrs -> blank long name and units.
    assert by_var["LONGITUDE"] == ["LONGITUDE", "", ""]


def test_variable_index_rows_units_none():
    #   Units of "None" (string) are blanked rather than printed literally.
    ds = xr.Dataset()
    ds["PHASE"] = xr.DataArray(np.zeros(3), attrs={"units": "None"})
    rows = wrp.variable_index_rows(ds)
    assert rows == [["PHASE", "", ""]]


### YAML config


def test_config_to_yaml_is_clean():
    config = {"pipeline": {"name": "demo"}, "steps": [{"name": "Load Data"}]}
    out = wrp.config_to_yaml(config)
    #   Insertion order preserved (sort_keys=False) and no document markers.
    assert out.startswith("pipeline:")
    assert "name: demo" in out
    assert "---" not in out


### Section builders (pdf is a recording mock)


def test_qc_section(qc_dataset):
    pdf = MagicMock()
    wrp.qc_section(pdf, qc_dataset)

    pdf.add_page.assert_called_once()
    pdf.section_heading.assert_called_once_with("Quality Control Summary")
    pdf.add_table.assert_called_once()

    headers = pdf.add_table.call_args[0][0]
    rows = pdf.add_table.call_args[0][1]
    assert headers == ["QC Variable", "Test", "Flag", "Count"]
    assert len(rows) > 0


def test_qc_section_no_rows(qc_dataset):
    pdf = MagicMock()
    with patch.object(wrp, "flatten_qc_dict", return_value=[]):
        wrp.qc_section(pdf, qc_dataset)

    pdf.body.assert_called_once_with("No QC tests found.")
    pdf.add_table.assert_not_called()


def test_add_log(tmp_path):
    log_content = (
        "2026-02-17 12:56:08 - INFO - pelagos_py.pipeline - Logging to file\n"
        "\n"  #   blank line -> ignored
        "2026-02-17 12:56:18 - WARNING - pelagos_py.pipeline.step.Apply QC - flagged\n"
        "not a properly formatted line\n"  #   wrong field count -> skipped
    )
    logfile = tmp_path / "processing.log"
    logfile.write_text(log_content)

    pdf = MagicMock()
    wrp.add_log(str(logfile), pdf)

    pdf.section_heading.assert_called_once_with("Logfile of run")
    rows = pdf.terminal_block.call_args[0][0]

    #   Date stripped, pelagos_py. prefix removed.
    assert rows[0] == ("12:56:08", "INFO", "pipeline", "Logging to file")
    assert rows[1][0] == "12:56:18"
    assert rows[1][1] == "WARNING"
    assert rows[1][2] == "pipeline.step.Apply QC"
    assert len(rows) == 2  #   blank + malformed lines dropped


def test_add_log_missing_file(tmp_path):
    pdf = MagicMock()
    wrp.add_log(str(tmp_path / "nope.log"), pdf)
    pdf.body.assert_called_once_with("Logfile not found.")
    pdf.terminal_block.assert_not_called()


def test_format_checker_section_json_file(tmp_path):
    cc_data = {
        "og": {
            "scored_points": 478,
            "possible_points": 524,
            "all_priorities": [
                {"name": "Check globals", "msgs": ["attr is missing"]},
                {"name": "No-message check", "msgs": []},  #   skipped (no msgs)
            ],
        }
    }
    ccfile = tmp_path / "cc.json"
    ccfile.write_text(json.dumps(cc_data))

    pdf = MagicMock()
    #   No structured cc_results -> falls back to the saved JSON file.
    wrp.format_checker_section(pdf, cc_results=None, ccfile=str(ccfile))

    pdf.section_heading.assert_called_once_with("Format Checker results")
    #   Known checker "og" -> OG1 label + docs link.
    label, url, score = pdf.cc_heading.call_args[0]
    assert label == "OG1"
    assert url == wrp.OG1_MANUAL_URL
    assert score == "Compliance score: 478/524"

    blocks = pdf.cc_checks.call_args[0][0]
    assert blocks == [("Check globals", ["attr is missing"])]


def test_format_checker_section_no_results():
    pdf = MagicMock()
    wrp.format_checker_section(pdf, cc_results=None, ccfile=None)
    pdf.body.assert_called_once_with(
        "Format Checker ran but produced no detailed results."
    )


### Full step (builds a real PDF; heavy plotting is patched out)


def _make_step(context, parameters):
    step = wrp.WriteDataReportPython.__new__(wrp.WriteDataReportPython)
    step.name = "Write Data Report"
    step.context = context
    step.parameters = parameters
    step.log = MagicMock()
    step.log_warn = MagicMock()
    return step


def test_write_data_report_python(tmp_path, qc_dataset):
    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "glider_test_run",
            "name": "Demo pipeline",
            "description": "A short description",
        },
        "data": qc_dataset,
        "pipeline_config": {
            "pipeline": {"name": "Demo pipeline"},
            "steps": [{"name": "Load Data"}, {"name": "Apply QC"}],
        },
    }
    (tmp_path / "run.log").write_text(
        "2026-02-17 12:00:00 - INFO - pelagos_py.pipeline - Test message\n"
    )

    step = _make_step(
        context,
        {"fname": "report", "title": "Test Report", "show_qc_plots": False},
    )

    #   Skip the expensive/figure-producing bits; the real FPDF still writes a PDF.
    with (
        patch.object(wrp, "make_plots") as mock_plots,
        patch.object(wrp, "glider_track_map", return_value=None),
    ):
        result = step.run()

    #   "report" -> "report.pdf", written to out_directory and actually exists.
    out = tmp_path / "report.pdf"
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")
    #   show_qc_plots=False, so make_plots is never called.
    mock_plots.assert_not_called()
    assert result is context


def test_write_data_report_python_defaults(tmp_path, qc_dataset):
    #   No fname/title -> derived from filename_core.
    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "test_filename",
        },
        "data": qc_dataset,
        "pipeline_config": {},
    }
    (tmp_path / "run.log").write_text("log line\n")

    step = _make_step(context, {"show_qc_plots": False})

    with (
        patch.object(wrp, "make_plots"),
        patch.object(wrp, "glider_track_map", return_value=None),
    ):
        step.run()

    assert (tmp_path / "test_filename.pdf").exists()


def test_write_data_report_python_missing_dataset_id(tmp_path, qc_dataset):
    qc_dataset = qc_dataset.copy()
    qc_dataset.attrs.pop("dataset_id", None)

    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "glider_test_run",
        },
        "data": qc_dataset,
        "pipeline_config": {},
    }
    (tmp_path / "run.log").write_text("log line\n")

    step = _make_step(context, {"fname": "report", "show_qc_plots": False})

    with (
        patch.object(wrp, "make_plots"),
        patch.object(wrp, "glider_track_map", return_value=None),
    ):
        step.run()

    step.log_warn.assert_any_call(
        "Dataset ID missing from OG1 file. Reporting with unknown platform information."
    )
    #   A placeholder ID is stamped onto the dataset for downstream sections.
    assert qc_dataset.attrs["dataset_id"] == wrp.UNKNOWN_DATASET_ID
