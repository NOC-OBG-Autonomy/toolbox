"""Parameter-schema behaviour for QC checks and the schema-aware pipeline validator."""

import logging

import numpy as np
import pytest
import xarray as xr

from pelagos_py.steps.quality_control.range_qc import range_qc
from pelagos_py.steps.quality_control.spike_qc import spike_qc
from pelagos_py.steps.quality_control.impossible_date_qc import impossible_date_qc
from pelagos_py.utils.valid_config_check import check_pipeline_variables


LOGGER = logging.getLogger("test")


# --- Dynamic QC resolve their I/O from parameters --------------------------------


def test_dynamic_qc_derives_required_variables():
    qc = range_qc(
        None,
        variable_ranges={"PRES": {4: [-5, 0]}},
        also_flag={"PRES": ["CNDC"]},
    )
    assert qc.required_variables == ["PRES"]
    assert set(qc.qc_outputs) == {"PRES_QC", "CNDC_QC"}


def test_optional_params_get_defaults():
    # also_flag/plot are optional; window_size defaults to 50.
    qc = spike_qc(None, variables={"PRES": 2})
    assert qc.window_size == 50
    assert qc.also_flag == {}
    assert "PROFILE_NUMBER" in qc.required_variables


def test_test_depth_range_adds_depth_requirement():
    without = range_qc(None, variable_ranges={"PRES": {4: [-5, 0]}})
    assert "DEPTH" not in without.required_variables

    with_range = range_qc(
        None, variable_ranges={"PRES": {4: [-5, 0]}}, test_depth_range=[-100, 0]
    )
    assert "DEPTH" in with_range.required_variables


def test_missing_required_param_raises():
    with pytest.raises(ValueError, match="variable_ranges"):
        range_qc(None, also_flag={})


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="bogus"):
        impossible_date_qc(None, bogus=1)


# --- Validator is schema-aware (steps and nested QC) -----------------------------


def test_validator_flags_missing_required_step_param():
    steps = [{"name": "Data Export", "parameters": {"export_format": "netcdf"}}]
    with pytest.raises(ValueError, match="output_path"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_flags_missing_required_qc_param():
    steps = [{"name": "Apply QC", "parameters": {"qc_settings": {"range qc": {}}}}]
    with pytest.raises(ValueError, match="variable_ranges"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_passes_valid_qc_config():
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"impossible date qc": {}}},
        },
    ]
    assert check_pipeline_variables(steps, LOGGER) is True


def test_validator_flags_missing_load_step():
    # No "Load OG1"/"Generate Data" step at all: TIME etc. are not just assumed
    # to be present, since nothing in the pipeline actually provides them.
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"impossible date qc": {}}},
        }
    ]
    with pytest.raises(ValueError, match="TIME"):
        check_pipeline_variables(steps, LOGGER)


# --- Validator catches QC variable ordering mistakes -----------------------------


def test_validator_flags_qc_var_produced_by_later_step():
    # PAR irregularity needs PROFILE_NUMBER, which "Find Profiles" produces — but
    # here it runs *after* the QC, so the requirement is unmet at that point.
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"PAR irregularity qc": {}}},
        },
        {"name": "Find Profiles", "parameters": {}},
    ]
    with pytest.raises(ValueError, match="PROFILE_NUMBER"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_passes_qc_var_in_correct_order():
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {"name": "Find Profiles", "parameters": {}},
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"PAR irregularity qc": {}}},
        },
    ]
    # DOWNWELLING_PAR is also required by PAR irregularity but is a file-native
    # variable no step produces, so it must not be flagged.
    assert check_pipeline_variables(steps, LOGGER) is True


def test_validator_flags_derived_only_qc_requirement_with_no_producing_step():
    # PROFILE_NUMBER is only ever produced by "Find Profiles" -- with no such
    # step anywhere in the pipeline (not just out of order), it can't be assumed
    # to come from the input file, so this must be flagged up front rather than
    # silently passing and failing at run time.
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"PAR irregularity qc": {}}},
        }
    ]
    with pytest.raises(ValueError, match="PROFILE_NUMBER"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_flags_missing_load_step_points_at_loader():
    # Same missing-TIME scenario as test_validator_flags_missing_load_step, but
    # the message should point specifically at adding a loader step, not the
    # generic "add the step that derives it" QC wording.
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"impossible date qc": {}}},
        }
    ]
    with pytest.raises(ValueError, match="Load OG1"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_flags_multiple_load_steps():
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {"name": "Generate Data", "parameters": {}},
    ]
    with pytest.raises(ValueError, match="Multiple data-loading steps"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_flags_blank_file_path():
    steps = [{"name": "Load OG1", "parameters": {"file_path": ""}}]
    with pytest.raises(ValueError, match="does not include a data file"):
        check_pipeline_variables(steps, LOGGER)


# --- Config-driven variable-name parameters (variable_parameters) ----------------


def test_validator_flags_out_of_order_variable_parameter():
    # Deep Correction's `apply_to` (a variable_parameters entry, not in its
    # static required_variables) is pointed at DEPTH, which "Derive CTD"
    # produces -- but here it runs *after* Deep Correction, so this is an
    # ordering mistake. Confirms variable_parameters values are resolved and
    # checked the same way required_variables are.
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {"name": "Deep Correction", "parameters": {"apply_to": "DEPTH", "depth_var": "PRES"}},
        {"name": "Derive CTD", "parameters": {"to_derive": ["DEPTH", "PRAC_SALINITY"]}},
    ]
    with pytest.raises(ValueError, match="DEPTH"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_ignores_self_produced_variable_parameter():
    # "BBP from Beta" defaults both apply_to and output_as to "BBP700" (reads
    # and overwrites the same variable) -- its own output_as must not make
    # its apply_to requirement look like an ordering mistake against itself.
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": "x.nc"}},
        {"name": "Derive CTD", "parameters": {"to_derive": ["DEPTH", "PRAC_SALINITY"]}},
        {"name": "BBP from Beta", "parameters": {}},
    ]
    # BBP700 isn't produced by any step and isn't known-derived, so with no
    # file to check it against, it's left for the run-time check.
    assert check_pipeline_variables(steps, LOGGER) is True


# --- Cross-checking against the actual input file (once Load OG1 has one) -------


def _write_og1_file(path, variables):
    ds = xr.Dataset({name: ("N", np.arange(3)) for name in variables})
    ds.to_netcdf(path)


def test_validator_flags_loader_variable_missing_from_file(tmp_path):
    file_path = tmp_path / "data.nc"
    _write_og1_file(file_path, ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP"])  # no CNDC
    steps = [{"name": "Load OG1", "parameters": {"file_path": str(file_path)}}]
    with pytest.raises(ValueError, match="CNDC"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_allows_loader_variable_renamed_by_later_step(tmp_path):
    # ALR-style raw files store latitude/longitude under different names --
    # a "Correct Values" step further down renames them to LATITUDE/LONGITUDE.
    # That must not be flagged just because the raw file lacks those names.
    file_path = tmp_path / "data.nc"
    _write_og1_file(
        file_path,
        ["TIME", "LATITUDE_GPS", "LONGITUDE_GPS", "PRES", "TEMP", "CNDC"],
    )
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": str(file_path)}},
        {
            "name": "Correct Values",
            "parameters": {"target_variable": "LATITUDE_GPS", "output_as": "LATITUDE"},
        },
        {
            "name": "Correct Values",
            "parameters": {"target_variable": "LONGITUDE_GPS", "output_as": "LONGITUDE"},
        },
    ]
    assert check_pipeline_variables(steps, LOGGER) is True


def test_validator_flags_variable_parameter_missing_from_file(tmp_path):
    file_path = tmp_path / "data.nc"
    _write_og1_file(
        file_path, ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"]
    )  # no BBP700
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": str(file_path)}},
        {"name": "Derive CTD", "parameters": {"to_derive": ["DEPTH", "PRAC_SALINITY"]}},
        {"name": "BBP from Beta", "parameters": {}},
    ]
    with pytest.raises(ValueError, match="BBP700"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_attributes_error_to_correct_step_with_duplicate_qc_names(tmp_path):
    # Two "Apply QC" steps each run a "range qc" test -- the first is fine
    # (PRES exists), the second isn't (BBP700 doesn't). Name-matching alone
    # can't tell these apart (both qc_settings have a "range qc" key), so the
    # raised error must carry the actual failing step's index directly.
    file_path = tmp_path / "data.nc"
    _write_og1_file(file_path, ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"])
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": str(file_path)}},
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"range qc": {"variable_ranges": {"PRES": {4: [-5, 0]}}}}},
        },
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"range qc": {"variable_ranges": {"BBP700": {4: [0, 10]}}}}},
        },
    ]
    with pytest.raises(ValueError, match="BBP700") as excinfo:
        check_pipeline_variables(steps, LOGGER)
    assert excinfo.value.step_index == 2


def test_validator_passes_when_file_has_everything(tmp_path):
    file_path = tmp_path / "data.nc"
    _write_og1_file(
        file_path,
        ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC", "BBP700"],
    )
    steps = [
        {"name": "Load OG1", "parameters": {"file_path": str(file_path)}},
        {"name": "Derive CTD", "parameters": {"to_derive": ["DEPTH", "PRAC_SALINITY"]}},
        {"name": "BBP from Beta", "parameters": {}},
    ]
    assert check_pipeline_variables(steps, LOGGER) is True
