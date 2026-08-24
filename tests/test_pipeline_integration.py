import pytest
import yaml
from pelagos_py.pipeline import Pipeline

GENERATE_YAML = """
pipeline:
  name: Synthetic Data Generator
  description: Generate synthetic OG1-style data and export it to netCDF

steps:
  - name: "Generate Data"
    parameters:
      sampling_info: ["2025-01-01", "2025-01-02", 20]
      additional_variables: []
      value_limits: {}
      gen_fixed_data: false
    diagnostics: false

  - name: "Data Export"
    parameters:
      export_format: "netcdf"
      output_path: "placeholder.nc"
"""

LOAD_PIPELINE_YAML = """
pipeline:
  name: Minimal Integration Pipeline
  description: Load a synthetic netCDF file and re-export it

steps:
  - name: Load OG1
    parameters:
      file_path: placeholder.nc
      filter_bad_time: false
    diagnostics: false

  - name: "Data Export"
    parameters:
      export_format: "netcdf"
      output_path: "placeholder_out.nc"
"""


@pytest.fixture(scope="session")
def synthetic_nc(tmp_path_factory):
    """Run gen_data + export once per session to produce a temp .nc input file."""
    tmp_path = tmp_path_factory.mktemp("synthetic_data")
    config = yaml.safe_load(GENERATE_YAML)

    config_file = tmp_path / "generate_pipeline.yaml"
    log_file = tmp_path / "generate.log"
    output_nc = tmp_path / "synthetic.nc"

    config["pipeline"]["out_directory"] = str(tmp_path)
    config["pipeline"]["log_file"] = str(log_file)
    for step in config["steps"]:
        if step["name"] == "Data Export":
            step["parameters"]["output_path"] = str(output_nc)

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    try:
        Pipeline(config_path=str(config_file)).run()
    except Exception as e:
        pytest.fail(f"Synthetic data pipeline (Generate Data + Data Export) failed: {e}")

    assert output_nc.exists(), (
        "Synthetic data pipeline ran without error but did not produce an output file."
    )
    return output_nc


@pytest.mark.filterwarnings("ignore:.*monotonically increasing.*")
def test_generate_and_export(synthetic_nc):
    """Sanity check that Generate Data + Data Export produced a readable netCDF."""
    import xarray as xr

    ds = xr.open_dataset(synthetic_nc)
    try:
        assert "TIME" in ds.variables or "TIME" in ds.coords, (
            "Synthetic netCDF is missing the TIME variable."
        )
        assert ds.sizes.get("N_MEASUREMENTS", 0) > 0, (
            "Synthetic netCDF has no measurements."
        )
    finally:
        ds.close()


@pytest.mark.filterwarnings("ignore:.*monotonically increasing.*")
@pytest.mark.filterwarnings("ignore:.*Removed.*records containing invalid or pre-deployment timestamps.*")
@pytest.mark.filterwarnings("ignore:.*Sanitised invalid flags to 0 in.*")
def test_full_pipeline_execution(tmp_path, synthetic_nc):

    config = yaml.safe_load(LOAD_PIPELINE_YAML)

    config_file = tmp_path / "temp_pipeline.yaml"
    log_file = tmp_path / "test.log"
    output_nc = tmp_path / "processed_output.nc"

    config["pipeline"]["out_directory"] = str(tmp_path)
    config["pipeline"]["log_file"] = str(log_file)

    for step in config["steps"]:
        if step["name"] == "Load OG1":
            step["parameters"]["file_path"] = str(synthetic_nc)
        elif step["name"] == "Data Export":
            step["parameters"]["output_path"] = str(output_nc)

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    try:
        p = Pipeline(config_path=str(config_file))
        p.run()

        assert output_nc.exists(), "The pipeline completed, but the output file was not created."

    except Exception as e:
        pytest.fail(f"Pipeline stopped unexpectedly: {e}")


FAILING_STEP_YAML = """
pipeline:
  name: Continue On Step Fail Test

steps:
  - name: Load OG1
    parameters:
      file_path: placeholder.nc
      filter_bad_time: false
    diagnostics: false

  - name: "Apply QC"
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            BBP700:
              4: [0, 10]

  - name: "Data Export"
    parameters:
      export_format: "netcdf"
      output_path: "placeholder_out.nc"
"""


def _build_failing_step_pipeline(tmp_path, synthetic_nc, continue_on_step_fail=None):
    config = yaml.safe_load(FAILING_STEP_YAML)
    config_file = tmp_path / "failing_step_pipeline.yaml"
    output_nc = tmp_path / "continue_on_fail_output.nc"

    config["pipeline"]["out_directory"] = str(tmp_path)
    config["pipeline"]["log_file"] = str(tmp_path / "test.log")
    if continue_on_step_fail is not None:
        config["pipeline"]["continue_on_step_fail"] = continue_on_step_fail

    for step in config["steps"]:
        if step["name"] == "Load OG1":
            step["parameters"]["file_path"] = str(synthetic_nc)
        elif step["name"] == "Data Export":
            step["parameters"]["output_path"] = str(output_nc)

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return Pipeline(config_path=str(config_file)), output_nc


@pytest.mark.filterwarnings("ignore:.*monotonically increasing.*")
def test_step_failure_stops_pipeline_when_disabled(tmp_path, synthetic_nc):
    p, output_nc = _build_failing_step_pipeline(
        tmp_path, synthetic_nc, continue_on_step_fail=False
    )
    with pytest.raises(SystemExit):
        p.run()
    assert not output_nc.exists()


@pytest.mark.filterwarnings("ignore:.*monotonically increasing.*")
def test_continue_on_step_fail_skips_failing_step(tmp_path, synthetic_nc):
    p, output_nc = _build_failing_step_pipeline(
        tmp_path, synthetic_nc, continue_on_step_fail=True
    )
    p.run()
    assert output_nc.exists(), (
        "continue_on_step_fail should let the pipeline finish after a step fails."
    )


@pytest.mark.filterwarnings("ignore:.*monotonically increasing.*")
def test_continue_on_step_fail_defaults_to_enabled(tmp_path, synthetic_nc):
    p, output_nc = _build_failing_step_pipeline(tmp_path, synthetic_nc)
    p.run()
    assert output_nc.exists(), (
        "continue_on_step_fail should default to enabled (skip on fail)."
    )
