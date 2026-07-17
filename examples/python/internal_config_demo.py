import yaml
from pelagos_py.pipeline import Pipeline

BASE_CONFIG_YAML = """
pipeline:
  name: Example CTD Processing Pipeline
  description: A pipeline for processing CTD data
  log_file: None

steps:
  - name: Load OG1
    parameters:
      file_path: examples/data/OG1/Nelson_646_R.nc
    diagnostics: false

  - name: Format Checker
    parameters:
      standards: ["og"]
      proceed_on_fail: true
    diagnostics: false

  - name: Correct Values
    parameters:
      target_variable: CNDC
      slope: 10.0
      intercept: 0.0
      expected_range: [20, 45]
      corrected_units: mS/cm
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
        impossible date qc: {}
        impossible location qc: {}
        position on land qc: {}
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:

        range qc:
          variable_ranges:
            PRES:
              3: [-2.4, -5]
              4: [-5, -.inf]
            TEMP:
              3: [0, 30]
              4: [-2.5, 40]
            CNDC:
              3: [5, 42]
              4: [2, 45]
          also_flag:
            PRES: [CNDC, TEMP]
            CNDC: [PRES, TEMP]
            TEMP: [PRES, CNDC]

        stuck value qc:
          variables:
            PRES: 2
          also_flag:
            PRES: [CNDC, TEMP]
          plot: [PRES]
    diagnostics: false

  - name: Interpolate Data
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          PRES: [3, 4, 9]
          LATITUDE: [3, 4, 9]
          LONGITUDE: [3, 4, 9]
        reconstruction_behaviour: replace
    diagnostics: false

  - name: Derive CTD
    parameters:
      to_derive: [DEPTH]
    diagnostics: false

  - name: Find Profiles
    parameters:
    diagnostics: false

  - name: Salinity Adjustment
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          CNDC: [3, 4, 9]
          TEMP: [3, 4, 9]
          PROFILE_NUMBER: [3, 4, 9]
        reconstruction_behaviour: reinsert
        flag_mapping:
          0: 5
          1: 5
          2: 5
      filter_window_size: 21
    diagnostics: false

  - name: Derive CTD
    parameters:
      to_derive: [PRAC_SALINITY, ABS_SALINITY, CONS_TEMP, DENSITY]
    diagnostics: false

  # ---------------------- BACKSCATTER ------------------------
  # Convert raw backscatter angle (beta) into the particulate backscattering
  # coefficient BBP, then split the smooth baseline from isolated particle
  # spikes. The quenching step below uses the despiked BBP700_BASELINE so that
  # noise-level backscatter can't blow up the fl:bbp ratio.
  - name: BBP from Beta
    parameters:
      theta: 124
      xfactor: 1.076
    diagnostics: false

  - name: Isolate BBP Spikes
    parameters:
      window_size: 50
      method: median
    diagnostics: false

  # ======================= CHLA SECTION =======================
  # Mixed layer depth (defaults: auto method -> DENSITY) - consumed by the
  # CHLA Quenching step below.
  - name: Mixed Layer Depth
    diagnostics: true

  # Range test on the raw CHLA before correction: probably-bad (3) outside
  # 0-15, bad (4) outside -1-20 mg m-3.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            CHLA:
              3: [0, 15, outside]
              4: [-1, 20, outside]
    diagnostics: false

  # Spike test on CHLA (per-profile MAD-style residual test).
  - name: Apply QC
    parameters:
      qc_settings:
        spike qc:
          variables:
            CHLA: 2
          window_size: 50
          plot: [CHLA]
    diagnostics: false

  - name: Deep Correction
    parameters:
      apply_to: CHLA
      dark_value: null
      depth_var: PRES
      depth_threshold: 950
    diagnostics: true

  - name: CHLA Quenching
    parameters:
      method: thomalla2017
    diagnostics: true

  # Re-run the range test on the corrected CHLA_ADJUSTED, in case the deep
  # and quenching corrections pushed any values out of range.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            CHLA_ADJUSTED:
              3: [0, 15, outside]
              4: [-1, 20, outside]
    diagnostics: false
"""


demo_config = yaml.safe_load(BASE_CONFIG_YAML)

p = Pipeline(config=demo_config)
p.run()
