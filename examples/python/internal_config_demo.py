import yaml
from pelagos_py.pipeline import Pipeline

# Same pipeline as examples/configs/example_config_nelson.yaml, but defined inline and
# passed to Pipeline(config=...) as a dict rather than loaded from a file. See
# external_config_demo.py for the config_path route.
BASE_CONFIG_YAML = """
pipeline:
  name: Complete processing pipeline
  description: A complete demonstration pipeline using Nelson Near Realtime data.
  out_directory: examples/data/OG1/
  log_file: complete_processing.log

steps:

# ===========================================================
#                       IMPORT DATA
# ===========================================================
  - name: Load OG1
    parameters:
      file_path: examples/data/OG1/Nelson_646_R.nc # Path to the input NetCDF file
    diagnostics: false

# Check the loaded file against the OG1 standard. With no 'src', it reuses the
# file from Load OG1 above. A short pass/fail summary is printed to the console;
# omit 'output_type' (as here) for console only, or set it to 'json'/'rst' (with
# an out_directory) to also save a detailed report file.
  - name: Format Checker
    parameters:
      standards: ['og']     # og = OG1
      proceed_on_fail: true # Keep going even if the file fails the checks
    diagnostics: false

# ===========================================================
#                  LATITUDE / LONGITUDE
# ===========================================================
# QC of the coordinate variables. These only depend on LATITUDE, LONGITUDE and
# TIME, so they run first, before anything touches the science variables.
  - name: Apply QC
    parameters:
      qc_settings:
        impossible date qc: {}     # Is the date between 1985 and now?
        impossible location qc: {} # Are the GPS coords within logical bounds?
        position on land qc: {}    # Are the coords on land?
        impossible speed qc: {}    # Is the horizontal speed reasonable?
    diagnostics: false # Plot all test outcomes

# ===========================================================
#                          CTD
# ===========================================================
# Nelson data has incorrect units for CNDC, so the values are corrected before
# any of the CTD tests below run against them.
  - name: Correct Values
    parameters:
      target_variable: CNDC
      slope: 10.0
      intercept: 0.0
      expected_range: [20, 45]
      corrected_units: mS/cm
    diagnostics: false

# Range and stuck-value tests on the CTD triad (PRES, TEMP, CNDC). The spike
# test on these same three lives further down: it is a per-profile test and so
# needs PROFILE_NUMBER, which does not exist until Find Profiles has run.
  - name: Apply QC
    parameters:
      qc_settings:

        range qc: # Flags the whole CTD triad (PRES, TEMP, CNDC) by range
          variable_ranges:
            PRES: # 'inside' -> flag PRES when it falls WITHIN the band
              3: [-2.4, -5, inside] # Pressure in this range is flagged as probably bad (3)
              4: [-5, -.inf, inside] # and bad (4) in this range.
            TEMP: # 'outside' -> flag TEMP when it falls OUTSIDE the band
              3: [0, 30, outside]
              4: [-2.5, 40, outside]
            CNDC:
              3: [5, 42, outside]
              4: [2, 45, outside]
          also_flag:
            # Cross-flag the CTD triad: PRES, TEMP & CNDC come from the same instrument, so
            # if one is untrustworthy the others probably are too.
            PRES: [CNDC, TEMP]
            CNDC: [PRES, TEMP]
            TEMP: [PRES, CNDC]

        stuck value qc: # Checks for consecutive same values
          variables:
            PRES: 2 # If pressure is stuck for >2 values, they are flagged as bad
          also_flag:
            PRES: [CNDC, TEMP] # Apply the bad flags to CNDC & TEMP too
          plot: [PRES] # Plot the outcome for just pressure
    diagnostics: false # Plot the CTD QC outcomes (set false to skip the blocking plots)

# ===========================================================
#              TEMPERATURE CROSS CALIBRATION
# ===========================================================
# Align TEMP against the reference sensor, as early as possible so that
# everything downstream (salinity, density, MLD) sees the calibrated value.
# PLACEHOLDER: slope 1.0 / intercept 0.0 is the identity (corrected =
# slope * value + intercept), so this currently does nothing. Put the real
# cross-calibration coefficients in here when they are available.
  - name: Correct Values
    parameters:
      target_variable: TEMP
      slope: 1.0        # PLACEHOLDER - identity, replace with real coefficient
      intercept: 0.0    # PLACEHOLDER - identity, replace with real offset
    diagnostics: false

# ===========================================================
#                      INTERPOLATION
# ===========================================================
# Since we are using near-realtime data there are a lot of missing values. Also, since
# the GPS and science measurement clocks do not sample at the same rate, there are many
# points that have a CTD measurement but nans for GPS location. Since we need both to
# process CTD, we need to intepolate to fill in the missing data.
  - name: Interpolate Data
    parameters:
      qc_handling_settings:
        # This QC filtering functionality can be specified in most processing steps by
        # adding the qc_handling_settings parameter. Here we are using it to nan-out bad (4)
        # probably bad (3) and missing (9) data so that it can be interpolated over.
        flag_filter_settings:
          PRES: [3, 4, 9]
          LATITUDE: [3, 4, 9]
          LONGITUDE: [3, 4, 9]
        reconstruction_behaviour: replace
        flag_mapping: # The QC flags where bad data has been replaced turn into 8 (interpolated)
          3: 8
          4: 8
          9: 8
    diagnostics: false # Plot the outcome after interpolation

# ===========================================================
#                    PROFILE FINDING
# ===========================================================
# We now use the derive CTD step to add the DEPTH variable to the dataset as it
# is needed for profiling.
  - name: Derive CTD
    parameters:
      to_derive: [DEPTH]
    diagnostics: false

# Now we assign profile numbers, phases and profile direction
  - name: Find Profiles
    diagnostics: false # When true, this step plots a static phase-map + profile/cycle figure.

# Now that profiles have been determined we should flag any that are too short or lack data at depth
  - name: Apply QC
    parameters:
      qc_settings:
          valid profile qc:
            profile_length: 50 # Profiles must be at least 100 points long
            depth_range: [0, 1000] # and must contain data within 0 and 1000 m
    diagnostics: false

# ===========================================================
#                    CTD SPIKE TESTS
# ===========================================================
# The remaining CTD test: a per-profile rolling-median/MAD residual test on the
# triad. It has to sit here rather than up in the CTD section because it needs
# PROFILE_NUMBER from Find Profiles above.
  - name: Apply QC
    parameters:
      qc_settings:
        spike qc:
          variables: # Variable -> spike sensitivity
            PRES: 2
            TEMP: 2
            CNDC: 2
          also_flag: # Cross-flag the triad, as in the range test above
            PRES: [CNDC, TEMP]
            CNDC: [PRES, TEMP]
            TEMP: [PRES, CNDC]
          window_size: 50 # Rolling-median window size in samples
          plot: [PRES, TEMP, CNDC]
    diagnostics: false

# ===========================================================
#                  SALINITY ADJUSTMENT
# ===========================================================
# Now we move onto adjustment of the CNDC variable and subsequent salinity derivations
  - name: Salinity Adjustment
    parameters:
      qc_handling_settings:
        flag_filter_settings: # Filter out bad PROFILE_NUMBER, TEMP & CNDC data
          CNDC: [3, 4, 9]
          TEMP: [3, 4, 9]
          PROFILE_NUMBER: [3, 4, 9]
        reconstruction_behaviour: reinsert # The bad data will be added back in after processing
        flag_mapping: # The data that goes through adjustment will be reflagged as processed (5)
          0: 5
          1: 5
          2: 5
      filter_window_size: 21
    diagnostics: false

# Derive practical salinity on its own, so that the cross calibration below can
# be applied to it before the remaining variables are derived from it.
  - name: Derive CTD
    parameters:
      to_derive: [PRAC_SALINITY]
    diagnostics: false

# ===========================================================
#                SALINITY CROSS CALIBRATION
# ===========================================================
# Align PRAC_SALINITY against the reference sensor. This sits between the two
# Derive CTD calls so ABS_SALINITY, CONS_TEMP and DENSITY below are derived from
# the calibrated salinity rather than the raw one.
# PLACEHOLDER: slope 1.0 / intercept 0.0 is the identity, so this currently does
# nothing. Put the real cross-calibration coefficients in here when available.
  - name: Correct Values
    parameters:
      target_variable: PRAC_SALINITY
      slope: 1.0        # PLACEHOLDER - identity, replace with real coefficient
      intercept: 0.0    # PLACEHOLDER - identity, replace with real offset
    diagnostics: false

# Now derive the remaining CTD variables from the calibrated salinity.
  - name: Derive CTD
    parameters:
      to_derive: [
        ABS_SALINITY,
        CONS_TEMP,
        DENSITY
      ]
    diagnostics: false

# ===========================================================
#              CTD INTERPOLATION (POST CALIBRATION)
# ===========================================================
# Fill out the calibrated CTD variables so every timestamp carries a TEMP, PRES
# and PRAC_SALINITY value. Bad (4), probably-bad (3) and missing (9) points are
# nan'd out, interpolated over, and reflagged as interpolated (8).
  - name: Interpolate Data
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          TEMP: [3, 4, 9]
          PRES: [3, 4, 9]
          PRAC_SALINITY: [3, 4, 9]
        reconstruction_behaviour: replace
        flag_mapping:
          3: 8
          4: 8
          9: 8
    diagnostics: false

# ===========================================================
#                      BACKSCATTER
# ===========================================================
# Backscatter is processed before chlorophyll because the quenching correction
# below relies on the despiked BBP baseline.

# QC the raw beta before converting it. Backscatter cannot physically be
# negative, but a small negative reading is just the sensor's dark-offset noise
# in clear water, so those are only probably-bad (3); the 4 band catches values
# far enough out to be nonsense. Note there is deliberately NO spike test here:
# BBP spikes are individual particles/aggregates, i.e. real signal, and the
# Isolate BBP Spikes step below separates them rather than discarding them.
# The BBP from Beta step writes its output back over BBP700, so these flags stay
# attached to the converted variable.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            BBP700:
              3: [0, 0.01, outside]     # Negative = dark-offset noise, high = suspect
              4: [-1.0e-4, 0.05, outside] # Physically absurd / sensor rail
        stuck value qc: # A frozen reading means a dead sensor
          variables:
            BBP700: 3
    diagnostics: false

# Convert the raw backscatter angle measurement (beta) into the particulate
# backscattering coefficient BBP, using the scattering angle and chi factor.
  - name: BBP from Beta
    parameters:
      theta: 124               # Effective optical backscatter scattering angle (degrees)
      xfactor: 1.076           # Chi factor scaling particulate scattering to total backscatter
    diagnostics: false

# Large isolated spikes in BBP correspond to individual particles/aggregates.
# This step separates that spike signal from the smooth baseline with a moving
# filter, leaving BBP700_BASELINE for the quenching step to use.
  - name: Isolate BBP Spikes
    parameters:
      window_size: 50          # Filter window size in samples
      method: median           # Filter method used to determine the baseline
    diagnostics: false

# ===========================================================
#                 CHLOROPHYLL CORRECTIONS
# ===========================================================
# QC the PAR input to the quenching step below. Deliberately NO lower bound:
# more than half the PAR record is negative, which is night and depth, not bad
# data - and thomalla2017 builds its fl:bbp ratio from the night profiles, so
# flagging those away would break the method. Only a physically impossible high
# reading (solar max is around 2500) and a frozen sensor are flagged.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            DOWNWELLING_PAR:
              4: [-5, 2500, outside] # Absurd high, or a huge negative excursion
        stuck value qc: # A frozen reading means a dead sensor
          variables:
            DOWNWELLING_PAR: 5
    diagnostics: false

# Mixed layer depth (defaults: auto method -> DENSITY) - consumed by the
# MLD-based quenching methods. Computed here so the method can be swapped
# without reordering steps.
  - name: Mixed Layer Depth
    diagnostics: false

# Global range test on the raw CHLA: probably-bad (3) outside 0.14-50, bad (4)
# outside -0.2-100. Most-severe flag wins on overlap, so e.g. a -0.5 value is 4
# and a 0.1 value is 3; anything in-band is good (1).
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            CHLA:
              4: [-0.2, 100, outside]
    diagnostics: false

# Spike-test CHLA (per-profile MAD-style residual test).
  - name: Apply QC
    parameters:
      qc_settings:
        spike qc:
          variables:
            CHLA: 2
          window_size: 50
          plot: [CHLA]
    diagnostics: false

# Chlorophyll fluorescence needs two corrections. First, a deep (dark-offset)
# correction removes the sensor's baseline drift. The dark value is the residual
# fluorescence measured deep down where there should be no chlorophyll, so it is
# subtracted from the whole profile. Here dark_value is null, so it is computed
# from the data using only measurements below depth_threshold.
  - name: Deep Correction
    parameters:
      apply_to: CHLA           # Variable to correct
      dark_value: null         # null -> compute the dark offset from the data
      depth_threshold: 950     # Only use data below this depth to estimate the dark value
    diagnostics: false         # Plot the computed dark value and corrected profiles

# Flag CHLA by DEPTH rather than by its own value: the top 5 m is unstable, so
# mark it probably-bad-but-correctable (3) and everything deeper probably-good
# (2). DEPTH itself is not flagged, and the Argo merge means the 2 can't
# downgrade a 3/4 from the range or spike tests above. The quenching step below
# then leaves the top 5 m out of its calculations while still correcting it.
# CHLA_ADJUSTED is flagged too, not just CHLA: Deep Correction has already run,
# so the quenching step reads CHLA_ADJUSTED and gates on CHLA_ADJUSTED_QC.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            DEPTH:
              3: [0, 5, inside]
              2: [0, 5, outside]
          flag_instead:
            DEPTH: [CHLA, CHLA_ADJUSTED]
    diagnostics: false

# Second, a non-photochemical quenching (NPQ) correction. Near the surface,
# daylight suppresses fluorescence, so chlorophyll reads artificially low. The
# thomalla2017 method builds a per-night fluorescence:backscatter ratio profile
# and applies it to the following day's profiles, so it needs the BBP and PAR
# variables produced/loaded above.
  - name: CHLA Quenching
    parameters:
      method: thomalla2017     # Per-night fl:bbp ratio profile (needs BBP + PAR)
      apply_to: CHLA           # Variable to correct (uses CHLA_ADJUSTED if it already exists)
    diagnostics: false

# Re-run the range test on the corrected CHLA_ADJUSTED, in case the deep and
# quenching corrections pushed any values out of range.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            CHLA_ADJUSTED:
              3: [0.14, 50, outside]
              4: [0, 100, outside]
    diagnostics: false

# ===========================================================
#                      EXPORT DATA
# ===========================================================
# Exporting the output data to the same folder as the input data
  - name: "Data Export"
    parameters:
      export_format: "netcdf"
      output_path: "examples/data/OG1/Nelson_646_R_Processed.nc"

# Report assembly
  - name: Write Data Report (Python)
"""


demo_config = yaml.safe_load(BASE_CONFIG_YAML)

p = Pipeline(config=demo_config)
p.run()
