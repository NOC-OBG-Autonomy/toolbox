Steps
=====

Steps are the individual operations that make up a pipeline. They are executed in the order they appear in your configuration file. Each step can be customised with parameters to suit your specific dataset.

A step is defined in Python and configured through the pipeline config file. Examples include:

* **I/O steps** — e.g. reading in data with :doc:`Load Data <api/pelagos_py/steps/input_output/load_data/index>`.
* **Variable processing steps** — e.g. adjusting existing salinity measurements with the :doc:`salinity <api/pelagos_py/steps/processing/salinity/index>` step.

Steps are not limited to one per file — a single module can define multiple steps — and any step can be called multiple times within a single config.

Some steps can filter out bad data before processing, using the QC handling mechanism. The pipeline preserves the original dataset dimensions, so filtered data is either replaced or reinserted once the step completes. For how to build a step with this behaviour, see :doc:`Extending pelagos_py <user_guide>`.

Non-specific Steps
------------------

These steps are general utility tools used for data management, preparation, and export. They are not tied to a specific sensor or variable.

* :doc:`Apply QC <api/pelagos_py/steps/quality_control/apply_qc/index>`: A container step used to run various quality control.
* :doc:`Deep Correction <api/pelagos_py/steps/processing/deep_correction/index>`: Subtracts a deep dark offset from a variable (e.g. chlorophyll fluorescence).
* :doc:`Derive CTD <api/pelagos_py/steps/processing/derive_ctd/index>`: Calculates derived physical properties such as density or potential temperature.
* :doc:`Export <api/pelagos_py/steps/input_output/export/index>`: Saves the final processed dataset to a specified format.
* :doc:`Find Profiles <api/pelagos_py/steps/processing/find_profiles/index>`: Segments a time series into individual vertical profiles.
* :doc:`Generate Data <api/pelagos_py/steps/input_output/gen_data/index>`: Creates synthetic data for testing and validation purposes.
* :doc:`Interpolate Data <api/pelagos_py/steps/processing/interpolate_data/index>`: Fills gaps in variables using mathematical interpolation.
* :doc:`Load Data <api/pelagos_py/steps/input_output/load_data/index>`: The entry point for importing data into the pelagos_py.
* :doc:`Find Profile Direction <api/pelagos_py/steps/processing/profile_direction/index>`: Identifies whether data was collected during an ascent or descent.

Variable Specific Steps
-----------------------

* :doc:`CTD <api/pelagos_py/steps/processing/salinity/index>`: Specialised adjustments for Conductivity, Temperature, and Depth data.
* :doc:`Oxygen <api/pelagos_py/steps/processing/oxygen/index>`: Corrections and calibrations for dissolved oxygen sensors.
* :doc:`Chlorophyll <api/pelagos_py/steps/processing/chla_quenching/index>`: Processing steps for fluorescence and chlorophyll-a concentration.
* :doc:`Backscatter <api/pelagos_py/steps/processing/bbp/index>`: Optical backscatter processing and scaling.

Quality Control (QC)
--------------------

Quality Control is handled differently from other steps. Rather than performing a single calculation, the ``Apply QC`` step acts as a manager for multiple sub-steps called **QC**.

When you add ``Apply QC`` to your configuration, you define a list of QC (such as range checks or spike tests) within its parameters. This step is responsible for:

1. **Running QC**: Executing individual quality checks on your variables.
2. **Merging Flags**: Combining results from multiple QC into a single QC column using standardised Argo flagging logic.
3. **Diagnostics**: Generating visualisations to show which data points were flagged and why.

For a full list of available QC and how to configure them, please refer to the :doc:`qc` page.
