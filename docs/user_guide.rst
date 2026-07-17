User Guide
==========

Detailed explanations of the processing steps, quality control tools, and how to extend the pipeline with your own functionality.

pelagos_py is built around two extensible concepts: **steps**, the stages that make up a pipeline, and **QC checks**, the sub-stages that flag data reliability. Both are defined in Python and configured through your YAML config file. The pages below describe the ones that ship with pelagos_py, and the rest of this page explains how to add your own.

----

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :text-align: center

      .. raw:: html

         <i class="fa-solid fa-list-check card-icon"></i>
         <span class="card-title">Steps</span>

      The individual operations that make up a pipeline — what they do and how to configure them.

      .. button-ref:: steps
         :ref-type: doc
         :expand:
         :color: primary
         :click-parent:

         To the steps guide

   .. grid-item-card::
      :text-align: center

      .. raw:: html

         <i class="fa-solid fa-magnifying-glass card-icon"></i>
         <span class="card-title">Quality Control</span>

      Sub-steps that flag data reliability without modifying values — how QC works and how to configure it.

      .. button-ref:: quality_control
         :ref-type: doc
         :expand:
         :color: primary
         :click-parent:

         To the QC guide

Extending pelagos_py
--------------------

One of the main goals of pelagos_py is to make adding your own processing as
painless as possible. This section explains how to add new **steps** and **QC
checks**, including the base-class interfaces and how new functionality is
registered so the pipeline can discover it.

Templates are provided for both: copy :doc:`blank_step.py <api/pelagos_py/steps/templates/blank_step/index>`
for a new step, or :doc:`blank_qc.py <api/pelagos_py/steps/templates/blank_qc/index>`
for a new QC check. It is still recommended that you read the instructions below
to avoid common implementation issues.

How to add a new step
~~~~~~~~~~~~~~~~~~~~~~~

1. Create a new Python file in the appropriate directory under
   ``src/pelagos_py/steps/``.

   .. note::
      If you are creating a step for specific variables, it should go in the
      ``variables`` subdirectory.

2. Define a new class for your step, inheriting from ``BaseStep`` and adding the
   ``@register_step`` decorator. This makes the step discoverable by the
   Pipeline Manager, while still letting you define other (unregistered) classes
   in the same file.

   .. code-block:: python

      from pelagos_py.steps.base_step import BaseStep, register_step

      @register_step
      class MyNewStep(BaseStep):
          ...

3. Define the ``step_name`` attribute. This is the name used in the pipeline
   config file to refer to the step.

   .. code-block:: python

      from pelagos_py.steps.base_step import BaseStep, register_step

      @register_step
      class MyNewStep(BaseStep):

          step_name = "My New Step"

4. Implement the ``run`` method, which contains the logic for your step. It
   should take no arguments other than ``self`` and should return the
   ``self.context`` object.

   .. code-block:: python

      from pelagos_py.steps.base_step import BaseStep, register_step

      @register_step
      class MyNewStep(BaseStep):
          step_name = "My New Step"

          def run(self):
              # Your processing logic here
              return self.context

5. Optionally, implement the ``generate_diagnostics`` method if your step
   produces any diagnostic plots or outputs.

   .. code-block:: python

      from pelagos_py.steps.base_step import BaseStep, register_step

      @register_step
      class MyNewStep(BaseStep):
          step_name = "My New Step"

          def run(self):
              # Your processing logic here
              return self.context

          def generate_diagnostics(self):
              # Your diagnostics logic here
              pass

   There are already default methods for generating common diagnostics, such as
   time series and scatter plots. See the
   :doc:`diagnostics documentation <api/pelagos_py/utils/diagnostics/index>`
   for more information.

6. Add the step to your pipeline config file, using the ``step_name`` you defined
   in step 3.

   .. code-block:: yaml

      # Pipeline configuration — only needed once at the top of the file
      pipeline:
        name: "My Pipeline"
        description: "A pipeline for demonstration purposes"

      # Steps in the pipeline
      steps:
        - name: "My New Step"
          parameters:
            param1: value1
            param2: value2

7. Any parameters defined in the ``parameters`` section of the config file are
   passed to your step as attributes. You can access them in your ``run`` method
   as ``self.param1``, ``self.param2``, and so on.

   .. note::
      This is handled automatically by the ``BaseStep`` class. More information
      can be found in the
      :doc:`BaseStep documentation <api/pelagos_py/steps/base_step/index>`.

Adding QC handling to a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some users may want to filter out bad data before a processing step runs. This
is implemented through the ``QCHandlingMixin`` (see
:doc:`qc_handling.py <api/pelagos_py/utils/qc_handling/index>`). The pipeline
preserves the original dataset dimensions, so filtered data is either replaced
or reinserted once the step completes.

To give your step QC handling, add ``QCHandlingMixin`` to the class inheritance
and call ``self.filter_qc()``, ``self.reconstruct_data()``, ``self.update_qc()``
and — if you add new variables — ``self.generate_qc({<QC_child>: [*<QC_parents>]})``:

.. code-block:: python

   from pelagos_py.steps.base_step import BaseStep, register_step
   from pelagos_py.utils.qc_handling import QCHandlingMixin

   @register_step
   class MyNewStep(BaseStep, QCHandlingMixin):
       step_name = "My New Step"

       def run(self):
           # Filter the specified QC out of this step's instance of self.data
           # and store it separately.
           self.filter_qc()

           # Your processing logic here. Always use self.data to access your
           # processing inputs, as this is what has been filtered.
           # --------- EXAMPLE ---------
           self.data["C"] = self.data["A"] * self.data["B"]
           # ---------------------------

           # Add the filtered-out data back in, or retain its replacement
           # depending on the user config.
           self.reconstruct_data()
           # Update the flags of the filtered data.
           self.update_qc()

           # If a new variable was added, give it its own QC column derived from
           # its parents' QC. Not necessary if no new variables were added.
           # --------- EXAMPLE ---------
           self.generate_qc({"C_QC": ["A_QC", "B_QC"]})
           # ---------------------------

           self.context["data"] = self.data
           return self.context

       def generate_diagnostics(self):
           # Your diagnostics logic here
           pass

To use QC filtering, the step config must specify ``qc_handling_settings``:

.. code-block:: yaml

   # Steps in the pipeline
   steps:
     - name: "My New Step"
       parameters:
         param1: value1
         param2: value2
         # [qc_handling_settings]:
         #   Can be specified in any step that has the QC filtering functionality.
         qc_handling_settings:
           # [flag_filter_settings]:
           #   {variable: flags to filter} pairs. Data flagged with any of the
           #   specified flags is replaced with a nan internally. All steps
           #   should be designed to operate with nans.
           flag_filter_settings:
             PRES: [3, 4]
           # [reconstruction_behaviour]:
           #   How data is reconstructed after processing (defaults to reinsert):
           #   "replace":  Filtered indices keep their post-processing value and
           #               the original "bad data" is deleted.
           #   "reinsert": The filtered "bad data" is reinserted into the
           #               post-processed data.
           reconstruction_behaviour: "replace"
           # [flag_mapping]:
           #   How flags should change for "bad data" indices when the pre- and
           #   post-processing data differ. E.g. interpolation replaces bad and
           #   missing values (3, 4, 9) with interpolated values (8).
           flag_mapping:
             3: 8
             4: 8
             9: 8
           # [calculation_flag_filter]:
           #   Flags a step's calculations ignore (defaults to [3, 4, 9]). Unlike
           #   flag_filter_settings this does not remove the data: the samples are
           #   still corrected, they just do not inform the correction. E.g. Deep
           #   Correction leaves bad values out of the dark-value estimate but
           #   still subtracts the dark value from them. Set to [] to opt out.
           #   Steps opt in by calling self.calculation_mask([...]) with the
           #   variables whose flags should gate the calculation.
           calculation_flag_filter: [3, 4, 9]

How to add a new QC check
~~~~~~~~~~~~~~~~~~~~~~~~~~~

QC checks operate exclusively on the QC flags of the data variables. This is
useful for researchers post-pipeline who want to remove bad or suspicious data,
or to exclude bad data from specific processing steps (see
`Adding QC handling to a step`_ above). All checks are run through the
:doc:`Apply QC <api/pelagos_py/steps/quality_control/apply_qc/index>` step, which
transfers the individual results onto the existing QC columns.

As described on the :doc:`Quality Control <quality_control>` page, there are two types of
check:

* **Static** checks always operate on the same variable(s) and produce the same
  QC outputs.
* **Dynamic** checks let the user specify which variables they apply to, so the
  QC output is not pre-determined.

A standard structure for dynamic checks is yet to be settled, so this section
covers only the implementation of static checks. Examples of dynamic checks can
be found in
:doc:`stuck_value_qc.py <api/pelagos_py/steps/quality_control/stuck_value_qc/index>`
and
:doc:`range_qc.py <api/pelagos_py/steps/quality_control/range_qc/index>`.
A template for a static check is provided in
:doc:`blank_qc.py <api/pelagos_py/steps/templates/blank_qc/index>`.

1. Create your QC file in ``src/pelagos_py/steps/quality_control/``.

2. Import the parent class and define your QC class. It must inherit from
   ``BaseQC`` and carry the ``@register_qc`` decorator so the pipeline can find
   and register it.

   .. code-block:: python

      from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

      @register_qc
      class MyNewCheck(BaseQC):
          ...

3. Specify the following attributes:

   .. code-block:: python

      from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

      @register_qc
      class MyNewCheck(BaseQC):
          qc_name = "my new check"  # How you refer to the check in config (see below)
          expected_parameters = {'A_cutoff': 1}  # Parameters the user may supply; the value is the default
          required_variables = ['A']  # Variables required for execution; cross-referenced against the data vars in context
          provided_variables = []  # Variables this check itself provides, if any
          qc_outputs = ['A_QC']  # QC outputs; references that help "Apply QC" update existing QC in the data

4. Add the ``return_qc`` method, which implements your check algorithm.
   Optionally add ``plot_diagnostics`` if the check should generate plots when
   ``diagnostics`` is true in the config.

   .. code-block:: python

      from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

      @register_qc
      class MyNewCheck(BaseQC):
          qc_name = "my new check"
          expected_parameters = {'A_cutoff': 1}
          required_variables = ['A']
          provided_variables = []
          qc_outputs = ['A_QC']

          def return_qc(self):
              # IMPORTANT: access the data with self.data.
              # self.flags should be an xarray Dataset whose data_vars hold the
              # "{variable}_QC" columns produced by the check.
              return self.flags

          def plot_diagnostics(self):
              # Add your diagnostic plotting here
              ...

   Access the data using ``self.data`` (an xarray ``Dataset``). The method must
   return an xarray ``Dataset`` (``self.flags``), which can contain any number
   of data variables — but those with the ``_QC`` suffix must be listed in the
   ``qc_outputs`` attribute.

5. Finally, add your new check to the config.

   .. code-block:: yaml

      - name: "Apply QC"
        parameters:
          # qc_settings can list multiple checks
          qc_settings:
            # The qc_name
            my new check:
              # Specify the A_cutoff setting
              A_cutoff: 100
          # If you want plotting:
          diagnostics: true

.. toctree::
   :maxdepth: 2
   :hidden:

   steps
   quality_control
