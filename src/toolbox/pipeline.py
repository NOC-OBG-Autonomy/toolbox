"""Base class for all toolbox classes."""

import yaml
from steps import create_step, STEP_CLASSES
from graphviz import Digraph


class Pipeline:
    def __init__(self, config_path=None):
        """Initialize pipeline with optional config file"""
        self.steps = []
        # Define the graph for visualiation
        self.graph = Digraph("Pipeline", format="png", graph_attr={"rankdir": "TB"})

        if config_path:
            # Load pipeline configuration from file
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            # Add steps from config
            self.steps = config["steps"]

    def add_step(
        self,
        step_name,
        parameters=None,
        diagnostics=False,
        parent_name=None,
        run_immediately=False,
    ):
        """Dynamically adds a step and optionally runs it immediately"""
        if step_name not in STEP_CLASSES:
            # Check if the step is recognised
            raise ValueError(f"Step '{step_name}' is not recognized.")

        step_config = {
            "name": step_name,
            "parameters": parameters or {},
            "diagnostics": diagnostics,
            "substeps": [],
        }

        if parent_name:
            # Add step as a substep of a parent if it exists
            parent = self._find_step(self.steps, parent_name)
            if parent:
                parent["substeps"].append(step_config)
            else:
                raise ValueError(f"Parent step '{parent_name}' not found.")
        else:
            self.steps.append(step_config)

        print(f"Step '{step_name}' added successfully!")

        if run_immediately:
            self.run_last_step()

    def _find_step(self, steps_list, step_name):
        """Recursively find a step by name"""
        for step in steps_list:
            if step["name"] == step_name:
                return step
            found = self._find_step(step.get("substeps", []), step_name)
            if found:
                return found
        return None

    def execute_step(self, step_config):
        """Executes a single step"""
        step = create_step(step_config)
        print(f"Executing: {step.name}")
        step.run()

    def run_last_step(self):
        """Runs only the most recently added step"""
        if not self.steps:
            print("No steps to run.")
            return

        last_step = self.steps[-1]
        print(f"Running last step: {last_step['name']}")
        self.execute_step(last_step)

    def run(self):
        """Runs the entire pipeline"""
        for step in self.steps:
            self.execute_step(step)

    def visualise_pipeline(self):
        """Generates a visualiation of the pipeline execution"""
        self.graph.clear()

        def add_to_graph(step_config, parent_name=None, step_order=None):
            step_name = step_config["name"]
            diagnostics = step_config.get("diagnostics", False)
            # ensure graphviz node colourway is clear that diagnostics are enabled
            color = "red" if diagnostics else "black"
            self.graph.node(
                step_name,
                step_name,
                color=color,
                style="filled",
                fillcolor="lightblue" if diagnostics else "white",
            )

            if parent_name:
                self.graph.edge(parent_name, step_name)

            # Add an edge to show the order/flow of substeps

            if step_order and len(step_order) > 1:
                for i in range(len(step_order) - 1):
                    self.graph.edge(step_order[i], step_order[i + 1])

            # Recursively add substeps

            substep_order = []
            for substep in step_config.get("substeps", []):
                substep_order.append(substep["name"])
                add_to_graph(substep, parent_name=step_name, step_order=substep_order)

        # Start by iterating through all top-level steps
        for step in self.steps:
            step_order = [step["name"]]  # Top-level step order
            add_to_graph(step, step_order=step_order)

        self.graph.render("pipeline_visualisation", view=True)
