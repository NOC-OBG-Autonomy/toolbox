"""Step Factory"""

import inspect
import os
import importlib
from .base_step import BaseStep

# Automatically populate STEP_CLASSES by inspecting custom step classes
STEP_CLASSES = {}


def discover_steps():
    """Dynamically populate the STEP_CLASSES dictionary from custom step classes."""
    # Get the path to the custom directory
    custom_dir = os.path.join(os.path.dirname(__file__), "custom")

    # Get all Python files in the custom directory (excluding __init__.py)
    module_files = [
        f for f in os.listdir(custom_dir) if f.endswith(".py") and f != "__init__.py"
    ]
    print(f"Discovered custom steps: {module_files}")

    # Loop through each file, import the module, and inspect its classes
    for module_file in module_files:
        # Get the module name by stripping the '.py' extension
        module_name = module_file[:-3]

        # Dynamically import the module from the custom folder
        module = importlib.import_module(f".custom.{module_name}", package="steps")

        # Inspect the module for classes
        for _, step_class in inspect.getmembers(module, inspect.isclass):
            # Ensure the class is a subclass of BaseStep
            if issubclass(step_class, BaseStep) and step_class is not BaseStep:
                # Get the step_name attribute, if it exists
                step_name = getattr(step_class, "step_name", None)
                if step_name:
                    STEP_CLASSES[step_name] = step_class
                    print(f"Registered step: {step_name} from {module_name}")


# Discover all steps during module import
discover_steps()


def create_step(step_config):
    """Dynamically create a step instance from config"""
    step_name = step_config["name"]
    step_class = STEP_CLASSES.get(step_name)
    print(STEP_CLASSES)
    if not step_class:
        raise ValueError(f"Step '{step_name}' not recognized.")

    return step_class(
        name=step_name,
        parameters=step_config.get("parameters", {}),
        diagnostics=step_config.get("diagnostics", False),
    )
