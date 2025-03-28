"""Step Factory"""

from .base_step import BaseStep
from .custom import STEP_CLASSES  # Import all custom steps


def create_step(step_config):
    """Dynamically create a step instance from config"""
    step_name = step_config["name"]
    step_class = STEP_CLASSES.get(step_name)

    if not step_class:
        raise ValueError(f"Step '{step_name}' not recognized.")

    return step_class(
        name=step_name,
        parameters=step_config.get("parameters", {}),
        diagnostics=step_config.get("diagnostics", False),
    )
