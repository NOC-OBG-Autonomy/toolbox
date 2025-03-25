"""Top-level parameters for Variable Quality Control Parameters."""

from dataclasses import dataclass
from toolbox.utils.dynamic_classes import DynamicConfig


@dataclass
class SalinityQC(DynamicConfig):
    """Salinity Quality Control Parameters."""

    pass


@dataclass
class TemperatureQC(DynamicConfig):
    """Temperature Quality Control Parameters."""

    pass
