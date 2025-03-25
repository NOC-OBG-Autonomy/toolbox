"""Data Import and Processing parameters dataclass.

This module contains the dataclass definition for the data import and processing
"""

from dataclasses import dataclass
from toolbox.utils.dynamic_classes import DynamicConfig


@dataclass
class FactoryCalibration(DynamicConfig):
    """Factory Calibration parameters."""

    pass


class SensorPecificCorrection(DynamicConfig):
    """Sensor Specific Correction parameters."""

    pass


class AutomatedCrossSensorAdjustment(DynamicConfig):
    """Automated Cross Sensor Adjustment parameters."""

    pass
