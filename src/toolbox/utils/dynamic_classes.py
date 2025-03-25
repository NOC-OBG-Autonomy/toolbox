"""Dynamic classes for configuration."""

from pathlib import Path


class DynamicConfig:
    """Dynamic configuration class."""

    def update(self, **kwargs: dict) -> None:
        """Update the configuration."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self, convert_path: bool = True) -> dict:
        """Convert the configuration to a dictionary.

        Args:
            convert_path (bool, optional): Whether to convert the path to
        a string. Defaults to True.

        Returns:
            dict: The configuration as a dictionary.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                if convert_path:
                    result[key] = str(value)
                else:
                    result[key] = value
            elif isinstance(value, DynamicConfig) or issubclass(
                type(value),
                DynamicConfig,
            ):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, DynamicConfig) else v for v in value
                ]
            else:
                result[key] = value
        return result
