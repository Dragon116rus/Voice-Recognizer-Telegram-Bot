from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class BotConfig:
    """
    Data class representing bot configuration.

    Attributes:
        model (str): Default model for speech-to-text conversion.
    """
    model: str = "Dragon116rus/whisper-small-distill-ru"

    @classmethod
    def from_yaml(cls, file_path: Path |str) -> 'BotConfig':
        """
        Create an instance of BotConfig from YAML data.

        Args:
            file_path (Path): Path to the YAML configuration file.

        Returns:
            BotConfig: An instance of BotConfig.
        """
        with open(file_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        return cls(**yaml_data)
