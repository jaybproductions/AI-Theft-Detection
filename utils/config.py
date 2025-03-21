import yaml
import os

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns it as a dictionary.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")