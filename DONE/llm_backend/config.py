import yaml
from .core.app_config import AppConfig

def load_app_configs(path: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return [AppConfig(**app) for app in raw["apps"]]
