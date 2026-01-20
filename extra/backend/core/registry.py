import yaml
import os
from typing import Dict, List
from backend.core.schemas import AppConfig
from backend.core.logger import get_logger

logger = get_logger(__name__)

class AppRegistry:
    def __init__(self, config_path: str = "backend/apps.yaml"):
        self.config_path = config_path
        self._apps: Dict[str, AppConfig] = {}
        self.load_apps()

    def load_apps(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}")
            return

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
                
            if not data or "apps" not in data:
                logger.warning("No 'apps' key found in config")
                return

            for app_data in data["apps"]:
                try:
                    app_config = AppConfig(**app_data)
                    self._apps[app_config.id] = app_config
                    logger.info(f"Loaded app: {app_config.id}")
                except Exception as e:
                    logger.error(f"Failed to load app config: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading apps.yaml: {e}")

    def get_app(self, app_id: str) -> AppConfig:
        if app_id not in self._apps:
            raise ValueError(f"App not found: {app_id}")
        return self._apps[app_id]

    def list_apps(self) -> List[AppConfig]:
        return list(self._apps.values())

# Global registry instance
registry = AppRegistry()
