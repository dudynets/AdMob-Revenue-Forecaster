"""
Configuration module for AdMob Revenue Forecaster
Handles settings, authentication, and application configuration
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class AppConfig:
    """Application configuration and settings management"""

    def __init__(self):
        # Support XDG_CONFIG_HOME environment variable
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")

        if xdg_config_home:
            self.config_dir = Path(xdg_config_home) / "admob_forecast"
        else:
            # Fallback to ~/.config on Unix-like systems, or ~/.admob_forecast on others
            if os.name == "posix":
                self.config_dir = Path.home() / ".config" / "admob_forecast"
            else:
                self.config_dir = Path.home() / ".admob_forecast"

        self.config_file = self.config_dir / "config.json"
        self.credentials_file = self.config_dir / "credentials.json"
        self.token_file = self.config_dir / "token.json"

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.default_config = {
            "api_settings": {
                "customer_id": "",
                "report_type": "mediation",  # "network" or "mediation"
            },
            "forecast_settings": {
                "default_forecast_days": 365,
                "default_backtest_days": 90,
                "sarima_order": [1, 1, 1],
                "seasonal_order": [1, 1, 1, 7],
                "confidence_interval": 0.95,
            },
            "ui_settings": {
                "window_width": 1400,
                "window_height": 900,
                "theme": "dark",
                "auto_refresh": True,
                "refresh_interval": 3600,  # 1 hour in seconds
                "local_currency": "UAH",
            },
            "data_settings": {
                "min_date": "2024-09-01",
                "cache_duration": 300,  # 5 minutes in seconds
            },
        }

        self.config = self.load_config()
        self.setup_logging()

    def setup_logging(self):
        """Setup application logging"""

        log_file = self.config_dir / "app.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(__name__)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""

        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)

                # Merge with default config to ensure all keys exist
                return self.merge_config(self.default_config, config)
            else:
                self.save_config(self.default_config)
                return self.default_config.copy()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.default_config.copy()

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to file"""

        try:
            config_to_save = config or self.config

            with open(self.config_file, "w") as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def merge_config(
        self, default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge user config with default config"""

        result = default.copy()

        for key, value in user.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'api_settings.client_id')"""

        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value
        self.save_config()

    def get_date_range(self) -> tuple:
        """Get the valid date range for data fetching"""

        min_date = datetime.strptime(self.get("data_settings.min_date"), "%Y-%m-%d")
        max_date = datetime.now() - timedelta(days=1)  # Yesterday

        return min_date.date(), max_date.date()

    def is_api_configured(self) -> bool:
        """Check if API credentials are configured"""

        # Check if credentials file exists and customer_id is set
        credentials_exist = self.credentials_file.exists()
        customer_id_set = bool(self.get("api_settings.customer_id"))

        return credentials_exist and customer_id_set

    def get_credentials_path(self) -> str:
        """Get path to Google OAuth2 credentials file"""

        return str(self.credentials_file)

    def get_token_path(self) -> str:
        """Get path to stored OAuth2 token file"""

        return str(self.token_file)

    def validate_sarima_parameters(self, order: list, seasonal_order: list) -> bool:
        """Validate SARIMA parameters"""

        try:
            # Check if orders are lists of 3 integers
            if not (
                isinstance(order, list)
                and len(order) == 3
                and all(isinstance(x, int) for x in order)
            ):
                return False

            if not (
                isinstance(seasonal_order, list)
                and len(seasonal_order) == 4
                and all(isinstance(x, int) for x in seasonal_order)
            ):
                return False

            # Check if values are non-negative
            if any(x < 0 for x in order + seasonal_order):
                return False

            return True
        except Exception:
            return False

    def get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for given cache key"""

        cache_dir = self.config_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        return cache_dir / f"{cache_key}.json"

    def is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid"""

        if not cache_file.exists():
            return False

        try:
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            return cache_age < self.get("data_settings.cache_duration")
        except Exception:
            return False

    def clear_cache(self):
        """Clear all cached data"""

        try:
            cache_dir = self.config_dir / "cache"

            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
                logging.info("Cache cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
