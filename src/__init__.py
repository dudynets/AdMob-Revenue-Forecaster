"""
AdMob Revenue Forecaster
A desktop application for forecasting AdMob revenue using SARIMA models
"""

__version__ = "1.0.0"
__author__ = "Oleksandr Dudynets"
__email__ = "hello@dudynets.dev"

from .config import AppConfig
from .admob_api import AdMobAPIClient
from .forecasting import SARIMAForecaster
from .data_processor import DataProcessor

__all__ = [
    "AppConfig",
    "AdMobAPIClient", 
    "SARIMAForecaster",
    "DataProcessor"
] 
