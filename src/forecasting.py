"""
SARIMA Forecasting Module
Handles time series forecasting and backtesting for AdMob revenue data
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings
warnings.filterwarnings("ignore")


class SARIMAForecaster:
    """SARIMA model for forecasting AdMob revenue"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.fitted_model = None
        self.original_data = None
        self.forecast_results = None

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for SARIMA modeling"""

        try:
            # Ensure data is properly indexed by date
            if not isinstance(data.index, pd.DatetimeIndex):
                data["date"] = pd.to_datetime(data["date"])
                data.set_index("date", inplace=True)

            # Sort by date
            data = data.sort_index()

            # Remove any duplicates
            data = data[~data.index.duplicated(keep="first")]

            # Ensure we have a revenue column
            if "revenue" not in data.columns:
                raise ValueError("Data must contain a 'revenue' column")

            # Handle missing values
            data["revenue"] = data["revenue"].fillna(0)

            # Handle negative values (set to 0)
            data["revenue"] = data["revenue"].clip(lower=0)

            # Add small constant to avoid zero values in log transformation
            data["revenue"] = data["revenue"] + 1e-6

            self.logger.info(f"Prepared {len(data)} data points for forecasting")
            return data

        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise

    def check_stationarity(self, series: pd.Series) -> Dict[str, bool]:
        """Check if the time series is stationary"""

        try:
            results = {}

            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            results["adf_stationary"] = adf_result[1] < 0.05  # p-value < 0.05

            # KPSS test
            kpss_result = kpss(series.dropna())
            results["kpss_stationary"] = kpss_result[1] > 0.05  # p-value > 0.05

            # Both tests should agree for strong evidence
            results["is_stationary"] = (
                results["adf_stationary"] and results["kpss_stationary"]
            )

            return results

        except Exception as e:
            self.logger.error(f"Error checking stationarity: {e}")
            return {
                "is_stationary": False,
                "adf_stationary": False,
                "kpss_stationary": False,
            }

    def auto_arima_order(
        self, data: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3
    ) -> Tuple[int, int, int]:
        """Automatically determine ARIMA order using AIC criterion"""

        try:
            best_aic = np.inf
            best_order = (1, 1, 1)

            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = SARIMAX(
                                data, order=(p, d, q), seasonal_order=(0, 0, 0, 0)
                            )
                            fitted = model.fit(disp=False)
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue

            self.logger.info(
                f"Auto-selected ARIMA order: {best_order} (AIC: {best_aic:.2f})"
            )
            return best_order

        except Exception as e:
            self.logger.error(f"Error in auto ARIMA selection: {e}")
            return (1, 1, 1)

    def fit_model(
        self,
        data: pd.DataFrame,
        order: Optional[Tuple] = None,
        seasonal_order: Optional[Tuple] = None,
    ) -> bool:
        """Fit SARIMA model to the data"""

        try:
            # Prepare data
            prepared_data = self.prepare_data(data)
            revenue_series = prepared_data["revenue"]

            # Store original data
            self.original_data = prepared_data.copy()

            # Use default parameters if not provided
            if order is None:
                order = self.config.get("forecast_settings.sarima_order", [1, 1, 1])
            if seasonal_order is None:
                seasonal_order = self.config.get(
                    "forecast_settings.seasonal_order", [1, 1, 1, 7]
                )

            # Validate parameters
            if not self.config.validate_sarima_parameters(order, seasonal_order):
                self.logger.warning("Invalid SARIMA parameters, using auto-selection")
                order = self.auto_arima_order(revenue_series)
                seasonal_order = (1, 1, 1, 7)

            self.logger.info(
                f"Fitting SARIMA model with order={order}, seasonal_order={seasonal_order}"
            )

            # Fit the model
            self.model = SARIMAX(
                revenue_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            self.fitted_model = self.model.fit(disp=False, maxiter=100)

            # Log model summary
            self.logger.info(
                f"Model fitted successfully. AIC: {self.fitted_model.aic:.2f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error fitting SARIMA model: {e}")
            return False

    def forecast(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate forecasts using the fitted model"""

        try:
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before forecasting")

            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps)

            # Get forecast values and confidence intervals
            forecast_values = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int(alpha=1 - confidence_level)

            # Create forecast dates
            last_date = self.original_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1), periods=steps, freq="D"
            )

            # Create forecast DataFrame
            forecast_df = pd.DataFrame(
                {
                    "date": forecast_dates,
                    "forecast": forecast_values.values,
                    "lower_ci": confidence_intervals.iloc[:, 0].values,
                    "upper_ci": confidence_intervals.iloc[:, 1].values,
                }
            )

            forecast_df.set_index("date", inplace=True)

            # Store forecast results
            self.forecast_results = forecast_df

            self.logger.info(f"Generated {steps} days of forecasts")
            return forecast_df

        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return pd.DataFrame()

    def backtest(self, data: pd.DataFrame, test_months: int = 3) -> Dict:
        """Perform backtesting by excluding last N months and forecasting them"""

        try:
            # Prepare data
            prepared_data = self.prepare_data(data)

            # Split data
            test_days = test_months * 30  # Approximate
            train_data = prepared_data.iloc[:-test_days]
            test_data = prepared_data.iloc[-test_days:]

            if len(train_data) < 30:
                raise ValueError("Insufficient training data for backtesting")

            self.logger.info(
                f"Backtesting: {len(train_data)} training days, {len(test_data)} test days"
            )

            # Fit model on training data
            temp_forecaster = SARIMAForecaster(self.config)
            if not temp_forecaster.fit_model(train_data):
                raise ValueError("Failed to fit model for backtesting")

            # Generate forecasts
            forecast_df = temp_forecaster.forecast(len(test_data))

            if forecast_df.empty:
                raise ValueError("Failed to generate forecasts for backtesting")

            # Align forecasts with actual data
            forecast_df = forecast_df.reindex(test_data.index)

            # Calculate metrics
            actual_values = test_data["revenue"].values
            forecast_values = forecast_df["forecast"].values

            # Handle NaN values
            valid_indices = ~(np.isnan(actual_values) | np.isnan(forecast_values))
            actual_values = actual_values[valid_indices]
            forecast_values = forecast_values[valid_indices]

            if len(actual_values) == 0:
                raise ValueError("No valid data points for backtesting")

            # Calculate metrics
            mae = mean_absolute_error(actual_values, forecast_values)
            mse = mean_squared_error(actual_values, forecast_values)
            rmse = np.sqrt(mse)
            mape = (
                np.mean(
                    np.abs((actual_values - forecast_values) / (actual_values + 1e-8))
                )
                * 100
            )

            # Create backtest results
            backtest_results = {
                "train_data": train_data,
                "test_data": test_data,
                "forecast_data": forecast_df,
                "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape},
                "test_period": f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
            }

            self.logger.info(
                f"Backtesting completed. RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%"
            )

            return backtest_results

        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return {}

    def get_model_diagnostics(self) -> Dict:
        """Get model diagnostics and statistics"""

        try:
            if self.fitted_model is None:
                return {}

            # Get residuals
            residuals = self.fitted_model.resid

            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

            # Basic statistics
            diagnostics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "log_likelihood": self.fitted_model.llf,
                "residual_mean": residuals.mean(),
                "residual_std": residuals.std(),
                "ljung_box_p_value": lb_test["lb_pvalue"].iloc[-1],
                "model_order": self.fitted_model.model.order,
                "seasonal_order": self.fitted_model.model.seasonal_order,
            }

            return diagnostics

        except Exception as e:
            self.logger.error(f"Error getting model diagnostics: {e}")
            return {}

    def seasonal_decomposition(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform seasonal decomposition of the time series"""

        try:
            prepared_data = self.prepare_data(data)
            revenue_series = prepared_data["revenue"]

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                revenue_series, model="additive", period=7  # Weekly seasonality
            )

            # Create decomposition DataFrame
            decomp_df = pd.DataFrame(
                {
                    "original": revenue_series,
                    "trend": decomposition.trend,
                    "seasonal": decomposition.seasonal,
                    "residual": decomposition.resid,
                }
            )

            return decomp_df

        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition: {e}")
            return pd.DataFrame()

    def get_feature_importance(self) -> Dict:
        """Get feature importance from the fitted model"""

        try:
            if self.fitted_model is None:
                return {}

            # Get model parameters
            params = self.fitted_model.params

            # Create feature importance dictionary
            importance = {}
            for i, param in enumerate(params):
                importance[f"parameter_{i}"] = abs(param)

            return importance

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
