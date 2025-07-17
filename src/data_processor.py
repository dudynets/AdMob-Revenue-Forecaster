"""
Data Processing Module
Handles data validation, transformation, and preparation for the application
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json


class DataProcessor:
    """Data processing and validation utilities"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_data(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Validate revenue data for quality and completeness"""

        try:
            validation_results = {
                "has_data": False,
                "has_revenue_column": False,
                "has_date_index": False,
                "no_negative_values": False,
                "sufficient_data": False,
                "no_excessive_gaps": False,
                "reasonable_values": False,
            }

            # Check if data exists
            if data.empty:
                self.logger.warning("No data provided for validation")
                return validation_results

            validation_results["has_data"] = True

            # Check for revenue column
            if "revenue" not in data.columns:
                self.logger.warning("No 'revenue' column found in data")
                return validation_results

            validation_results["has_revenue_column"] = True

            # Check for date index
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning("Data index is not a DatetimeIndex")
                return validation_results

            validation_results["has_date_index"] = True

            # Check for negative values
            if (data["revenue"] < 0).any():
                self.logger.warning("Found negative revenue values")
            else:
                validation_results["no_negative_values"] = True

            # Check for sufficient data (at least 30 days)
            if len(data) < 30:
                self.logger.warning(
                    f"Insufficient data: {len(data)} days (minimum 30 required)"
                )
            else:
                validation_results["sufficient_data"] = True

            # Check for excessive gaps (more than 7 consecutive days with 0 revenue)
            consecutive_zeros = (
                (data["revenue"] == 0)
                .astype(int)
                .groupby((data["revenue"] != 0).cumsum())
                .sum()
            )

            if consecutive_zeros.max() > 7:
                self.logger.warning(
                    f"Found {consecutive_zeros.max()} consecutive days with zero revenue"
                )
            else:
                validation_results["no_excessive_gaps"] = True

            # Check for reasonable values (not too high or too low)
            max_revenue = data["revenue"].max()
            mean_revenue = data["revenue"].mean()

            if max_revenue > mean_revenue * 100:  # Max > 100x mean
                self.logger.warning(
                    f"Potentially unreasonable max revenue: ${max_revenue:.2f}"
                )
            else:
                validation_results["reasonable_values"] = True

            # Log overall validation status
            passed_checks = sum(validation_results.values())
            total_checks = len(validation_results)

            self.logger.info(
                f"Data validation: {passed_checks}/{total_checks} checks passed"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            return validation_results

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess revenue data"""

        try:
            if data.empty:
                return data

            # Create a copy to avoid modifying original
            cleaned_data = data.copy()

            # Ensure proper date index
            if not isinstance(cleaned_data.index, pd.DatetimeIndex):
                if "date" in cleaned_data.columns:
                    cleaned_data["date"] = pd.to_datetime(cleaned_data["date"])
                    cleaned_data.set_index("date", inplace=True)
                else:
                    self.logger.error("Cannot create date index - no date column found")
                    return data

            # Sort by date
            cleaned_data = cleaned_data.sort_index()

            # Remove duplicates (keep first occurrence)
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep="first")]

            # Handle missing values in revenue
            if "revenue" in cleaned_data.columns:
                # Fill NaN values with 0
                cleaned_data["revenue"] = cleaned_data["revenue"].fillna(0)

                # Handle negative values (set to 0)
                cleaned_data["revenue"] = cleaned_data["revenue"].clip(lower=0)

                # Handle extreme outliers (values > 10x the 95th percentile)
                percentile_95 = cleaned_data["revenue"].quantile(0.95)
                outlier_threshold = percentile_95 * 10

                outliers = cleaned_data["revenue"] > outlier_threshold
                if outliers.any():
                    self.logger.warning(
                        f"Found {outliers.sum()} outliers, capping at ${outlier_threshold:.2f}"
                    )
                    cleaned_data.loc[outliers, "revenue"] = outlier_threshold

            # Add derived features
            cleaned_data = self.add_derived_features(cleaned_data)

            self.logger.info(f"Data cleaned: {len(cleaned_data)} records")
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data

    def add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset"""

        try:
            if data.empty or "revenue" not in data.columns:
                return data

            # Add day of week
            data["day_of_week"] = data.index.dayofweek
            data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

            # Add month and day of month
            data["month"] = data.index.month
            data["day_of_month"] = data.index.day

            # Add rolling averages
            data["revenue_ma_7"] = (
                data["revenue"].rolling(window=7, min_periods=1).mean()
            )
            data["revenue_ma_30"] = (
                data["revenue"].rolling(window=30, min_periods=1).mean()
            )

            # Add rolling standard deviations
            data["revenue_std_7"] = (
                data["revenue"].rolling(window=7, min_periods=1).std()
            )
            data["revenue_std_30"] = (
                data["revenue"].rolling(window=30, min_periods=1).std()
            )

            # Add lag features
            data["revenue_lag_1"] = data["revenue"].shift(1)
            data["revenue_lag_7"] = data["revenue"].shift(7)

            # Add percentage change
            data["revenue_pct_change"] = data["revenue"].pct_change()

            # Add cumulative revenue
            data["revenue_cumulative"] = data["revenue"].cumsum()

            return data

        except Exception as e:
            self.logger.error(f"Error adding derived features: {e}")
            return data

    def resample_data(self, data: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
        """Resample data to different frequency"""

        try:
            if data.empty:
                return data

            # Define aggregation functions
            agg_funcs = {
                "revenue": "sum",
                "revenue_ma_7": "mean",
                "revenue_ma_30": "mean",
                "revenue_std_7": "mean",
                "revenue_std_30": "mean",
                "day_of_week": "first",
                "is_weekend": "max",
                "month": "first",
                "day_of_month": "first",
            }

            # Only use functions for columns that exist
            existing_agg_funcs = {
                col: func for col, func in agg_funcs.items() if col in data.columns
            }

            # Resample
            resampled = data.resample(frequency).agg(existing_agg_funcs)

            # Forward fill missing values
            resampled = resampled.ffill()

            self.logger.info(
                f"Data resampled to {frequency} frequency: {len(resampled)} records"
            )
            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data

    def split_data(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""

        try:
            if data.empty:
                return data, data

            # Calculate split point
            n_test = int(len(data) * test_size)
            n_train = len(data) - n_test

            # Split data
            train_data = data.iloc[:n_train]
            test_data = data.iloc[n_train:]

            self.logger.info(
                f"Data split: {len(train_data)} training, {len(test_data)} testing"
            )
            return train_data, test_data

        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            return data, pd.DataFrame()

    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Get summary statistics for the data"""

        try:
            if data.empty:
                return {}

            summary = {
                "total_records": len(data),
                "date_range": {
                    "start": data.index.min().strftime("%Y-%m-%d"),
                    "end": data.index.max().strftime("%Y-%m-%d"),
                },
                "revenue_stats": {},
                "data_quality": {},
            }

            # Revenue statistics
            if "revenue" in data.columns:
                revenue_stats = {
                    "total_revenue": data["revenue"].sum(),
                    "average_daily_revenue": data["revenue"].mean(),
                    "median_daily_revenue": data["revenue"].median(),
                    "max_daily_revenue": data["revenue"].max(),
                    "min_daily_revenue": data["revenue"].min(),
                    "std_daily_revenue": data["revenue"].std(),
                    "zero_revenue_days": (data["revenue"] == 0).sum(),
                    "positive_revenue_days": (data["revenue"] > 0).sum(),
                }
                summary["revenue_stats"] = revenue_stats

            # Data quality metrics
            data_quality = {
                "missing_values": data.isnull().sum().sum(),
                "duplicate_dates": data.index.duplicated().sum(),
                "negative_values": (
                    (data["revenue"] < 0).sum() if "revenue" in data.columns else 0
                ),
                "completeness_pct": (
                    len(data)
                    / self._get_expected_days(data.index.min(), data.index.max())
                )
                * 100,
            }
            summary["data_quality"] = data_quality

            return summary

        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}

    def _get_expected_days(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate expected number of days between two dates"""

        try:
            return (end_date - start_date).days + 1
        except Exception:
            return 1

    def export_data(
        self, data: pd.DataFrame, filepath: str, format: str = "csv"
    ) -> bool:
        """Export data to file"""

        try:
            if data.empty:
                self.logger.warning("No data to export")
                return False

            if format.lower() == "csv":
                data.to_csv(filepath)
            elif format.lower() == "json":
                data.to_json(filepath, orient="index", date_format="iso")
            elif format.lower() == "excel":
                data.to_excel(filepath)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False

            self.logger.info(f"Data exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False

    def import_data(self, filepath: str, format: str = "csv") -> pd.DataFrame:
        """Import data from file"""

        try:
            if format.lower() == "csv":
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format.lower() == "json":
                data = pd.read_json(filepath, orient="index")
                data.index = pd.to_datetime(data.index)
            elif format.lower() == "excel":
                data = pd.read_excel(filepath, index_col=0, parse_dates=True)
            else:
                self.logger.error(f"Unsupported import format: {format}")
                return pd.DataFrame()

            self.logger.info(f"Data imported from {filepath}: {len(data)} records")
            return data

        except Exception as e:
            self.logger.error(f"Error importing data: {e}")
            return pd.DataFrame()
