"""
Main UI Module
PyQt6 interface for the AdMob Revenue Forecaster
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QDateEdit,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QSplitter,
    QMessageBox,
    QFileDialog,
    QStatusBar,
    QMenuBar,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QDate
from PyQt6.QtGui import QFont, QIcon, QAction

# Try to import QWebEngineView, fall back to QTextEdit if not available
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    WEB_ENGINE_AVAILABLE = True
except ImportError:
    WEB_ENGINE_AVAILABLE = False
    QWebEngineView = None

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from admob_api import AdMobAPIClient
from forecasting import SARIMAForecaster
from data_processor import DataProcessor


def create_table(title, content):
    """Create a table with the given title and content"""

    return f"""
<h3>{title}</h3>
<table style="border-collapse: collapse; table-layout: fixed; margin: 16px 0 0 0;">
{content}
</table>
"""


def create_th(text):
    """Create a table header cell with the given text"""

    return f"<th style='padding: 8px; border: 1px solid white; text-align: left;'>{text}</th>"


def create_td(text):
    """Create a table data cell with the given text"""

    return f"<td style='padding: 8px; border: 1px solid white; text-align: left;'>{text}</td>"


class DataFetchingThread(QThread):
    """Background thread for fetching data from AdMob API"""

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    data_ready = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_client, start_date, end_date):
        super().__init__()
        self.api_client = api_client
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            self.progress_update.emit(10)
            self.status_update.emit("Authenticating with AdMob API...")

            # Check if authentication succeeds
            if not self.api_client.authenticate():
                self.error_occurred.emit(
                    "Authentication failed. Please check your OAuth2 credentials and try again."
                )
                return

            self.progress_update.emit(30)
            self.status_update.emit("Fetching revenue data...")

            # Fetch data
            data = self.api_client.fetch_revenue_data(
                self.start_date, self.end_date, use_cache=True
            )

            self.progress_update.emit(100)

            if data.empty:
                self.error_occurred.emit(
                    "No data retrieved from AdMob API. Please check your date range and account access."
                )
            else:
                self.status_update.emit(
                    f"Successfully fetched {len(data)} days of data"
                )
                self.data_ready.emit(data)

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower():
                self.error_occurred.emit(f"Authentication error: {error_msg}")
            elif "403" in error_msg or "permission" in error_msg.lower():
                self.error_occurred.emit(
                    f"Permission error: {error_msg}. Please check your AdMob account access."
                )
            else:
                self.error_occurred.emit(f"Error fetching data: {error_msg}")


class ForecastingThread(QThread):
    """Background thread for running forecasting models"""

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    forecast_ready = pyqtSignal(pd.DataFrame, dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self, data, config, forecast_days, run_backtest=True, backtest_days=90
    ):
        super().__init__()
        self.data = data
        self.config = config
        self.forecast_days = forecast_days
        self.run_backtest = run_backtest
        self.backtest_days = backtest_days

    def run(self):
        try:
            self.progress_update.emit(10)
            self.status_update.emit("Preparing data for forecasting...")

            forecaster = SARIMAForecaster(self.config)

            self.progress_update.emit(30)
            self.status_update.emit("Fitting SARIMA model...")

            if not forecaster.fit_model(self.data):
                self.error_occurred.emit("Failed to fit SARIMA model")
                return

            self.progress_update.emit(60)
            self.status_update.emit("Generating forecasts...")

            forecast_data = forecaster.forecast(self.forecast_days)

            results = {
                "forecast_data": forecast_data,
                "diagnostics": forecaster.get_model_diagnostics(),
            }

            if self.run_backtest:
                self.progress_update.emit(80)
                self.status_update.emit("Running backtest...")

                # Convert backtest days to months (approximately)
                test_months = max(1, self.backtest_days // 30)
                backtest_results = forecaster.backtest(
                    self.data, test_months=test_months
                )
                results["backtest"] = backtest_results

            self.progress_update.emit(100)
            self.status_update.emit("Forecasting completed successfully")

            self.forecast_ready.emit(forecast_data, results)

        except Exception as e:
            self.error_occurred.emit(f"Error in forecasting: {str(e)}")


class SettingsTab(QWidget):
    """Settings tab for API configuration and application settings"""

    def __init__(self, config, currency_formatter):
        super().__init__()
        self.config = config
        self.currency_formatter = currency_formatter
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # API Settings Group
        api_group = QGroupBox("AdMob API Settings")
        api_layout = QFormLayout()

        # Add instructions
        instructions_label = QLabel(
            """
<b>Setup Instructions:</b><br><br>
1. Go to <a href="https://console.cloud.google.com/apis/credentials">Google Cloud Console</a><br>
2. Create OAuth2 credentials (Desktop application type)<br>
3. Download the JSON file and upload it below<br>
4. Find your Customer ID (Publisher ID) in AdMob console<br>
5. Select the appropriate report type<br>
6. Test the connection
        """
        )
        instructions_label.setWordWrap(True)
        instructions_label.setOpenExternalLinks(True)
        instructions_label.setStyleSheet("QLabel { font-size: 12px; color: #888; }")
        api_layout.addRow("", instructions_label)

        # OAuth2 Credentials File Upload
        credentials_layout = QHBoxLayout()
        self.credentials_status_label = QLabel("No credentials file uploaded")
        self.credentials_status_label.setStyleSheet(
            "QLabel { color: #888; font-style: italic; }"
        )
        self.upload_credentials_btn = QPushButton("Upload OAuth2 JSON File")
        self.upload_credentials_btn.clicked.connect(self.upload_credentials_file)
        credentials_layout.addWidget(self.credentials_status_label)
        credentials_layout.addWidget(self.upload_credentials_btn)

        api_layout.addRow("OAuth2 Credentials:", credentials_layout)

        # Customer ID (manual entry required)
        self.customer_id_edit = QLineEdit(
            self.config.get("api_settings.customer_id", "")
        )
        self.customer_id_edit.setPlaceholderText(
            "pub-1234567890123456 (from AdMob console)"
        )
        api_layout.addRow("Customer ID:", self.customer_id_edit)

        # Report Type Selection
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItem("Mediation Report (recommended)", "mediation")
        self.report_type_combo.addItem("Network Report", "network")

        # Set current report type
        current_report_type = self.config.get("api_settings.report_type", "mediation")
        report_type_index = self.report_type_combo.findData(current_report_type)
        if report_type_index >= 0:
            self.report_type_combo.setCurrentIndex(report_type_index)

        # Connect report type change to clear cache
        self.report_type_combo.currentTextChanged.connect(self.on_report_type_changed)

        api_layout.addRow("Report Type:", self.report_type_combo)

        # Add report type explanation
        report_type_explanation = QLabel(
            """
<b>Report Types:</b><br><br>
• <b>Mediation Report:</b> Includes revenue from all sources (AdMob Network + Third-party mediation)<br>
• <b>Network Report:</b> Only includes revenue from AdMob Network (direct ads)<br><br>
<i>Use Mediation Report for complete revenue data including mediated ads.</i>
        """
        )
        report_type_explanation.setWordWrap(True)
        report_type_explanation.setStyleSheet(
            "QLabel { font-size: 12px; color: #888; }"
        )
        # dynamically set the height of the label to the height of the text
        api_layout.addRow("", report_type_explanation)

        # Test Connection Button
        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_connection)
        api_layout.addRow("", self.test_connection_btn)

        api_group.setLayout(api_layout)

        # Update credentials status on initialization
        self.update_credentials_status()

        # Forecast Settings Group
        forecast_group = QGroupBox("Forecast Settings")
        forecast_layout = QFormLayout()

        self.forecast_days_spin = QSpinBox()
        self.forecast_days_spin.setRange(1, 365 * 10)  # 10 years
        self.forecast_days_spin.setValue(
            self.config.get("forecast_settings.default_forecast_days", 30)
        )
        self.forecast_days_spin.setSuffix(" days")

        self.backtest_days_spin = QSpinBox()
        self.backtest_days_spin.setRange(1, 365 * 10)  # 10 years
        self.backtest_days_spin.setValue(
            self.config.get("forecast_settings.default_backtest_days", 90)
        )
        self.backtest_days_spin.setSuffix(" days")

        # SARIMA Parameters
        sarima_order = self.config.get("forecast_settings.sarima_order", [1, 1, 1])
        seasonal_order = self.config.get(
            "forecast_settings.seasonal_order", [1, 1, 1, 7]
        )

        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 5)
        self.p_spin.setValue(sarima_order[0])

        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 2)
        self.d_spin.setValue(sarima_order[1])

        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 5)
        self.q_spin.setValue(sarima_order[2])

        self.sp_spin = QSpinBox()
        self.sp_spin.setRange(0, 5)
        self.sp_spin.setValue(seasonal_order[0])

        self.sd_spin = QSpinBox()
        self.sd_spin.setRange(0, 2)
        self.sd_spin.setValue(seasonal_order[1])

        self.sq_spin = QSpinBox()
        self.sq_spin.setRange(0, 5)
        self.sq_spin.setValue(seasonal_order[2])

        self.seasonal_period_spin = QSpinBox()
        self.seasonal_period_spin.setRange(2, 365)
        self.seasonal_period_spin.setValue(seasonal_order[3])

        forecast_layout.addRow("Default Forecast Days:", self.forecast_days_spin)
        forecast_layout.addRow("Default Backtest Days:", self.backtest_days_spin)
        forecast_layout.addRow("SARIMA p:", self.p_spin)
        forecast_layout.addRow("SARIMA d:", self.d_spin)
        forecast_layout.addRow("SARIMA q:", self.q_spin)
        forecast_layout.addRow("Seasonal P:", self.sp_spin)
        forecast_layout.addRow("Seasonal D:", self.sd_spin)
        forecast_layout.addRow("Seasonal Q:", self.sq_spin)
        forecast_layout.addRow("Seasonal Period:", self.seasonal_period_spin)

        # Add SARIMA explanation
        explanation_label = QLabel(
            """
<b>SARIMA Parameters Explanation:</b><br><br>

<b>p (AutoRegressive):</b> Number of lag observations in the model (0-5)<br>
<i>- Higher values capture more historical patterns</i><br>
<i>- Start with 1-2 for daily revenue data</i>

<br><br>

<b>d (Differencing):</b> Degree of differencing to make data stationary (0-2)<br>
<i>- 0 = data is already stationary</i><br>
<i>- 1 = first difference (most common)</i><br>
<i>- 2 = second difference (rarely needed)</i>

<br><br>

<b>q (Moving Average):</b> Size of moving average window (0-5)<br>
<i>- Captures error correction from previous periods</i><br>
<i>- Start with 1-2 for daily revenue data</i>

<br><br>

<b>P, D, Q (Seasonal):</b> Same as p, d, q but for seasonal patterns<br>
<i>- Use when data has recurring patterns (weekly, monthly)</i><br>
<i>- Seasonal Period: 7 for weekly, 30 for monthly patterns</i>

<br><br>

<i>Default values (1,1,1)(1,1,1,7) work well for daily revenue with weekly patterns.</i>
        """
        )
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet("QLabel { font-size: 12px; color: #888; }")

        forecast_layout.addRow("", explanation_label)

        forecast_group.setLayout(forecast_layout)

        # Application Settings Group
        app_group = QGroupBox("Application Settings")
        app_layout = QFormLayout()

        self.auto_refresh_check = QCheckBox()
        self.auto_refresh_check.setChecked(
            self.config.get("ui_settings.auto_refresh", True)
        )

        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(300, 86400)  # 5 minutes to 24 hours
        self.refresh_interval_spin.setValue(
            self.config.get("ui_settings.refresh_interval", 3600)
        )
        self.refresh_interval_spin.setSuffix(" seconds")

        # Currency Settings
        self.currency_combo = QComboBox()
        available_currencies = self.currency_formatter.get_available_currencies()
        for currency_code, currency_name in available_currencies.items():
            self.currency_combo.addItem(currency_name, currency_code)

        # Set current currency
        current_currency = self.currency_formatter.get_local_currency()
        currency_index = self.currency_combo.findData(current_currency)
        if currency_index >= 0:
            self.currency_combo.setCurrentIndex(currency_index)

        self.currency_combo.currentTextChanged.connect(self.on_currency_changed)

        # Exchange Rate Display
        self.exchange_rate_label = QLabel("Loading...")
        self.exchange_rate_label.setStyleSheet(
            "QLabel { color: #888; font-size: 12px; }"
        )
        self.update_exchange_rate_display()

        app_layout.addRow("Auto Refresh:", self.auto_refresh_check)
        app_layout.addRow("Refresh Interval:", self.refresh_interval_spin)
        app_layout.addRow("Local Currency:", self.currency_combo)
        app_layout.addRow("Current Exchange Rate:", self.exchange_rate_label)

        app_group.setLayout(app_layout)

        # Save Settings Button
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)

        # Layout
        layout.addWidget(api_group)
        layout.addWidget(forecast_group)
        layout.addWidget(app_group)
        layout.addWidget(self.save_settings_btn)
        layout.addStretch()

        self.setLayout(layout)

    def upload_credentials_file(self):
        """Upload OAuth2 credentials JSON file to app directory"""

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select OAuth2 Credentials File", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import json
                import shutil

                # Validate JSON file first
                with open(file_path, "r") as f:
                    credentials = json.load(f)

                # Check if it's a valid OAuth2 credentials file
                valid_formats = ["installed", "web"]
                is_valid = False

                for format_type in valid_formats:
                    if format_type in credentials:
                        client_info = credentials[format_type]
                        if (
                            "client_id" in client_info
                            and "client_secret" in client_info
                        ):
                            is_valid = True
                            break

                # Also check direct format
                if (
                    not is_valid
                    and "client_id" in credentials
                    and "client_secret" in credentials
                ):
                    is_valid = True

                if not is_valid:
                    QMessageBox.warning(
                        self,
                        "Invalid Credentials File",
                        "The selected file does not appear to be a valid OAuth2 credentials JSON file.\n\n"
                        "Please ensure you downloaded the correct file from Google Cloud Console.",
                    )
                    return

                # Copy file to app directory
                target_path = self.config.credentials_file
                shutil.copy2(file_path, target_path)

                # Update status
                self.update_credentials_status()

                # Update tab states in main window
                self.update_main_window_tabs()

                # Show success message
                QMessageBox.information(
                    self,
                    "Credentials Uploaded",
                    "OAuth2 credentials uploaded successfully!\n\n"
                    "Please enter your Customer ID and test the connection.",
                )

            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Upload Error",
                    f"Could not upload credentials file: {str(e)}\n\n"
                    "Please ensure the file is a valid OAuth2 credentials JSON file.",
                )

    def update_credentials_status(self):
        """Update the credentials status label"""

        if self.config.credentials_file.exists():
            self.credentials_status_label.setText("✓ Credentials file uploaded")
            self.credentials_status_label.setStyleSheet(
                "QLabel { color: #28a745; font-weight: bold; }"
            )
        else:
            self.credentials_status_label.setText("No credentials file uploaded")
            self.credentials_status_label.setStyleSheet(
                "QLabel { color: #888; font-style: italic; }"
            )

    def update_main_window_tabs(self):
        """Update tab states in main window"""

        # Find the main window and update tab states
        main_window = self.window()
        if hasattr(main_window, "update_tab_states"):
            main_window.update_tab_states()

    def on_currency_changed(self):
        """Handle currency change"""

        try:
            selected_currency = self.currency_combo.currentData()
            if selected_currency:
                self.currency_formatter.set_local_currency(selected_currency)
                # Update exchange rate display
                self.update_exchange_rate_display()
                # Auto-save settings when currency changes
                self.save_settings()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to change currency: {str(e)}")

    def update_exchange_rate_display(self):
        """Update the exchange rate display for the selected currency"""

        try:
            current_currency = self.currency_formatter.get_local_currency()

            if current_currency == "USD":
                self.exchange_rate_label.setText("1.00 USD = 1.00 USD (Base currency)")
                self.exchange_rate_label.setStyleSheet(
                    "QLabel { color: #888; font-size: 12px; }"
                )
            else:
                exchange_rate = self.currency_formatter.get_current_exchange_rate(
                    current_currency
                )

                if exchange_rate is not None:
                    self.exchange_rate_label.setText(
                        f"1.00 USD = {exchange_rate:,.2f} {current_currency}"
                    )
                    self.exchange_rate_label.setStyleSheet(
                        "QLabel { color: #28a745; font-size: 12px; font-weight: bold; }"
                    )
                else:
                    self.exchange_rate_label.setText("Exchange rate not available")
                    self.exchange_rate_label.setStyleSheet(
                        "QLabel { color: #dc3545; font-size: 12px; font-style: italic; }"
                    )

        except Exception as e:
            self.exchange_rate_label.setText("Error loading exchange rate")
            self.exchange_rate_label.setStyleSheet(
                "QLabel { color: #dc3545; font-size: 12px; font-style: italic; }"
            )
            logging.error(f"Error updating exchange rate display: {str(e)}")

    def test_connection(self):
        """Test AdMob API connection"""

        try:
            # Save current settings temporarily
            self.save_settings()

            # Create API client and test connection
            api_client = AdMobAPIClient(self.config)

            if api_client.authenticate():
                QMessageBox.information(
                    self, "Success", "Successfully connected to AdMob API!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to connect to AdMob API. Please check your credentials.",
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Connection test failed: {str(e)}")

    def save_settings(self):
        """Save all settings to configuration"""

        try:
            # API Settings
            self.config.set("api_settings.customer_id", self.customer_id_edit.text())
            self.config.set(
                "api_settings.report_type", self.report_type_combo.currentData()
            )

            # Forecast Settings
            self.config.set(
                "forecast_settings.default_forecast_days",
                self.forecast_days_spin.value(),
            )
            self.config.set(
                "forecast_settings.default_backtest_days",
                self.backtest_days_spin.value(),
            )
            self.config.set(
                "forecast_settings.sarima_order",
                [self.p_spin.value(), self.d_spin.value(), self.q_spin.value()],
            )
            self.config.set(
                "forecast_settings.seasonal_order",
                [
                    self.sp_spin.value(),
                    self.sd_spin.value(),
                    self.sq_spin.value(),
                    self.seasonal_period_spin.value(),
                ],
            )

            # Application Settings
            self.config.set(
                "ui_settings.auto_refresh", self.auto_refresh_check.isChecked()
            )
            self.config.set(
                "ui_settings.refresh_interval", self.refresh_interval_spin.value()
            )

            # Update tab states in main window
            self.update_main_window_tabs()

            QMessageBox.information(self, "Success", "Settings saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def on_report_type_changed(self):
        """Handle report type change"""

        current_report_type = self.report_type_combo.currentData()
        saved_report_type = self.config.get("api_settings.report_type", "mediation")

        # Only update if actually changing from saved value
        if current_report_type != saved_report_type:
            # Save the new report type
            self.config.set("api_settings.report_type", current_report_type)
            self.update_main_window_tabs()

            # Update status in main window if it exists
            main_window = self.window()
            if hasattr(main_window, "data_tab") and hasattr(
                main_window.data_tab, "status_label"
            ):
                main_window.data_tab.status_label.setText(
                    f"Changed to {current_report_type.title()} Report - fetch new data"
                )
                main_window.data_tab.status_label.setStyleSheet(
                    "QLabel { color: #ff8c00; font-size: 12px; }"
                )


class DataTab(QWidget):
    """Data tab for fetching and viewing revenue data"""

    def __init__(self, config, currency_formatter):
        super().__init__()
        self.config = config
        self.currency_formatter = currency_formatter
        self.data_processor = DataProcessor(config)
        self.current_data = pd.DataFrame()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Data Fetching Controls
        fetch_group = QGroupBox("Data Fetching")
        fetch_layout = QFormLayout()

        # Date Range
        min_date, max_date = self.config.get_date_range()

        self.start_date_edit = QDateEdit(
            QDate.fromString(min_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
        )
        self.start_date_edit.setCalendarPopup(True)
        # Remove minimum date restriction - allow any date
        self.start_date_edit.setMaximumDate(
            QDate.fromString(max_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
        )

        self.end_date_edit = QDateEdit(
            QDate.fromString(max_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
        )
        self.end_date_edit.setCalendarPopup(True)
        # Remove minimum date restriction - allow any date
        self.end_date_edit.setMaximumDate(
            QDate.fromString(max_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
        )

        fetch_layout.addRow("Start Date:", self.start_date_edit)
        fetch_layout.addRow("End Date:", self.end_date_edit)

        # Add reset button
        self.reset_dates_btn = QPushButton("Reset to Default")
        self.reset_dates_btn.clicked.connect(self.reset_dates)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Fetch Button
        self.fetch_btn = QPushButton("Fetch Data")
        self.fetch_btn.clicked.connect(self.fetch_data)
        buttons_layout.addWidget(self.fetch_btn)

        buttons_layout.addWidget(self.reset_dates_btn)

        # Progress Bar
        self.progress_bar = QProgressBar()

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #888; font-size: 12px; }")

        fetch_layout.addRow("", buttons_layout)
        fetch_layout.addRow("Progress:", self.progress_bar)
        fetch_layout.addRow("Status:", self.status_label)

        fetch_group.setLayout(fetch_layout)

        # Export/Import
        io_group = QGroupBox("Data Import/Export")
        io_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        self.export_btn.setEnabled(False)

        self.import_btn = QPushButton("Import Data")
        self.import_btn.clicked.connect(self.import_data)

        io_layout.addWidget(self.export_btn)
        io_layout.addWidget(self.import_btn)
        io_group.setLayout(io_layout)

        # Data Summary
        summary_group = QGroupBox("Data Summary")
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)

        summary_layout = QVBoxLayout()
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)

        # Layout
        layout.addWidget(fetch_group)
        layout.addWidget(io_group)
        layout.addWidget(summary_group)

        self.setLayout(layout)

    def fetch_data(self):
        """Fetch data from AdMob API"""

        try:
            if not self.config.is_api_configured():
                QMessageBox.warning(
                    self,
                    "Configuration Error",
                    "Please configure AdMob API settings first.",
                )
                return

            start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
            end_date = self.end_date_edit.date().toString("yyyy-MM-dd")

            # Show current report type in status
            report_type = self.config.get("api_settings.report_type", "mediation")
            self.status_label.setText(f"Fetching {report_type} report data...")

            # Create API client
            api_client = AdMobAPIClient(self.config)

            # Start fetching thread
            self.fetch_thread = DataFetchingThread(api_client, start_date, end_date)
            self.fetch_thread.progress_update.connect(self.update_progress)
            self.fetch_thread.status_update.connect(self.update_status)
            self.fetch_thread.data_ready.connect(self.on_data_ready)
            self.fetch_thread.error_occurred.connect(self.on_fetch_error)

            self.progress_bar.setVisible(True)
            self.fetch_btn.setEnabled(False)
            self.fetch_thread.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to start data fetching: {str(e)}"
            )
            self.status_label.setText("Error starting data fetch")

    def reset_dates(self):
        """Reset date range to default values"""

        try:
            min_date, max_date = self.config.get_date_range()

            self.start_date_edit.setDate(
                QDate.fromString(min_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
            )
            self.end_date_edit.setDate(
                QDate.fromString(max_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
            )

            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("QLabel { color: #888; font-size: 12px; }")

            QMessageBox.information(
                self, "Success", "Date range reset to default values!"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reset dates: {str(e)}")
            self.status_label.setText("Error resetting dates")
            self.status_label.setStyleSheet(
                "QLabel { color: #dc3545; font-size: 12px; }"
            )

    def update_progress(self, value):
        """Update progress bar"""

        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status label with current message"""

        self.status_label.setText(message)

    def on_data_ready(self, data):
        """Handle data ready signal"""

        self.current_data = data
        self.progress_bar.setValue(0)
        self.fetch_btn.setEnabled(True)

        # Update status with report type and data info
        report_type = self.config.get("api_settings.report_type", "mediation")
        self.status_label.setText(
            f"✓ {report_type.title()} report data loaded ({len(data)} days)"
        )
        self.status_label.setStyleSheet("QLabel { color: #28a745; font-size: 12px; }")

        # Update data display
        self.update_data_display()

    def update_data_display(self):
        """Update data display after successful fetch"""

        try:
            # Clean and validate data
            self.current_data = self.data_processor.clean_data(self.current_data)

            # Update summary
            self.update_data_summary()

            # Enable export
            self.export_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process data: {str(e)}")
            self.status_label.setText("Error processing data")
            self.status_label.setStyleSheet(
                "QLabel { color: #dc3545; font-size: 12px; }"
            )

    def on_fetch_error(self, error_message):
        """Handle fetch error signal"""

        self.progress_bar.setValue(0)
        self.fetch_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("QLabel { color: #dc3545; font-size: 12px; }")
        QMessageBox.critical(self, "Data Fetch Error", error_message)

    def update_data_summary(self):
        """Update data summary display"""

        try:
            if self.current_data.empty:
                self.summary_text.setText("No data available")
                return

            summary = self.data_processor.get_data_summary(self.current_data)

            # Format currency values
            revenue_stats = summary.get("revenue_stats", {})
            total_revenue = self.currency_formatter.format_currency(
                revenue_stats.get("total_revenue", 0)
            )
            avg_daily_revenue = self.currency_formatter.format_currency(
                revenue_stats.get("average_daily_revenue", 0)
            )
            median_daily_revenue = self.currency_formatter.format_currency(
                revenue_stats.get("median_daily_revenue", 0)
            )
            max_daily_revenue = self.currency_formatter.format_currency(
                revenue_stats.get("max_daily_revenue", 0)
            )
            min_daily_revenue = self.currency_formatter.format_currency(
                revenue_stats.get("min_daily_revenue", 0)
            )

            data_summary_text = create_table(
                "Data Summary",
                f"""
<tr>
{create_th("Date Range")}
{create_td(summary.get('date_range', {}).get('start', 'N/A') + " to " + summary.get('date_range', {}).get('end', 'N/A'))}
{create_td("Date range of the data")}
</tr>

<tr>
{create_th("Total Days")}
{create_td(str(summary.get('total_records', 0)))}
{create_td("Total number of days in the data")}
</tr>

<tr>
{create_th("Total Revenue")}
{create_td(total_revenue)}
{create_td("Total revenue from the data")}
</tr>

<tr>
{create_th("Average Daily")}
{create_td(avg_daily_revenue)}
{create_td("Average daily revenue from the data")}
</tr>

<tr>
{create_th("Highest Day")}
{create_td(max_daily_revenue)}
{create_td("Highest daily revenue from the data")}
</tr>

<tr>
{create_th("Lowest Day")}
{create_td(min_daily_revenue)}
{create_td("Lowest daily revenue from the data")}
</tr>

<tr>
{create_th("Zero Revenue Days")}
{create_td(str(summary.get('revenue_stats', {}).get('zero_revenue_days', 0)))}
{create_td("Number of days with zero revenue")}
</tr>
""",
            )

            data_quality_text = create_table(
                "Data Quality",
                f"""
<tr>
{create_th("Missing Values")}
{create_td(str(summary.get('data_quality', {}).get('missing_values', 0)))}
{create_td("Number of missing values in the data")}
</tr>

<tr>
{create_th("Duplicates")}
{create_td(str(summary.get('data_quality', {}).get('duplicate_dates', 0)))}
{create_td("Number of duplicate rows in the data")}
</tr>

<tr>
{create_th("Completeness")}
{create_td(str(round(summary.get('data_quality', {}).get('completeness_pct', 0), 1)) + "%")}
{create_td("Percentage of complete data")}
</tr>
""",
            )

            summary_text = f"""
{data_summary_text}
<br>
{data_quality_text}
"""
            self.summary_text.setText(summary_text)

        except Exception as e:
            self.summary_text.setText(f"Error updating summary: {str(e)}")
            logging.error(f"Error updating data summary: {e}")

    def export_data(self):
        """Export current data to file"""

        try:
            if self.current_data.empty:
                QMessageBox.warning(self, "No Data", "No data to export.")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Data",
                "",
                "CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx)",
            )

            if file_path:
                file_format = file_path.split(".")[-1].lower()

                if self.data_processor.export_data(
                    self.current_data, file_path, file_format
                ):
                    QMessageBox.information(
                        self, "Success", "Data exported successfully!"
                    )
                else:
                    QMessageBox.warning(self, "Error", "Failed to export data.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def import_data(self):
        """Import data from file"""

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Data",
                "",
                "CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx)",
            )

            if file_path:
                file_format = file_path.split(".")[-1].lower()

                imported_data = self.data_processor.import_data(file_path, file_format)

                if not imported_data.empty:
                    self.on_data_ready(imported_data)
                    QMessageBox.information(
                        self, "Success", "Data imported successfully!"
                    )
                else:
                    QMessageBox.warning(self, "Error", "Failed to import data.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Import failed: {str(e)}")

    def get_current_data(self):
        """Get current data for use in other tabs"""

        return self.current_data


class ForecastTab(QWidget):
    """Forecast tab for running forecasts and viewing results"""

    def __init__(self, config, data_tab, currency_formatter):
        super().__init__()
        self.config = config
        self.data_tab = data_tab
        self.currency_formatter = currency_formatter
        self.forecast_results = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Forecast Controls
        forecast_group = QGroupBox("Forecast Controls")
        forecast_layout = QFormLayout()

        self.forecast_days_spin = QSpinBox()
        self.forecast_days_spin.setRange(1, 365 * 10)  # 10 years
        self.forecast_days_spin.setValue(
            self.config.get("forecast_settings.default_forecast_days", 30)
        )
        self.forecast_days_spin.setSuffix(" days")

        self.run_backtest_check = QCheckBox()
        self.run_backtest_check.setChecked(True)

        # Backtest duration
        self.backtest_days_spin = QSpinBox()
        self.backtest_days_spin.setRange(1, 365 * 10)  # 10 years
        self.backtest_days_spin.setValue(
            self.config.get("forecast_settings.default_backtest_days", 90)
        )
        self.backtest_days_spin.setSuffix(" days")

        forecast_layout.addRow("Forecast Days:", self.forecast_days_spin)
        forecast_layout.addRow("Run Backtest:", self.run_backtest_check)
        forecast_layout.addRow("Backtest Duration:", self.backtest_days_spin)

        # Run Forecast Button
        self.run_forecast_btn = QPushButton("Run Forecast")
        self.run_forecast_btn.clicked.connect(self.run_forecast)

        # Progress Bar
        self.progress_bar = QProgressBar()

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #888; font-size: 12px; }")

        forecast_layout.addRow("", self.run_forecast_btn)
        forecast_layout.addRow("Progress:", self.progress_bar)
        forecast_layout.addRow("Status:", self.status_label)

        forecast_group.setLayout(forecast_layout)

        # Results
        results_group = QGroupBox("Forecast Results")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        results_layout = QVBoxLayout()
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)

        # Layout
        layout.addWidget(forecast_group)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def run_forecast(self):
        """Run forecasting model"""

        try:
            # Get current data
            current_data = self.data_tab.get_current_data()
            if current_data.empty:
                QMessageBox.warning(self, "No Data", "Please fetch data first.")
                return

            forecast_days = self.forecast_days_spin.value()
            run_backtest = self.run_backtest_check.isChecked()
            backtest_days = self.backtest_days_spin.value()

            # Show initial status
            self.status_label.setText("Preparing forecast...")
            self.status_label.setStyleSheet("QLabel { color: #888; font-size: 12px; }")

            # Validate backtest duration
            if run_backtest:
                data_days = len(current_data)
                min_training_days = (
                    backtest_days + 30
                )  # Need at least 30 days for training

                if data_days < min_training_days:
                    QMessageBox.warning(
                        self,
                        "Insufficient Data",
                        f"Need at least {min_training_days} days of data for backtesting with {backtest_days} days test period.\n"
                        f"Current data has {data_days} days. Please reduce backtest duration or get more data.",
                    )
                    self.status_label.setText(
                        "Error: Insufficient data for backtesting"
                    )
                    self.status_label.setStyleSheet(
                        "QLabel { color: #dc3545; font-size: 12px; }"
                    )
                    return

            # Start forecasting thread
            self.forecast_thread = ForecastingThread(
                current_data, self.config, forecast_days, run_backtest, backtest_days
            )
            self.forecast_thread.progress_update.connect(self.update_progress)
            self.forecast_thread.status_update.connect(self.update_status)
            self.forecast_thread.forecast_ready.connect(self.on_forecast_ready)
            self.forecast_thread.error_occurred.connect(self.on_forecast_error)

            self.progress_bar.setVisible(True)
            self.run_forecast_btn.setEnabled(False)
            self.forecast_thread.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to start forecasting: {str(e)}"
            )
            self.status_label.setText("Error starting forecast")
            self.status_label.setStyleSheet(
                "QLabel { color: #dc3545; font-size: 12px; }"
            )

    def update_progress(self, value):
        """Update progress bar"""

        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status label with current message"""

        self.status_label.setText(message)

    def on_forecast_ready(self, forecast_data, results):
        """Handle successful forecasting"""

        try:
            self.forecast_results = results

            # Update results display
            self.update_results_display()

            # Update status with success
            forecast_days = len(forecast_data)
            self.status_label.setText(f"✓ Forecast completed ({forecast_days} days)")
            self.status_label.setStyleSheet(
                "QLabel { color: #28a745; font-size: 12px; }"
            )

            # Reset progress bar
            self.progress_bar.setValue(0)
            self.run_forecast_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to process forecast results: {str(e)}"
            )
            self.status_label.setText("Error processing forecast results")
            self.status_label.setStyleSheet(
                "QLabel { color: #dc3545; font-size: 12px; }"
            )

    def on_forecast_error(self, error_message):
        """Handle forecasting errors"""

        QMessageBox.critical(self, "Forecast Error", error_message)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("QLabel { color: #dc3545; font-size: 12px; }")
        self.progress_bar.setValue(0)
        self.run_forecast_btn.setEnabled(True)

    def update_results_display(self):
        """Update forecast results display"""

        try:
            if not self.forecast_results:
                self.results_text.setText("No forecast results available.")
                return

            diagnostics = self.forecast_results.get("diagnostics", {})
            forecast_data = self.forecast_results.get("forecast_data", pd.DataFrame())

            def format_numeric(value, default="N/A"):
                """Safely format numeric values"""

                if value is None or value == "N/A":
                    return default
                try:
                    return f"{float(value):.2f}"
                except (ValueError, TypeError):
                    return default

            forecast_results = create_table(
                "Forecast Results",
                f"""
<tr>
{create_th("Forecast Period")}
{create_td(str(len(forecast_data)) + " days")}
{create_td("Days to forecast")}
</tr>

<tr>
{create_th("Model AIC")}
{create_td(format_numeric(diagnostics.get('aic')))}
{create_td("Akaike Information Criterion: Lower values indicate better model fit")}
</tr>

<tr>
{create_th("Model BIC")}
{create_td(format_numeric(diagnostics.get('bic')))}
{create_td("Bayesian Information Criterion: Lower values indicate better model fit")}
</tr>

<tr>
{create_th("Log Likelihood")}
{create_td(format_numeric(diagnostics.get('log_likelihood')))}
{create_td("Higher values indicate better model fit")}
</tr>

<tr>
{create_th("Model Order")}
{create_td(diagnostics.get('model_order', 'N/A'))}
{create_td("p, d, q: AutoRegressive, Differencing, Moving Average terms")}
</tr>

<tr>
{create_th("Seasonal Order")}
{create_td(diagnostics.get('seasonal_order', 'N/A'))}
{create_td("P, D, Q, s: Seasonal AR, Differencing, MA terms, seasonal period")}
</tr>
""",
            )

            # Format the results similar to data summary
            results_text = f"""
{forecast_results}
"""

            if "backtest" in self.forecast_results:
                backtest = self.forecast_results["backtest"]
                metrics = backtest.get("metrics", {})

                backtest_results = create_table(
                    "Backtest Results",
                    f"""
<tr>
{create_th("Test Period")}
{create_td(backtest.get('test_period', 'N/A'))}
{create_td("Days used for backtesting")}
</tr>

<tr>
{create_th("RMSE")}
{create_td(format_numeric(metrics.get('rmse')))}
{create_td("Root Mean Square Error: Lower values indicate better accuracy")}
</tr>

<tr>
{create_th("MAE")}
{create_td(format_numeric(metrics.get('mae')))}
{create_td("Mean Absolute Error: Lower values indicate better accuracy")}
</tr>

<tr>
{create_th("MAPE")}
{create_td(format_numeric(metrics.get('mape')) + "%")}
{create_td("Mean Absolute Percentage Error: Lower percentages indicate better accuracy")}
</tr>
""",
                )

                results_text += f"""
<br>
{backtest_results}
"""

            self.results_text.setText(results_text)

        except Exception as e:
            self.results_text.setText(f"Error displaying results: {str(e)}")
            logging.error(f"Error updating forecast results display: {e}")

    def get_forecast_results(self):
        """Get forecast results for use in visualization"""

        return self.forecast_results


class VisualizationTab(QWidget):
    """Visualization tab for interactive charts"""

    def __init__(self, config, data_tab, forecast_tab, currency_formatter):
        super().__init__()
        self.config = config
        self.data_tab = data_tab
        self.forecast_tab = forecast_tab
        self.currency_formatter = currency_formatter
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(0)  # Remove spacing

        # Web view for Plotly chart (or fallback to text display)
        if WEB_ENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            self.web_view.setContentsMargins(0, 0, 0, 0)
        else:
            self.web_view = QTextEdit()
            self.web_view.setReadOnly(True)
            self.web_view.setContentsMargins(0, 0, 0, 0)
            self.web_view.setHtml(
                """
                <h2>Web Engine Not Available</h2>
                <p>PyQt6-WebEngine is not installed. Interactive charts are not available.</p>
                <p>To enable interactive charts, install PyQt6-WebEngine:</p>
                <pre>pip install PyQt6-WebEngine</pre>
                <p>Chart data will be displayed as text when available.</p>
            """
            )

        # Layout
        layout.addWidget(self.web_view)

        self.setLayout(layout)

        # Chart options (internal settings, no UI controls)
        self.show_confidence_intervals = True
        self.show_backtest = True

        # Initial empty chart
        self.show_empty_chart()

        # Auto-update chart when forecast data is available
        self.auto_update_timer = QTimer()
        self.auto_update_timer.timeout.connect(self.check_and_update_chart)
        self.auto_update_timer.start(1000)  # Check every second

        # Track last update state
        self.last_data_hash = None
        self.last_forecast_hash = None

    def check_and_update_chart(self):
        """Check if data/forecast has changed and update chart if needed"""

        try:
            current_data = self.data_tab.get_current_data()
            forecast_results = self.forecast_tab.get_forecast_results()

            # Create hashes to detect changes
            data_hash = hash(
                str(current_data.values.tobytes() if not current_data.empty else "")
            )
            forecast_hash = hash(str(forecast_results))

            # Check if data or forecast has changed
            if (
                data_hash != self.last_data_hash
                or forecast_hash != self.last_forecast_hash
            ):

                self.last_data_hash = data_hash
                self.last_forecast_hash = forecast_hash

                # Update chart
                self.update_chart()

        except Exception as e:
            # Silently ignore errors in auto-update
            pass

    def show_empty_chart(self):
        """Show empty chart when no data is available"""

        if WEB_ENGINE_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available. Please fetch data and run forecast.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(
                title="AdMob Revenue Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue (USD)",
                template="plotly_dark",
                paper_bgcolor="black",  # Black background
                plot_bgcolor="black",  # Black plot area
            )

            html = fig.to_html(include_plotlyjs=True)

            # Add custom CSS to remove all margins and padding
            css = """
            <style>
            html, body {
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                height: 100% !important;
                background: transparent !important;
            }
            </style>
            """

            # Insert CSS into the HTML head
            html = html.replace("<head>", f"<head>{css}")

            self.web_view.setHtml(html)
        else:
            self.web_view.setHtml(
                """
                <h2>AdMob Revenue Forecast</h2>
                <p><strong>Status:</strong> No data available</p>
                <p>Please fetch data and run forecast to see results.</p>
                <p><em>Note: Interactive charts require PyQt6-WebEngine installation.</em></p>
            """
            )

    def update_chart(self):
        """Update the chart with current data and forecasts"""

        try:
            # Get current data
            current_data = self.data_tab.get_current_data()

            if current_data.empty:
                self.show_empty_chart()
                return

            # Get forecast results
            forecast_results = self.forecast_tab.get_forecast_results()

            # Create figure
            fig = go.Figure()

            # Add actual revenue data
            customdata = (
                current_data["revenue_cumulative"]
                if "revenue_cumulative" in current_data.columns
                else current_data["revenue"].cumsum()
            )

            # Format hover template with currency formatter
            local_currency = self.currency_formatter.get_local_currency()

            if local_currency == "USD":
                hover_template = (
                    "<b>%{x}</b><br>"
                    + "Daily Revenue: %{y:,.2f} USD<br>"
                    + "Cumulative Revenue: %{customdata:,.2f} USD<br>"
                    + "<extra></extra>"
                )

                # For USD, we can use the original data
                customdata_converted = customdata
                daily_converted = current_data["revenue"]
            else:
                exchange_rate = self.currency_formatter.get_current_exchange_rate(
                    local_currency
                )
                if exchange_rate is not None:
                    # Pre-calculate converted values
                    daily_converted = current_data["revenue"] * exchange_rate
                    customdata_converted = customdata * exchange_rate

                    hover_template = (
                        "<b>%{x}</b><br>"
                        + f"Daily Revenue: %{{y:,.2f}} USD (%{{customdata:,.2f}} {local_currency})<br>"
                        + f"Cumulative Revenue: %{{customdata2:,.2f}} USD (%{{customdata3:,.2f}} {local_currency})<br>"
                        + "<extra></extra>"
                    )
                else:
                    # If no exchange rate available, show only USD
                    hover_template = (
                        "<b>%{x}</b><br>"
                        + "Daily Revenue: %{y:,.2f} USD<br>"
                        + "Cumulative Revenue: %{customdata:,.2f} USD<br>"
                        + "<extra></extra>"
                    )

                    customdata_converted = customdata
                    daily_converted = current_data["revenue"]

            # Prepare custom data arrays for hover template
            if local_currency == "USD":
                fig.add_trace(
                    go.Scatter(
                        x=current_data.index,
                        y=current_data["revenue"],
                        mode="lines",
                        name="Actual Revenue",
                        line=dict(color="#1f77b4", width=2),
                        customdata=customdata,
                        hovertemplate=hover_template,
                    )
                )
            else:
                exchange_rate = self.currency_formatter.get_current_exchange_rate(
                    local_currency
                )
                if exchange_rate is not None:
                    # Create array of custom data: [daily_local, cumulative_usd, cumulative_local]
                    custom_data_array = list(
                        zip(daily_converted, customdata, customdata_converted)
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=current_data.index,
                            y=current_data["revenue"],
                            mode="lines",
                            name="Actual Revenue",
                            line=dict(color="#1f77b4", width=2),
                            customdata=custom_data_array,
                            hovertemplate="<b>%{x}</b><br>"
                            + f"Daily Revenue: %{{y:,.2f}} USD (%{{customdata[0]:,.2f}} {local_currency})<br>"
                            + f"Cumulative Revenue: %{{customdata[1]:,.2f}} USD (%{{customdata[2]:,.2f}} {local_currency})<br>"
                            + "<extra></extra>",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=current_data.index,
                            y=current_data["revenue"],
                            mode="lines",
                            name="Actual Revenue",
                            line=dict(color="#1f77b4", width=2),
                            customdata=customdata,
                            hovertemplate=hover_template,
                        )
                    )

            # Add forecast data if available
            if forecast_results and "forecast_data" in forecast_results:
                forecast_data = forecast_results["forecast_data"]

                if not forecast_data.empty:
                    # Calculate cumulative forecast values (continuing from actual data)
                    last_cumulative = customdata.iloc[-1] if len(customdata) > 0 else 0
                    forecast_cumulative = (
                        last_cumulative + forecast_data["forecast"].cumsum()
                    )

                    # Add forecast line
                    # Format forecast hover template with currency formatter
                    if local_currency == "USD":
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_data.index,
                                y=forecast_data["forecast"],
                                mode="lines",
                                name="Forecast",
                                line=dict(color="#ff7f0e", width=2),
                                customdata=forecast_cumulative,
                                hovertemplate="<b>%{x}</b><br>"
                                + "Forecast Revenue: %{y:,.2f} USD<br>"
                                + "Cumulative Revenue: %{customdata:,.2f} USD<br>"
                                + "<extra></extra>",
                            )
                        )
                    else:
                        exchange_rate = (
                            self.currency_formatter.get_current_exchange_rate(
                                local_currency
                            )
                        )
                        if exchange_rate is not None:
                            # Pre-calculate converted values for forecast
                            forecast_daily_converted = (
                                forecast_data["forecast"] * exchange_rate
                            )
                            forecast_cumulative_converted = (
                                forecast_cumulative * exchange_rate
                            )

                            # Create array of custom data: [daily_local, cumulative_usd, cumulative_local]
                            forecast_custom_data_array = list(
                                zip(
                                    forecast_daily_converted,
                                    forecast_cumulative,
                                    forecast_cumulative_converted,
                                )
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_data.index,
                                    y=forecast_data["forecast"],
                                    mode="lines",
                                    name="Forecast",
                                    line=dict(color="#ff7f0e", width=2),
                                    customdata=forecast_custom_data_array,
                                    hovertemplate="<b>%{x}</b><br>"
                                    + f"Forecast Revenue: %{{y:,.2f}} USD (%{{customdata[0]:,.2f}} {local_currency})<br>"
                                    + f"Cumulative Revenue: %{{customdata[1]:,.2f}} USD (%{{customdata[2]:,.2f}} {local_currency})<br>"
                                    + "<extra></extra>",
                                )
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_data.index,
                                    y=forecast_data["forecast"],
                                    mode="lines",
                                    name="Forecast",
                                    line=dict(color="#ff7f0e", width=2),
                                    customdata=forecast_cumulative,
                                    hovertemplate="<b>%{x}</b><br>"
                                    + "Forecast Revenue: %{y:,.2f} USD<br>"
                                    + "Cumulative Revenue: %{customdata:,.2f} USD<br>"
                                    + "<extra></extra>",
                                )
                            )

                    # Add confidence intervals if enabled
                    if (
                        self.show_confidence_intervals
                        and "upper_ci" in forecast_data.columns
                        and "lower_ci" in forecast_data.columns
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_data.index,
                                y=forecast_data["upper_ci"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=forecast_data.index,
                                y=forecast_data["lower_ci"],
                                mode="lines",
                                fill="tonexty",
                                fillcolor="rgba(255, 127, 14, 0.2)",
                                line=dict(width=0),
                                name="Confidence Interval",
                                hoverinfo="skip",
                            )
                        )

            # Add backtest data if available and enabled
            if (
                self.show_backtest
                and forecast_results
                and "backtest" in forecast_results
            ):
                backtest = forecast_results["backtest"]
                if "forecast_data" in backtest:
                    backtest_forecast = backtest["forecast_data"]

                    if not backtest_forecast.empty:
                        # Calculate cumulative backtest forecast values
                        # Find the cumulative value at the start of backtest period
                        backtest_start_date = backtest_forecast.index[0]

                        # Get cumulative revenue up to backtest start
                        backtest_start_cumulative = 0
                        if backtest_start_date in customdata.index:
                            backtest_start_cumulative = (
                                customdata.loc[:backtest_start_date].iloc[-2]
                                if len(customdata.loc[:backtest_start_date]) > 1
                                else 0
                            )
                        elif len(customdata) > 0:
                            # If backtest date is not in actual data, use the last cumulative value
                            actual_end_date = customdata.index[-1]
                            if backtest_start_date > actual_end_date:
                                backtest_start_cumulative = customdata.iloc[-1]
                            else:
                                # Find the cumulative value at the closest date before backtest start
                                backtest_start_cumulative = (
                                    customdata[
                                        customdata.index <= backtest_start_date
                                    ].iloc[-1]
                                    if len(
                                        customdata[
                                            customdata.index <= backtest_start_date
                                        ]
                                    )
                                    > 0
                                    else 0
                                )

                        backtest_cumulative = (
                            backtest_start_cumulative
                            + backtest_forecast["forecast"].cumsum()
                        )

                        # Format backtest hover template with currency formatter
                        if local_currency == "USD":
                            fig.add_trace(
                                go.Scatter(
                                    x=backtest_forecast.index,
                                    y=backtest_forecast["forecast"],
                                    mode="lines",
                                    name="Backtest Forecast",
                                    line=dict(color="#2ca02c", width=2, dash="dot"),
                                    customdata=backtest_cumulative,
                                    hovertemplate="<b>%{x}</b><br>"
                                    + "Backtest Forecast: %{y:,.2f} USD<br>"
                                    + "Cumulative Revenue: %{customdata:,.2f} USD<br>"
                                    + "<extra></extra>",
                                )
                            )
                        else:
                            exchange_rate = (
                                self.currency_formatter.get_current_exchange_rate(
                                    local_currency
                                )
                            )
                            if exchange_rate is not None:
                                # Pre-calculate converted values for backtest
                                backtest_daily_converted = (
                                    backtest_forecast["forecast"] * exchange_rate
                                )
                                backtest_cumulative_converted = (
                                    backtest_cumulative * exchange_rate
                                )

                                # Create array of custom data: [daily_local, cumulative_usd, cumulative_local]
                                backtest_custom_data_array = list(
                                    zip(
                                        backtest_daily_converted,
                                        backtest_cumulative,
                                        backtest_cumulative_converted,
                                    )
                                )

                                fig.add_trace(
                                    go.Scatter(
                                        x=backtest_forecast.index,
                                        y=backtest_forecast["forecast"],
                                        mode="lines",
                                        name="Backtest Forecast",
                                        line=dict(color="#2ca02c", width=2, dash="dot"),
                                        customdata=backtest_custom_data_array,
                                        hovertemplate="<b>%{x}</b><br>"
                                        + f"Backtest Forecast: %{{y:,.2f}} USD (%{{customdata[0]:,.2f}} {local_currency})<br>"
                                        + f"Cumulative Revenue: %{{customdata[1]:,.2f}} USD (%{{customdata[2]:,.2f}} {local_currency})<br>"
                                        + "<extra></extra>",
                                    )
                                )
                            else:
                                fig.add_trace(
                                    go.Scatter(
                                        x=backtest_forecast.index,
                                        y=backtest_forecast["forecast"],
                                        mode="lines",
                                        name="Backtest Forecast",
                                        line=dict(color="#2ca02c", width=2, dash="dot"),
                                        customdata=backtest_cumulative,
                                        hovertemplate="<b>%{x}</b><br>"
                                        + "Backtest Forecast: %{y:,.2f} USD<br>"
                                        + "Cumulative Revenue: %{customdata:,.2f} USD<br>"
                                        + "<extra></extra>",
                                    )
                                )

            # Update layout
            fig.update_layout(
                title="AdMob Revenue Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue (USD)",
                template="plotly_dark",
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                paper_bgcolor="black",  # Black background
                plot_bgcolor="black",  # Black plot area
            )

            # Update y-axis to start from 0
            fig.update_yaxes(rangemode="tozero")

            # Show chart
            if WEB_ENGINE_AVAILABLE:
                try:
                    # Generate HTML with config to avoid potential issues
                    config = {
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["toImage", "downloadPlot"],
                    }

                    html = fig.to_html(
                        include_plotlyjs=True, config=config, div_id="plotly-chart"
                    )

                    # Add custom CSS to remove all margins and padding
                    css = """
                    <style>
                    html, body {
                        margin: 0 !important;
                        padding: 0 !important;
                        width: 100% !important;
                        height: 100% !important;
                        background: transparent !important;
                    }
                    #plotly-chart {
                        margin: 0 !important;
                        padding: 0 !important;
                        width: 100% !important;
                        height: 100% !important;
                    }
                    </style>
                    """

                    # Insert CSS into the HTML head
                    html = html.replace("<head>", f"<head>{css}")

                    # Force webview to load the HTML
                    self.web_view.setHtml(html)

                    # Alternative: try loading from a temporary file if setHtml fails
                    import tempfile
                    import os

                    # Create temporary HTML file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".html", delete=False
                    ) as f:
                        f.write(html)
                        temp_path = f.name

                    # Load from file URL
                    from PyQt6.QtCore import QUrl

                    file_url = QUrl.fromLocalFile(os.path.abspath(temp_path))
                    self.web_view.load(file_url)

                except Exception as e:
                    # Fallback to simple HTML
                    self.web_view.setHtml(
                        f"""
                        <html>
                        <head><title>Chart Loading Error</title></head>
                        <body>
                            <h2>Chart Generation Successful</h2>
                            <p>Chart data processed successfully, but display failed.</p>
                            <p>Error: {str(e)}</p>
                            <p>Try refreshing the chart or check browser console.</p>
                        </body>
                        </html>
                    """
                    )
            else:
                # Create text summary for non-web engine display
                total_revenue = self.currency_formatter.format_currency(
                    current_data["revenue"].sum()
                )
                avg_daily_revenue = self.currency_formatter.format_currency(
                    current_data["revenue"].mean()
                )
                max_daily_revenue = self.currency_formatter.format_currency(
                    current_data["revenue"].max()
                )

                summary = f"""
                <h2>AdMob Revenue Forecast</h2>
                <p><strong>Data Points:</strong> {len(current_data)}</p>
                <p><strong>Date Range:</strong> {current_data.index.min().strftime('%Y-%m-%d')} to {current_data.index.max().strftime('%Y-%m-%d')}</p>
                <p><strong>Total Revenue:</strong> {total_revenue}</p>
                <p><strong>Average Daily Revenue:</strong> {avg_daily_revenue}</p>
                <p><strong>Max Daily Revenue:</strong> {max_daily_revenue}</p>
                """

                if forecast_results and "forecast_data" in forecast_results:
                    forecast_data = forecast_results["forecast_data"]
                    if not forecast_data.empty:
                        predicted_avg_daily = self.currency_formatter.format_currency(
                            forecast_data["forecast"].mean()
                        )
                        predicted_total = self.currency_formatter.format_currency(
                            forecast_data["forecast"].sum()
                        )

                        summary += f"""
                        <h3>Forecast Summary</h3>
                        <p><strong>Forecast Period:</strong> {len(forecast_data)} days</p>
                        <p><strong>Predicted Average Daily Revenue:</strong> {predicted_avg_daily}</p>
                        <p><strong>Predicted Total Revenue:</strong> {predicted_total}</p>
                        """

                        if "backtest" in forecast_results:
                            backtest = forecast_results["backtest"]
                            metrics = backtest.get("metrics", {})
                            summary += f"""
                            <h3>Backtest Results</h3>
                            <p><strong>RMSE:</strong> {metrics.get('rmse', 0):.2f}</p>
                            <p><strong>MAE:</strong> {metrics.get('mae', 0):.2f}</p>
                            <p><strong>MAPE:</strong> {metrics.get('mape', 0):.2f}%</p>
                            """

                summary += """
                <p><em>Note: For interactive charts, install PyQt6-WebEngine: pip install PyQt6-WebEngine</em></p>
                """

                self.web_view.setHtml(summary)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update chart: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, config, currency_formatter):
        super().__init__()
        self.config = config
        self.currency_formatter = currency_formatter
        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        self.setWindowTitle("AdMob Revenue Forecaster")
        self.setGeometry(
            100,
            100,
            self.config.get("ui_settings.window_width", 1400),
            self.config.get("ui_settings.window_height", 900),
        )

        # Create menu bar
        self.create_menu_bar()

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create tabs
        self.data_tab = DataTab(self.config, self.currency_formatter)
        self.forecast_tab = ForecastTab(
            self.config, self.data_tab, self.currency_formatter
        )
        self.visualization_tab = VisualizationTab(
            self.config, self.data_tab, self.forecast_tab, self.currency_formatter
        )
        self.settings_tab = SettingsTab(self.config, self.currency_formatter)

        # Wrap tabs in scroll areas for smaller screens
        data_scroll = QScrollArea()
        data_scroll.setWidgetResizable(True)
        data_scroll.setWidget(self.data_tab)

        forecast_scroll = QScrollArea()
        forecast_scroll.setWidgetResizable(True)
        forecast_scroll.setWidget(self.forecast_tab)

        visualization_scroll = QScrollArea()
        visualization_scroll.setWidgetResizable(True)
        visualization_scroll.setWidget(self.visualization_tab)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self.settings_tab)

        # Add tabs
        self.tab_widget.addTab(data_scroll, "Data")
        self.tab_widget.addTab(forecast_scroll, "Forecast")
        self.tab_widget.addTab(visualization_scroll, "Visualization")
        self.tab_widget.addTab(settings_scroll, "Settings")

        # Store tab indices for enabling/disabling
        self.data_tab_index = 0
        self.forecast_tab_index = 1
        self.visualization_tab_index = 2
        self.settings_tab_index = 3

        # Initially disable tabs that require data
        self.update_tab_states()

        # Connect to data and forecast events
        self.connect_tab_events()

        layout.addWidget(self.tab_widget)

    def update_tab_states(self):
        """Update tab enabled/disabled state based on data availability"""

        # Check if API is configured
        is_configured = self.config.is_api_configured()

        # Check if data is available
        has_data = not self.data_tab.get_current_data().empty

        # Check if forecast is available
        has_forecast = bool(self.forecast_tab.get_forecast_results())

        # Enable/disable tabs
        self.tab_widget.setTabEnabled(self.data_tab_index, is_configured)
        self.tab_widget.setTabEnabled(self.forecast_tab_index, has_data)
        self.tab_widget.setTabEnabled(
            self.visualization_tab_index, has_data and has_forecast
        )

        # Update tab text to indicate status
        if is_configured:
            self.tab_widget.setTabText(self.data_tab_index, "Data")
        else:
            self.tab_widget.setTabText(self.data_tab_index, "Data (No API Config)")

        if has_data:
            self.tab_widget.setTabText(self.forecast_tab_index, "Forecast")
        else:
            self.tab_widget.setTabText(self.forecast_tab_index, "Forecast (No Data)")

        if has_data and has_forecast:
            self.tab_widget.setTabText(self.visualization_tab_index, "Visualization")
        else:
            self.tab_widget.setTabText(
                self.visualization_tab_index, "Visualization (No Forecast)"
            )

    def connect_tab_events(self):
        """Connect to tab events to update tab states"""

        # Store original methods
        original_data_ready = self.data_tab.on_data_ready
        original_forecast_ready = self.forecast_tab.on_forecast_ready

        # Create wrapper methods
        def data_ready_wrapper(data):
            result = original_data_ready(data)
            self.update_tab_states()
            # Auto-trigger forecasting
            self.forecast_tab.run_forecast()
            return result

        def forecast_ready_wrapper(forecast_data, results):
            result = original_forecast_ready(forecast_data, results)
            self.update_tab_states()
            # Auto-update visualization
            self.visualization_tab.update_chart()
            return result

        # Replace methods
        self.data_tab.on_data_ready = data_ready_wrapper
        self.forecast_tab.on_forecast_ready = forecast_ready_wrapper

        # Also update tab states when settings tab updates credentials
        self.settings_tab.update_main_window_tabs = self.update_tab_states

    def create_menu_bar(self):
        """Create application menu bar"""

        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Export action
        export_action = QAction("Export Data", self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)

        # Import action
        import_action = QAction("Import Data", self)
        import_action.triggered.connect(self.import_data)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Refresh action
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_data)
        view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_timers(self):
        """Setup auto-refresh timer"""

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)

        if self.config.get("ui_settings.auto_refresh", True):
            interval = (
                self.config.get("ui_settings.refresh_interval", 3600) * 1000
            )  # Convert to milliseconds
            self.refresh_timer.start(interval)

    def export_data(self):
        """Export data from data tab"""

        self.data_tab.export_data()

    def import_data(self):
        """Import data to data tab"""

        self.data_tab.import_data()

    def refresh_data(self):
        """Refresh data if auto-refresh is enabled"""

        if self.config.get("ui_settings.auto_refresh", True):
            self.data_tab.fetch_data()

    def show_about(self):
        """Show about dialog"""

        QMessageBox.about(
            self,
            "About AdMob Revenue Forecaster",
            "AdMob Revenue Forecaster\n\n"
            "A desktop application for forecasting AdMob revenue using SARIMA models.",
        )

    def closeEvent(self, event):
        """Handle application close event"""

        # Save current window size
        self.config.set("ui_settings.window_width", self.width())
        self.config.set("ui_settings.window_height", self.height())

        # Stop timers
        if hasattr(self, "refresh_timer"):
            self.refresh_timer.stop()

        event.accept()
