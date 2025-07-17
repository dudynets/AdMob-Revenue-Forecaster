#!/usr/bin/env python3
"""
AdMob Revenue Forecaster
Main entry point for the desktop application
"""

import sys
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import warnings

warnings.filterwarnings("ignore")

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from ui import MainWindow
    from config import AppConfig
    from currency_formatter import CurrencyFormatter
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main application entry point"""

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("AdMob Revenue Forecaster")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Oleksandr Dudynets")

    # Set application style
    app.setStyle("Fusion")

    try:
        # Initialize configuration
        config = AppConfig()

        # Initialize currency formatter
        currency_formatter = CurrencyFormatter(config)
        currency_formatter.initialize()

        # Create and show main window
        window = MainWindow(config, currency_formatter)
        window.show()

        # Run application
        sys.exit(app.exec())

    except Exception as e:
        QMessageBox.critical(
            None, "Application Error", f"Failed to start application: {str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
