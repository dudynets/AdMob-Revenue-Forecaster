"""
AdMob API integration module
Handles authentication and data fetching from Google AdMob API
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class AdMobAPIClient:
    """Client for interacting with Google AdMob API"""

    SCOPES = ["https://www.googleapis.com/auth/admob.readonly"]
    API_SERVICE_NAME = "admob"
    API_VERSION = "v1"

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.service = None
        self.credentials = None

    def authenticate(self) -> bool:
        """Authenticate with Google AdMob API using OAuth2"""

        try:
            creds = None
            token_file = self.config.get_token_path()

            # Load existing token if available
            if Path(token_file).exists():
                creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)

            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        self.logger.error(f"Error refreshing credentials: {e}")
                        creds = None

                if not creds:
                    credentials_file = self.config.get_credentials_path()
                    if not Path(credentials_file).exists():
                        self.logger.error(
                            "OAuth2 credentials file not found. Please add your Google OAuth2 credentials."
                        )
                        return False

                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_file, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save the credentials for the next run
                with open(token_file, "w") as token:
                    token.write(creds.to_json())

            self.credentials = creds
            self.service = build(
                self.API_SERVICE_NAME, self.API_VERSION, credentials=creds
            )

            # Test the connection
            return self.test_connection()

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test the API connection by making a simple request"""

        try:
            # Get publisher account info
            request = self.service.accounts().list()
            response = request.execute()

            if "account" in response and len(response["account"]) > 0:
                self.logger.info("Successfully connected to AdMob API")
                return True
            else:
                self.logger.error("No AdMob accounts found")
                return False

        except HttpError as e:
            self.logger.error(f"API connection test failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection test: {e}")
            return False

    def get_publisher_account(self) -> Optional[str]:
        """Get the publisher account ID"""

        try:
            request = self.service.accounts().list()
            response = request.execute()

            if "account" in response and len(response["account"]) > 0:
                # Return the first account ID
                account_info = response["account"][0]
                return account_info.get("publisherId", account_info.get("name", ""))
            return None

        except Exception as e:
            self.logger.error(f"Error getting publisher account: {e}")
            return None

    def get_apps(self, account_id: str) -> List[Dict]:
        """Get list of apps for the publisher account"""

        try:
            request = (
                self.service.accounts().apps().list(parent=f"accounts/{account_id}")
            )
            response = request.execute()

            apps = []
            if "apps" in response:
                for app in response["apps"]:
                    apps.append(
                        {
                            "app_id": app.get("appId", ""),
                            "name": app.get("name", ""),
                            "platform": app.get("platform", ""),
                            "app_store_id": app.get("appStoreId", ""),
                        }
                    )

            return apps

        except Exception as e:
            self.logger.error(f"Error getting apps: {e}")
            return []

    def generate_report(
        self, account_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate revenue report (network or mediation) based on user settings"""

        try:
            # Get report type from settings
            report_type = self.config.get("api_settings.report_type", "mediation")

            # Create the request body
            request_body = {
                "reportSpec": {
                    "dateRange": {
                        "startDate": {
                            "year": int(start_date[:4]),
                            "month": int(start_date[5:7]),
                            "day": int(start_date[8:10]),
                        },
                        "endDate": {
                            "year": int(end_date[:4]),
                            "month": int(end_date[5:7]),
                            "day": int(end_date[8:10]),
                        },
                    },
                    "dimensions": ["DATE"],
                    "metrics": ["ESTIMATED_EARNINGS"],
                    "sortConditions": [{"dimension": "DATE", "order": "ASCENDING"}],
                    "localizationSettings": {
                        "currencyCode": "USD",
                        "languageCode": "en-US",
                    },
                }
            }

            self.logger.info(
                f"Generating {report_type} report for {start_date} to {end_date}"
            )

            # Make the API request based on report type
            if report_type == "network":
                request = (
                    self.service.accounts()
                    .networkReport()
                    .generate(parent=f"accounts/{account_id}", body=request_body)
                )
            else:  # mediation
                request = (
                    self.service.accounts()
                    .mediationReport()
                    .generate(parent=f"accounts/{account_id}", body=request_body)
                )

            response = request.execute()

            # Parse the response
            data = []

            # Handle different response formats
            if isinstance(response, list):
                # Response is a list of rows (new format)
                for row_data in response:
                    # Skip header and footer rows
                    if "row" not in row_data:
                        continue

                    row = row_data["row"]

                    # Extract date - can be in different formats
                    date_info = row["dimensionValues"]["DATE"]
                    if "value" in date_info:
                        # Format: "20231024" -> "2023-10-24"
                        date_value = date_info["value"]
                        date_str = (
                            f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                        )
                    else:
                        # Format: {year: 2023, month: 10, day: 24}
                        date_str = f"{date_info['year']}-{date_info['month']:02d}-{date_info['day']:02d}"

                    # Extract earnings - can be microsAmount or microsValue
                    earnings_info = row["metricValues"]["ESTIMATED_EARNINGS"]
                    if "microsValue" in earnings_info:
                        earnings = float(earnings_info["microsValue"]) / 1000000
                    else:
                        earnings = float(earnings_info["microsAmount"]) / 1000000

                    data.append({"date": pd.to_datetime(date_str), "revenue": earnings})

            elif isinstance(response, dict) and "rows" in response:
                # Response is a dict with 'rows' key (old format)
                for row_data in response["rows"]:
                    # Handle the response format: each row is wrapped in a "row" object
                    row = row_data.get("row", row_data)

                    # Extract date - can be in different formats
                    date_info = row["dimensionValues"]["DATE"]
                    if "value" in date_info:
                        # Format: "20231024" -> "2023-10-24"
                        date_value = date_info["value"]
                        date_str = (
                            f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                        )
                    else:
                        # Format: {year: 2023, month: 10, day: 24}
                        date_str = f"{date_info['year']}-{date_info['month']:02d}-{date_info['day']:02d}"

                    # Extract earnings - can be microsAmount or microsValue
                    earnings_info = row["metricValues"]["ESTIMATED_EARNINGS"]
                    if "microsValue" in earnings_info:
                        earnings = float(earnings_info["microsValue"]) / 1000000
                    else:
                        earnings = float(earnings_info["microsAmount"]) / 1000000

                    data.append({"date": pd.to_datetime(date_str), "revenue": earnings})

            self.logger.info(
                f"Retrieved {len(data)} data points using {report_type} report"
            )

            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values("date").reset_index(drop=True)
                df.set_index("date", inplace=True)

            return df

        except HttpError as e:
            self.logger.error(f"HTTP error generating {report_type} report: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error generating {report_type} report: {e}")
            return pd.DataFrame()

    def get_revenue_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get revenue data for the specified date range"""

        try:
            # Check if authenticated
            if not self.service:
                if not self.authenticate():
                    raise Exception("Authentication failed")

            # Get publisher account
            account_id = self.get_publisher_account()
            if not account_id:
                raise Exception("Could not get publisher account")

            self.logger.info(f"Fetching revenue data from {start_date} to {end_date}")

            # Generate report
            df = self.generate_report(account_id, start_date, end_date)

            if df.empty:
                self.logger.warning(
                    "No revenue data found for the specified date range"
                )
                return pd.DataFrame()

            # Fill missing dates with 0 revenue
            df = self.fill_missing_dates(df, start_date, end_date)

            self.logger.info(f"Successfully fetched {len(df)} days of revenue data")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching revenue data: {e}")
            return pd.DataFrame()

    def fill_missing_dates(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fill missing dates with 0 revenue"""

        try:
            # Create a complete date range
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")

            # Reindex the dataframe to include all dates
            df = df.reindex(date_range, fill_value=0)
            df.index.name = "date"

            # Ensure revenue column exists
            if "revenue" not in df.columns:
                df["revenue"] = 0

            return df

        except Exception as e:
            self.logger.error(f"Error filling missing dates: {e}")
            return df

    def get_cached_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get cached revenue data if available and valid"""

        try:
            report_type = self.config.get("api_settings.report_type", "mediation")
            cache_key = f"revenue_{report_type}_{start_date}_{end_date}"
            cache_file = self.config.get_cache_file(cache_key)

            if self.config.is_cache_valid(cache_file):
                with open(cache_file, "r") as f:
                    data = json.load(f)

                df = pd.DataFrame(data)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    self.logger.info(
                        f"Using cached {report_type} revenue data from {start_date} to {end_date}"
                    )
                    return df

            return None

        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            return None

    def cache_data(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Cache revenue data for future use"""

        try:
            report_type = self.config.get("api_settings.report_type", "mediation")
            cache_key = f"revenue_{report_type}_{start_date}_{end_date}"
            cache_file = self.config.get_cache_file(cache_key)

            # Convert DataFrame to JSON-serializable format
            data = df.reset_index().to_dict("records")
            for record in data:
                record["date"] = record["date"].isoformat()

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(
                f"Cached {report_type} revenue data for {start_date} to {end_date}"
            )

        except Exception as e:
            self.logger.error(f"Error caching data: {e}")

    def fetch_revenue_data(
        self, start_date: str, end_date: str, use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch revenue data with caching support"""

        try:
            # Try to get cached data first
            if use_cache:
                cached_data = self.get_cached_data(start_date, end_date)
                if cached_data is not None:
                    return cached_data

            # Fetch fresh data from API
            df = self.get_revenue_data(start_date, end_date)

            # Cache the data
            if not df.empty:
                self.cache_data(df, start_date, end_date)

            return df

        except Exception as e:
            self.logger.error(f"Error in fetch_revenue_data: {e}")
            return pd.DataFrame()
