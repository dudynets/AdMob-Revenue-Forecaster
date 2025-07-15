<a href="https://dudynets.dev">
  <img src="https://user-images.githubusercontent.com/39008921/191470114-c074b17f-1c88-4af3-b089-1b14418cabf5.png" alt="drawing" width="128"/>
</a>

# AdMob Revenue Forecaster

<p><strong>A desktop application for forecasting AdMob revenue using SARIMA time series models.</strong></p>

> Developed by [Oleksandr Dudynets](https://dudynets.dev)

## Features

- Fetch AdMob revenue data via Google AdMob API
- Generate revenue forecasts using SARIMA models
- Interactive charts with historical data and predictions
- Backtesting to validate model performance
- Currency conversion with exchange rate updates
- Export data and results

## Requirements

- Python (tested on 3.13.3)
- Google AdMob account with API access
- OAuth2 credentials from Google Cloud Console

## Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the application:
   ```bash
   python src/main.py
   ```

## Setup

1. Create OAuth2 credentials in Google Cloud Console (Desktop application type)
2. Download the credentials JSON file
3. In the Settings tab, upload the JSON file
4. Enter your AdMob Customer ID (Publisher ID)
5. Test the connection

## Usage

1. Go to Data tab and fetch your AdMob revenue data
2. Go to Forecast tab and run the forecasting model
3. View results in the Visualization tab

The application will guide you through the process.

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License.
See [LICENSE](LICENSE) for more information.
