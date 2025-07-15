"""
Currency Formatter Module
Handles currency conversion and formatting for the AdMob Revenue Forecaster
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging


class CurrencyFormatter:
    """Currency formatter with exchange rate support"""
    
    def __init__(self, config):
        self.config = config
        self.rates = {}
        self.currencies = {}
        self.last_updated = None
        self.base_currency = "USD"
        
        # Use free fawazahmed0/exchange-api - no rate limits, 200+ currencies
        self.api_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
        self.fallback_api_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
        self.currencies_api_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies.json"
        self.currencies_fallback_url = "https://latest.currency-api.pages.dev/v1/currencies.json"
        
    def get_local_currency(self) -> str:
        """Get the user's local currency setting"""
        return self.config.get('ui_settings.local_currency', 'UAH')
    
    def set_local_currency(self, currency: str):
        """Set the user's local currency"""
        self.config.set('ui_settings.local_currency', currency)
        # Clear rates to force refresh
        self.rates = {}
        self.last_updated = None
    
    def needs_update(self) -> bool:
        """Check if exchange rates need to be updated"""
        if not self.rates or not self.last_updated:
            return True
        
        # Update rates daily
        return datetime.now() - self.last_updated > timedelta(days=1)
    
    def fetch_currencies(self) -> bool:
        """Fetch available currencies from API"""
        try:
            logging.info("Fetching currency list...")
            
            # Try primary API
            response = requests.get(self.currencies_api_url, timeout=10)
            
            if response.status_code == 200:
                self.currencies = response.json()
                logging.info(f"Currency list updated successfully. {len(self.currencies)} currencies loaded.")
                return True
            else:
                logging.warning(f"Primary currencies API failed with status {response.status_code}")
                return self._try_fallback_currencies_api()
                
        except Exception as e:
            logging.error(f"Error fetching currencies: {str(e)}")
            return self._try_fallback_currencies_api()
    
    def _try_fallback_currencies_api(self) -> bool:
        """Try fallback currencies API"""
        try:
            response = requests.get(self.currencies_fallback_url, timeout=10)
            
            if response.status_code == 200:
                self.currencies = response.json()
                logging.info(f"Currency list updated using fallback API. {len(self.currencies)} currencies loaded.")
                return True
            else:
                logging.warning("Fallback currencies API also failed")
                return False
                
        except Exception as e:
            logging.error(f"Fallback currencies API error: {str(e)}")
            return False
    
    def fetch_exchange_rates(self) -> bool:
        """Fetch exchange rates from API"""
        try:
            logging.info("Fetching exchange rates...")
            
            # Try primary API
            response = requests.get(self.api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # The API returns data in format: {"date": "2024-01-15", "usd": {"eur": 0.85, ...}}
                if 'usd' in data:
                    self.rates = data['usd']
                    self.last_updated = datetime.now()
                    logging.info(f"Exchange rates updated successfully. {len(self.rates)} currencies loaded.")
                    return True
                else:
                    logging.warning("Unexpected API response format")
                    return self._try_fallback_api()
            else:
                logging.warning(f"Primary API failed with status {response.status_code}")
                return self._try_fallback_api()
                
        except Exception as e:
            logging.error(f"Error fetching exchange rates: {str(e)}")
            return self._try_fallback_api()
    
    def _try_fallback_api(self) -> bool:
        """Try fallback exchange rate API"""
        try:
            response = requests.get(self.fallback_api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'usd' in data:
                    self.rates = data['usd']
                    self.last_updated = datetime.now()
                    logging.info("Exchange rates updated using fallback API")
                    return True
                else:
                    logging.warning("Unexpected fallback API response format")
                    return False
            else:
                logging.warning("Fallback API also failed")
                return False
                
        except Exception as e:
            logging.error(f"Fallback API error: {str(e)}")
            return False
    
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get exchange rate between two currencies. Returns None if rate not available."""
        if from_currency == to_currency:
            return 1.0
        
        # Update rates if needed
        if self.needs_update():
            success = self.fetch_exchange_rates()
            if not success:
                logging.warning("Failed to fetch exchange rates")
                return None
        
        # If no rates available, return None
        if not self.rates:
            return None
        
        # Convert currency codes to lowercase for API
        from_curr = from_currency.lower()
        to_curr = to_currency.lower()
        
        if from_curr == "usd":
            return self.rates.get(to_curr)
        elif to_curr == "usd":
            rate = self.rates.get(from_curr)
            return 1.0 / rate if rate and rate != 0 else None
        else:
            # Convert via USD
            usd_to_target = self.rates.get(to_curr)
            usd_to_source = self.rates.get(from_curr)
            if usd_to_target and usd_to_source and usd_to_source != 0:
                return usd_to_target / usd_to_source
            return None
    
    def format_currency(self, amount: float, show_local: bool = True) -> str:
        """Format currency amount with local currency conversion"""
        if not show_local:
            return f"{amount:,.2f} USD"
        
        local_currency = self.get_local_currency()
        
        # If local currency is USD, just show USD
        if local_currency == "USD":
            return f"{amount:,.2f} USD"
        
        # Convert to local currency
        exchange_rate = self.get_exchange_rate("USD", local_currency)
        
        # If exchange rate is not available, show only USD
        if exchange_rate is None:
            return f"{amount:,.2f} USD"
        
        local_amount = amount * exchange_rate
        
        # Format both currencies with currency codes
        usd_formatted = f"{amount:,.2f} USD"
        local_formatted = f"{local_amount:,.2f} {local_currency}"
        
        return f"{usd_formatted} ({local_formatted})"
    
    def format_currency_short(self, amount: float) -> str:
        """Format currency amount in short form (local currency only if different from USD)"""
        local_currency = self.get_local_currency()
        
        if local_currency == "USD":
            return f"{amount:,.2f} USD"
        
        exchange_rate = self.get_exchange_rate("USD", local_currency)
        
        # If exchange rate is not available, show USD
        if exchange_rate is None:
            return f"{amount:,.2f} USD"
        
        local_amount = amount * exchange_rate
        
        return f"{local_amount:,.2f} {local_currency}"
    
    def get_available_currencies(self) -> Dict[str, str]:
        """Get list of available currencies"""
        # Update currencies if needed
        if not self.currencies:
            self.fetch_currencies()
        
        # If still no currencies, return basic USD
        if not self.currencies:
            return {"USD": "USD - United States Dollar"}
        
        available = {}
        for currency_code, currency_name in self.currencies.items():
            currency_upper = currency_code.upper()

            if currency_name:
                available[currency_upper] = f"{currency_upper} - {currency_name}"
            else:
                available[currency_upper] = f"{currency_upper}"
        
        # Always include USD
        available["USD"] = "USD - United States Dollar"
        
        return dict(sorted(available.items()))
    
    def get_current_exchange_rate(self, currency: str) -> Optional[float]:
        """Get current exchange rate for a specific currency from USD"""
        if currency == "USD":
            return 1.0
        
        return self.get_exchange_rate("USD", currency)
    
    def initialize(self):
        """Initialize currency formatter by fetching rates and currencies"""
        # Fetch currencies first
        currencies_success = self.fetch_currencies()
        if not currencies_success:
            logging.warning("Failed to fetch currencies list")
        
        # Fetch exchange rates
        if self.needs_update():
            rates_success = self.fetch_exchange_rates()
            if not rates_success:
                logging.warning("Failed to fetch exchange rates")
        
        logging.info(f"Currency formatter initialized. Local currency: {self.get_local_currency()}") 
