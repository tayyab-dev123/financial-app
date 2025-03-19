# app/services/stock_data_factory.py
from typing import Optional
from app.core.config import settings

# Import all service classes
from app.services.alpaca_service import AlpacaService
from app.services.yahoo_finance_service import YahooFinanceService
from app.services.twelve_data_service import TwelveDataService
from app.services.alpha_vantage_service import AlphaVantageService
from app.services.fmp_service import FMPService


class StockDataFactory:
    """Factory for creating stock data service instances."""

    @staticmethod
    def create_service(provider: str = None):
        """
        Create a stock data service based on the specified provider.

        Args:
            provider: Name of the service provider (alpaca, yahoo, twelvedata, alphavantage, fmp)

        Returns:
            An instance of the requested stock data service
        """
        # Use provider from settings if not specified
        if not provider:
            provider = settings.STOCK_DATA_PROVIDER.lower()
        else:
            provider = provider.lower()

        if provider == "alpaca":
            return AlpacaService(
                api_key=settings.ALPACA_API_KEY,
                api_secret=settings.ALPACA_SECRET_KEY,
                base_url=settings.ALPACA_BASE_URL,
            )
        # elif provider == "yahoo":
        #     return YahooFinanceService()
        elif provider == "twelvedata":
            return TwelveDataService(api_key=settings.TWELVE_DATA_API_KEY)
        elif provider == "alphavantage":
            return AlphaVantageService(api_key=settings.ALPHA_VANTAGE_API_KEY)
        elif provider == "fmp":
            return FMPService(api_key=settings.FMP_API_KEY)
        else:

            # Default to Yahoo Finance as it's the most accessible
            return YahooFinanceService()
