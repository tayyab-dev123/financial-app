# app/agents/data_agent.py
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd

from app.core.config import settings
from app.services.alpaca_service import AlpacaService
from app.models.stock import StockData, NewsItem


class DataAgent:
    """Agent responsible for fetching stock data, news, and financial reports."""

    def __init__(self):
        self.alpaca_service = AlpacaService(
            api_key=settings.ALPACA_API_KEY,
            api_secret=settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
        )

    def _format_date(self, date):
        """Helper method to ensure dates are properly formatted for Alpaca API.

        Args:
            date: Can be a datetime object or a string in YYYY-MM-DD format

        Returns:
            str: A properly formatted date string for Alpaca API
        """
        if isinstance(date, datetime):
            return date.strftime("%Y-%m-%d")
        return date

    async def get_stock_data(
        self,
        symbol: str,
        timeframe: str = "1D",
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
    ) -> StockData:
        """
        Fetch stock data for a given symbol and timeframe.

        Args:
            symbol: Stock ticker symbol
            timeframe: Time interval ('1D', '1H', '15Min', etc.)
            start_date: Start date for historical data (datetime or string)
            end_date: End date for historical data (datetime or string)

        Returns:
            StockData object containing OHLCV data
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)

        if not end_date:
            # Set end_date to current time
            current_time = datetime.now()
            # For free Alpaca accounts, add a buffer period to avoid SIP data restrictions
            # Subtract 20 minutes from current time
            end_date = current_time - timedelta(minutes=20)

        try:
            # Convert dates to properly formatted strings
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime("%Y-%m-%d")
            else:
                start_date_str = start_date

            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime("%Y-%m-%d")
            else:
                end_date_str = end_date

            print(f"Fetching {symbol} data from {start_date_str} to {end_date_str}")

            # Fetch data from Alpaca
            stock_data = await self.alpaca_service.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date_str,
                end=end_date_str,
            )

            # Process and return as StockData model
            return StockData(
                symbol=symbol,
                data=stock_data,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            if "subscription does not permit" in str(e):
                print(f"Using fallback for {symbol} due to subscription limitations")
                # Create an empty dataframe but return a valid StockData object
                empty_df = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
                return StockData(
                    symbol=symbol,
                    data=empty_df,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    error="Data limited due to subscription tier",
                )
            # Add more context to the error
            raise Exception(f"Failed to fetch stock data: {str(e)}") from e

    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a stock symbol."""
        return await self.alpaca_service.get_latest_quote(symbol)

    async def get_company_news(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        """Fetch recent news articles about a company."""
        news_data = await self.alpaca_service.get_news(symbol, limit)
        return [NewsItem(**item) for item in news_data]

    async def get_market_news(self, limit: int = 5) -> List[NewsItem]:
        """Fetch general market news."""
        news_data = await self.alpaca_service.get_news(None, limit)
        return [NewsItem(**item) for item in news_data]

    async def get_multiple_stocks(
        self, symbols: List[str], timeframe: str = "1D"
    ) -> Dict[str, StockData]:
        """Fetch data for multiple stocks at once."""
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_stock_data(symbol, timeframe)
        return result

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Fetch company financial statements if available."""
        return await self.alpaca_service.get_company_financials(symbol)
