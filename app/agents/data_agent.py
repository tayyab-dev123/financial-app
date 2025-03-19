# app/agents/data_agent.py
from typing import Dict, List, Optional, Any
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

    async def get_stock_data(
        self,
        symbol: str,
        timeframe: str = "1D",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> StockData:
        """
        Fetch stock data for a given symbol and timeframe.

        Args:
            symbol: Stock ticker symbol
            timeframe: Time interval ('1D', '1H', '15Min', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            StockData object containing OHLCV data
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)

        if not end_date:
            end_date = datetime.now()

        # Fetch data from Alpaca
        stock_data = await self.alpaca_service.get_bars(
            symbol=symbol, timeframe=timeframe, start=start_date, end=end_date
        )

        # Process and return as StockData model
        return StockData(
            symbol=symbol,
            data=stock_data,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

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
