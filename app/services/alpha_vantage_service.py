# app/services/alpha_vantage_service.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
import asyncio
import os


class AlphaVantageService:
    """Service for fetching stock data from Alpha Vantage API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.base_url = "https://www.alphavantage.co/query"

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        start: Union[datetime, str, None] = None,
        end: Union[datetime, str, None] = None,
    ) -> pd.DataFrame:
        """Get historical price data for a symbol."""
        function_map = {
            "1Min": "TIME_SERIES_INTRADAY",
            "5Min": "TIME_SERIES_INTRADAY",
            "15Min": "TIME_SERIES_INTRADAY",
            "30Min": "TIME_SERIES_INTRADAY",
            "1H": "TIME_SERIES_INTRADAY",
            "1D": "TIME_SERIES_DAILY",
            "1W": "TIME_SERIES_WEEKLY",
            "1M": "TIME_SERIES_MONTHLY",
        }

        interval_map = {
            "1Min": "1min",
            "5Min": "5min",
            "15Min": "15min",
            "30Min": "30min",
            "1H": "60min",
        }

        function = function_map.get(timeframe, "TIME_SERIES_DAILY")

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",
        }

        # Add interval for intraday data
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval_map.get(timeframe, "5min")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API error: {response.status} - {text}")

                    data = await response.json()

                    # Extract the time series data
                    time_series_key = None
                    for key in data.keys():
                        if "Time Series" in key:
                            time_series_key = key
                            break

                    if not time_series_key:
                        return pd.DataFrame()

                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(data[time_series_key], orient="index")

                    # Rename columns
                    df = df.rename(
                        columns={
                            "1. open": "open",
                            "2. high": "high",
                            "3. low": "low",
                            "4. close": "close",
                            "5. volume": "volume",
                        }
                    )

                    # Convert types
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])

                    # Set index
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)

                    # Filter by date range if provided
                    if start:
                        start_dt = (
                            pd.to_datetime(start) if isinstance(start, str) else start
                        )
                        df = df[df.index >= start_dt]

                    if end:
                        end_dt = pd.to_datetime(end) if isinstance(end, str) else end
                        df = df[df.index <= end_dt]

                    return df

            except Exception as e:
                print(f"Error fetching time series data: {e}")
                return pd.DataFrame()

    async def get_latest_quote(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        return 0.0

                    data = await response.json()

                    if "Global Quote" in data and "05. price" in data["Global Quote"]:
                        return float(data["Global Quote"]["05. price"])
                    return 0.0

            except Exception as e:
                print(f"Error getting quote for {symbol}: {e}")
                return 0.0

    async def get_news(
        self, symbol: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get market news (Alpha Vantage has limited news capabilities)."""
        # Alpha Vantage doesn't have a great news API, so returning empty for now
        return []

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data for a company."""
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": self.api_key}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        return {"symbol": symbol, "financials_available": False}

                    data = await response.json()

                    if not data or "Symbol" not in data:
                        return {"symbol": symbol, "financials_available": False}

                    return {
                        "symbol": symbol,
                        "company_name": data.get("Name", ""),
                        "sector": data.get("Sector", ""),
                        "industry": data.get("Industry", ""),
                        "market_cap": float(data.get("MarketCapitalization", 0)),
                        "pe_ratio": float(data.get("PERatio", 0)),
                        "dividend_yield": float(data.get("DividendYield", 0)),
                        "financials_available": True,
                        "description": data.get("Description", ""),
                        "exchange": data.get("Exchange", ""),
                        "currency": data.get("Currency", "USD"),
                    }

            except Exception as e:
                print(f"Error getting company overview: {e}")
                return {"symbol": symbol, "financials_available": False}
