# app/services/twelve_data_service.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
import asyncio
import os


class TwelveDataService:
    """Service for fetching stock data from Twelve Data API."""

    def __init__(self, api_key: str = None):
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY", "")
        self.base_url = "https://api.twelvedata.com"

    async def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make a request to the Twelve Data API."""
        url = f"{self.base_url}/{endpoint}"
        params["apikey"] = self.api_key

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API error: {response.status} - {text}")

                    data = await response.json()
                    return data
            except Exception as e:
                raise Exception(f"Request failed: {str(e)}")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1day",
        start: Union[datetime, str, None] = None,
        end: Union[datetime, str, None] = None,
    ) -> pd.DataFrame:
        """Get historical price data for a symbol."""
        # Map timeframes to Twelve Data format
        interval_map = {
            "1D": "1day",
            "1H": "1h",
            "15Min": "15min",
            "5Min": "5min",
            "1Min": "1min",
        }
        interval = interval_map.get(timeframe, "1day")

        # Calculate date range
        if end is None:
            end = datetime.now()
        if start is None:
            # Default to 30 days for daily, adjust for other timeframes
            if interval == "1day":
                start = end - timedelta(days=30)
            else:
                start = end - timedelta(days=7)

        # Format dates
        start_str = (
            start.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(start, datetime)
            else start
        )
        end_str = (
            end.strftime("%Y-%m-%d %H:%M:%S") if isinstance(end, datetime) else end
        )

        # Set up params for time series endpoint
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_str,
            "end_date": end_str,
            "format": "json",
            "timezone": "UTC",
        }

        try:
            data = await self._make_request("time_series", params)

            if "values" not in data:
                print(
                    f"No data returned for {symbol}: {data.get('message', 'Unknown error')}"
                )
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data["values"])

            # Convert and rename columns
            df = df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "datetime": "timestamp",
                }
            )

            # Convert types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"])

            # Set index
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            # Sort by date (oldest to newest)
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"Error fetching time series data: {e}")
            return pd.DataFrame()

    async def get_latest_quote(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        params = {"symbol": symbol, "format": "json"}

        try:
            data = await self._make_request("price", params)

            if "price" in data:
                return float(data["price"])
            return 0.0

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return 0.0

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data for a company."""
        # Basic company profile endpoint
        params = {"symbol": symbol, "format": "json"}

        try:
            data = await self._make_request("profile", params)

            if not isinstance(data, dict) or "symbol" not in data:
                return {"symbol": symbol, "financials_available": False}

            return {
                "symbol": symbol,
                "company_name": data.get("name", ""),
                "sector": data.get("sector", ""),
                "industry": data.get("industry", ""),
                "market_cap": data.get("market_cap", 0),
                "pe_ratio": data.get("pe", 0),
                "dividend_yield": data.get("dividend_yield", 0),
                "financials_available": True,
            }

        except Exception as e:
            print(f"Error getting company profile: {e}")
            return {"symbol": symbol, "financials_available": False}
