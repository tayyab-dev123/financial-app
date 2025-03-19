# app/services/fmp_service.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
import asyncio
import os


class FMPService:
    """Service for fetching stock data from Financial Modeling Prep API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    async def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """Make a request to the FMP API."""
        if params is None:
            params = {}

        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API error: {response.status} - {text}")

                    return await response.json()
            except Exception as e:
                raise Exception(f"Request failed: {str(e)}")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        start: Union[datetime, str, None] = None,
        end: Union[datetime, str, None] = None,
    ) -> pd.DataFrame:
        """Get historical price data for a symbol."""
        # Calculate date range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)

        # Format dates
        start_str = start.strftime("%Y-%m-%d") if isinstance(start, datetime) else start
        end_str = end.strftime("%Y-%m-%d") if isinstance(end, datetime) else end

        # Determine endpoint based on timeframe
        if timeframe in ["1Min", "5Min", "15Min", "30Min", "1H"]:
            # Intraday data (only available for premium)
            print("Intraday data is limited in free tier. Using daily data instead.")
            endpoint = f"historical-price-full/{symbol}"
        else:
            endpoint = f"historical-price-full/{symbol}"

        try:
            data = await self._make_request(endpoint)

            if not data or "historical" not in data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data["historical"])

            # Rename columns
            df = df.rename(
                columns={
                    "date": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            # Set index
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            # Sort by date (newest to oldest)
            df = df.sort_index(ascending=False)

            # Filter by date range
            if start_str:
                df = df[df.index >= start_str]
            if end_str:
                df = df[df.index <= end_str]

            return df

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    async def get_latest_quote(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        try:
            endpoint = f"quote/{symbol}"
            data = await self._make_request(endpoint)

            if data and isinstance(data, list) and len(data) > 0:
                return float(data[0].get("price", 0))
            return 0.0

        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return 0.0

    async def get_news(
        self, symbol: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get news for a stock or general market news."""
        news_items = []

        try:
            if symbol:
                endpoint = f"stock_news?tickers={symbol}&limit={limit}"
            else:
                endpoint = f"stock_news?limit={limit}"

            data = await self._make_request(endpoint)

            if data and isinstance(data, list):
                for item in data[:limit]:
                    news_items.append(
                        {
                            "id": str(item.get("id", "")),
                            "headline": item.get("title", ""),
                            "summary": item.get("text", ""),
                            "url": item.get("url", ""),
                            "source": item.get("site", "FMP"),
                            "created_at": (
                                datetime.fromtimestamp(
                                    item.get("publishedDate", 0) / 1000
                                )
                                if "publishedDate" in item
                                else datetime.now()
                            ),
                            "updated_at": None,
                            "symbols": [symbol] if symbol else [],
                        }
                    )

            return news_items

        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data for a company."""
        try:
            # Get company profile
            profile_endpoint = f"profile/{symbol}"
            profile_data = await self._make_request(profile_endpoint)

            if (
                not profile_data
                or not isinstance(profile_data, list)
                or len(profile_data) == 0
            ):
                return {"symbol": symbol, "financials_available": False}

            profile = profile_data[0]

            # Get income statement
            income_endpoint = f"income-statement/{symbol}?limit=4"
            income_data = await self._make_request(income_endpoint)

            # Get balance sheet
            balance_endpoint = f"balance-sheet-statement/{symbol}?limit=4"
            balance_data = await self._make_request(balance_endpoint)

            return {
                "symbol": symbol,
                "company_name": profile.get("companyName", ""),
                "sector": profile.get("sector", ""),
                "industry": profile.get("industry", ""),
                "market_cap": profile.get("mktCap", 0),
                "pe_ratio": profile.get("pe", 0),
                "dividend_yield": profile.get("lastDiv", 0),
                "financials_available": True,
                "description": profile.get("description", ""),
                "exchange": profile.get("exchange", ""),
                "income_statement": income_data if income_data else [],
                "balance_sheet": balance_data if balance_data else [],
            }

        except Exception as e:
            print(f"Error getting company financials: {e}")
            return {"symbol": symbol, "financials_available": False}
