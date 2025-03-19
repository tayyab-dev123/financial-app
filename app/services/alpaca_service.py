# app/services/alpaca_service.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import httpx
import json


class AlpacaService:
    """Service to interact with Alpaca API for stock market data."""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """Initialize with Alpaca API credentials."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.data_url = "https://data.alpaca.markets/v2"
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        base: str = None,
    ) -> Dict:
        """Make an HTTP request to Alpaca API."""
        if base is None:
            base = self.base_url

        url = f"{base}{endpoint}"

        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = await client.post(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code != 200:
                raise Exception(
                    f"Alpaca API error: {response.status_code} - {response.text}"
                )

            return response.json()

    async def get_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch historical price bars for a symbol."""
        endpoint = f"/stocks/{symbol}/bars"

        params = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "all",
        }

        response = await self._make_request(endpoint, params=params, base=self.data_url)
        bars = response.get("bars", [])

        # Convert to DataFrame
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )

        df.set_index("timestamp", inplace=True)
        return df

    async def get_latest_quote(self, symbol: str) -> float:
        """Get the latest quote for a symbol."""
        endpoint = f"/stocks/{symbol}/quotes/latest"
        response = await self._make_request(endpoint, base=self.data_url)
        return float(response["quote"]["ap"])  # ask price

    async def get_news(
        self, symbol: Optional[str] = None, limit: int = 5
    ) -> List[Dict]:
        """Fetch news articles for a symbol or market news if symbol is None."""
        endpoint = "/news"
        params = {"limit": limit}

        if symbol:
            params["symbols"] = symbol

        response = await self._make_request(endpoint, params=params, base=self.data_url)
        return response.get("news", [])

    async def get_account(self) -> Dict:
        """Get account information."""
        endpoint = "/account"
        return await self._make_request(endpoint)

    async def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = "market",
        time_in_force: str = "day",
    ) -> Dict:
        """Place a trade order."""
        endpoint = "/orders"
        data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }

        return await self._make_request(endpoint, method="POST", data=data)

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company financial statements.
        Note: Requires premium Alpaca subscription for full data.
        """
        endpoint = f"/fundamentals/{symbol}"
        try:
            return await self._make_request(endpoint, base=self.data_url)
        except Exception:
            # Fallback to basic company information if financials not available
            return {"symbol": symbol, "financials_available": False}
