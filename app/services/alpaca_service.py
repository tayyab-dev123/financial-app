# app/services/alpaca_service.py
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import httpx
import json


class AlpacaService:
    """Service for interacting with the Alpaca API."""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

        # Set up data API URL - different from trading API
        # Alpaca's data API is at a different domain
        self.data_url = "https://data.alpaca.markets"

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Dict = None,
        data: Dict = None,
        base: str = None,
    ) -> Dict:
        """Make an HTTP request to the Alpaca API."""
        base_url = base if base else self.base_url
        url = f"{base_url}{endpoint}"

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                if method == "GET":
                    response = await client.get(url, headers=headers, params=params)
                elif method == "POST":
                    response = await client.post(url, headers=headers, json=data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json() if response.content else {}

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if "message" in error_data:
                        error_msg = f"{error_msg}: {error_data['message']}"
                except:
                    pass
                raise Exception(error_msg) from e
            except Exception as e:
                raise Exception(f"API request failed: {str(e)}") from e

    def _format_date(self, date_input: Union[datetime, str, None]) -> Optional[str]:
        """Format date objects to strings in YYYY-MM-DD format for Alpaca API.

        Args:
            date_input: Can be a datetime object, str or None

        Returns:
            str or None: A properly formatted date string or None
        """
        if date_input is None:
            return None

        if isinstance(date_input, datetime):
            return date_input.strftime("%Y-%m-%d")

        # If it's already a string, return as is (assuming it's properly formatted)
        return date_input

    # Modify get_bars method in app/services/alpaca_service.py
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Union[datetime, str, None] = None,
        end: Union[datetime, str, None] = None,
    ):
        """Get historical price data for a symbol.

        Args:
            symbol: Stock ticker symbol
            timeframe: Time interval ('1D', '1H', etc.)
            start: Start date (datetime or string in YYYY-MM-DD format)
            end: End date (datetime or string in YYYY-MM-DD format)

        Returns:
            DataFrame with OHLCV data
        """
        # Format dates to strings if they're datetime objects
        start_str = self._format_date(start)

        # Handle the 15-minute delay limitation for free tier
        current_time = datetime.now()
        buffer_time = datetime.now().replace(
            minute=(
                current_time.minute - 16
                if current_time.minute >= 16
                else current_time.minute + 44
            ),
            hour=(
                current_time.hour - 1 if current_time.minute < 16 else current_time.hour
            ),
        )

        # If end time is not provided or is too recent, adjust it
        if end is None or (isinstance(end, datetime) and end > buffer_time):
            end = buffer_time
            print(f"Adjusting end time to account for 15-minute data delay: {end}")

        end_str = self._format_date(end)

        try:
            # Call Alpaca API with properly formatted date strings
            bars = self.api.get_bars(symbol, timeframe, start=start_str, end=end_str)

            # Convert to dataframe
            if hasattr(bars, "df"):
                return bars.df
            else:
                import pandas as pd

                return pd.DataFrame(bars)

        except Exception as e:
            if "subscription does not permit" in str(e):
                # Handle subscription limitation elegantly
                print(f"Subscription limitation for {symbol}: {e}")

                # Return empty dataframe with expected columns for graceful degradation
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            else:
                # Add helpful context to other errors
                raise Exception(f"Alpaca API error: {e}") from e

    async def get_latest_quote(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        try:
            # Use the API client for quotes, which is more reliable
            latest_quote = self.api.get_latest_quote(symbol)
            if hasattr(latest_quote, "ask_price"):
                return float(latest_quote.ask_price)
            elif hasattr(latest_quote, "ap"):
                return float(latest_quote.ap)
            # Fallback to last trade price
            latest_trade = self.api.get_latest_trade(symbol)
            return float(latest_trade.price)
        except Exception as e:
            if "subscription does not permit" in str(e):
                # Fallback to alternative data sources or cached value
                print(
                    f"Using delayed quote for {symbol} due to subscription limitations"
                )

                # Try to get the last known price from historical data
                try:
                    # Get data from yesterday to avoid the 15-minute restriction
                    end_time = datetime.now().replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    start_time = end_time - timedelta(days=1)

                    bars_df = await self.get_bars(
                        symbol, "1D", start=start_time, end=end_time
                    )
                    if not bars_df.empty:
                        return float(bars_df["close"].iloc[-1])
                    return 0.0  # Return a default value when no data is available
                except:
                    return 0.0
            raise Exception(f"Failed to get latest quote: {str(e)}")

    async def get_news(
        self, symbol: str = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get news for a symbol or general market news."""
        # Updated to use correct news endpoint path
        endpoint = "/v1beta1/news"  # Newer Alpaca news API endpoint
        params = {"limit": limit}

        if symbol:
            params["symbol"] = symbol  # Changed from 'symbols' to 'symbol'

        try:
            # Print request details for debugging
            print(
                f"Requesting news from: {self.data_url}{endpoint} with params: {params}"
            )

            response = await self._make_request(
                endpoint, params=params, base=self.data_url
            )

            # Return empty list if no news found or if structure is unexpected
            if not isinstance(response, dict):
                print(f"Unexpected news API response type: {type(response)}")
                return []

            # Handle different response structures that Alpaca might return
            if "news" in response:
                return response["news"]
            elif isinstance(response, list):
                return response
            else:
                print(
                    f"News response keys: {response.keys() if hasattr(response, 'keys') else 'No keys'}"
                )
                return []

        except Exception as e:
            # Try alternative news endpoint format as fallback
            try:
                alt_endpoint = "/v2/news"
                print(
                    f"Trying alternative news endpoint: {self.data_url}{alt_endpoint}"
                )
                response = await self._make_request(
                    alt_endpoint, params=params, base=self.data_url
                )
                return (
                    response if isinstance(response, list) else response.get("news", [])
                )
            except Exception as fallback_error:
                raise Exception(
                    f"Failed to fetch news: {str(e)} (fallback error: {str(fallback_error)})"
                )

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
            "time_in_force": "day",
        }

        return await self._make_request(endpoint, method="POST", data=data)

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data for a company."""
        endpoint = f"/fundamentals/{symbol}"
        try:
            return await self._make_request(endpoint, base=self.data_url)
        except Exception:
            # Fallback to basic company information if financials not available
            return {"symbol": symbol, "financials_available": False}
