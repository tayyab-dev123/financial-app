# app/services/yahoo_finance_service.py
import yfinance as yf
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
import asyncio
import json


class YahooFinanceService:
    """Service for fetching stock data from Yahoo Finance."""

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",  # 1d, 1h, etc.
        start: Union[datetime, str, None] = None,
        end: Union[datetime, str, None] = None,
    ) -> pd.DataFrame:
        """Get historical price bars for a symbol."""
        # Convert timeframe from Alpaca format to Yahoo format
        interval_map = {
            "1D": "1d",
            "1H": "1h",
            "15Min": "15m",
            "5Min": "5m",
            "1Min": "1m",
        }
        yahoo_interval = interval_map.get(timeframe, "1d")

        # Run in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(
                symbol, start=start, end=end, interval=yahoo_interval, auto_adjust=True
            ),
        )

        # Standardize column names to match what our app expects
        if not df.empty:
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

        return df

    async def get_latest_quote(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))

        # Get the most recent data point
        try:
            data = await loop.run_in_executor(None, lambda: ticker.history(period="1d"))
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"Error getting latest price: {e}")
        return 0.0

    async def get_news(
        self, symbol: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get news articles for a symbol or market news."""
        news_items = []

        if symbol:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))

            try:
                # Get news from Yahoo Finance
                news = await loop.run_in_executor(None, lambda: ticker.news)

                for item in news[:limit]:
                    news_items.append(
                        {
                            "id": str(item.get("uuid", "")),
                            "headline": item.get("title", ""),
                            "summary": item.get("summary", ""),
                            "url": item.get("link", ""),
                            "source": item.get("publisher", "Yahoo Finance"),
                            "created_at": datetime.fromtimestamp(
                                item.get("providerPublishTime", 0)
                            ),
                            "updated_at": None,
                            "symbols": [symbol],
                        }
                    )
            except Exception as e:
                print(f"Error fetching news for {symbol}: {e}")

        # For market news when no symbol is provided
        if not symbol or not news_items:
            try:
                # Get general market news
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
                    ) as response:
                        if response.status == 200:
                            import xml.etree.ElementTree as ET
                            from io import StringIO

                            text = await response.text()
                            root = ET.fromstring(text)

                            for item in root.findall(".//item")[:limit]:
                                title = (
                                    item.find("title").text
                                    if item.find("title") is not None
                                    else ""
                                )
                                link = (
                                    item.find("link").text
                                    if item.find("link") is not None
                                    else ""
                                )
                                pub_date = (
                                    item.find("pubDate").text
                                    if item.find("pubDate") is not None
                                    else ""
                                )
                                description = (
                                    item.find("description").text
                                    if item.find("description") is not None
                                    else ""
                                )

                                try:
                                    pub_datetime = datetime.strptime(
                                        pub_date, "%a, %d %b %Y %H:%M:%S %z"
                                    )
                                except:
                                    pub_datetime = datetime.now()

                                news_items.append(
                                    {
                                        "id": link[-20:],
                                        "headline": title,
                                        "summary": description,
                                        "url": link,
                                        "source": "Yahoo Finance",
                                        "created_at": pub_datetime,
                                        "updated_at": None,
                                        "symbols": [],
                                    }
                                )
            except Exception as e:
                print(f"Error fetching market news: {e}")

        return news_items

    async def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data for a company."""
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))

        try:
            # Get basic info
            info = await loop.run_in_executor(None, lambda: ticker.info)

            # Get financial statements
            financials = await loop.run_in_executor(None, lambda: ticker.financials)
            balance_sheet = await loop.run_in_executor(
                None, lambda: ticker.balance_sheet
            )

            return {
                "symbol": symbol,
                "company_name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "financials_available": True,
                "income_statement": (
                    financials.to_dict() if not financials.empty else {}
                ),
                "balance_sheet": (
                    balance_sheet.to_dict() if not balance_sheet.empty else {}
                ),
            }
        except Exception as e:
            print(f"Error getting company financials: {e}")
            return {"symbol": symbol, "financials_available": False}
