# app/routes/stocks.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd

from app.core.dependencies import get_data_agent
from app.agents.data_agent import DataAgent
from app.models.stock import StockData, NewsItem
from app.utils.serialization import clean_response

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.get("/{symbol}/price")
async def get_stock_price(
    symbol: str,
    timeframe: str = Query(
        "1D", description="Time interval for data (1D, 1H, 15Min, etc.)"
    ),
    days: int = Query(30, description="Number of days of historical data"),
    data_agent: DataAgent = Depends(get_data_agent),
):
    """Get historical price data for a stock."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        stock_data = await data_agent.get_stock_data(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
        )

        # Convert pandas DataFrame to dict for JSON serialization
        if stock_data and hasattr(stock_data, "data") and not stock_data.data.empty:
            # Reset index to include timestamp in the records
            df = stock_data.data.reset_index()

            # Convert DataFrame to records
            records = df.to_dict(orient="records")

            # Ensure dates are properly formatted
            for record in records:
                for key, value in record.items():
                    if isinstance(value, (pd.Timestamp, datetime)):
                        record[key] = value.isoformat()

            response_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "data": records,
            }
        else:
            response_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "data": [],
            }

        # Clean the response data to ensure it's serializable
        return clean_response(response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch stock data: {str(e)}"
        )


@router.get("/{symbol}/quote")
async def get_stock_quote(symbol: str, data_agent: DataAgent = Depends(get_data_agent)):
    """Get the latest quote for a stock."""
    try:
        price = await data_agent.get_latest_price(symbol)
        timestamp = datetime.now()
        response_data = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp.isoformat(),
        }
        return clean_response(response_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch stock quote: {str(e)}"
        )


@router.get("/{symbol}/news")
async def get_stock_news(
    symbol: str,
    limit: int = Query(5, description="Number of news articles to return"),
    data_agent: DataAgent = Depends(get_data_agent),
):
    """Get recent news for a stock."""
    try:
        news = await data_agent.get_company_news(symbol, limit)
        return {"symbol": symbol, "news": news}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")


@router.get("/market/news")
async def get_market_news(
    limit: int = Query(5, description="Number of news articles to return"),
    data_agent: DataAgent = Depends(get_data_agent),
):
    """Get general market news."""
    try:
        news = await data_agent.get_market_news(limit)
        return {"news": news}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch market news: {str(e)}"
        )


@router.get("/batch")
async def get_multiple_stocks(
    symbols: str = Query(..., description="Comma-separated list of stock symbols"),
    timeframe: str = Query("1D", description="Time interval for data"),
    days: int = Query(30, description="Number of days of historical data"),
    data_agent: DataAgent = Depends(get_data_agent),
):
    """Get data for multiple stocks in a single request."""
    symbol_list = [s.strip() for s in symbols.split(",")]

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        results = {}
        for symbol in symbol_list:
            stock_data = await data_agent.get_stock_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if stock_data and hasattr(stock_data, "data") and not stock_data.data.empty:
                # Reset index to include timestamp in the records
                df = stock_data.data.reset_index()

                # Convert to records with proper date formatting
                records = df.to_dict(orient="records")

                # Format date/time objects
                for record in records:
                    for key, value in record.items():
                        if isinstance(value, (pd.Timestamp, datetime)):
                            record[key] = value.isoformat()

                results[symbol] = {"timeframe": timeframe, "data": records}
            else:
                results[symbol] = {"timeframe": timeframe, "data": []}

        response_data = {
            "symbols": symbol_list,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "results": results,
        }

        # Clean the response data to ensure it's serializable
        return clean_response(response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch batch stock data: {str(e)}"
        )
