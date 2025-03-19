# app/routes/analysis.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import get_data_agent, get_analysis_agent
from app.agents.data_agent import DataAgent
from app.agents.analysis_agent import AnalysisAgent
from app.models.stock import StockAnalysis, SentimentAnalysis

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.get("/{symbol}/technical")
async def get_technical_analysis(
    symbol: str,
    timeframe: str = Query(
        "1D", description="Time interval for analysis (1D, 1H, etc.)"
    ),
    days: int = Query(90, description="Number of days of historical data to analyze"),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
):
    """Get technical analysis for a stock."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # No need to format dates here as the data_agent will handle it
        stock_data = await data_agent.get_stock_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if not stock_data or not hasattr(stock_data, "data") or stock_data.data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol {symbol}"
            )

        analysis = await analysis_agent.analyze_stock_trend(stock_data)

        # Convert the Pydantic model to dict for response
        return analysis.dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to perform technical analysis: {str(e)}"
        )


@router.get("/{symbol}/sentiment")
async def get_sentiment_analysis(
    symbol: str,
    limit: int = Query(10, description="Number of news articles to analyze"),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
):
    """Get sentiment analysis of recent news for a stock."""
    try:
        news = await data_agent.get_company_news(symbol, limit)

        if not news:
            return {
                "symbol": symbol,
                "overall_sentiment": "neutral",
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "summary": "No recent news found for analysis.",
            }

        sentiment = await analysis_agent.analyze_news_sentiment(news)

        # Combine the results
        result = sentiment.dict()
        result["symbol"] = symbol
        result["news_count"] = len(news)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to perform sentiment analysis: {str(e)}"
        )


@router.get("/{symbol}/complete")
async def get_complete_analysis(
    symbol: str,
    timeframe: str = Query("1D", description="Time interval for analysis"),
    days: int = Query(90, description="Number of days of historical data to analyze"),
    news_limit: int = Query(10, description="Number of news articles to analyze"),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
):
    """Get both technical and sentiment analysis for a stock."""
    try:
        # Fetch stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # No need to format dates here as the data_agent will handle it
        stock_data = await data_agent.get_stock_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if not stock_data or not hasattr(stock_data, "data") or stock_data.data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol {symbol}"
            )

        # Fetch news data
        news = await data_agent.get_company_news(symbol, news_limit)

        # Perform technical analysis
        technical_analysis = await analysis_agent.analyze_stock_trend(stock_data)

        # Perform sentiment analysis
        sentiment_analysis = await analysis_agent.analyze_news_sentiment(news)

        # Combine the results
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "technical_analysis": technical_analysis.dict(),
            "sentiment_analysis": sentiment_analysis.dict(),
            "news_count": len(news),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to perform complete analysis: {str(e)}"
        )


@router.get("/batch")
async def get_batch_analysis(
    symbols: str = Query(..., description="Comma-separated list of stock symbols"),
    analysis_type: str = Query(
        "technical", description="Type of analysis: technical, sentiment, or complete"
    ),
    timeframe: str = Query("1D", description="Time interval for analysis"),
    days: int = Query(90, description="Number of days of historical data to analyze"),
    news_limit: int = Query(
        5, description="Number of news articles to analyze per symbol"
    ),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
):
    """Get analysis for multiple stocks in a single request."""
    symbol_list = [s.strip() for s in symbols.split(",")]

    try:
        results = {}

        # Determine the end and start dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for symbol in symbol_list:
            symbol_result = {"symbol": symbol}

            # Fetch stock data for technical analysis
            if analysis_type in ["technical", "complete"]:
                stock_data = await data_agent.get_stock_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )

                if (
                    stock_data
                    and hasattr(stock_data, "data")
                    and not stock_data.data.empty
                ):
                    technical_analysis = await analysis_agent.analyze_stock_trend(
                        stock_data
                    )
                    symbol_result["technical_analysis"] = technical_analysis.dict()
                else:
                    symbol_result["technical_analysis"] = None

            # Fetch news data for sentiment analysis
            if analysis_type in ["sentiment", "complete"]:
                news = await data_agent.get_company_news(symbol, news_limit)
                sentiment_analysis = await analysis_agent.analyze_news_sentiment(news)
                symbol_result["sentiment_analysis"] = sentiment_analysis.dict()
                symbol_result["news_count"] = len(news)

            results[symbol] = symbol_result

        return {
            "symbols": symbol_list,
            "analysis_type": analysis_type,
            "timeframe": (
                timeframe if analysis_type in ["technical", "complete"] else None
            ),
            "timestamp": datetime.now(),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to perform batch analysis: {str(e)}"
        )
