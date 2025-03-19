# app/routes/strategy.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Body

from app.core.dependencies import get_data_agent, get_analysis_agent, get_strategy_agent
from app.agents.data_agent import DataAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.strategy_agent import StrategyAgent
from app.models.stock import TradingRecommendation

router = APIRouter(prefix="/strategy", tags=["Strategy"])


@router.get("/{symbol}/recommendation")
async def get_trading_recommendation(
    symbol: str,
    risk_profile: str = Query(
        "moderate", description="Risk profile: conservative, moderate, or aggressive"
    ),
    investment_horizon: str = Query(
        "medium", description="Investment horizon: short, medium, or long"
    ),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
    strategy_agent: StrategyAgent = Depends(get_strategy_agent),
):
    """Get a trading recommendation for a stock."""
    try:
        # Determine timeframe based on investment horizon
        timeframe_map = {"short": "1H", "medium": "1D", "long": "1W"}
        timeframe = timeframe_map.get(investment_horizon, "1D")

        # Determine date range
        end_date = datetime.now()

        if investment_horizon == "short":
            start_date = end_date - timedelta(days=5)
        elif investment_horizon == "medium":
            start_date = end_date - timedelta(days=90)
        else:  # long
            start_date = end_date - timedelta(days=365)

        # Fetch stock data
        stock_data = await data_agent.get_stock_data(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
        )

        if not stock_data or not hasattr(stock_data, "data") or stock_data.data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol {symbol}"
            )

        # Perform technical analysis
        stock_analysis = await analysis_agent.analyze_stock_trend(stock_data)

        # Fetch news and perform sentiment analysis
        news = await data_agent.get_company_news(symbol, limit=10)
        sentiment_analysis = await analysis_agent.analyze_news_sentiment(news)

        # Generate recommendation
        recommendation = await strategy_agent.generate_recommendation(
            symbol=symbol,
            stock_analysis=stock_analysis,
            sentiment_analysis=sentiment_analysis,
            risk_profile=risk_profile,
            investment_horizon=investment_horizon,
        )

        return recommendation.dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate recommendation: {str(e)}"
        )


@router.post("/portfolio")
async def get_portfolio_recommendations(
    portfolio: Dict[str, float] = Body(
        ..., description="Portfolio holdings as symbol:weight mapping"
    ),
    risk_profile: str = Query(
        "moderate", description="Risk profile: conservative, moderate, or aggressive"
    ),
    cash_available: float = Query(
        0.0, description="Amount of cash available for new investments"
    ),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
    strategy_agent: StrategyAgent = Depends(get_strategy_agent),
):
    """Get recommendations for a portfolio of stocks."""
    try:
        if not portfolio:
            raise HTTPException(status_code=400, detail="Portfolio cannot be empty")

        # Fetch data and analyze all stocks in the portfolio
        analyses = {}
        for symbol in portfolio.keys():
            # Fetch stock data
            stock_data = await data_agent.get_stock_data(
                symbol=symbol,
                timeframe="1D",
                start_date=datetime.now() - timedelta(days=90),
            )

            if (
                not stock_data
                or not hasattr(stock_data, "data")
                or stock_data.data.empty
            ):
                continue

            # Perform technical analysis
            stock_analysis = await analysis_agent.analyze_stock_trend(stock_data)

            # Fetch news and perform sentiment analysis
            news = await data_agent.get_company_news(symbol, limit=5)
            sentiment_analysis = await analysis_agent.analyze_news_sentiment(news)

            analyses[symbol] = {
                "stock_analysis": stock_analysis,
                "sentiment_analysis": sentiment_analysis,
            }

        # Generate portfolio recommendations
        recommendations = await strategy_agent.generate_portfolio_recommendations(
            portfolio=portfolio,
            analyses=analyses,
            risk_profile=risk_profile,
            cash_available=cash_available,
        )

        # Convert recommendations to dictionaries for response
        recommendation_dicts = [rec.dict() for rec in recommendations]

        return {
            "portfolio": portfolio,
            "recommendations": recommendation_dicts,
            "timestamp": datetime.now(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate portfolio recommendations: {str(e)}",
        )


@router.get("/opportunities")
async def get_investment_opportunities(
    excluded_symbols: str = Query(
        "", description="Comma-separated list of symbols to exclude"
    ),
    risk_profile: str = Query(
        "moderate", description="Risk profile: conservative, moderate, or aggressive"
    ),
    max_results: int = Query(
        5, description="Maximum number of opportunities to return"
    ),
    universe: str = Query(
        "SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,BRK.B",
        description="Comma-separated list of symbols to analyze for opportunities",
    ),
    data_agent: DataAgent = Depends(get_data_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
    strategy_agent: StrategyAgent = Depends(get_strategy_agent),
):
    """Find new investment opportunities."""
    try:
        excluded_list = (
            [s.strip() for s in excluded_symbols.split(",")] if excluded_symbols else []
        )
        universe_list = [s.strip() for s in universe.split(",")]

        # Fetch data and analyze stocks in the universe
        analyses = {}
        for symbol in universe_list:
            if symbol in excluded_list:
                continue

            # Fetch stock data
            stock_data = await data_agent.get_stock_data(
                symbol=symbol,
                timeframe="1D",
                start_date=datetime.now() - timedelta(days=90),
            )

            if (
                not stock_data
                or not hasattr(stock_data, "data")
                or stock_data.data.empty
            ):
                continue

            # Perform technical analysis
            stock_analysis = await analysis_agent.analyze_stock_trend(stock_data)

            # Fetch news and perform sentiment analysis
            news = await data_agent.get_company_news(symbol, limit=5)
            sentiment_analysis = await analysis_agent.analyze_news_sentiment(news)

            analyses[symbol] = {
                "stock_analysis": stock_analysis,
                "sentiment_analysis": sentiment_analysis,
            }

        # Find investment opportunities
        opportunities = await strategy_agent.find_investment_opportunities(
            analyses=analyses,
            excluded_symbols=excluded_list,
            risk_profile=risk_profile,
            max_results=max_results,
        )

        # Convert opportunities to dictionaries for response
        opportunity_dicts = [opp.dict() for opp in opportunities]

        return {
            "opportunities": opportunity_dicts,
            "count": len(opportunity_dicts),
            "timestamp": datetime.now(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to find investment opportunities: {str(e)}"
        )
